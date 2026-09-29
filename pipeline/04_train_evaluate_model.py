#!/usr/bin/env python3
"""
Leaf-presence model: L2-penalised logistic regression on DINOv2 features (section 3.c, Fig. 4).

  - positive class = leaf absence (photo-interpretation label "D"; "L" and "R" = leaves present);
    leaf-presence probability = 1 - p(leaf absence)
  - no class weighting, no post-hoc calibration (natural prevalence, 31 % leaf-absent)
  - C = 0.01 (selected by nested cross-validation, 03_select_C_nested_cv.py)
  - 5-fold cross-validation grouped by acquisition date (GroupKFold): an entire date is
    either in training or in testing
  - metrics on held-out folds: AUC-ROC, AUC-PR, macro-F1, Brier, confusion matrices at 0.5
    (most-likely class) and at the class prevalence of the training fold
  - final model fitted on all points of Luki and Yangambi

Input : data/model/photointerpretation_points_features.csv.gz
        (one row per photo-interpreted point and date: site, date, label, label_raw,
         crown_id, feat_0000 ... feat_0383)
Output: results/model/Fig4_model_performance.png, validation_metrics.csv
        results/model/leaf_presence_logreg.joblib (dict: model, scaler, feature_cols, ...)
        (the model used in the paper is shipped in model/leaf_presence_logreg.joblib)
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, confusion_matrix,
                             brier_score_loss, roc_curve, precision_score, recall_score,
                             accuracy_score, cohen_kappa_score)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg
warnings.filterwarnings("ignore")

OUT = cfg.RESULTS_DIR / "model"
N_SPLITS = 5
C_VALUE = 0.01
RANDOM_STATE = 42
CLASSES = np.array([0, 1])
LABEL_NAMES = {0: "L", 1: "D"}
LABEL_LONG = {0: "leaf-present", 1: "leaf-absent"}
POS_LABEL = 1
THRESHOLD_MODE = "0.5"            # main threshold: most-likely class
THRESHOLD_COMPARE = "prevalence"  # second threshold: class prevalence in the training fold


# ----------------------------------------------------------------------------- model
def fit_predict(df_tr, df_te, feature_cols, C=C_VALUE):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(df_tr[feature_cols].values)
    X_te = scaler.transform(df_te[feature_cols].values)
    glm = LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE)
    glm.fit(X_tr, df_tr["label"].values)
    return glm.predict_proba(X_te)[:, 1], glm.predict_proba(X_tr)[:, 1], glm, scaler


def choose_threshold(y_tr, mode):
    """Decision threshold, set on the training fold only."""
    return 0.5 if mode == "0.5" else float((y_tr == POS_LABEL).mean())


def metrics_summary(y_true, y_proba, label, threshold, threshold_alt, extra=None):
    y_pred = (y_proba >= threshold).astype(int)
    y_alt = (y_proba >= threshold_alt).astype(int)
    m = dict(label=label, n_test=len(y_true), prevalence=float((y_true == POS_LABEL).mean()),
             threshold=float(threshold), threshold_alt=float(threshold_alt),
             auc_roc=roc_auc_score(y_true, y_proba), pr_auc=average_precision_score(y_true, y_proba),
             f1_macro=f1_score(y_true, y_pred, average="macro"), brier=brier_score_loss(y_true, y_proba),
             cm=confusion_matrix(y_true, y_pred, labels=CLASSES),
             cm_alt=confusion_matrix(y_true, y_alt, labels=CLASSES),
             y_true=y_true, y_proba=y_proba, y_pred=y_pred, y_pred_alt=y_alt)
    if extra:
        m.update(extra)
    return m


def cross_validate(df, feature_cols, context):
    d = df.reset_index(drop=True)
    folds = []
    n_groups = d["date"].nunique()
    k = int(min(N_SPLITS, n_groups))
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=k).split(np.arange(len(d)), d["label"].values,
                                                                 d["date"].values), 1):
        df_tr, df_te = d.iloc[tr], d.iloc[te]
        if df_tr["label"].nunique() < 2 or df_te["label"].nunique() < 2:
            continue
        y_proba, _, _, _ = fit_predict(df_tr, df_te, feature_cols)
        y_tr = df_tr["label"].values
        m = metrics_summary(df_te["label"].values, y_proba, f"Fold {fold}",
                            choose_threshold(y_tr, THRESHOLD_MODE), choose_threshold(y_tr, THRESHOLD_COMPARE),
                            extra={"context": context, "n_train": len(df_tr)})
        folds.append(m)
        print(f"  [{context}] fold {fold}: n_test={m['n_test']:,} AUC={m['auc_roc']:.4f} "
              f"PR-AUC={m['pr_auc']:.4f} F1={m['f1_macro']:.4f} Brier={m['brier']:.4f}")
    return folds


def calibration_by_fold(folds, n_bins=10, min_n=20):
    edges = np.linspace(0, 1, n_bins + 1)
    pred = np.full((len(folds), n_bins), np.nan)
    obs = np.full((len(folds), n_bins), np.nan)
    for i, m in enumerate(folds):
        y, p = m["y_true"], m["y_proba"]
        b = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
        for k in range(n_bins):
            sel = b == k
            if sel.sum() >= min_n:
                pred[i, k] = p[sel].mean()
                obs[i, k] = (y[sel] == POS_LABEL).mean()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmean(pred, 0), np.nanmean(obs, 0), np.nanstd(obs, 0)


# ----------------------------------------------------------------------------- figure 4
def plot_validation(results, sites, path):
    """(a) calibration, (b) ROC, (c) confusion matrices; all values from held-out folds,
    +/- = standard deviation across folds."""
    COL_DIAG = "#aaaaaa"
    COLS = {"Global": "#c0392b", "Luki": "#1a7a4a", "Yangambi": "#154360"}
    STYLES = {"Global": "-", "Luki": "--", "Yangambi": ":"}
    contexts = [(c, results[c]) for c in ["Global"] + sites if results.get(c)]

    fig = plt.figure(figsize=(17.5, 7.0))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.28, width_ratios=[1.0, 1.0, 1.02])

    ax = fig.add_subplot(gs[0, 0])                                      # (a) calibration
    ax.plot([0, 1], [0, 1], "--", color=COL_DIAG, lw=1.2, alpha=0.7, label="Perfect calibration", zorder=1)
    for ctx, folds in contexts:
        pp, pt, sd = calibration_by_fold(folds)
        ok = ~np.isnan(pp)
        ax.errorbar(pp[ok], pt[ok], yerr=sd[ok], fmt="o" + STYLES.get(ctx, "-"), color=COLS.get(ctx, "#555"),
                    lw=2, markersize=5, capsize=2.5, elinewidth=1.0, zorder=3, label=ctx)
    gm = results["Global"]
    y_g0 = np.concatenate([m["y_true"] for m in gm])
    mp = [m["y_proba"].mean() for m in gm]
    of = [(m["y_true"] == POS_LABEL).mean() for m in gm]
    ax.text(0.97, 0.06, f"Mean predicted     {np.mean(mp):.3f} +/- {np.std(mp):.3f}\n"
                        f"Observed frequency {np.mean(of):.3f} +/- {np.std(of):.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, family="monospace", color="#2c3e50",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fafafa", ec="#dddddd", lw=0.6))
    ax.set_title(f"(a) Calibration — {LABEL_LONG[POS_LABEL]} ({LABEL_NAMES[POS_LABEL]})", fontsize=10, fontweight="bold")
    ax.set_xlabel("Mean predicted probability", fontsize=9); ax.set_ylabel("Observed fraction of positives", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=8, loc="upper left", framealpha=0.9); ax.grid(True, alpha=0.2)

    ax = fig.add_subplot(gs[0, 1])                                      # (b) ROC
    mean_fpr = np.linspace(0, 1, 200)
    ax.plot([0, 1], [0, 1], "--", color=COL_DIAG, lw=1.2, alpha=0.7)
    for ctx, folds in contexts:
        tprs = []
        for m in folds:
            fpr, tpr, _ = roc_curve(m["y_true"], m["y_proba"], pos_label=POS_LABEL)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
        aucs = [m["auc_roc"] for m in folds]; prs = [m["pr_auc"] for m in folds]
        mt, st = np.mean(tprs, axis=0), np.std(tprs, axis=0)
        ax.plot(mean_fpr, mt, color=COLS.get(ctx, "#555"), lw=2.2, ls=STYLES.get(ctx, "-"),
                label=f"{ctx:<8s} {np.mean(aucs):.3f}+/-{np.std(aucs):.3f}  {np.mean(prs):.3f}+/-{np.std(prs):.3f}")
        ax.fill_between(mean_fpr, np.clip(mt - st, 0, 1), np.clip(mt + st, 0, 1), alpha=0.10, color=COLS.get(ctx, "#555"))
    prev0 = float((y_g0 == POS_LABEL).mean())
    leg = ax.legend(loc="lower right", framealpha=0.95, prop={"family": "monospace", "size": 7},
                    title=f"{'':<8s}   AUC-ROC        PR-AUC\n{'no skill':<8s}   0.500          {prev0:.3f}")
    leg.get_title().set_fontsize(7); leg.get_title().set_fontfamily("monospace"); leg.get_title().set_color("#777777")
    ax.set_title("(b) Discrimination — ROC curves", fontsize=10, fontweight="bold")
    ax.set_xlabel("False positive rate", fontsize=9); ax.set_ylabel("True positive rate", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(True, alpha=0.2)

    y_g = np.concatenate([m["y_true"] for m in gm]); pred_g = np.concatenate([m["y_pred"] for m in gm])
    cm_abs = sum(m["cm"] for m in gm)                                   # encoding guard
    assert np.isclose(cm_abs[POS_LABEL, POS_LABEL] / cm_abs[POS_LABEL, :].sum(),
                      recall_score(y_g, pred_g, pos_label=POS_LABEL), atol=1e-3)
    gs_cm = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 2], hspace=0.72)
    ticks = [LABEL_NAMES[c] for c in CLASSES]
    for i_p, (k_cm, k_pred, k_thr, rule) in enumerate([("cm", "y_pred", "threshold", "most-likely class"),
                                                       ("cm_alt", "y_pred_alt", "threshold_alt", "class prevalence")]):
        cms = np.array([m[k_cm] / m[k_cm].sum(axis=1, keepdims=True) for m in gm])
        mu, sd = cms.mean(axis=0), cms.std(axis=0)
        tot = sum(m[k_cm] for m in gm)
        thr = np.mean([m[k_thr] for m in gm])
        accs = [accuracy_score(m["y_true"], m[k_pred]) for m in gm]
        kaps = [cohen_kappa_score(m["y_true"], m[k_pred]) for m in gm]
        p0 = [precision_score(m["y_true"], m[k_pred], pos_label=0) for m in gm]
        p1 = [precision_score(m["y_true"], m[k_pred], pos_label=1) for m in gm]
        ax = fig.add_subplot(gs_cm[i_p, 0])                             # (c) confusion matrices
        ax.imshow(mu, cmap="YlOrBr", vmin=0, vmax=1, interpolation="nearest")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{mu[i, j]*100:.1f}±{sd[i, j]*100:.1f}%\n(n={tot[i, j]:,})", ha="center", va="center",
                        color="white" if mu[i, j] > 0.55 else "black", fontsize=8.6, fontweight="bold")
        ax.set_xticks(range(2)); ax.set_xticklabels(ticks, fontsize=9.5)
        ax.set_yticks(range(2)); ax.set_yticklabels(ticks, fontsize=9.5, rotation=90, va="center")
        ax.set_xlabel("Predicted", fontsize=8.5, labelpad=1); ax.set_ylabel("True", fontsize=8.5, labelpad=1)
        ax.set_title(("(c) Confusion matrices\n\n" if i_p == 0 else "") + f"threshold {thr:.2f} — {rule}",
                     fontsize=9.5, fontweight="bold" if i_p == 0 else "normal")
        ax.text(0.5, -0.32, f"Accuracy {np.mean(accs):.3f}+/-{np.std(accs):.3f}   kappa {np.mean(kaps):.3f}+/-{np.std(kaps):.3f}\n"
                            f"Precision  {LABEL_NAMES[0]} {np.mean(p0):.3f}+/-{np.std(p0):.3f}   "
                            f"{LABEL_NAMES[1]} {np.mean(p1):.3f}+/-{np.std(p1):.3f}",
                transform=ax.transAxes, ha="center", va="top", fontsize=7.2, family="monospace", color="#2c3e50")
    plt.suptitle("L2 logistic regression (no class weighting) — GroupKFold by acquisition date (unseen date)"
                 f"\nall values from held-out folds; +/- = standard deviation across the {len(gm)} folds",
                 fontsize=11.5, fontweight="bold", y=1.06)
    plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()


# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(cfg.POINT_FEATURES)
    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    df = df[~df[feature_cols].isna().any(axis=1)].reset_index(drop=True)
    assert "D" in set(df.loc[df["label"] == POS_LABEL, "label_raw"])
    print(f"{len(df):,} points ({df.site.value_counts().to_dict()}), {len(feature_cols)} features, "
          f"prevalence of leaf absence {df.label.mean():.4f}")

    sites = sorted(df["site"].unique())
    results = {"Global": cross_validate(df, feature_cols, "Global")}
    for site in sites:
        results[site] = cross_validate(df[df["site"] == site].reset_index(drop=True), feature_cols, site)

    rows = [dict(context=ctx, fold=m["label"], n_test=m["n_test"], prevalence=m["prevalence"],
                 auc_roc=m["auc_roc"], pr_auc=m["pr_auc"], f1_macro=m["f1_macro"], brier=m["brier"])
            for ctx, folds in results.items() for m in folds]
    met = pd.DataFrame(rows); met.to_csv(OUT / "validation_metrics.csv", index=False)
    print(met.groupby("context")[["auc_roc", "pr_auc", "f1_macro", "brier"]].mean().round(4))
    plot_validation(results, sites, OUT / "Fig4_model_performance.png")

    # final model on all points
    scaler = StandardScaler(); X = scaler.fit_transform(df[feature_cols].values)
    glm = LogisticRegression(penalty="l2", C=C_VALUE, solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE)
    glm.fit(X, df["label"].values)
    joblib.dump({"model": glm, "scaler": scaler, "feature_cols": feature_cols, "label_names": LABEL_NAMES,
                 "label_long": LABEL_LONG, "pos_label": POS_LABEL, "C": C_VALUE, "class_weight": None,
                 "threshold": 0.5, "threshold_mode": THRESHOLD_MODE, "prevalence": float(df["label"].mean())},
                OUT / "leaf_presence_logreg.joblib")
    print("->", OUT)

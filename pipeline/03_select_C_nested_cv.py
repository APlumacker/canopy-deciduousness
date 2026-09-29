#!/usr/bin/env python3
"""
Selection of the regularisation strength C of the L2 logistic regression by nested
cross-validation (section 3.c).

Outer loop: 5-fold GroupKFold by acquisition date, used only to report performance.
Inner loop: 3-fold GroupKFold by date on the outer training set, used only to choose C
(grid 1e-3 ... 1e1, selection on the area under the precision-recall curve).
The C most often selected is used in 04_train_evaluate_model.py (C = 0.01).

Input : data/model/photointerpretation_points_features.csv.gz
Output: results/model/nested_cv.csv
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg
warnings.filterwarnings("ignore")

OUT = cfg.RESULTS_DIR / "model"
C_GRID = np.logspace(-3, 1, 5)
N_OUTER, N_INNER = 5, 3
SELECT_METRIC = "pr_auc"
RANDOM_STATE = 42


def date_splits(d, k):
    k = int(min(k, d["date"].nunique()))
    yield from GroupKFold(n_splits=k).split(np.arange(len(d)), d["label"].values, d["date"].values)


def fit_eval(df_tr, df_te, feature_cols, C):
    sc = StandardScaler()
    X_tr = sc.fit_transform(df_tr[feature_cols].values); X_te = sc.transform(df_te[feature_cols].values)
    clf = LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=2000, random_state=RANDOM_STATE)
    clf.fit(X_tr, df_tr["label"].values)
    p = clf.predict_proba(X_te)[:, 1]; y = df_te["label"].values
    return {"auc_roc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p), "brier": brier_score_loss(y, p)}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(cfg.POINT_FEATURES)
    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    df = df[~df[feature_cols].isna().any(axis=1)].reset_index(drop=True)
    rows = []
    for fold, (tr, te) in enumerate(date_splits(df, N_OUTER), 1):
        df_tr, df_te = df.iloc[tr].reset_index(drop=True), df.iloc[te]
        scores = {}
        for C in C_GRID:
            vals = [fit_eval(df_tr.iloc[a], df_tr.iloc[b], feature_cols, C)[SELECT_METRIC]
                    for a, b in date_splits(df_tr, N_INNER)]
            scores[C] = float(np.mean(vals))
        C_best = max(scores, key=scores.get)
        r = fit_eval(df_tr, df_te, feature_cols, C_best)
        r.update(fold=fold, C_selected=C_best, inner_score=scores[C_best], n_train=len(df_tr), n_test=len(df_te))
        rows.append(r)
        print(f"fold {fold}: C={C_best:.0e}  AUC={r['auc_roc']:.4f}  PR-AUC={r['pr_auc']:.4f}")
    res = pd.DataFrame(rows); res.to_csv(OUT / "nested_cv.csv", index=False)
    print(f"C selected: {res.C_selected.value_counts().to_dict()} | "
          f"AUC {res.auc_roc.mean():.4f} +/- {res.auc_roc.std():.4f}")

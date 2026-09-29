#!/usr/bin/env python3
"""
Fig. 6 and Table S2: seasonal course of leaf loss and water availability (section 3.g).

For the crowns with strong leaf loss (magnitude >= 0.6) at each site, the mean crown leaf
presence at each flight date is set against:
  - 30-day rainfall (IMERG V07, cumulative over the 30 days preceding each date),
  - consecutive dry days (CDD, daily rainfall < 2 mm),
  - vapour pressure deficit (NASA POWER / MERRA-2, 7-day centred mean).
Spearman correlations at flight dates are descriptive only (no p-value: serial correlation).

Inputs : results/crown_metrics_all.csv (step 1), data/crowns/crown_timeseries_all.csv.gz,
         data/climate/climate_daily_<site>.csv
Outputs: results/Fig6_climate.png/.pdf
         results/fig6_correlations.csv          (rho at magnitude >= 0.6, shown in Fig. 6)
         results/tableS2_threshold_sensitivity.csv (thresholds 0.4-0.8)
         results/fig6_share_below_half.csv      (share of crowns simultaneously below 0.5)
"""
import sys, warnings
from datetime import timedelta
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg
warnings.filterwarnings("ignore")

OUT = cfg.RESULTS_DIR
GAP_DAYS = 35          # beyond this gap between flights, the mean is only interpolated (dashed)
WINDOW_DAYS = 30       # rainfall window
CDD_DRY_MM = 2.0       # daily rainfall below which a day is dry
AMP_THRESH = cfg.STRONG_LOSS
THRESHOLDS_S2 = [0.4, 0.5, 0.6, 0.7, 0.8]
ORDER = ["Luki", "Mbalmayo", "Yangambi"]
COLORS = {
    "Luki": {"main": "#08519C", "light": "#6BAED6", "right": "#5B6770"},
    "Yangambi": {"main": "#41B6E6", "light": "#9FDBF3", "right": "#5B6770"},
    "Mbalmayo": {"main": "#CC79A7", "light": "#E9B8D2", "right": "#5B6770"},
}
FIG_W_CM, FIG_H_CM = 17.0, 20.0
FSA = 0.78             # typographic scale factor


# ----------------------------------------------------------------------------- climate
def read_climate(site):
    df = pd.read_csv(cfg.CLIMATE[site], parse_dates=["date"])
    return df.rename(columns={"precip_imerg_mm": "precip_mm", "vpd_nasapower_kpa": "vpd_kpa"})


def load_p30(site, dates_str):
    """30-day rainfall preceding each flight date: {YYYYMMDD: mm}."""
    df = read_climate(site)
    df["date"] = df["date"].dt.date
    df = df[df["precip_mm"] >= 0].sort_values("date").reset_index(drop=True)
    precip = {row.date: float(row.precip_mm) for _, row in df.iterrows()}
    out = {}
    for ds in dates_str:
        dt = pd.to_datetime(ds, format="%Y%m%d").date()
        vals = [precip[d] for d in (dt - timedelta(days=k) for k in range(WINDOW_DAYS - 1, -1, -1)) if d in precip]
        out[ds] = float(sum(vals)) if vals else np.nan
    return out


def load_climate_daily(site):
    """Daily series: rainfall, CDD, continuous 30-day rainfall, 7-day VPD."""
    df = read_climate(site)
    df = df[df["precip_mm"] >= 0].sort_values("date").reset_index(drop=True)
    cdd = np.zeros(len(df), dtype=int); cnt = 0
    for i, p in enumerate(df["precip_mm"].values):
        cnt = cnt + 1 if p < CDD_DRY_MM else 0
        cdd[i] = cnt
    df["cdd"] = cdd
    df["p30_cont"] = df["precip_mm"].rolling(window=WINDOW_DAYS, min_periods=15).sum()
    df["vpd_smooth"] = df["vpd_kpa"].rolling(7, min_periods=3, center=True).mean()
    return df.set_index("date")[["precip_mm", "cdd", "p30_cont", "vpd_smooth"]]


# ----------------------------------------------------------------------------- panel
def build_panel(ts, metrics, thresh):
    strong = metrics[metrics.magnitude >= thresh][["site", "crown_id"]]
    d = ts.merge(strong, on=["site", "crown_id"], how="inner")
    panels = []
    for site in ORDER:
        s = d[d.site == site].copy()
        if s.empty:
            continue
        s["p30"] = s["date"].map(load_p30(site, sorted(s["date"].unique())))
        panels.append(s)
    p = pd.concat(panels, ignore_index=True)
    p["date_dt"] = pd.to_datetime(p["date"], format="%Y%m%d")
    return p


def site_series(sub, clim):
    """Mean, quartiles across crowns at each flight date, and descriptive correlations."""
    pivot = sub.pivot_table(index="crown_id", columns="date_dt", values="q90", aggfunc="mean").sort_index(axis=1)
    dates = pivot.columns.tolist()
    mean = np.nanmean(pivot.values, axis=0)
    rho = {}
    p30 = sub.groupby("date_dt")["p30"].first()
    m = pd.DataFrame({"mean": mean, "p30": p30.reindex(dates).values}).dropna()
    if len(m) >= 3:
        rho["P30"] = (spearmanr(m["mean"], m["p30"])[0], len(m))
    for drv, lab in [("vpd_smooth", "VPD"), ("cdd", "CDD")]:
        t = pd.DataFrame({"mean": mean, "d": clim[drv].reindex(dates).values}, index=dates).dropna()
        if len(t) >= 5:
            rho[lab] = (spearmanr(t["mean"], t["d"])[0], len(t))
    return pivot, dates, mean, rho


# ----------------------------------------------------------------------------- figure
def draw_site(ax, sub, site, clim, show_xlabel):
    col_main, col_light, col_right = COLORS[site]["main"], COLORS[site]["light"], COLORS[site]["right"]
    axr = ax.twinx()
    pivot, dates_c, med, rho = site_series(sub, clim)
    n_crowns = pivot.shape[0]
    p25 = np.nanpercentile(pivot.values, 25, axis=0)
    p75 = np.nanpercentile(pivot.values, 75, axis=0)

    def _norm_inv(x):
        mn, mx = x.min(), x.max()
        return x * 0 if mx - mn < 1e-6 else 1.0 - (x - mn) / (mx - mn)

    p30 = clim[["p30_cont"]].dropna()
    axr.plot(p30.index, p30["p30_cont"], color=col_right, lw=0.9, ls="-", alpha=0.8, zorder=3)
    axr.set_ylabel("30-day rainfall (mm)", fontsize=8 * FSA, color=col_right)
    axr.tick_params(axis="y", colors=col_right, labelsize=7 * FSA)
    if clim["vpd_smooth"].notna().sum() > 30:
        v = _norm_inv(clim["vpd_smooth"].dropna())
        ax.plot(v.index, v.values, color="#8B4513", lw=0.9, ls="-.", alpha=0.7, zorder=3)
    if clim["cdd"].notna().sum() > 30:
        c = _norm_inv(clim["cdd"].dropna()).rolling(7, min_periods=3, center=True).mean()
        ax.plot(c.index, c.values, color="#e67e22", lw=0.9, ls=":", alpha=0.7, zorder=3)

    # individual crowns, interrupted across gaps > GAP_DAYS
    a_line = max(0.006, min(0.03, 1.5 / n_crowns))
    for _, row in pivot.iterrows():
        v = row.values.astype(float); ok = ~np.isnan(v)
        if ok.sum() >= 5:
            vv = v.copy(); vv[np.r_[False, np.diff(np.array(dates_c, dtype="datetime64[D]")).astype(int) > GAP_DAYS]] = np.nan
            vv[~ok] = np.nan
            ax.plot(np.array(dates_c), vv, color=col_main, lw=0.4, alpha=a_line, zorder=2)
    # mean and IQR: solid within observed segments, dashed across gaps
    dts = pd.to_datetime(pd.Series(dates_c)); gap = dts.diff().dt.days.fillna(0).values > GAP_DAYS
    blocks = np.cumsum(gap)
    for b in np.unique(blocks):
        k = np.where(blocks == b)[0]
        dd = [dates_c[i] for i in k]
        ax.fill_between(dd, p25[k], p75[k], color=col_light, alpha=0.35, zorder=4, lw=0)
        ax.plot(dd, med[k], color="white", lw=2.4, zorder=5)
        ax.plot(dd, med[k], color=col_main, lw=1.7, zorder=6)
    for i in np.where(gap)[0]:
        ax.plot([dates_c[i - 1], dates_c[i]], [med[i - 1], med[i]], color=col_main, lw=1.1, ls=(0, (3, 3)), alpha=0.8, zorder=6)
    ax.plot(dates_c, med, ls="", marker="o", ms=2.6, mfc=col_main, mec="white", mew=0.5, zorder=7)
    for d in dates_c:
        ax.axvline(d, color="gray", lw=0.4, ls=":", alpha=0.30, zorder=2)

    lines = []
    if "P30" in rho:
        lines.append(f"$\\rho$(leaf presence, 30-d rainfall) = {rho['P30'][0]:+.2f}")
    for lab in ("VPD", "CDD"):
        if lab in rho:
            lines.append(f"$\\rho$(leaf presence, {lab}) = {rho[lab][0]:+.2f}")
    if lines:
        axr.text(0.012, 0.045, "\n".join(lines), transform=axr.transAxes, fontsize=7.2 * FSA, linespacing=1.5,
                 va="bottom", zorder=50, bbox=dict(boxstyle="round,pad=0.30", fc="white", alpha=1.0, ec=col_main, lw=0.7))

    t0, t1 = min(dates_c) - pd.Timedelta(days=15), max(dates_c) + pd.Timedelta(days=15)
    ax.set_xlim(t0, t1); axr.set_xlim(t0, t1); ax.set_ylim(-0.02, 1.13)
    for k, yr in enumerate(range(t0.year, t1.year + 1)):       # calendar years, alternating bands
        a_, b_ = max(pd.Timestamp(f"{yr}-01-01"), t0), min(pd.Timestamp(f"{yr + 1}-01-01"), t1)
        if b_ <= a_:
            continue
        if k % 2 == 1:
            ax.axvspan(a_, b_, color="#f2f2f2", lw=0, zorder=0)
        if k > 0:
            ax.axvline(pd.Timestamp(f"{yr}-01-01"), color="#9a9a9a", lw=0.8, zorder=0.5)
        if (b_ - a_).days > 40:
            ax.text(a_ + (b_ - a_) / 2, 1.095, str(yr), ha="center", va="center", fontsize=8 * FSA,
                    fontweight="bold", color="#666666", zorder=8)
    ax.set_ylabel("Crown leaf presence", fontsize=8.5 * FSA)
    ax.tick_params(labelsize=7 * FSA)
    ax.set_xlabel("Date" if show_xlabel else "", fontsize=9 * FSA)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha="right")
    ax.grid(True, alpha=0.15)
    return n_crowns


def figure6(panel, climates):
    fig, axes = plt.subplots(len(ORDER), 1, figsize=(FIG_W_CM / 2.54, FIG_H_CM / 2.54))
    for k, (site, ax) in enumerate(zip(ORDER, axes)):
        n = draw_site(ax, panel[panel.site == site], site, climates[site], show_xlabel=(k == len(ORDER) - 1))
        ax.set_title(f"({'abc'[k]}) {site} — n = {n:,} crowns with strong leaf loss (magnitude ≥ {AMP_THRESH})",
                     fontsize=9.5 * FSA, fontweight="bold", loc="left", pad=4)
    h = [plt.Rectangle((0, 0), 1, 1, fc="#999999", alpha=0.45),
         Line2D([0], [0], color="#444444", lw=1.7, marker="o", ms=3),
         Line2D([0], [0], color="#444444", lw=1.1, ls=(0, (3, 3))),
         Line2D([0], [0], color="#5B6770", lw=0.9),
         Line2D([0], [0], color="#8B4513", lw=1.0, ls="-."),
         Line2D([0], [0], color="#e67e22", lw=1.0, ls=":")]
    l = ["IQR 25–75th percentile", "Mean crown leaf presence",
         f"Mean leaf presence interpolated across a gap > {GAP_DAYS} d without acquisition",
         "30-day rainfall (right axis, mm)", "Inverted VPD [0–1]  (↑ = lower VPD)",
         "Inverted CDD [0–1]  (↑ = few dry days)"]
    fig.legend(h, l, fontsize=7.5 * FSA, ncol=3, loc="lower center", framealpha=0.9, bbox_to_anchor=(0.5, -0.035))
    fig.tight_layout(h_pad=1.4)
    for ext, dpi in (("png", 400), ("pdf", None)):
        fig.savefig(OUT / f"Fig6_climate.{ext}", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    ts = pd.read_csv(cfg.TS_ALL, usecols=["site", "crown_id", "date", "q90"])
    ts["date"] = ts["date"].astype(str).str.zfill(8)
    metrics = pd.read_csv(cfg.METRICS_ALL)
    climates = {s: load_climate_daily(s) for s in ORDER}

    rows = []
    for thr in THRESHOLDS_S2:
        panel = build_panel(ts, metrics, thr)
        for site in ORDER:
            sub = panel[panel.site == site]
            pivot, dates, mean, rho = site_series(sub, climates[site])
            rows.append(dict(threshold=thr, site=site, n_crowns=pivot.shape[0],
                             **{f"rho_{k}": v[0] for k, v in rho.items()},
                             **{f"n_dates_{k}": v[1] for k, v in rho.items()}))
    S2 = pd.DataFrame(rows)
    S2.to_csv(OUT / "tableS2_threshold_sensitivity.csv", index=False)
    S2[S2.threshold == AMP_THRESH].to_csv(OUT / "fig6_correlations.csv", index=False)
    print(S2.round(2).to_string(index=False))

    # crown leaf presence below 0.5: maximum share of crowns at one date, and mean share of each
    # crown's dates (text, results iii)
    q = ts.dropna(subset=["q90"])
    strong = metrics[metrics.magnitude >= AMP_THRESH][["site", "crown_id"]]
    rows = []
    for site in ORDER:
        for label, sel in [("strong leaf loss", q.merge(strong, on=["site", "crown_id"])), ("all crowns", q)]:
            s = sel[sel.site == site]
            low = s.assign(low=s.q90 < 0.5)
            frac = low.groupby("date").low.mean()                 # share of crowns below 0.5 at each date
            per_crown = low.groupby("crown_id").low.mean()        # share of each crown's dates below 0.5
            rows.append(dict(site=site, crowns=label, max_share_of_crowns_pct=frac.max() * 100,
                             mean_share_of_dates_per_crown_pct=per_crown.mean() * 100))
    SH = pd.DataFrame(rows); SH.to_csv(OUT / "fig6_share_below_half.csv", index=False)
    print(SH.round(1).to_string(index=False))

    figure6(build_panel(ts, metrics, AMP_THRESH), climates)
    print("->", OUT / "Fig6_climate.png")

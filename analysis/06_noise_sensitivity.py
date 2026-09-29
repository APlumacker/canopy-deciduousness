#!/usr/bin/env python3
"""
Noise level of the magnitude and sensitivity of the results to it (Appendix S1, Fig. S2).

1. Measurement noise from the replicate flights of 22 and 23 August 2025 at Yangambi; the
   magnitude of a leaf-stable crown simulated with this noise has a median of about 0.12,
   the noise level used in the detectability factor g.
2. Sensitivity of duration statistics to the noise magnitude (0.08-0.16).

Inputs : results/crown_metrics_{all,field}.csv, data/crowns/crown_timeseries_all.csv.gz,
         data/crowns/field_crowns.csv
Outputs: results/noise_null_magnitude.csv, sensitivity_noise_magnitude.csv,
         FigS2_noise_sensitivity.png/.pdf
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg

OUT = cfg.RESULTS_DIR
SITES = cfg.SITES; LBL = {s: s for s in SITES}
COL_SITE = {"Luki": "#17a2a2", "Mbalmayo": "#a6a021", "Yangambi": "#c2258f"}
COL_HAB = {"deciduous": "#e0531f", "semi-deciduous": "#e3b21a", "evergreen": "#1d7a3e"}
GRID_T = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

ts = pd.read_csv(cfg.TS_ALL, usecols=["site", "crown_id", "date", "q90"]).dropna()
ts["dt"] = pd.to_datetime(ts.date.astype(str), format="%Y%m%d")

# 1. measurement noise: replicate flights one day apart (Yangambi, 22-23 Aug 2025), leafy crowns;
#    magnitude of a leaf-stable crown simulated with this noise over the median number of dates
y = ts[(ts.site == "Yangambi") & ts.date.isin([20250822, 20250823])].pivot(index="crown_id", columns="date", values="q90").dropna()
a, b = y[20250822], y[20250823]; leafy = (a > 0.8) & (b > 0.8)
e = ((a - b) / ((a + b) / 2))[leafy].values / np.sqrt(2)
rng = np.random.default_rng(0); rows = []
for s in SITES:
    n = int(ts[ts.site == s].groupby("crown_id").size().median())
    sim = 1 + rng.choice(e, (20000, n)); amp0 = (sim.max(1) - sim.min(1)) / sim.max(1)
    d0 = np.clip(1 - sim / np.percentile(sim, 95, axis=1, keepdims=True), 0, None)
    rows.append(dict(site=s, n_dates=n, n_crowns_noise=int(leafy.sum()), sd_noise=e.std(),
                     null_q50=np.percentile(amp0, 50), null_q95=np.percentile(amp0, 95), null_q99=np.percentile(amp0, 99),
                     d_date_q95=np.percentile(d0, 95), d_date_q99=np.percentile(d0, 99)))
NZ = pd.DataFrame(rows); NZ.to_csv(OUT / "noise_null_magnitude.csv", index=False); print(NZ.round(3).to_string(index=False))

GRID = [0.08, 0.10, 0.12, 0.14, 0.16]
KEEP = ["site", "crown_id", "magnitude", "duration_uncorrected"]
A = pd.read_csv(cfg.METRICS_ALL)[KEEP].rename(columns={"magnitude": "amp", "duration_uncorrected": "duration_raw"})
F = pd.read_csv(cfg.METRICS_FIELD)[KEEP].rename(columns={"magnitude": "amp", "duration_uncorrected": "duration_raw"}).merge(
    pd.read_csv(cfg.FIELD_CROWNS, usecols=["site", "crown_id", "species", "leaf_habit"]),
    on=["site", "crown_id"]); F = F[F.species.notna()]
for d in (A, F):
    d["duration_raw"] = d.duration_raw.fillna(0)
    for a0 in GRID: d[f"dur_{a0}"] = d.duration_raw * d.amp ** 2 / (d.amp ** 2 + a0 ** 2)
def eta2(v, g):
    d = pd.DataFrame({"v": v, "g": g}).dropna(); m = d.v.mean()
    return d.groupby("g").v.apply(lambda x: len(x) * (x.mean() - m) ** 2).sum() / ((d.v - m) ** 2).sum()
rows = []
sp = F.groupby("species").filter(lambda g: len(g) >= 4)
for a0 in GRID:
    c = f"dur_{a0}"
    r = dict(A0=a0, eta2_species=eta2(sp[c], sp.species), eta2_site=eta2(A[c], A.site),
             rho_duration_amplitude_A020=A[A.amp >= 0.2][[c, "amp"]].corr("spearman").iloc[0, 1],
             rho_duration_amplitude_all=A[[c, "amp"]].corr("spearman").iloc[0, 1])
    for s_ in SITES: r[f"dur_med_{s_}"] = A[A.site == s_][c].median()
    for h in COL_HAB: r[f"dur_med_{h}"] = F[F.leaf_habit == h][c].median()
    rows.append(r)
S = pd.DataFrame(rows); S.to_csv(OUT / "sensitivity_noise_magnitude.csv", index=False)
pd.set_option("display.width", 250); print(S.round(3).to_string(index=False))

# Fig. S2
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})
fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.2), constrained_layout=True)
# (a) detectability factor and magnitude under noise alone
xa = np.linspace(0, 0.6, 300)
rng2 = np.random.default_rng(1); sim = 1 + rng2.choice(e, (20000, 40)); a_null = (sim.max(1) - sim.min(1)) / sim.max(1)
h_, ed = np.histogram(a_null, np.linspace(0, 0.6, 61), density=True)
ax[0].fill_between(ed[:-1], 0, h_ / h_.max(), step="post", color="#dddddd", lw=0, label="magnitude, noise only")
for a0 in GRID:
    ax[0].plot(xa, xa ** 2 / (xa ** 2 + a0 ** 2), color="#222222" if a0 == 0.12 else "#aaaaaa", lw=1.4 if a0 == 0.12 else 0.7)
ax[0].set_xlabel("Magnitude of leaf loss"); ax[0].set_ylabel("Detectability g"); ax[0].legend(fontsize=5.5, loc="lower right")
for k, a_ in enumerate(ax):
    if k: a_.axvline(0.12, color="#222222", lw=0.8, ls="--"); a_.set_xlabel("Noise magnitude")
    a_.text(-0.02, 1.04, "abcd"[k], transform=a_.transAxes, fontweight="bold", fontsize=9, ha="right")
for s_ in SITES:
    ax[1].plot(S.A0, S[f"dur_med_{s_}"], color=COL_SITE[s_], lw=1.3, label=LBL[s_], marker="o", ms=2.5)
for h in COL_HAB:
    ax[2].plot(S.A0, S[f"dur_med_{h}"], color=COL_HAB[h], lw=1.3, label=h.capitalize(), marker="o", ms=2.5)
ax[3].plot(S.A0, S.eta2_species, color="#222222", lw=1.3, marker="o", ms=2.5, label="η² species")
ax[3].plot(S.A0, S.eta2_site, color="#1f6f8b", lw=1.3, marker="o", ms=2.5, label="η² site")
ax[3].plot(S.A0, S.rho_duration_amplitude_A020, color="#999999", lw=1.3, marker="o", ms=2.5, label="ρ(duration, magnitude), magnitude ≥ 0.2")
ax[1].set_ylabel("Median duration (d/yr)"); ax[2].set_ylabel("Median duration (d/yr)"); ax[3].set_ylabel("Statistic")
ax[1].legend(fontsize=6); ax[2].legend(fontsize=6); ax[3].legend(fontsize=5.5); ax[3].axhline(0, color="#cccccc", lw=0.5)
fig.savefig(OUT / "FigS2_noise_sensitivity.png", dpi=400); fig.savefig(OUT / "FigS2_noise_sensitivity.pdf")
print("->", OUT / "FigS2_noise_sensitivity.png")

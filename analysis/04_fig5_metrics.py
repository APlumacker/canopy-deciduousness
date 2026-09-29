#!/usr/bin/env python3
"""
Fig. 5: diversity and synchrony of leaf loss.
  (a) magnitude x duration of all crowns (by site) and of field-identified crowns (by leaf habit)
  (b) the 15 most frequent species (median and interquartile box)
  (c) timing of leaf loss per site (monthly shares weighted by g), with R1 and phi

Inputs : results/crown_metrics_{all,field}.csv, results/synchrony.csv (steps 1-2),
         data/crowns/field_crowns.csv, crown_timeseries_all.csv.gz (flight dates)
Outputs: results/Fig5_metrics.png / .pdf
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg
warnings.filterwarnings("ignore")

OUT = cfg.RESULTS_DIR
SYN = OUT / "synchrony.csv"
SITES = cfg.SITES
LBL = {s: s for s in SITES}
COL_SITE = {"Luki": "#17a2a2", "Mbalmayo": "#a6a021", "Yangambi": "#c2258f"}
COL_HAB = {"deciduous": "#e0531f", "semi-deciduous": "#e3b21a", "evergreen": "#1d7a3e"}
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"
MIN_SP, N_SP = 4, 15

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": "#9a9a9a",
                     "axes.linewidth": 0.6, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})

A = pd.read_csv(cfg.METRICS_ALL)
F = pd.read_csv(cfg.METRICS_FIELD).merge(pd.read_csv(cfg.FIELD_CROWNS), on=["site", "crown_id"], how="inner")
F = F[F.species.notna()]
F["hab"] = F.leaf_habit.astype(str).str.lower()
for d in (A, F):
    d["w"] = 1.0                           # duration already multiplied by g
    d["wt"] = d.detectability.fillna(0.0)  # weight of timings


def panel(ax, s):
    ax.text(-0.02, 1.02, s, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="right", va="bottom")

def wmed(x, w):
    m = np.isfinite(x) & (w > 0); x, w = np.asarray(x)[m], np.asarray(w)[m]
    o = np.argsort(x); c = np.cumsum(w[o]); return x[o][np.searchsorted(c, c[-1] / 2)]
def wq(x, w, q):
    m = np.isfinite(x) & (w > 0); x, w = np.asarray(x)[m], np.asarray(w)[m]
    o = np.argsort(x); c = np.cumsum(w[o]) / w.sum(); return np.interp(q, c, x[o])

fig = plt.figure(figsize=(7.2, 7.8))
gs = GridSpec(2, 3, figure=fig, height_ratios=[1.55, 0.85], hspace=0.62, wspace=0.42)
gtop = gs[0, :].subgridspec(1, 2, width_ratios=[1.3, 1], wspace=0.22)
rng = np.random.default_rng(1)
YMAX = 200
HABS = ["deciduous", "semi-deciduous", "evergreen"]

def rgba(hexc, alpha):
    r, g, b = [int(hexc[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    return np.c_[np.full(len(alpha), r), np.full(len(alpha), g), np.full(len(alpha), b), alpha]

# ------------------------------------------------------------------ (a) amplitude x duree, pondere A^2
ga = gtop[0].subgridspec(2, 2, height_ratios=[0.8, 4], width_ratios=[4, 0.8], hspace=0.03, wspace=0.03)
ax = fig.add_subplot(ga[1, 0]); axt = fig.add_subplot(ga[0, 0], sharex=ax); axs = fig.add_subplot(ga[1, 1], sharey=ax)
Aa = A.dropna(subset=["duration"]).sample(frac=1, random_state=0)
col = np.vstack([rgba(COL_SITE[s_], np.array([0.0]))[0] for s_ in Aa.site])
col[:, 3] = 0.13
ax.scatter(Aa.magnitude.clip(0, 1), Aa.duration.clip(0, YMAX), c=col, s=2.2, lw=0, rasterized=True, zorder=1)
Fh = F[F.hab.isin(HABS)].dropna(subset=["duration"]).sample(frac=1, random_state=0)
fc = np.vstack([rgba(COL_HAB[h], np.array([0.0]))[0] for h in Fh.hab]); fc[:, 3] = 0.7
ax.scatter(Fh.magnitude.clip(0, 1), Fh.duration.clip(0, YMAX), c=fc, s=7, lw=0, zorder=5, rasterized=True)
for h in HABS:   # croix : mediane +/- IQR de toutes les couronnes de terrain
    g = F[F.hab == h].dropna(subset=["duration"])
    if len(g) < 10: continue
    qx = np.percentile(g.magnitude.clip(0, 1), [25, 50, 75]); qy = [wq(g.duration.values, g.w.values, q) for q in (.25, .5, .75)]
    for dx in (-0.004, 0.004):
        pass
    ax.plot([qx[0], qx[2]], [qy[1]] * 2, color="white", lw=4.2, zorder=6, solid_capstyle="butt")
    ax.plot([qx[1]] * 2, [qy[0], qy[2]], color="white", lw=4.2, zorder=6, solid_capstyle="butt")
    ax.plot([qx[0], qx[2]], [qy[1]] * 2, color=COL_HAB[h], lw=2.2, zorder=7, solid_capstyle="butt")
    ax.plot([qx[1]] * 2, [qy[0], qy[2]], color=COL_HAB[h], lw=2.2, zorder=7, solid_capstyle="butt")
    ax.scatter(qx[1], qy[1], s=46, color=COL_HAB[h], edgecolors="white", lw=1.0, zorder=8)
xs = np.linspace(0, 1, 200); ys = np.linspace(0, YMAX, 200)
for s_ in SITES:
    g = A[A.site == s_]
    axt.plot(xs, gaussian_kde(g.magnitude.clip(0, 1), 0.08)(xs), color=COL_SITE[s_], lw=1.3)
    g = g.dropna(subset=["duration"]); g = g[g.w > 0]
    axs.plot(gaussian_kde(g.duration.clip(0, YMAX), 0.15)(ys), ys, color=COL_SITE[s_], lw=1.3)
for a_ in (axt, axs): a_.axis("off")
ax.set_xlim(0, 1); ax.set_ylim(-5, YMAX)
ax.set_xlabel("Magnitude of leaf loss")
ax.set_ylabel("Duration of leaf loss (d/yr)")
ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
axt.text(-0.07, 1.0, "a", transform=axt.transAxes, fontsize=9, fontweight="bold", ha="right", va="top")


# ------------------------------------------------------------------ (b) especes, medianes ponderees
ax = fig.add_subplot(gtop[1])
ax.text(-0.16, 1.0, "b", transform=ax.transAxes, fontsize=9, fontweight="bold", ha="right", va="bottom")
top = F.species.value_counts().head(N_SP).index
hab = F.assign(hab=F.hab.fillna("unknown")).groupby("species").hab.agg(lambda s: s.value_counts().index[0])
ax.scatter(Aa.magnitude.clip(0, 1), Aa.duration.clip(0, YMAX), c=np.c_[np.full((len(Aa), 3), 0.75), np.full(len(Aa), 0.18)],
           s=1, lw=0, rasterized=True, zorder=1)
def abbr(sp):
    p_ = sp.split(); return f"{p_[0][0]}. {' '.join(p_[1:])}" if len(p_) > 1 else sp
P = []
for s_ in top:
    g = F[(F.species == s_)].dropna(subset=["duration"]); c = COL_HAB.get(hab[s_], "#9a9a9a")
    if len(g) < MIN_SP:        # especes surtout sans perte mesurable : marqueur creux sur l'axe, a leur amplitude mediane
        xm = F.loc[F.species == s_, "magnitude"].median()
        ax.scatter(xm, 3, s=8 + 1.6 * (F.species == s_).sum(), facecolors="white", edgecolors=c, lw=1.0, zorder=4, clip_on=False)
        P.append([xm, 3, abbr(s_) + f" ({len(g)}/{(F.species == s_).sum()})"]); continue
    qx = np.percentile(F.loc[F.species == s_, "magnitude"].clip(0, 1), [25, 50, 75])
    qy = [wq(g.duration.values, g.w.values, q) for q in (0.25, 0.5, 0.75)]
    ax.add_patch(plt.Rectangle((qx[0], qy[0]), qx[2] - qx[0], qy[2] - qy[0], facecolor=c, alpha=0.10, edgecolor=c, lw=0.6, zorder=2))
    ax.scatter(qx[1], qy[1], s=8 + 1.6 * (F.species == s_).sum(), color=c, edgecolors="white", lw=0.6, zorder=4)
    P.append([qx[1], qy[1], abbr(s_)])
print({t: (round(x, 2), round(y_)) for x, y_, t in P})
LAB_POS = {"T. superba": (0.85, 110, "center"), "R. heudelotii": (0.70, 92, "center"), "P. angolensis": (0.41, 116, "left"),
           "P. oxyphylla": (0.25, 88, "right"), "C. mildbraedii": (0.28, 58, "right"), "H. gabunensis": (0.60, 60, "left"),
           "P. africanum": (0.60, 76, "left"), "S. zenkeri": (0.18, 52, "left"), "D. buettneri": (0.01, 82, "left"),
           "P. macrocarpus": (0.40, 31, "left"), "D. pachyphylium": (0.40, 55, "left"), "P. balsamifera": (0.56, 41, "left"),
           "G. giganteum": (0.01, 29, "left"), "C. hankei": (0.01, 12, "left"), "M. cecropioides": (0.01, 68, "left")}
for x, y_, t in P:
    lx, ly, ha = LAB_POS.get(t, (x, y_ + 5, "center"))
    far = np.hypot((lx - x) / 0.06, (ly - y_) / 4) > 1.6
    ax.annotate(t, (x, y_), (lx, ly), fontsize=5.6, style="italic", ha=ha, va="center", color=INK, zorder=6,
                arrowprops=dict(arrowstyle="-", color="#9a9a9a", lw=0.45, shrinkA=1, shrinkB=3) if far else None)
ax.set_xlim(0, 1); ax.set_ylim(-3, 125)
ax.set_xlabel("Magnitude of leaf loss"); ax.set_ylabel("Duration of leaf loss (d/yr)")
ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.set_title(f"{len(top)} most frequent species: median (size = n crowns), box = interquartile range;\n"
             "all field crowns; duration = annual deficit area / maximum observed deficit × detectability g", fontsize=6.0, color=INK, loc="left")

h1 = [plt.Line2D([], [], marker="o", ls="", color=COL_HAB[h], markersize=4.5, label=f"{h.capitalize()}") for h in HABS]
h2 = [plt.Line2D([], [], color=COL_SITE[x], lw=1.6, label=LBL[x]) for x in SITES]
fig.legend(handles=h1, title="Field-identified crowns: species leaf habit (CoForTrait)", loc="upper left",
           bbox_to_anchor=(0.08, 0.435), ncol=3, fontsize=6.3, title_fontsize=6.3)
fig.legend(handles=h2, title="All crowns: site (points and marginal densities)", loc="upper left",
           bbox_to_anchor=(0.52, 0.435), ncol=3, fontsize=6.3, title_fontsize=6.3, handlelength=1.4)

# ------------------------------------------------------------------ (c) timing
syn = pd.read_csv(SYN).set_index("site")
flights = {}
ts = pd.read_csv(cfg.TS_ALL, usecols=["site", "date"]).drop_duplicates()
for s in SITES:
    d = pd.to_datetime(ts.loc[ts.site == s, "date"].astype(str), format="%Y%m%d")
    flights[s] = d.dt.dayofyear.values
edges = np.linspace(0, 2 * np.pi, 13)
for j, s in enumerate(SITES):
    ax = fig.add_subplot(gs[1, j], projection="polar")
    if j == 0: ax.text(-0.18, 1.40, "c", transform=ax.transAxes, fontsize=9, fontweight="bold")
    if j == 0: ax.text(-0.10, 1.40, "Timing of leaf loss (share of crowns per month, weighted by g) and synchrony (R$_1$, φ)", transform=ax.transAxes, fontsize=7.0, color=INK, va="baseline")
    gg = A[(A.site == s)].dropna(subset=["timing"]); gg = gg[gg.wt > 0]
    doy = gg.timing; ww = gg.wt.values
    th = 2 * np.pi * doy.values / 365.25
    h, _ = np.histogram(th, edges, weights=ww); h = h / h.sum() * 100
    ax.bar(edges[:-1], h, width=2 * np.pi / 12, align="edge", color=COL_SITE[s], alpha=0.75,
           edgecolor="white", lw=0.8)
    rmax = 30
    if s in flights:
        ax.scatter(2 * np.pi * flights[s] / 365.25, np.full(len(flights[s]), rmax * 1.02), marker="|",
                   s=18, color=INK, lw=0.6, clip_on=False)
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_xticks(edges[:-1] + np.pi / 12)
    ax.set_xticklabels(list("JFMAMJJASOND"), fontsize=6.3, color=MUTED)
    ax.set_ylim(0, rmax); ax.set_yticks([10, 20]); ax.set_yticklabels(["10 %", "20 %"], fontsize=5.5, color=MUTED)
    ax.set_rlabel_position(100); ax.grid(color=GRID, lw=0.6); ax.spines["polar"].set_color("#bdbdbd")
    r = syn.loc[LBL[s]]
    ax.set_title(f"{LBL[s]}  ·  R$_1$ = {r.R1_timing:.2f}, φ = {r.phi:.3f}", fontsize=7.0, pad=12, color=INK)
    R1 = np.hypot((ww * np.cos(th)).sum(), (ww * np.sin(th)).sum()) / ww.sum()
    R2 = np.hypot((ww * np.cos(2 * th)).sum(), (ww * np.sin(2 * th)).sum()) / ww.sum()


fig.savefig(OUT / "Fig5_metrics.png", dpi=400, bbox_inches="tight")
fig.savefig(OUT / "Fig5_metrics.pdf", bbox_inches="tight")
print("->", OUT / "Fig5_metrics.png")

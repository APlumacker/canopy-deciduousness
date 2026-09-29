#!/usr/bin/env python3
"""
Fig. 3: from the leaf-presence time series to the metrics of leaf loss, for one crown
(Piptadeniastrum africanum, crown 132 of the field inventory, Yangambi).
  (a) crown leaf presence, leaf deficit, magnitude, and each year a rectangle of the same
      area as the mean annual deficit, centred on the timing (width = duration)
  (b) the 44 crown thumbnails on the same time axis, framed by the leaf deficit
  (c) mean annual deficit profile on the circle (arrow = timing, arc = duration)
Also Fig. S1 (detectability factor g and the magnitude of a leaf-stable crown under the
measurement noise of the replicate flights of 22 and 23 August 2025 at Yangambi).

Inputs : data/crowns/crown_timeseries_all.csv.gz, assets/fig3_crown132_yangambi/
Outputs: results/Fig3_crown_example.png/.pdf, results/FigS1_detectability.png/.pdf
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colorbar
import matplotlib.dates as mdates
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from leafloss import crown_metrics
warnings.filterwarnings("ignore")

OUT = cfg.RESULTS_DIR
VIG = cfg.FIG3_THUMBS
# crown 132 of the field inventory = crown 14926 of the automatic delineation
SITE, TS_CROWN_ID, CROWN_LABEL, SPECIES = "Yangambi", 14926, 132, "Piptadeniastrum africanum"
A0 = cfg.NOISE_MAGNITUDE

INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"
C_SERIES, C_D, C_DD, C_M = "#2b2b2b", "#e08a2c", "#a15b12", "#1f6f8b"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#9a9a9a",
                     "axes.linewidth": 0.6, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "mathtext.fontset": "dejavusans"})

# ------------------------------------------------------------------ metrics
ts = pd.read_csv(cfg.TS_ALL, usecols=["site", "crown_id", "date", "q90"])
c = ts[(ts.site == SITE) & (ts.crown_id == TS_CROWN_ID)].dropna().sort_values("date")
c["dt"] = pd.to_datetime(c.date.astype(str), format="%Y%m%d")
q, dt = c.q90.values, c.dt.reset_index(drop=True)
M = crown_metrics(q, dt, noise=A0, return_profile=True)
B, amp, cent, W, g = M["full_foliage"], M["magnitude"], M["timing"], M["duration_uncorrected"], M["detectability"]
d = M["deficit"]; dmax = d.max(); prof = M["profile"]
cent_date = pd.Timestamp("2001-01-01") + pd.Timedelta(days=cent - 1)
print(f"A={amp:.2f} timing={cent_date:%d %b} W={W:.1f} g={g:.2f} W*g={W*g:.1f} R1={M['R1']:.2f} R2={M['R2']:.2f}")

fig = plt.figure(figsize=(11.69, 8.27))                       # A4 paysage
L, Wd = 0.065, 0.905
fig.text(L, 0.962, f"{SPECIES}, crown {CROWN_LABEL}, Yangambi ({len(q)} acquisitions, Oct 2023 – Oct 2025)",
         fontsize=10.5, style="italic", color=INK)

# ------------------------------------------------------------------ (a) serie temporelle
ax = fig.add_axes([L, 0.42, Wd, 0.49])
x0, x1 = dt.min() - pd.Timedelta(days=12), dt.max() + pd.Timedelta(days=12)
ax.fill_between(dt, np.minimum(q, B), B, color=C_D, alpha=0.35, lw=0, zorder=1)
ax.plot(dt, q, color=C_SERIES, lw=1.1, zorder=3); ax.scatter(dt, q, s=9, color=C_SERIES, zorder=4, lw=0)
ax.axhline(B, color=MUTED, lw=0.6, ls="--", zorder=2)
ax.text(x0 + pd.Timedelta(days=6), B + 0.015, "full foliage", fontsize=8.5, color=MUTED, va="bottom")
ybot = B * (1 - dmax)
for yr in range(dt.min().year, dt.max().year + 1):
    t0 = pd.Timestamp(f"{yr}-01-01") + pd.Timedelta(days=cent - 1)
    if not (x0 <= t0 <= x1): continue
    ax.add_patch(plt.Rectangle((mdates.date2num(t0 - pd.Timedelta(days=W / 2)), ybot), W, B - ybot, fill=False,
                               edgecolor=C_M, lw=1.0, ls=(0, (3, 2)), zorder=5))
    ax.plot([t0, t0], [ybot, B], color=C_M, lw=1.0, zorder=2)
    ax.text(t0, B + 0.015, f"timing {cent_date:%d %b}", color=C_M, fontsize=8.5, ha="center", va="bottom")
    ax.annotate("", (t0 + pd.Timedelta(days=W / 2), ybot - 0.05), (t0 - pd.Timedelta(days=W / 2), ybot - 0.05),
                arrowprops=dict(arrowstyle="<->", color=C_M, lw=0.9, shrinkA=0, shrinkB=0))
    ax.text(t0, ybot - 0.08, f"duration {W:.0f} d/yr", color=C_M, fontsize=9, ha="center", va="top")
imin = int(np.argmin(q)); t_last = pd.Timestamp(f"{dt.iloc[imin].year}-01-01") + pd.Timedelta(days=cent - 1)
xa = t_last - pd.Timedelta(days=W / 2 + 14)
ax.annotate("", (xa, q.min()), (xa, q.max()), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9, shrinkA=0, shrinkB=0))  # magnitude = (max - min)/max de q90
ax.text(xa - pd.Timedelta(days=6), ybot + 0.14, f"magnitude\n{amp:.2f}", ha="right", va="center", fontsize=9.5, color=INK)
_j = int(np.argmin(np.where(dt.dt.year.values == dt.min().year + 1, q, 9)))
ax.annotate("leaf deficit", (dt.iloc[_j] - pd.Timedelta(days=14), 0.8), (dt.iloc[_j] - pd.Timedelta(days=75), 0.6),
            color=C_DD, fontsize=9, ha="right", va="center", arrowprops=dict(arrowstyle="-", color=C_DD, lw=0.5))
# annees : bandes alternees (a et b) + etiquette de l'annee en haut de a
YEARS = list(range(dt.min().year, dt.max().year + 1))
def _year_span(yr):
    return max(pd.Timestamp(f"{yr}-01-01"), x0), min(pd.Timestamp(f"{yr + 1}-01-01"), x1)
for k_, yr in enumerate(YEARS):
    s0, s1 = _year_span(yr)
    if k_ % 2 == 1: ax.axvspan(s0, s1, color="#f2f2f2", lw=0, zorder=0)
    if yr != YEARS[0]: ax.axvline(pd.Timestamp(f"{yr}-01-01"), color="#9a9a9a", lw=0.8, zorder=0.5)
    ax.text(s0 + (s1 - s0) / 2, 1.075, str(yr), ha="center", va="center", fontsize=10.5, fontweight="bold", color="#555555")
ax.set_xlim(x0, x1); ax.set_ylim(0, 1.1)
ax.set_ylabel("Crown leaf presence")
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b")); ax.xaxis.set_minor_locator(mdates.MonthLocator())
ax.grid(True, axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.text(-0.035, 1.0, "a", transform=ax.transAxes, fontsize=12, fontweight="bold", ha="right", va="top")

# ------------------------------------------------------------------ (b) vignettes alignees sur l'axe du temps
axv = fig.add_axes([L, 0.075, Wd, 0.25], sharex=ax)
idx = pd.read_csv(VIG / "index.csv"); idx["dt"] = pd.to_datetime(idx.date.astype(str), format="%Y%m%d")
idx["d"] = np.clip(1 - idx.q90 / B, 0, None)
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("gr", ["#1d7a3e", "#9cc54a", "#f2c12e", "#e0531f", "#9e1b12"]); NROW = 2
ROWS = [0.7, 0.2]
ax_w_in = Wd * fig.get_figwidth(); span = mdates.date2num(x1) - mdates.date2num(x0)
TILE_IN = 0.37; TILE_D = (TILE_IN + 0.02) * span / ax_w_in          # largeur d'une vignette (+ marge) en jours d'axe
lo_b, hi_b = mdates.date2num(x0) + TILE_D / 2, mdates.date2num(x1) - TILE_D / 2
# chaque vignette a sa date (meme axe que a) ; rangees alternees ; quand deux dates d'une meme
# rangee sont trop proches (grappe juillet-octobre 2025), on les ecarte le minimum necessaire
# et un trait gris relie la vignette a sa date
X = mdates.date2num(idx["dt"]).astype(float); ROW = np.arange(len(idx)) % NROW; XP = X.copy()
for _ in range(600):
    for r_ in range(NROW):
        ii = np.where(ROW == r_)[0]
        for a_, b_ in zip(ii[:-1], ii[1:]):
            ov = TILE_D - (XP[b_] - XP[a_])
            if ov > 0: XP[a_] -= ov / 2; XP[b_] += ov / 2
    XP = np.clip(XP, lo_b, hi_b)
place = list(zip(XP, ROW))
for k in range(len(idx)):
    r = idx.iloc[k]; t_ = r["dt"]; frac = min(r["d"] / dmax, 1); xp, r_ = place[k]; y = ROWS[r_]
    x = mdates.date2num(t_)
    im = Image.open(VIG / r["file"]).convert("RGBA"); bb = im.getbbox()
    if bb: im = im.crop(bb)
    im.thumbnail((160, 160))
    ab = AnnotationBbox(OffsetImage(np.asarray(im), zoom=TILE_IN * 72 / max(im.size)), (xp, y), xycoords=("data", "axes fraction"),
                        frameon=True, pad=0.05, box_alignment=(0.5, 0.5),
                        bboxprops=dict(edgecolor=cmap(frac), lw=2.4, facecolor="white"))
    axv.add_artist(ab)
    strong = r["d"] >= 0.5 * dmax
    newyr = k == 0 or t_.year != idx["dt"].iloc[k - 1].year
    # repere de la date exacte au-dessus de la vignette (la vignette peut etre legerement decalee)
    axv.plot([x], [y + 0.215], marker="v", ms=4, color=INK, lw=0, transform=axv.get_xaxis_transform(), clip_on=False)
    axv.text(xp, y - 0.155, f"{t_:%d %b}", transform=axv.get_xaxis_transform(),
             ha="center", va="top", fontsize=6.3, color=INK if strong else MUTED, fontweight="bold" if strong else "normal")
for k_, yr in enumerate(YEARS):
    s0, s1 = _year_span(yr)
    if k_ % 2 == 1: axv.axvspan(s0, s1, color="#f2f2f2", lw=0, zorder=0)
    if yr != YEARS[0]: axv.axvline(pd.Timestamp(f"{yr}-01-01"), color="#9a9a9a", lw=0.8, zorder=0.5)
axv.set_ylim(0, 1); axv.axis("off")
axv.text(-0.035, 1.02, "b", transform=axv.transAxes, fontsize=12, fontweight="bold", ha="right", va="top")
cax = fig.add_axes([0.70, 0.035, 0.18, 0.012])
cb = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, orientation="horizontal", ticks=[0, 0.5, 1])
cb.ax.set_xticklabels(["0", "half", "magnitude"], fontsize=8); cb.outline.set_linewidth(0.4)
fig.text(0.695, 0.041, "frame: leaf deficit", fontsize=8.5, color=INK, ha="right", va="center")
fig.text(L, 0.041, "All 44 acquisitions on the time axis of a (▼ exact date); bold dates: deficit ≥ half the magnitude", fontsize=8.5, color=MUTED, va="center")

# ------------------------------------------------------------------ (c) profil annuel sur le cercle
_fx = lambda t: L + Wd * (mdates.date2num(pd.Timestamp(t)) - mdates.date2num(x0)) / (mdates.date2num(x1) - mdates.date2num(x0))
_cx = _fx("2025-01-12"); _h = 0.25; _w = _h * fig.get_figheight() / fig.get_figwidth()
axp = fig.add_axes([_cx - _w / 2, 0.515, _w, _h], projection="polar")
axp.set_theta_zero_location("N"); axp.set_theta_direction(-1)
bins = 73; mb = prof[:365].reshape(bins, 5).mean(1); rmax = dmax
thb = 2 * np.pi * (np.arange(bins) * 5 + 2.5) / 365
axp.bar(thb, mb, width=2 * np.pi / bins, color=C_D, alpha=0.9, lw=0)
mu = 2 * np.pi * (cent - 1) / 365.25
axp.annotate("", (mu, 0.95 * rmax), (mu, 0), arrowprops=dict(arrowstyle="-|>", color=C_M, lw=1.8, mutation_scale=11))
half = np.pi * W / 365.25; tt = np.linspace(mu - half, mu + half, 100)
axp.plot(tt, np.full_like(tt, 1.05 * rmax), color=C_M, lw=3.0, solid_capstyle="butt", clip_on=False)
axp.set_ylim(0, rmax); axp.set_yticks([])
MONTHS = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])
axp.set_xticks(2 * np.pi * MONTHS / 365.25); axp.set_xticklabels([])
for mth, lab in zip(MONTHS, list("JFMAMJJASOND")):
    axp.text(2 * np.pi * (mth + 15) / 365.25, 1.16 * rmax, lab, fontsize=8.5, color=MUTED, ha="center", va="center")
axp.grid(color=GRID, lw=0.6); axp.spines["polar"].set_color("#bdbdbd")
axp.text(0.5, 1.14, "Mean annual deficit profile", transform=axp.transAxes, fontsize=9.5, color=INK, ha="center")
axp.text(0.5, -0.16, f"arrow: centre of mass → timing ({cent_date:%d %b})\narc: duration ({W:.0f} d/yr)",
         transform=axp.transAxes, fontsize=8.5, color=C_M, ha="center", va="top", linespacing=1.4)
axp.text(-0.25, 1.16, "c", transform=axp.transAxes, fontsize=12, fontweight="bold", ha="right", va="top")

# ------------------------------------------------------------------ figure supplementaire : detectabilite g
# bruit : ecart de q90 entre les deux vols a 1 jour d'ecart (Yangambi 22/23-08-2025), couronnes feuillees ;
# amplitude d'une couronne plate simulee avec ce bruit sur 44 dates
_y = ts[(ts.site == SITE) & ts.date.isin([20250822, 20250823])].pivot_table(index="crown_id", columns="date", values="q90").dropna()
_a, _b = _y[20250822], _y[20250823]; _k = (_a > 0.8) & (_b > 0.8)
_e = ((_a - _b) / ((_a + _b) / 2))[_k].values / np.sqrt(2)
_sim = 1 + np.random.default_rng(0).choice(_e, (20000, len(q))); a_null = (_sim.max(1) - _sim.min(1)) / _sim.max(1)
figS = plt.figure(figsize=(5.0, 3.4)); axg = figS.add_axes([0.14, 0.16, 0.82, 0.72])
xa_ = np.linspace(0, 1, 400); gg_ = xa_ ** 2 / (xa_ ** 2 + A0 ** 2)
h_, ed_ = np.histogram(a_null, np.linspace(0, 1, 101), density=True)
axg.fill_between(ed_[:-1], 0, h_ / h_.max(), step="post", color="#d9d9d9", lw=0, label="magnitude of a leaf-stable crown\n(measurement noise only)")
axg.plot(xa_, gg_, color=C_M, lw=1.8, label="detectability g")
axg.plot([A0, A0], [0, 0.5], color=MUTED, lw=0.8, ls="--"); axg.plot([0, A0], [0.5, 0.5], color=MUTED, lw=0.8, ls="--")
axg.text(A0 + 0.03, 0.42, f"noise magnitude {A0:.2f} → g = 0.5", fontsize=8, color=MUTED, va="top")
axg.text(0.21, 0.16, "← magnitude of a leaf-stable crown\n    (measurement noise only)", fontsize=7.5, color="#8a8a8a", va="center")
axg.scatter([amp], [g], s=34, color=C_M, zorder=5, edgecolor="white", lw=0.8)
axg.annotate(f"crown 132: magnitude {amp:.2f}, g = {g:.2f}\nduration × g = {W * g:.0f} d/yr", (amp, g), (0.98, 0.62), ha="right", va="center",
             fontsize=8, color=C_M, arrowprops=dict(arrowstyle="-", color=C_M, lw=0.6))
axg.set_xlim(0, 1); axg.set_ylim(0, 1.05); axg.set_xlabel("Magnitude of leaf loss"); axg.set_ylabel("g")
axg.set_title("Detectability g", fontsize=9.5, color=INK, loc="left")
figS.savefig(OUT / "FigS1_detectability.png", dpi=300); figS.savefig(OUT / "FigS1_detectability.pdf")
print("->", OUT / "FigS1_detectability.png")

fig.savefig(OUT / "Fig3_crown_example.png", dpi=300)
fig.savefig(OUT / "Fig3_crown_example.pdf")
print("->", OUT / "Fig3_crown_example.png")

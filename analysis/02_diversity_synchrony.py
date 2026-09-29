#!/usr/bin/env python3
"""
Step 2: diversity and synchrony of leaf loss (sections 3.f-3.g, Tables 2, 3 and S1).

Inputs : results/crown_metrics_{all,field}.csv (step 1)
         data/crowns/field_crowns.csv, field_crowns_topography.csv, crown_timeseries_all.csv.gz
Outputs (results/):
  table2_variance_partition.csv  eta2 of site / species / genus / family (magnitude, duration,
                                 circular eta2 of timing weighted by detectability, Eq. 2)
  habit_concordance.csv          AUC of magnitude vs leaf habit, Hartigan's dip test
  intraspecific.csv              range and SD of magnitude within species
  shared_species.csv             share of within-species variance separating Luki and Yangambi
  tableS1_topography.csv         mixed model magnitude ~ topography + (1 | species)
  synchrony.csv                  R1, R2, bimodal profiles, community synchrony phi per site
  table3_sites.csv               Table 3
  summary.txt                    all numbers quoted in the text
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
from scipy.stats import kruskal, chi2 as _chi2
from sklearn.metrics import roc_auc_score
from diptest import diptest as _dip

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg
warnings.filterwarnings("ignore")

OUT = cfg.RESULTS_DIR
SITES = cfg.SITES
MIN_SP, MIN_GEN, MIN_FAM = 4, 5, 5        # minimum crowns per species / genus / family
TVARS = ["topo_elevation", "topo_slope", "topo_twi", "topo_pisr", "topo_curv_profile", "topo_curv_plan"]
LINES = []


def say(s=""):
    print(s); LINES.append(s)


# ------------------------------------------------------------------ helpers
def eta2(v, g):
    d = pd.DataFrame({"v": v, "g": g}).dropna(); gm = d.v.mean()
    sst = ((d.v - gm) ** 2).sum()
    return d.groupby("g").v.apply(lambda x: len(x) * (x.mean() - gm) ** 2).sum() / sst


def var_comp(d, y, grp):
    """Share of variance of the random intercept (REML), standardised response."""
    z = d[[y, grp]].dropna().copy(); z["yz"] = (z[y] - z[y].mean()) / z[y].std()
    m = smf.mixedlm("yz ~ 1", z, groups=z[grp]).fit(reml=True)
    s2 = float(m.cov_re.iloc[0, 0]); return s2 / (s2 + m.scale)


def ang(doy, k=1):
    return 2 * np.pi * k * np.asarray(doy, float) / 365.25


def wmed(x, w):
    m = x.notna() & (w > 0); x, w = x[m], w[m]
    o = np.argsort(x.values); c = np.cumsum(np.asarray(w)[o]); return float(x.values[o][np.searchsorted(c, c[-1] / 2)])


def weta2(v, g, w):
    d = pd.DataFrame({"v": v, "g": g, "w": w}).dropna(); d = d[d.w > 0]; m = np.average(d.v, weights=d.w)
    b = d.groupby("g").apply(lambda x: x.w.sum() * (np.average(x.v, weights=x.w) - m) ** 2).sum()
    return b / (d.w * (d.v - m) ** 2).sum()


def wRk(doy, w, k=1):
    a = ang(doy, k); w = np.asarray(w); return float(np.hypot((w * np.cos(a)).sum(), (w * np.sin(a)).sum()) / w.sum())


def wcirc_eta2(doy, g, w):
    """Circular eta2 weighted by detectability (Eq. 2): 1 - sum_g W_g (1 - R_g) / (W (1 - R))."""
    d = pd.DataFrame({"a": doy, "g": g, "w": w}).dropna(); d = d[d.w > 0]; W = d.w.sum()
    within = sum(x.w.sum() * (1 - wRk(x.a, x.w)) for _, x in d.groupby("g"))
    return 1 - within / (W * (1 - wRk(d.a, d.w)))


def keep_groups(d, col, n):
    c = d[col].value_counts(); return d[d[col].isin(c[c >= n].index)]


def kw_p(d, y, col):
    gr = [x[y].dropna().values for _, x in d.groupby(col)]
    gr = [x for x in gr if len(x) >= 2]
    return kruskal(*gr).pvalue if len(gr) >= 2 else np.nan


# ------------------------------------------------------------------ data
A = pd.read_csv(cfg.METRICS_ALL)
F = pd.read_csv(cfg.METRICS_FIELD).merge(pd.read_csv(cfg.FIELD_CROWNS), on=["site", "crown_id"], how="inner")
F = F[F.species.notna()]
for d in (A, F):
    d["wt"] = d.detectability.fillna(0.0)        # weight of each crown's timing
    d["w"] = 1.0                                  # duration already multiplied by g
say(f"Crowns: {len(A)} ({A.site.value_counts().to_dict()}) | field-identified: {len(F)}")
say(f"Strong leaf loss, magnitude >= {cfg.STRONG_LOSS} (% of crowns): "
    f"{A.groupby('site').magnitude.apply(lambda m: round((m >= cfg.STRONG_LOSS).mean() * 100, 1)).to_dict()}")
say(f"Magnitude above the noise level {cfg.NOISE_MAGNITUDE} (% of crowns): "
    f"{A.groupby('site').magnitude.apply(lambda m: round((m > cfg.NOISE_MAGNITUDE).mean() * 100, 1)).to_dict()}")
say(f"Bimodal profiles, R2 > R1 (weighted by g, %): "
    f"{A.groupby('site').apply(lambda g: round(np.average(g.bimodal == True, weights=g.wt) * 100, 1)).to_dict()}")
say(f"Median duration (d/yr, all crowns): {A.groupby('site').apply(lambda g: round(wmed(g.duration, g.w), 0)).to_dict()}")
say(f"Field crowns, median duration by leaf habit: "
    f"{F.groupby(F.leaf_habit.astype(str)).apply(lambda g: round(wmed(g.duration, g.w), 0)).to_dict()}")
say(f"Spearman duration-magnitude (magnitude >= 0.2): "
    f"{A[A.magnitude >= 0.2][['duration', 'magnitude']].dropna().corr('spearman').iloc[0, 1]:.2f}")
say(f"Spearman duration-magnitude (all crowns): {A[['duration', 'magnitude']].dropna().corr('spearman').iloc[0, 1]:.2f}")

# ------------------------------------------------------------------ Table 2
rows = []


def add(level, d, col, n_used, n_rec):
    r = dict(level=level, groups=f"{n_used} / {n_rec}", crowns=len(d))
    r["magnitude_eta2"] = eta2(d.magnitude, d[col]); r["magnitude_kw_p"] = kw_p(d, "magnitude", col)
    r["duration_eta2"] = weta2(d.duration, d[col], d.w)
    r["n_duration"] = int(d.duration.notna().sum())
    r["timing_circ_eta2"] = wcirc_eta2(d.timing, d[col], d.wt)
    r["n_timing"] = int(d.timing.notna().sum())
    rows.append(r)


add("Site", A, "site", 3, 3)
for lvl, col, n in [("Species", "species", MIN_SP), ("Genus", "genus", MIN_GEN), ("Family", "family", MIN_FAM)]:
    k = keep_groups(F[F[col].notna() & (F[col] != "Indetermine")], col, n)
    add(lvl, k, col, k[col].nunique(), F[col].nunique())
T2 = pd.DataFrame(rows); T2.to_csv(OUT / "table2_variance_partition.csv", index=False)
say("\n== Table 2 (eta2; timing: circular analogue weighted by g) =="); say(T2.round(4).to_string(index=False))

sp = keep_groups(F, "species", MIN_SP)
top = F.species.value_counts().head(15).index
say("Median duration of the 15 most frequent species (d/yr): " + ", ".join(
    f"{s_} {F[F.species == s_].duration.median():.0f}" for s_ in top))
say(f"\nSpecies variance component (mixed model): magnitude {var_comp(sp, 'magnitude', 'species'):.3f} "
    f"(n={len(sp)}, {sp.species.nunique()} sp.) | duration {var_comp(sp, 'duration', 'species'):.3f}")

# ------------------------------------------------------------------ leaf habit + dip test
lab = F.leaf_habit.astype(str).str.lower()
H = F[lab.isin(["deciduous", "evergreen"])].copy(); H["dec"] = (H.leaf_habit.str.lower() == "deciduous").astype(int)
hs = H.groupby("species").agg(dec=("dec", "first"), mag=("magnitude", "median"))
med = A.magnitude.median()
conc = dict(n_crowns=len(H), n_species=len(hs), auc_crown=roc_auc_score(H.dec, H.magnitude),
            auc_species=roc_auc_score(hs.dec, hs.mag), magnitude_median_all=med,
            magnitude_mean_deciduous=H[H.dec == 1].magnitude.mean(),
            magnitude_mean_evergreen=H[H.dec == 0].magnitude.mean(),
            magnitude_mean_semideciduous=F[lab == "semi-deciduous"].magnitude.mean(),
            pct_deciduous_above_median=(H[H.dec == 1].magnitude > med).mean() * 100,
            pct_evergreen_below_median=(H[H.dec == 0].magnitude <= med).mean() * 100)
lo, hi = H[H.dec == 1].magnitude.quantile(.05), H[H.dec == 0].magnitude.quantile(.95)
conc["overlap_interval"] = f"{lo:.3f}-{hi:.3f}"
conc["pct_field_in_overlap"] = F.magnitude.between(lo, hi).mean() * 100
for nm, x in [("all", A.magnitude), ("field", F.magnitude)]:
    s, p = _dip(np.asarray(x, float)); conc[f"dip_{nm}"] = s; conc[f"dip_p_{nm}"] = p
for s_ in SITES:
    s, p = _dip(A[A.site == s_].magnitude.values); conc[f"dip_{s_}"] = s; conc[f"dip_p_{s_}"] = p
pd.Series(conc).to_csv(OUT / "habit_concordance.csv", header=["value"])
say("\n== Magnitude vs leaf habit (CoForTrait) + Hartigan's dip test ==")
for k, v in conc.items():
    say(f"  {k:28s} {v:.4g}" if isinstance(v, float) else f"  {k:28s} {v}")

# ------------------------------------------------------------------ intraspecific variation
g = sp.groupby("species").magnitude
I = pd.DataFrame({"n": g.size(), "min": g.min(), "max": g.max(), "sd": g.std()})
I["range"] = I["max"] - I["min"]; tot_rng = sp.magnitude.max() - sp.magnitude.min()
I["frac_gradient"] = I["range"] / tot_rng
I.sort_values("n", ascending=False).to_csv(OUT / "intraspecific.csv")
say(f"\n== Intraspecific variation of magnitude ({len(I)} species with >= {MIN_SP} crowns) ==")
say(f"  median within-species range {I['range'].median():.3f} vs {tot_rng:.3f} overall; "
    f"median within-species SD {I.sd.median():.3f} vs {sp.magnitude.std():.3f}")
for s_, r in I[(I.n >= 15) & (I.frac_gradient > .75)].sort_values("n", ascending=False).iterrows():
    say(f"    {s_} (n={r.n:.0f}) {r['min']:.2f}-{r['max']:.2f} ({r.frac_gradient:.0%} of the gradient)")

# ------------------------------------------------------------------ species shared by Luki and Yangambi
sh = sp[sp.site.isin(["Luki", "Yangambi"])]
both = sh.groupby("species").site.nunique(); both = both[both == 2].index
S = sh[sh.species.isin(both)].copy()
out = []
for y in ["magnitude", "duration"]:
    z = S.dropna(subset=[y]).copy()
    z["r"] = z[y] - z.groupby("species")[y].transform("mean")
    ss_w = (z.r ** 2).sum()
    ss_site = z.groupby(["species", "site"]).apply(lambda x: len(x) * x.r.mean() ** 2).sum()
    out.append(dict(metric=y, n_species=z.species.nunique(), n_crowns=len(z), site_share_within=ss_site / ss_w))
out = pd.DataFrame(out); out.to_csv(OUT / "shared_species.csv", index=False)
say("\n== Species recorded at both Luki and Yangambi: share of within-species variance due to site ==")
say(out.round(4).to_string(index=False))

# ------------------------------------------------------------------ Table S1: topography
T = pd.read_csv(cfg.FIELD_TOPO)[["site", "crown_id"] + TVARS]
M = sp.merge(T, on=["site", "crown_id"], how="inner")
rowsT = []
for label, d in [("Luki", M[M.site == "Luki"]), ("Yangambi", M[M.site == "Yangambi"]), ("Pooled", M)]:
    d = keep_groups(d, "species", MIN_SP).copy()
    for v in TVARS:
        d["z_" + v] = d.groupby("site")[v].transform(lambda x: (x - x.mean()) / x.std(ddof=0))
    zt = ["z_" + v for v in TVARS]; d["y"] = d.magnitude / d.magnitude.std()
    a = sm.MixedLM.from_formula("y ~ " + "+".join(zt), groups="species", data=d).fit(reml=False)
    b = sm.MixedLM.from_formula("y ~ 1", groups="species", data=d).fit(reml=False)
    chi = 2 * (a.llf - b.llf)
    fx = np.var(a.fe_params[zt].values @ d[zt].values.T)                 # Nakagawa marginal R2
    r2m = fx / (fx + float(a.cov_re.iloc[0, 0]) + a.scale)
    m0 = smf.ols("y ~ C(species)", d).fit(); m1 = smf.ols("y ~ C(species) + " + "+".join(zt), d).fit()
    rowsT.append(dict(label=label, n=len(d), n_species=d.species.nunique(), lrt_chi2=chi,
                      lrt_p=_chi2.sf(chi, len(zt)), R2_marginal_topography=r2m,
                      dR2_after_species=m1.rsquared - m0.rsquared,
                      beta_twi_sd=a.fe_params["z_topo_twi"], p_twi=a.pvalues["z_topo_twi"]))
TT = pd.DataFrame(rowsT); TT.to_csv(OUT / "tableS1_topography.csv", index=False)
say("\n== Table S1: magnitude ~ topography + (1 | species) (magnitude in SD units) ==")
say(TT.round(4).to_string(index=False))


# ------------------------------------------------------------------ synchrony
def phi(Mx):
    """Community synchrony (Loreau & de Mazancourt 2008)."""
    return Mx.sum(1).var() / Mx.std(0).sum() ** 2


rowsS = []
ts = pd.read_csv(cfg.TS_ALL, usecols=["site", "crown_id", "date", "q90"])
rng = np.random.default_rng(0)
for s_ in SITES:
    g = A[A.site == s_].dropna(subset=["timing"])
    mk = A[(A.site == s_) & (A.wt >= 0.5)]                    # magnitude above the noise level (g >= 0.5)
    r = dict(site=s_, n=len(g), R1_timing=wRk(g.timing, g.wt), R2_timing=wRk(g.timing, g.wt, 2),
             pct_bimodal=np.average(g.bimodal == True, weights=g.wt) * 100,
             pct_bimodal_unweighted=(g.bimodal == True).mean() * 100,
             R1_first_moment=wRk(g.timing_first_moment.dropna(), g.loc[g.timing_first_moment.notna(), "wt"]))
    t = ts[(ts.site == s_) & ts.crown_id.isin(mk.crown_id)].pivot_table(index="date", columns="crown_id", values="q90")
    t = t[t.notna().mean(axis=1) >= .8]; t = t.loc[:, t.notna().all(axis=0)]   # complete series
    Mx = t.values; o = phi(Mx)
    nl = np.mean([phi(np.column_stack([np.roll(Mx[:, j], rng.integers(Mx.shape[0])) for j in range(Mx.shape[1])]))
                  for _ in range(100)])
    r.update(phi=o, phi_null_circular_shift=nl, n_dates_phi=Mx.shape[0], n_crowns_phi=Mx.shape[1])
    rowsS.append(r)
SY = pd.DataFrame(rowsS); SY.to_csv(OUT / "synchrony.csv", index=False)
say("\n== Synchrony (timings weighted by g; phi on complete series of crowns with g >= 0.5) ==")
say(SY.round(3).to_string(index=False))


# ------------------------------------------------------------------ Table 3
def _date(doy):
    return (pd.Timestamp("2001-01-01") + pd.Timedelta(days=float(doy) % 365)).strftime("%d %b")


rows3 = []
for s_ in SITES:
    g = A[A.site == s_].dropna(subset=["timing"]); w_ = g.wt.values; th = ang(g.timing)
    C1, S1 = (w_ * np.cos(th)).sum(), (w_ * np.sin(th)).sum(); C2, S2 = (w_ * np.cos(2 * th)).sum(), (w_ * np.sin(2 * th)).sum()
    m1 = np.degrees(np.arctan2(S1, C1)) % 360 * 365.25 / 360
    m2 = (np.degrees(np.arctan2(S2, C2)) / 2) % 180 * 365.25 / 360
    sy = SY.set_index("site").loc[s_]
    rows3.append({"Site": s_, "Crowns": int((A.site == s_).sum()),
                  "Magnitude, median": round(A[A.site == s_].magnitude.median(), 2),
                  "Duration, median (d/yr)": round(A[A.site == s_].duration.median()),
                  "Mean timing (R1 direction)": _date(m1),
                  "Axial timing (R2 direction)": f"{_date(m2)} / {_date(m2 + 182.6)}",
                  "R1": round(np.hypot(C1, S1) / w_.sum(), 2), "R2": round(np.hypot(C2, S2) / w_.sum(), 2),
                  "Bimodal profiles, weighted (%)": round(sy.pct_bimodal),
                  "Bimodal profiles, unweighted (%)": round(sy.pct_bimodal_unweighted),
                  "R1, first-moment timing (sensitivity)": round(sy.R1_first_moment, 2),
                  "phi (community synchrony)": round(sy.phi, 3)})
T3 = pd.DataFrame(rows3); T3.to_csv(OUT / "table3_sites.csv", index=False)
say("\n== Table 3 =="); say(T3.to_string(index=False))

(OUT / "summary.txt").write_text("\n".join(LINES), encoding="utf-8")
print(f"\nOutputs -> {OUT}")

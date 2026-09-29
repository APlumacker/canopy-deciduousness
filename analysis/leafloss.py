"""
Metrics of leaf loss for one crown (section 3.e and Appendix S1 of the paper).

Input: the crown leaf-presence series (90th percentile of leaf-presence probability
within the crown, "q90") and the acquisition dates.

    full-foliage level  B    = 95th percentile of the series
    leaf deficit        d(t) = max(0, 1 - q90(t) / B)
    magnitude                = (max - min) / max of the series
    annual deficit profile   : each acquisition covers the period up to the midpoints with
                               its neighbours (capped at +/- 17.5 days); for each day of the
                               year the deficit is averaged over the observed years.
    timing                   = direction of the circular centre of mass of the profile;
                               if the profile is bimodal (R2 > R1), centre of the window
                               (half circle) that holds most of the deficit.
    duration                 = 365 * mean(profile over covered days) / max_i d_i   (Eq. 1)
                               multiplied by the detectability factor
                               g = magnitude^2 / (magnitude^2 + 0.12^2).
"""
import numpy as np
import pandas as pd

HALF_WIN = 17.5          # days: maximum half-window represented by one acquisition
NOISE_MAGNITUDE = 0.12   # median magnitude of a leaf-stable crown under measurement noise


def detectability(magnitude, noise=NOISE_MAGNITUDE):
    return magnitude ** 2 / (magnitude ** 2 + noise ** 2)


def crown_metrics(q, dt, noise=NOISE_MAGNITUDE, return_profile=False):
    """q: crown leaf presence at each date (sorted by date); dt: pandas Series of Timestamps."""
    q = np.asarray(q, float)
    B = np.percentile(q, 95)
    d = np.clip(1 - q / B, 0, None)
    mag = (q.max() - q.min()) / q.max()
    g = detectability(mag, noise)
    out = dict(magnitude=mag, detectability=g, full_foliage=B,
               timing=np.nan, timing_first_moment=np.nan, timing_second_window=np.nan,
               duration=np.nan, duration_uncorrected=np.nan, bimodal=np.nan,
               R1=np.nan, R2=np.nan, coverage=np.nan)
    if d.max() <= 0:
        return out
    t = (dt - dt.iloc[0]).dt.days.values.astype(float)
    mids = (t[1:] + t[:-1]) / 2
    lo = np.r_[t[0] - HALF_WIN, np.maximum(mids, t[1:] - HALF_WIN)]
    hi = np.r_[np.minimum(mids, t[:-1] + HALF_WIN), t[-1] + HALF_WIN]
    lo = np.maximum(lo, t - HALF_WIN)
    hi = np.minimum(hi, t + HALF_WIN)
    doy0 = dt.iloc[0].dayofyear - 1
    obs = np.zeros(365)
    dsum = np.zeros(365)
    for di, a, b in zip(d, lo, hi):
        days = (np.arange(int(np.floor(a)), int(np.ceil(b))) + doy0) % 365
        np.add.at(obs, days, 1.0)
        np.add.at(dsum, days, di)
    ok = obs > 0
    profile = np.where(ok, dsum / np.maximum(obs, 1), 0.0)     # mean annual deficit profile
    if profile.sum() <= 0:
        return out
    width = 365 * profile[ok].mean() / d.max()                   # Eq. 1 (before detectability)
    ang = 2 * np.pi * (np.arange(365) + 0.5) / 365
    C = (profile * np.cos(ang)).sum() / profile.sum()
    S = (profile * np.sin(ang)).sum() / profile.sum()
    R1 = np.hypot(C, S)
    R2 = np.hypot((profile * np.cos(2 * ang)).sum(), (profile * np.sin(2 * ang)).sum()) / profile.sum()
    bimodal = R2 > R1
    first = np.degrees(np.arctan2(S, C)) % 360                   # first circular moment
    second = np.nan
    if bimodal:
        # two opposite windows (axial direction th2 and th2 + 180 deg): timing = centre of
        # the window (+/- 90 deg) that holds most of the deficit
        th2 = (np.degrees(np.arctan2((profile * np.sin(2 * ang)).sum(),
                                     (profile * np.cos(2 * ang)).sum())) / 2) % 180

        def _mass(theta):
            dd = np.abs(((np.degrees(ang) - theta) + 180) % 360 - 180)
            return profile[dd <= 90].sum()

        main, other = (th2, th2 + 180) if _mass(th2) >= _mass(th2 + 180) else (th2 + 180, th2)
        timing = main * 365.25 / 360
        second = (other % 360) * 365.25 / 360
    else:
        timing = first * 365.25 / 360
    out.update(timing=timing, timing_first_moment=first * 365.25 / 360, timing_second_window=second,
               duration=width * g, duration_uncorrected=width, bimodal=bool(bimodal),
               R1=R1, R2=R2, coverage=float(ok.mean()))
    if return_profile:
        out.update(profile=profile, deficit=d)
    return out


def metrics_table(ts, min_dates=5, noise=NOISE_MAGNITUDE):
    """ts: long table site, crown_id, date (YYYYMMDD), q90 -> one row of metrics per crown."""
    ts = ts[["site", "crown_id", "date", "q90"]].dropna().copy()
    ts["dt"] = pd.to_datetime(ts.date.astype(str), format="%Y%m%d")
    rows = []
    for (s, c), g in ts.sort_values("dt").groupby(["site", "crown_id"]):
        if len(g) < min_dates:
            continue
        rows.append(dict(site=s, crown_id=c, n_dates=len(g),
                         **crown_metrics(g.q90.values, g.dt.reset_index(drop=True), noise)))
    return pd.DataFrame(rows)

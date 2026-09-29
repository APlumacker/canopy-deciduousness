#!/usr/bin/env python3
"""
Step 1 of the analyses: metrics of leaf loss for every crown with at least five dates.

Inputs : data/crowns/crown_timeseries_{all,field}.csv.gz
Outputs: results/crown_metrics_{all,field}.csv
         (magnitude, timing, duration, detectability g, R1, R2, bimodal profile, ...)
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from leafloss import metrics_table

for src, dst in [(cfg.TS_ALL, cfg.METRICS_ALL), (cfg.TS_FIELD, cfg.METRICS_FIELD)]:
    ts = pd.read_csv(src, usecols=["site", "crown_id", "date", "q90"])
    m = metrics_table(ts, min_dates=cfg.MIN_DATES, noise=cfg.NOISE_MAGNITUDE)
    m.to_csv(dst, index=False)
    print(f"{dst.name}: {len(m)} crowns {m.site.value_counts().to_dict()}")

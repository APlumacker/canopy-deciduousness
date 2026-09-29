#!/usr/bin/env python3
"""
DINOv2 features at the photo-interpreted points (training data of the leaf-presence model).

Each photo-interpretation file holds the points of one site and one date, with the label
in column II_default: L (leaves), R (leaves, renewal), D (no leaves, "deciduous"), NA
(cloud, orthomosaic defect: excluded). Labels are coded 0 = leaf present (L, R) and
1 = leaf absent (D). For each point, the 384 DINOv2 features of the 14 x 14-pixel patch
containing it are read from the feature raster of the same date.

Inputs : raw/photointerpretation/<site>/*<YYYYMMDD>*.gpkg
         raw/dinov2_features/<site>/<YYYYMMDD>*features*.tif  (pipeline/01)
Output : results/model/photointerpretation_points_features.csv.gz
         (the data release ships this table, with an additional informational crown_id column:
          field-inventory crown containing the point, -1 outside; unused by the analyses)
"""
import re, sys, warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd
import geopandas as gpd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg
warnings.filterwarnings("ignore")

SITES = ["Luki", "Yangambi"]            # photo-interpretation is available at these two sites
LABEL_COL = "II_default"
LABEL_MAP = {"L": 0, "R": 0, "D": 1}    # NA -> excluded
N_WORKERS = 4


def process_one(args):
    gpkg_path, tif_path, site = args
    date = re.search(r"(\d{8})", Path(gpkg_path).stem).group(1)
    gdf = gpd.read_file(gpkg_path)
    gdf = gdf[gdf[LABEL_COL].isin(LABEL_MAP)].copy()
    gdf["label"] = gdf[LABEL_COL].map(LABEL_MAP)
    if gdf.empty:
        return None
    with rasterio.open(tif_path) as src:
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)
        values = np.array(list(src.sample([(g.x, g.y) for g in gdf.geometry])), dtype=np.float32)
        nodata = src.nodata
    valid = ~np.all(values == (nodata if nodata is not None else 0), axis=1)   # outside the raster
    gdf, values = gdf.iloc[valid].reset_index(drop=True), values[valid]
    if gdf.empty:
        return None
    df = pd.DataFrame({"site": site, "date": date, "label": gdf["label"].values,
                       "label_raw": gdf[LABEL_COL].values,
                       **{f"feat_{i:04d}": values[:, i] for i in range(values.shape[1])}})
    print(f"  {site} {date}: {len(df)} points (leaf absent: {int(df.label.sum())})")
    return df


def tasks():
    out = []
    for site in SITES:
        for g in sorted((cfg.RAW_POINTS / site).glob("*.gpkg")):
            m = re.search(r"(\d{8})", g.stem)
            if not m:
                continue
            tifs = [f for f in (cfg.RAW_FEATURES / site).glob(f"{m.group(1)}*.tif") if "feature" in f.name.lower()]
            if tifs:
                out.append((str(g), str(sorted(tifs)[0]), site))
            else:
                print(f"  no feature raster for {site} {m.group(1)}")
    return out


if __name__ == "__main__":
    res = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for f in as_completed([ex.submit(process_one, t) for t in tasks()]):
            if f.result() is not None:
                res.append(f.result())
    df = pd.concat(res, ignore_index=True)
    feat = [c for c in df.columns if c.startswith("feat_")]
    df = df[~df[feat].isna().any(axis=1)].reset_index(drop=True)
    out = cfg.RESULTS_DIR / "model" / "photointerpretation_points_features.csv.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, compression="gzip")
    print(f"{len(df):,} points -> {out}")

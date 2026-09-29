#!/usr/bin/env python3
"""
Per-crown leaf-presence time series (section 3.d).

For each crown polygon and each probability map:
  - the polygon is eroded by a 2-m inner buffer (edge pixels mixing crown and neighbours);
  - crowns with fewer than 5 valid cells on the first map of the site are excluded;
  - the within-crown distribution of leaf-presence probability is summarised
    (n_pix, mean, std, q10, q25, q50, q75, q90); q90 is "crown leaf presence".
    A crown with fewer than 5 valid cells at a date gets n_pix = 0 and missing values.

Usage : python 06_crown_timeseries.py all     # automatically delineated crowns (3 sites)
        python 06_crown_timeseries.py field   # field-identified crowns (Luki, Yangambi)
Inputs: data/crowns/crowns_<site>.gpkg or field_crowns_<site>.gpkg (column crown_id)
        raw/leaf_probability/<site>/<YYYYMMDD>_proba_L.tif (pipeline/05)
Output: results/crown_timeseries_<all|field>.csv.gz (same format as the data release)

Note: on the raw Detectree2SAM outputs the crown identifier was the row index of the layer
(for the field crowns, after keeping crowns covered > 50 % by the inventory polygon); the
released polygons carry that identifier in the crown_id column.
"""
import re, sys, gc, warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rasterio_mask
from shapely.geometry import mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg
warnings.filterwarnings("ignore")

BUFFER_M = -2.0
MIN_PIX = 5
NODATA = -9999.0
N_WORKERS = 4
COLS = ["n_pix", "mean", "std", "q10", "q25", "q50", "q75", "q90"]


def load_crowns(gpkg, raster_crs):
    gdf = gpd.read_file(gpkg)
    if "pct_recouvrement" in gdf.columns:                  # raw inventory layer (not the release)
        gdf = gdf[gdf["pct_recouvrement"] > 0.5].reset_index(drop=True)
    gdf["crown_id"] = gdf["crown_id"].astype(int) if "crown_id" in gdf.columns else gdf.index.astype(int)
    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)
    buf = gdf.geometry.buffer(BUFFER_M)
    ok = ~(buf.is_empty | ~buf.is_valid | (buf.area == 0))
    gdf = gdf[ok].copy().reset_index(drop=True)
    gdf["geometry"] = buf[ok].values
    return gdf[["crown_id", "geometry"]]


def crown_values(src, geom):
    out, _ = rasterio_mask(src, [mapping(geom)], crop=True, nodata=NODATA, all_touched=False)
    px = out[0].flatten()
    return px[(px != NODATA) & np.isfinite(px)]


def n_valid(src, geom):
    try:
        return len(crown_values(src, geom))
    except Exception:
        return 0


def extract_date(args):
    tif, gpkg, site, keep = args
    date = re.search(r"(\d{8})", Path(tif).stem).group(1)
    with rasterio.open(tif) as src:
        gdf = load_crowns(gpkg, src.crs)
        gdf = gdf[gdf.crown_id.isin(keep)]
        recs = []
        for cid, geom in zip(gdf.crown_id, gdf.geometry):
            rec = dict(site=site, crown_id=cid, date=date, n_pix=0,
                       **{c: np.nan for c in COLS[1:]})
            try:
                v = crown_values(src, geom)
                if len(v) >= MIN_PIX:
                    q = np.percentile(v, [10, 25, 50, 75, 90])
                    rec.update(n_pix=len(v), mean=float(np.mean(v)), std=float(np.std(v)),
                               q10=float(q[0]), q25=float(q[1]), q50=float(q[2]), q75=float(q[3]), q90=float(q[4]))
            except Exception:
                pass
            recs.append(rec)
    gc.collect()
    return pd.DataFrame(recs)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    polygons = cfg.CROWN_POLYGONS if which == "all" else cfg.FIELD_CROWN_POLYGONS
    parts = []
    for site, gpkg in polygons.items():
        tifs = sorted((cfg.RAW_PROBA / site).glob("*_proba_L.tif"))
        if not tifs:
            print(f"{site}: no probability map"); continue
        with rasterio.open(tifs[0]) as src:                  # crown filter on the first map
            gdf = load_crowns(gpkg, src.crs)
            keep = {cid for cid, g in zip(gdf.crown_id, gdf.geometry) if n_valid(src, g) >= MIN_PIX}
        print(f"{site}: {len(keep)} crowns x {len(tifs)} dates")
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            for f in as_completed([ex.submit(extract_date, (str(t), str(gpkg), site, keep)) for t in tifs]):
                parts.append(f.result())
    df = pd.concat(parts, ignore_index=True).sort_values(["site", "crown_id", "date"]).reset_index(drop=True)
    out = cfg.RESULTS_DIR / f"crown_timeseries_{which}.csv.gz"
    df.to_csv(out, index=False, compression="gzip")
    print(f"{len(df):,} rows -> {out}")

#!/usr/bin/env python3
"""
Terrain descriptors of the field-identified crowns (Table S1).

Mean of each terrain raster (from 07a_terrain_rasters.R: elevation, slope, TWI, PISR, profile
and plan curvature, 1-m LiDAR DTM) within each field crown polygon (no buffer; at least 3
valid cells).

Inputs : data/crowns/field_crowns_<site>.gpkg, raw/dtm/<site>/{elev,slope,twi,pisr,
         curvature_profile,curvature_plan}.tif
Output : results/field_crowns_topography.csv (same format as the data release)
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window, from_bounds as window_from_bounds
from rasterio.features import rasterize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg

FILES = {"topo_elevation": "elev.tif", "topo_slope": "slope.tif", "topo_twi": "twi.tif", "topo_pisr": "pisr.tif",
         "topo_curv_profile": "curvature_profile.tif", "topo_curv_plan": "curvature_plan.tif"}
MIN_PIX = 3


def zonal_mean(raster, geoms):
    out = []
    with rasterio.open(raster) as src:
        nd, H, W = src.nodata, src.height, src.width
        for g in geoms:
            win = window_from_bounds(*g.bounds, src.transform)
            r0, c0 = max(0, int(np.floor(win.row_off))), max(0, int(np.floor(win.col_off)))
            r1, c1 = min(H, int(np.ceil(win.row_off + win.height))), min(W, int(np.ceil(win.col_off + win.width)))
            if r1 <= r0 or c1 <= c0:
                out.append(np.nan); continue
            w = Window(c0, r0, c1 - c0, r1 - r0)
            sub = src.read(1, window=w).astype(float)
            if nd is not None:
                sub[sub == nd] = np.nan
            m = rasterize([(g, 1)], out_shape=sub.shape, transform=src.window_transform(w), fill=0, dtype="uint8")
            px = sub[m == 1]; px = px[np.isfinite(px)]
            out.append(float(px.mean()) if len(px) >= MIN_PIX else np.nan)
    return np.array(out, float)


if __name__ == "__main__":
    parts = []
    for site, gpkg in cfg.FIELD_CROWN_POLYGONS.items():
        gdf = gpd.read_file(gpkg)
        with rasterio.open(cfg.RAW_DTM / site / FILES["topo_elevation"]) as s:
            gdf = gdf.to_crs(s.crs) if gdf.crs != s.crs else gdf
        gdf = gdf[~(gdf.geometry.is_empty | ~gdf.geometry.is_valid | (gdf.geometry.area == 0))]
        df = pd.DataFrame({"site": site, "crown_id": gdf.crown_id.values})
        for v, f in FILES.items():
            df[v] = zonal_mean(cfg.RAW_DTM / site / f, list(gdf.geometry))
        parts.append(df.dropna())
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(cfg.RESULTS_DIR / "field_crowns_topography.csv", index=False)
    print(f"{len(out)} crowns -> {cfg.RESULTS_DIR / 'field_crowns_topography.csv'}")

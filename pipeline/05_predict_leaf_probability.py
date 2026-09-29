#!/usr/bin/env python3
"""
Leaf-presence probability maps: the model is applied to every 14 x 14-pixel patch of every
DINOv2 feature raster (section 3.c). Leaf-presence probability = 1 - p(leaf absence).

Inputs : raw/dinov2_features/<site>/*features*.tif (pipeline/01), model/leaf_presence_logreg.joblib
Output : raw/leaf_probability/<site>/<YYYYMMDD>_proba_L.tif (float32, nodata -9999, LZW)
         Maps are read in tiles to limit memory; existing outputs are skipped.
"""
import gc, re, sys, warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import rasterio
from rasterio.windows import Window
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg
warnings.filterwarnings("ignore")

TILE_SIZE = 1024
N_WORKERS = 4
NODATA = -9999.0


def predict_one(args):
    tif_path, out_dir, model_path = map(Path, args)
    date = re.search(r"(\d{8})", tif_path.stem).group(1)
    out_path = out_dir / f"{date}_proba_L.tif"
    if out_path.exists():
        return f"skip {out_path.name}"
    bundle = joblib.load(model_path)
    model, scaler = bundle["model"], bundle["scaler"]
    try:
        with rasterio.open(tif_path) as src:
            nb, H, W, nodata = src.count, src.height, src.width, src.nodata
            meta = dict(driver="GTiff", dtype="float32", count=1, height=H, width=W, crs=src.crs,
                        transform=src.transform, nodata=NODATA, compress="lzw", tiled=True,
                        blockxsize=256, blockysize=256)
            with rasterio.open(out_path, "w", **meta) as dst:
                for r0 in range(0, H, TILE_SIZE):
                    for c0 in range(0, W, TILE_SIZE):
                        h, w = min(TILE_SIZE, H - r0), min(TILE_SIZE, W - c0)
                        win = Window(c0, r0, w, h)
                        X = src.read(window=win).reshape(nb, h * w).T.astype(np.float32)
                        valid = ~np.all(X == (nodata if nodata is not None else 0), axis=1)
                        p = np.full(h * w, NODATA, dtype=np.float32)
                        if valid.any():
                            p[valid] = (1.0 - model.predict_proba(scaler.transform(X[valid]))[:, 1]).astype(np.float32)
                        dst.write(p.reshape(1, h, w), window=win)
                        del X, p, valid
                        gc.collect()
        return f"ok {out_path.name}"
    except Exception as e:
        if out_path.exists():
            out_path.unlink()
        return f"error {tif_path.name}: {e}"


if __name__ == "__main__":
    jobs = []
    for site in cfg.SITES:
        out_dir = cfg.RAW_PROBA / site; out_dir.mkdir(parents=True, exist_ok=True)
        for t in sorted((cfg.RAW_FEATURES / site).glob("*.tif")):
            if "feature" in t.name.lower():
                jobs.append((str(t), str(out_dir), str(cfg.MODEL)))
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for f in as_completed([ex.submit(predict_one, j) for j in jobs]):
            print(" ", f.result())

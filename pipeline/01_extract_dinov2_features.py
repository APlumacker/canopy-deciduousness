#!/usr/bin/env python3
"""
DINOv2 features of every orthomosaic (section 3.b).

Orthomosaics are tiled into 518 x 518-pixel tiles with a 126-pixel overlap and passed through
DINOv2-ViT-S/14 (Oquab et al., 2024); each 14 x 14-pixel patch receives a 384-dimensional
feature vector. One 384-band raster is written per orthomosaic.

Input : raw/orthomosaics/<site>/<YYYYMMDD>*.tif        (RGB, Time-SIFT + AROSICS)
        optional vector mask of the study area: raw/orthomosaics/<site>/mask.gpkg
Output: raw/dinov2_features/<site>/<YYYYMMDD>_features.tif
Requires a GPU for reasonable run times (torch, torchvision; the model is downloaded
from torch.hub, facebookresearch/dinov2).
"""
import re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config as cfg
from dinov2_extractor import process_geotiff_with_dinov2

MODEL, TILE, OVERLAP, BATCH = "dinov2_vits14", 518, 126, 8

if __name__ == "__main__":
    sites = sys.argv[1:] or cfg.SITES
    for site in sites:
        src_dir, out_dir = cfg.RAW_ORTHO / site, cfg.RAW_FEATURES / site
        out_dir.mkdir(parents=True, exist_ok=True)
        mask = src_dir / "mask.gpkg"
        for tif in sorted(src_dir.glob("*.tif")):
            m = re.search(r"(\d{8})", tif.stem)
            if not m:
                continue
            out = out_dir / f"{m.group(1)}_features.tif"
            if out.exists():
                continue
            print(f"{site} {m.group(1)}")
            process_geotiff_with_dinov2(str(tif), str(out), model_name=MODEL, tile_size=TILE, overlap=OVERLAP,
                                        batch_size=BATCH, mask_path=str(mask) if mask.exists() else None)

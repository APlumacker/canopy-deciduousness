"""
Paths and constants shared by all scripts.

DATA_DIR    : the data folder Article2_data (see README). Default: ./data
RESULTS_DIR : where analysis outputs are written. Default: ./results
Both can be overridden with the environment variables of the same name.

Raw inputs of the pipeline/ scripts (orthomosaics, DINOv2 feature rasters,
photo-interpretation points per date, LiDAR DTM) are not part of the data
release; their locations are set with RAW_DIR (default ./raw), see README.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
RAW_DIR = Path(os.environ.get("RAW_DIR", ROOT / "raw"))

# --- data release layout -----------------------------------------------------
CROWNS = DATA_DIR / "crowns"
TS_ALL = CROWNS / "crown_timeseries_all.csv.gz"          # automatically delineated crowns
TS_FIELD = CROWNS / "crown_timeseries_field.csv.gz"      # field-identified crowns
FIELD_CROWNS = CROWNS / "field_crowns.csv"                # species, genus, family, leaf habit
FIELD_TOPO = CROWNS / "field_crowns_topography.csv"       # six LiDAR terrain descriptors
CLIMATE = {s: DATA_DIR / "climate" / f"climate_daily_{s}.csv" for s in ("Luki", "Mbalmayo", "Yangambi")}
POINT_FEATURES = DATA_DIR / "model" / "photointerpretation_points_features.csv.gz"
MODEL = ROOT / "model" / "leaf_presence_logreg.joblib"
FIG3_THUMBS = ROOT / "assets" / "fig3_crown132_yangambi"

# --- analysis outputs ----------------------------------------------------------
METRICS_ALL = RESULTS_DIR / "crown_metrics_all.csv"
METRICS_FIELD = RESULTS_DIR / "crown_metrics_field.csv"

# --- constants of the paper ----------------------------------------------------
SITES = ["Luki", "Mbalmayo", "Yangambi"]
NOISE_MAGNITUDE = 0.12      # median magnitude of a leaf-stable crown under measurement noise alone (Appendix S1)
STRONG_LOSS = 0.6           # magnitude threshold for "strong leaf loss" (section 3.g)
MIN_DATES = 5               # minimum number of dates per crown


# --- raw inputs of the pipeline/ scripts (not distributed, see README) --------------------
RAW_ORTHO = RAW_DIR / "orthomosaics"           # <site>/<YYYYMMDD>*.tif, RGB orthomosaics
RAW_FEATURES = RAW_DIR / "dinov2_features"     # <site>/<YYYYMMDD>*features*.tif, 384 bands
RAW_POINTS = RAW_DIR / "photointerpretation"   # <site>/*<YYYYMMDD>*.gpkg, column II_default (L, R, D, NA)
RAW_PROBA = RAW_DIR / "leaf_probability"       # <site>/<YYYYMMDD>_proba_L.tif (output of pipeline/05)
RAW_DTM = RAW_DIR / "dtm"                      # <site>/dtm.tif (+ terrain rasters from pipeline/07a)
CROWN_POLYGONS = {s: CROWNS / f"crowns_{s}.gpkg" for s in SITES}
FIELD_CROWN_POLYGONS = {s: CROWNS / f"field_crowns_{s}.gpkg" for s in ("Luki", "Yangambi")}

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

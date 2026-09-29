# Article2_data: crown-scale canopy deciduousness from UAV time series in Central Africa

Data supporting Plumacker et al., *Continuous UAV monitoring across canopy observatories reveals
diversity and climatic control of canopy deciduousness in Central Africa*. The analysis code is at
https://github.com/APlumacker/canopy-deciduousness. Copy the content of this folder into the `data/`
folder of the code repository and run `bash analysis/run_all.sh`.

Sites: Luki (DR Congo), Mbalmayo (Cameroon), Yangambi (DR Congo), monitored within the CanObs network
(https://www.canobs.net). Dates are `YYYYMMDD`. Coordinates are in the UTM zone of each site
(Luki EPSG:32733, Mbalmayo EPSG:32632, Yangambi EPSG:32635).

Stored in the University of Liège data repository; contact: Antoine Plumacker (antoine.plumacker@uliege.be).

## Contents

| File | Content |
|---|---|
| `crowns/crown_timeseries_all.csv.gz` | Leaf-presence time series of the 23,163 automatically delineated crowns retained after erosion (23,119 with ≥ 5 dates are analysed): one row per crown and date |
| `crowns/crown_timeseries_field.csv.gz` | Same for the 969 field-identified crowns (Luki, Yangambi) |
| `crowns/crowns_<site>.gpkg` | Polygons of the automatically delineated crowns (Detectree2SAM), before the 2-m inner buffer |
| `crowns/field_crowns_<site>.gpkg` | Polygons of the field-identified crowns |
| `crowns/field_crowns.csv` | Species, genus, family and leaf habit of the 968 field-identified crowns with ≥ 5 dates |
| `crowns/field_crowns_topography.csv` | Six terrain descriptors of the field-identified crowns (1-m LiDAR DTM) |
| `climate/climate_daily_<site>.csv` | Daily rainfall (IMERG V07) and vapour pressure deficit (NASA POWER) |
| `model/photointerpretation_points_features.csv.gz` | Labels and DINOv2 features of the 199,681 photo-interpreted points with a valid label and feature value (Luki, Yangambi; 207,178 annotated points before excluding NA labels and points outside the rasters) used to train and validate the leaf-presence model |
| `model/leaf_presence_logreg.joblib` | The leaf-presence model (also in the code repository) |
| `Leaf_presence_probability_maps.docx` | The leaf-presence probability map of every acquisition (27 at Luki, 46 at Mbalmayo, 44 at Yangambi), down-sampled for display |

Orthomosaics and full-resolution probability rasters are not included; they are available through
the CanObs network on request.

## Columns

**crown_timeseries_*.csv.gz** — `site`, `crown_id`, `date`; `n_pix`: number of valid cells of the
probability map within the crown eroded by 2 m (0 when fewer than 5: statistics missing); `mean`,
`std`, `q10`, `q25`, `q50`, `q75`, `q90`: distribution of leaf-presence probability within the crown.
`q90` is the *crown leaf presence* used in all analyses.

**crowns_<site>.gpkg / field_crowns_<site>.gpkg** — `crown_id` (joins the time series), `area_m2`,
geometry.

**field_crowns.csv** — `site`, `crown_id`; `species`: species name used in the analyses (field
inventory); `species_cofortrait`, `match_method`: matched name in CoForTrait (Bénédet et al., 2022)
and matching method (exact, fuzzy, genus); `genus`, `family` (APG IV); `leaf_habit`: deciduous,
semi-deciduous, evergreen or unknown (CoForTrait); `leaf_habit_source`: CoForTrait, or the
source used when the species was missing from CoForTrait.

**field_crowns_topography.csv** — `site`, `crown_id`, mean within the crown of `topo_elevation` (m),
`topo_slope` (degrees), `topo_twi` (topographic wetness index, D-infinity), `topo_pisr` (potential
incoming solar radiation, annual), `topo_curv_profile`, `topo_curv_plan`.

**climate_daily_<site>.csv** — `date`; `precip_imerg_mm`: daily precipitation from IMERG V07, mean
over the grid cells within 5 km of the site; `vpd_nasapower_kpa`: daily vapour pressure deficit from
the 2-m air and dew-point temperatures of NASA POWER (MERRA-2).

**photointerpretation_points_features.csv.gz** — `site`, `date`; `label`: 1 = leaves absent
(photo-interpretation code D), 0 = leaves present (codes L and R); `label_raw`; `crown_id`:
field-inventory crown containing the point (-1 outside; not used in the analyses);
`feat_0000`–`feat_0383`: DINOv2-ViT-S/14 features of the 14 × 14-pixel patch containing the point.

**leaf_presence_logreg.joblib** — Python dictionary (joblib, scikit-learn 1.7.2): `model`
(LogisticRegression, L2, C = 0.01), `scaler` (StandardScaler), `feature_cols`; the positive class is
leaf absence, so leaf-presence probability = 1 − `model.predict_proba(scaler.transform(X))[:, 1]`.

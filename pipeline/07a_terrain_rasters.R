# =============================================================================
# Derivation of topographic variables from a DEM
# Author: Antoine Plumacker - CANOPI / Gembloux Agro-Bio Tech
#
# Computed variables:
#   - Elevation  (taken directly from the DEM)
#   - Slope      (degrees; aspect is also computed and used for PISR)
#   - TWI        (Topographic Wetness Index, WhiteboxTools D-infinity)
#   - Curvature  (profile + plan)
#   - PISR       (Potential Incoming Solar Radiation - annual insolation)
#
# Output: GeoTIFF rasters in the same folder as the DEM
# =============================================================================

library(terra)
library(whitebox)  # whitebox::wbt_*  - install.packages("whitebox"); whitebox::install_whitebox()

# =============================================================================
# 0. PARAMETERS - one site per list element
# =============================================================================

RAW_DIR <- Sys.getenv("RAW_DIR", "raw")

# The 1-m LiDAR DTM of each site is expected at raw/dtm/<site>/dtm.tif; outputs are written next to it.
SITES <- list(
  Luki = list(
    DEM_PATH = file.path(RAW_DIR, "dtm", "Luki", "dtm.tif"),
    SITE_LAT_DEG = -5.6     # Luki, DRC
  ),
  Yangambi = list(
    DEM_PATH = file.path(RAW_DIR, "dtm", "Yangambi", "dtm.tif"),
    SITE_LAT_DEG = 0.85     # Yangambi, DRC
  ),
  Mbalmayo = list(
    DEM_PATH = file.path(RAW_DIR, "dtm", "Mbalmayo", "dtm.tif"),
    SITE_LAT_DEG = 3.5      # Mbalmayo, Cameroon
  )
)

# Months for PISR integration (annual integration over 12 months)
PISR_MONTHS <- 1:12

# =============================================================================
# PIPELINE - run for each site in SITES
# =============================================================================

run_topo_site <- function(site_name, DEM_PATH, SITE_LAT_DEG, PISR_MONTHS = 1:12) {

OUT_DIR <- dirname(DEM_PATH)   # same folder as the DEM

message(sprintf("\n========== Site: %s ==========", site_name))

# =============================================================================
# 1. LOAD DEM
# =============================================================================

message("-- 1. Loading DEM...")
dem <- rast(DEM_PATH)
message(sprintf("   Resolution: %.1f m | Dimensions: %d x %d | CRS: %s",
                res(dem)[1], nrow(dem), ncol(dem), crs(dem, describe = TRUE)$name))

# Output paths
path_elev <- file.path(OUT_DIR, "elev.tif")
path_slope <- file.path(OUT_DIR, "slope.tif")
path_aspect <- file.path(OUT_DIR, "aspect.tif")
path_twi <- file.path(OUT_DIR, "twi.tif")
path_curv_prof <- file.path(OUT_DIR, "curvature_profile.tif")
path_curv_plan <- file.path(OUT_DIR, "curvature_plan.tif")
path_pisr <- file.path(OUT_DIR, "pisr.tif")

# =============================================================================
# 2. ELEVATION (direct copy)
# =============================================================================

message("-- 2. Elevation...")
writeRaster(dem, path_elev, overwrite = TRUE)
elev <- dem

# =============================================================================
# 3. SLOPE & ASPECT (native terra)
# =============================================================================

message("-- 3. Slope & Aspect...")

slope <- terrain(dem, v = "slope", unit = "degrees", neighbors = 8)
writeRaster(slope, path_slope, overwrite = TRUE)

aspect <- terrain(dem, v = "aspect", unit = "degrees", neighbors = 8)
writeRaster(aspect, path_aspect, overwrite = TRUE)

message(sprintf("   Slope: min=%.1f deg max=%.1f deg",
                global(slope, "min", na.rm=TRUE)$min,
                global(slope, "max", na.rm=TRUE)$max))

# =============================================================================
# 4. TWI - via WhiteboxTools (hydrological, D-infinity flow)
# =============================================================================
# WhiteboxTools gives a hydrologically consistent TWI (D-inf flow accumulation),
# better than the simplified ln(As/tan(beta)) approach

message("-- 4. TWI (WhiteboxTools - D-infinity)...")

# Initialise Whitebox
whitebox::wbt_init()

# Intermediate files
path_dem_filled   <- file.path(OUT_DIR, "dem_filled.tif")
path_flow_dir     <- file.path(OUT_DIR, "flow_dir_dinf.tif")
path_flow_acc     <- file.path(OUT_DIR, "flow_acc_dinf.tif")
path_slope_rad    <- file.path(OUT_DIR, "slope_rad.tif")

# 4a. Fill depressions (required before flow accumulation)
whitebox::wbt_fill_depressions_wang_and_liu(
  dem    = DEM_PATH,
  output = path_dem_filled
)

# 4b. D-infinity flow accumulation
whitebox::wbt_d_inf_flow_accumulation(
  input  = path_dem_filled,
  output = path_flow_acc,
  out_type = "Specific Contributing Area"  # As in m2/m
)

# 4c. TWI = ln(As / tan(slope_radians))
flow_acc  <- rast(path_flow_acc)
slope_rad <- terrain(rast(path_dem_filled), v = "slope", unit = "radians", neighbors = 8)

# Avoid tan(0): clamp slope to a minimum of 0.001 rad
slope_rad_safe <- max(slope_rad, 0.001)
twi <- log(flow_acc / tan(slope_rad_safe))

writeRaster(twi, path_twi, overwrite = TRUE)

# Remove intermediate files
file.remove(path_dem_filled, path_flow_dir, path_flow_acc)

message(sprintf("   TWI: min=%.2f max=%.2f",
                global(twi, "min", na.rm=TRUE)$min,
                global(twi, "max", na.rm=TRUE)$max))

# =============================================================================
# 5. CURVATURE (profile + plan) - native terra
# =============================================================================

message("-- 5. Curvature (profile + plan)...")

curv <- terrain(dem, v = c("TPI", "TRI"), neighbors = 8)
# terra does not compute profile/plan curvature natively, so they are
# derived analytically from the second derivatives of the DEM

# Derivatives via convolution (Zevenbergen & Thorne 1987 kernel)
dem_mat <- as.matrix(dem, wide = TRUE)
res_m   <- res(dem)[1]

# Per-pixel curvature computation via focal
# Uses the Evans (1980) / Zevenbergen & Thorne (1987) formulation
compute_curvatures <- function(dem, res_m) {
  # Second-order partial derivatives via 3x3 focal
  # z1..z9: 3x3 window centred on the pixel
  # D = (z4 + z6 - 2z5) / (2 * res^2)   -> d2z/dx2
  # E = (z2 + z8 - 2z5) / (2 * res^2)   -> d2z/dy2
  # F = (-z1 + z3 + z7 - z9) / (4*res^2) -> d2z/dxdy
  
  # 3x3 kernel coefficients
  kern_D <- matrix(c(0, 0, 0,
                     1,-2, 1,
                     0, 0, 0), 3, 3) / (2 * res_m^2)
  kern_E <- matrix(c(0, 1, 0,
                     0,-2, 0,
                     0, 1, 0), 3, 3) / (2 * res_m^2)
  kern_F <- matrix(c(-1, 0, 1,
                     0, 0, 0,
                     1, 0,-1), 3, 3) / (4 * res_m^2)
  # First derivatives
  kern_G <- matrix(c(0, 0, 0,
                     -1, 0, 1,
                     0, 0, 0), 3, 3) / (2 * res_m)  # dz/dx
  kern_H <- matrix(c(0,-1, 0,
                     0, 0, 0,
                     0, 1, 0), 3, 3) / (2 * res_m)  # dz/dy
  
  D <- focal(dem, w = kern_D)
  E <- focal(dem, w = kern_E)
  F_ <- focal(dem, w = kern_F)
  G <- focal(dem, w = kern_G)
  H <- focal(dem, w = kern_H)
  
  denom <- G^2 + H^2
  
  # Profile curvature (along flow direction) - negative = convex
  curv_prof <- -2 * (D * G^2 + E * H^2 + F_ * G * H) / (denom + 1e-10)
  
  # Plan curvature (perpendicular to flow) - positive = divergent
  curv_plan <- 2 * (D * H^2 + E * G^2 - F_ * G * H) / (denom + 1e-10)
  
  list(profile = curv_prof, plan = curv_plan)
}

curvs <- compute_curvatures(dem, res_m)

writeRaster(curvs$profile, path_curv_prof, overwrite = TRUE)
writeRaster(curvs$plan,    path_curv_plan, overwrite = TRUE)

message("   Curvatures computed (profile + plan)")

# =============================================================================
# 6. PISR - Potential Incoming Solar Radiation (annual integration)
# =============================================================================
# FAO-56 formula / Allen et al. 1998
# Ra = (24*60/pi) * Gsc * dr * (ws*sin(phi)*sin(delta) + cos(phi)*cos(delta)*sin(ws))
# Corrected for slope and aspect (surface inclination)

message("-- 6. PISR (annual insolation, FAO-56 formula + topographic correction)...")

compute_pisr_annual <- function(dem, slope_deg, aspect_deg, lat_deg, months = 1:12) {
  # Constants
  Gsc <- 0.0820  # MJ m-2 min-1 (solar constant)
  lat_rad <- lat_deg * pi / 180
  
  # Convert rasters to radians
  slope_rad  <- slope_deg  * (pi / 180)
  aspect_rad <- aspect_deg * (pi / 180)
  
  # Accumulator raster initialised to zero
  pisr_annual <- slope_deg * 0
  values(pisr_annual) <- 0
  
  for (m in months) {
    # Mid-month day of year
    doy <- round(30.4 * (m - 0.5))
    
    # Solar declination (scalar, radians)
    delta <- 0.409 * sin(2 * pi * doy / 365 - 1.39)
    
    # Inverse relative Earth-Sun distance (scalar)
    dr <- 1 + 0.033 * cos(2 * pi * doy / 365)
    
    # Sunset hour angle - horizontal surface (scalar)
    cos_ws_h <- -tan(lat_rad) * tan(delta)
    cos_ws_h <- min(max(cos_ws_h, -1), 1)
    ws <- acos(cos_ws_h)
    
    # cos(incidence angle) on the inclined surface - raster
    # Iqbal (1983) / Allen (2006) formula
    cos_theta <- sin(delta) * (sin(lat_rad) * cos(slope_rad) -
                                 cos(lat_rad) * sin(slope_rad) * cos(aspect_rad)) +
      cos(delta) * cos(ws) * (cos(lat_rad) * cos(slope_rad) +
                                sin(lat_rad) * sin(slope_rad) * cos(aspect_rad)) +
      cos(delta) * sin(slope_rad) * sin(aspect_rad) * sin(ws)
    
    # Ra on inclined surface (MJ m-2 day-1) - clamped at 0
    Ra_topo <- (24 * 60 / pi) * Gsc * dr * cos_theta * ws
    Ra_topo <- max(Ra_topo, 0)  # terra::max - works on SpatRaster
    
    pisr_annual <- pisr_annual + Ra_topo
    message(sprintf("   Month %2d/12 - mean Ra: %.1f MJ/m2/day",
                    m, global(Ra_topo, "mean", na.rm = TRUE)$mean))
  }
  
  return(pisr_annual)
}

pisr <- compute_pisr_annual(dem, slope, aspect, SITE_LAT_DEG, PISR_MONTHS)
writeRaster(pisr, path_pisr, overwrite = TRUE)

message(sprintf("   PISR: min=%.0f max=%.0f MJ/m2/yr",
                global(pisr, "min", na.rm=TRUE)$min,
                global(pisr, "max", na.rm=TRUE)$max))

# =============================================================================
# 7. SUMMARY & FINAL STACK
# =============================================================================

message("\n-- 7. Final stack & check...")

stack_topo <- c(
  rast(path_elev),
  rast(path_slope),
  rast(path_twi),
  rast(path_curv_prof),
  rast(path_curv_plan),
  rast(path_pisr)
)
names(stack_topo) <- c("Elevation", "Slope", "TWI", "Curv_Profile", "Curv_Plan", "PISR")

# Quick statistics
stats <- global(stack_topo, c("min", "max", "mean"), na.rm = TRUE)
message("\n   Variable statistics:")
print(round(stats, 3))

# Export full stack (optional)
writeRaster(stack_topo,
            file.path(OUT_DIR, "topo_stack.tif"),
            overwrite = TRUE)

message(sprintf("\nDone - all rasters in:\n   %s", OUT_DIR))
message("  ├── elev.tif")
message("  ├── slope.tif")
message("  ├── aspect.tif")
message("  ├── twi.tif")
message("  ├── curvature_profile.tif")
message("  ├── curvature_plan.tif")
message("  ├── pisr.tif")
message("  └── topo_stack.tif  (multi-layer stack)")
message("\nThese paths can be passed directly to station_topo.R")

} # end run_topo_site()

# =============================================================================
# RUN FOR ALL SITES
# =============================================================================

for (site_name in names(SITES)) {
  cfg <- SITES[[site_name]]
  run_topo_site(site_name, cfg$DEM_PATH, cfg$SITE_LAT_DEG, PISR_MONTHS)
}


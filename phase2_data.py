#!/usr/bin/env python3
"""
PHASE 2: Align layers, derive the eight flood-conditioning factors, standardise to 0-1.
Lokoja LGA flood susceptibility project.

Input  : GeoTIFFs exported by phase1_data_acquisition.js (EPSG:32632), in DATA_DIR.
         The DEM, LGA mask and water occurrence are the BUFFERED exports (*_buf.tif):
         they cover the LGA plus a buffer so that flow accumulation, streams, distance
         to drainage and drainage density are not cut off at the LGA boundary.
Output : in OUT_DIR
           raw_*.tif   factor layers on the common 30 m grid (physical units)
           std_*.tif   the eight factors rescaled to 0-1 (1 = more susceptible)
           perm_water.tif, streams.tif, lga_valid.tif
           factor_summary.csv, factor_correlation.csv, phase2_log.json, factors_overview.png
         All outputs are masked to the LGA (nodata outside it), but they sit on the
         larger buffered grid.

Install : pip install rasterio scipy numpy pysheds matplotlib
Run     : python phase2_factors.py

Edit the CONFIG block first. Phase 3 (AHP weights and the susceptibility index)
reads the std_*.tif files.
"""
import json
import os
import sys

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy import ndimage as ndi
from scipy.signal import fftconvolve

# pysheds 0.5 calls np.in1d, which NumPy 2.4 removed. This shim keeps it working.
if not hasattr(np, "in1d"):
    np.in1d = np.isin

# ------------------------------ CONFIG ---------------------------------
DATA_DIR = "data/phase1/Lokoja_flood_phase1"
OUT_DIR = "data/phase2"

DEM_FILE = "dem_cop30_30m_buf.tif"               # buffered; master grid and DEM used for all terrain work
MASK_FILE = "lga_mask_buf.tif"                   # LGA-only mask (1 inside, 0 outside) on the buffered grid
WATER_FILE = "water_occurrence_30m_buf.tif"      # buffered JRC water occurrence
DEM_ALT_FILE = "dem_srtm_30m.tif"                # optional: comparison only (set None to skip)
RAIN_FILE = "rain_chirps_mean_annual_native.tif"
NDVI_FILE = "ndvi_s2_median_10m.tif"
LULC_DW_FILE = "lulc_dynamicworld_10m.tif"
LULC_WC_FILE = "lulc_worldcover_10m.tif"

LULC_SOURCE = "dynamicworld"            # "worldcover" or "dynamicworld"
PERM_WATER_OCC = 90                     # JRC occurrence (%) treated as permanent water
STREAM_THRESHOLD_KM2 = 2.0              # contributing area that starts a stream (calibrate!)
DRAINAGE_DENSITY_RADIUS_M = 1500        # circular window for drainage density
RAIN_RESAMPLING = "nearest"             # "nearest" keeps the native blocky CHIRPS pixels; "bilinear" smooths
SLOPE_GENTLE_IS_HIGH = True             # gentler slope = higher susceptibility (common convention)
CLIP_PCT = (1, 99)                      # (0, 100) = pure min-max; (1, 99) trims outliers
SAMPLE_N = 200000                       # cells sampled for the correlation / VIF check
RANDOM_SEED = 42

# Susceptibility score (0-1) per land-cover class. EDIT to match your methodology.
LULC_SCORES_DW = {0: 1.0, 1: 0.2, 2: 0.5, 3: 0.9, 4: 0.6, 5: 0.4, 6: 0.9, 7: 0.8}
LULC_SCORES_WC = {10: 0.2, 20: 0.4, 30: 0.5, 40: 0.6, 50: 0.9, 60: 0.8,
                  70: 0.0, 80: 1.0, 90: 0.9, 95: 0.5, 100: 0.3}
LULC_NAMES_DW = {0: "water", 1: "trees", 2: "grass", 3: "flooded veg", 4: "crops",
                 5: "shrub", 6: "built", 7: "bare", 8: "snow/ice"}
LULC_NAMES_WC = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built", 60: "bare",
                 70: "snow/ice", 80: "water", 90: "wetland", 95: "mangrove", 100: "moss"}
# -----------------------------------------------------------------------

NODATA_F = -9999.0
NODATA_U = 255


def path(name):
    return os.path.join(DATA_DIR, name)


def need(name):
    p = path(name)
    if not os.path.exists(p):
        sys.exit("Missing input file: %s" % p)
    return p


def log(msg=""):
    print(msg, flush=True)


# --------------------------- master grid -------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
with rasterio.open(need(DEM_FILE)) as src:
    M_PROFILE = src.profile.copy()
    M_TRANSFORM = src.transform
    M_CRS = src.crs
    M_SHAPE = (src.height, src.width)
    dem_raw = src.read(1).astype("float32")
CELL = abs(M_TRANSFORM.a)
CELL_KM2 = (CELL / 1000.0) ** 2
log("Master grid: %d x %d cells, %.1f m, CRS %s" % (M_SHAPE[1], M_SHAPE[0], CELL, M_CRS))
if M_CRS is None or M_CRS.to_epsg() != 32632:
    log("WARNING: master CRS is not EPSG:32632. Distances assume a metric CRS.")


def align(fname, resampling, nodata, dtype):
    """Resample a Phase 1 raster onto the master grid."""
    dst = np.full(M_SHAPE, nodata, dtype=dtype)
    with rasterio.open(need(fname)) as src:
        reproject(
            source=rasterio.band(src, 1), destination=dst,
            src_transform=src.transform, src_crs=src.crs, src_nodata=nodata,
            dst_transform=M_TRANSFORM, dst_crs=M_CRS, dst_nodata=nodata,
            resampling=resampling,
        )
    return dst


def write(name, arr, nodata=NODATA_F, dtype="float32"):
    prof = M_PROFILE.copy()
    prof.update(dtype=dtype, count=1, nodata=nodata, compress="deflate")
    out = arr.copy()
    if np.issubdtype(out.dtype, np.floating):
        out = np.where(np.isfinite(out), out, nodata)
    with rasterio.open(os.path.join(OUT_DIR, name), "w", **prof) as dst:
        dst.write(out.astype(dtype), 1)


# ---------------------------- valid area -------------------------------
# dem_full / region cover the LGA plus the buffer and are used for hydrology.
# valid covers the LGA only and is used for everything that is reported or scored.
dem = np.where(dem_raw == NODATA_F, np.nan, dem_raw).astype("float32")
if os.path.exists(path(MASK_FILE)):
    lga = align(MASK_FILE, Resampling.nearest, 0, "uint8") == 1
else:
    lga = np.ones(M_SHAPE, bool)
    log("WARNING: mask file not found; the whole DEM extent is treated as the LGA.")
valid = lga & np.isfinite(dem)
dem_full = dem.copy()            # buffered DEM, used for hydrology and slope
region = np.isfinite(dem_full)   # LGA plus buffer
dem[~valid] = np.nan
log("Valid LGA cells: %d (%.0f km2) | cells incl. buffer: %d (%.0f km2)" % (
    valid.sum(), valid.sum() * CELL_KM2, region.sum(), region.sum() * CELL_KM2))

# --------------------------- permanent water ---------------------------
occ = align(WATER_FILE, Resampling.nearest, NODATA_U, "uint8").astype("float32")
occ[occ == NODATA_U] = np.nan
perm_water_full = region & np.isfinite(occ) & (occ >= PERM_WATER_OCC)   # incl. buffer
perm_water = perm_water_full & valid                                    # LGA only
land = valid & ~perm_water
log("Permanent water: %.1f km2 | land used for the index: %.1f km2" % (
    perm_water.sum() * CELL_KM2, land.sum() * CELL_KM2))

# ------------------------------ DEM check ------------------------------
log("\n-- DEM check --")
log("Primary DEM (%s): min %.1f, mean %.1f, max %.1f m" % (
    DEM_FILE, np.nanmin(dem), np.nanmean(dem), np.nanmax(dem)))
if DEM_ALT_FILE and os.path.exists(path(DEM_ALT_FILE)):
    alt = align(DEM_ALT_FILE, Resampling.nearest, NODATA_F, "float32")
    alt[alt == NODATA_F] = np.nan
    both = valid & np.isfinite(alt)
    diff = (alt - dem)[both]
    log("Alternative DEM (%s) minus primary: mean %.2f, std %.2f m, 1st/99th pct %.1f / %.1f m" % (
        DEM_ALT_FILE, diff.mean(), diff.std(), np.percentile(diff, 1), np.percentile(diff, 99)))
    log("Alternative DEM cells below 20 m: %d" % int(np.sum(alt[both] < 20)))
    pw = perm_water & np.isfinite(alt)
    if pw.any():
        log("Elevation over permanent water, std: primary %.2f m, alternative %.2f m" % (
            np.nanstd(dem[pw]), np.nanstd(alt[pw])))

# --------------------------- slope (Horn) ------------------------------
log("\nDeriving slope ...")
if np.isnan(dem_full).any():
    idx = ndi.distance_transform_edt(np.isnan(dem_full), return_distances=False, return_indices=True)
    dem_filled = dem_full[tuple(idx)]
else:
    dem_filled = dem_full
kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype="float32") / (8.0 * CELL)
ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype="float32") / (8.0 * CELL)
dzdx = ndi.correlate(dem_filled, kx, mode="nearest")
dzdy = ndi.correlate(dem_filled, ky, mode="nearest")
slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy))).astype("float32")
slope[~valid] = np.nan

# ----------------- hydrology: fill, flow direction, accumulation -------
# Runs on the whole buffered DEM so that water entering the LGA from outside is counted.
log("Conditioning the DEM and computing flow accumulation (this can take a few minutes) ...")
from pysheds.grid import Grid  # imported here so the earlier steps run without it

tmp_dem = os.path.join(OUT_DIR, "_dem_for_hydrology.tif")
prof = M_PROFILE.copy()
prof.update(dtype="float32", count=1, nodata=NODATA_F, compress="deflate")
with rasterio.open(tmp_dem, "w", **prof) as dst:
    dst.write(np.where(region, dem_full, NODATA_F).astype("float32"), 1)
grid = Grid.from_raster(tmp_dem)
dem_r = grid.read_raster(tmp_dem)
pit_filled = grid.fill_pits(dem_r)
flooded = grid.fill_depressions(pit_filled)
inflated = grid.resolve_flats(flooded)
fdir = grid.flowdir(inflated)
acc = np.asarray(grid.accumulation(fdir), dtype="float64")
os.remove(tmp_dem)
acc_km2_full = (acc * CELL_KM2).astype("float32")   # LGA plus buffer
acc_km2 = acc_km2_full.copy()
acc_km2[~valid] = np.nan                            # LGA only
log("Max contributing area inside the LGA: %.1f km2" % np.nanmax(acc_km2))

# Edge-effect check: contributing area in the outer 2 cells of the LGA.
edge = valid & ~ndi.binary_erosion(valid, iterations=2)
if edge.any():
    e = acc_km2[edge]
    log("Contributing area on the LGA boundary ring: median %.2f, 99th pct %.1f, max %.1f km2" % (
        np.nanmedian(e), np.nanpercentile(e, 99), np.nanmax(e)))

# ------------------------------ streams --------------------------------
streams_full = region & (np.nan_to_num(acc_km2_full) >= STREAM_THRESHOLD_KM2)
streams_full |= perm_water_full
streams = streams_full & valid
log("Stream cells in the LGA: %d (%.0f km of channel, approx.), threshold %.2f km2 plus permanent water" % (
    streams.sum(), streams.sum() * CELL / 1000.0, STREAM_THRESHOLD_KM2))

# --------------------------- distance to drainage ----------------------
# Uses streams in the buffer too, so cells near the boundary see channels just outside it.
dist = ndi.distance_transform_edt(~streams_full, sampling=CELL).astype("float32")
dist[~valid] = np.nan

# ----------------------------- drainage density ------------------------
r = int(np.ceil(DRAINAGE_DENSITY_RADIUS_M / CELL))
yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
kernel = ((xx ** 2 + yy ** 2) <= r ** 2).astype("float32")
stream_km = (streams_full.astype("float32") * CELL / 1000.0)
len_sum = fftconvolve(stream_km, kernel, mode="same")
area_km2 = fftconvolve(region.astype("float32"), kernel, mode="same") * CELL_KM2
with np.errstate(divide="ignore", invalid="ignore"):
    drain_density = np.where(area_km2 > 1e-6, len_sum / area_km2, np.nan).astype("float32")
drain_density = np.clip(drain_density, 0, None)
drain_density[~valid] = np.nan

# ----------------------- rainfall, NDVI, land cover --------------------
rs = Resampling.nearest if RAIN_RESAMPLING == "nearest" else Resampling.bilinear
rain = align(RAIN_FILE, rs, NODATA_F, "float32")
rain[rain == NODATA_F] = np.nan
rain[~valid] = np.nan

ndvi = align(NDVI_FILE, Resampling.average, NODATA_F, "float32")   # 10 m -> 30 m mean
ndvi[ndvi == NODATA_F] = np.nan
ndvi[~valid] = np.nan

lulc_dw = align(LULC_DW_FILE, Resampling.mode, NODATA_U, "uint8") if os.path.exists(path(LULC_DW_FILE)) else None
lulc_wc = align(LULC_WC_FILE, Resampling.mode, NODATA_U, "uint8") if os.path.exists(path(LULC_WC_FILE)) else None

log("\n-- Land cover class shares on the valid area (%) --")
for label, arr, names in (("Dynamic World", lulc_dw, LULC_NAMES_DW), ("ESA WorldCover", lulc_wc, LULC_NAMES_WC)):
    if arr is None:
        continue
    vals = arr[valid & (arr != NODATA_U)]
    parts = []
    for k in sorted(set(np.unique(vals).tolist())):
        parts.append("%s %.1f" % (names.get(k, k), 100.0 * np.mean(vals == k)))
    log("%s: %s" % (label, ", ".join(parts)))

if LULC_SOURCE == "worldcover":
    lulc, scores = lulc_wc, LULC_SCORES_WC
else:
    lulc, scores = lulc_dw, LULC_SCORES_DW
if lulc is None:
    sys.exit("Selected land-cover file for LULC_SOURCE='%s' not found." % LULC_SOURCE)
lut = np.full(256, np.nan, dtype="float32")
for k, v in scores.items():
    lut[k] = v
lulc_score = lut[lulc]
lulc_score[~valid] = np.nan
lulc_out = np.where(valid, lulc, NODATA_U).astype("uint8")   # class map, LGA only

# ------------------------- standardise to 0-1 --------------------------
def rescale(a, invert=False, use_log=False):
    x = np.log10(np.clip(a, 1e-6, None)) if use_log else a.copy()
    ok = land & np.isfinite(x)
    lo, hi = np.percentile(x[ok], CLIP_PCT)
    if hi <= lo:
        hi = lo + 1e-9
    z = np.clip((x - lo) / (hi - lo), 0, 1)
    if invert:
        z = 1 - z
    z = z.astype("float32")
    z[~ok] = np.nan
    return z, float(lo), float(hi)


factors = {
    # name: (raw array, invert, log10 first)
    "elevation": (dem, True, False),
    "slope": (slope, SLOPE_GENTLE_IS_HIGH, False),
    "rainfall": (rain, False, False),
    "dist_drainage": (dist, True, False),
    "flow_acc": (acc_km2, False, True),
    "drainage_density": (drain_density, False, False),
    "ndvi": (ndvi, True, False),
}
std, bounds = {}, {}
for name, (arr, inv, lg) in factors.items():
    std[name], lo, hi = rescale(arr, inv, lg)
    bounds[name] = (lo, hi)
std["lulc"] = np.where(land, lulc_score, np.nan).astype("float32")
bounds["lulc"] = (0.0, 1.0)

# ------------------------------ outputs --------------------------------
log("\nWriting rasters ...")
raw_out = {
    "elevation": dem, "slope_deg": slope, "rainfall_mm": rain, "dist_drainage_m": dist,
    "flow_acc_km2": acc_km2, "drainage_density_km_km2": drain_density, "ndvi": ndvi,
    "lulc_score": lulc_score,
}
for name, arr in raw_out.items():
    write("raw_%s.tif" % name, arr)
for name, arr in std.items():
    write("std_%s.tif" % name, arr)
write("perm_water.tif", perm_water.astype("uint8"), nodata=NODATA_U, dtype="uint8")
write("streams.tif", streams.astype("uint8"), nodata=NODATA_U, dtype="uint8")
write("lga_valid.tif", valid.astype("uint8"), nodata=NODATA_U, dtype="uint8")
write("lulc_class_aligned.tif", lulc_out, nodata=NODATA_U, dtype="uint8")

# --------------------------- factor summary ----------------------------
log("\n-- Factor summary (land cells) --")
rows = ["factor,raw_min,raw_mean,raw_max,scale_lo,scale_hi,std_mean,std_std"]
raw_for_summary = {"elevation": dem, "slope": slope, "rainfall": rain, "dist_drainage": dist,
                   "flow_acc": acc_km2, "drainage_density": drain_density, "ndvi": ndvi, "lulc": lulc_score}
for name in std:
    a = raw_for_summary[name][land]
    a = a[np.isfinite(a)]
    s = std[name][land]
    s = s[np.isfinite(s)]
    lo, hi = bounds[name]
    rows.append("%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f" % (
        name, a.min(), a.mean(), a.max(), lo, hi, s.mean(), s.std()))
    log("%-17s raw %.3g to %.3g (mean %.3g) | standardised mean %.2f, std %.2f" % (
        name, a.min(), a.max(), a.mean(), s.mean(), s.std()))
with open(os.path.join(OUT_DIR, "factor_summary.csv"), "w") as f:
    f.write("\n".join(rows) + "\n")

# --------------------- collinearity: correlation and VIF ---------------
log("\n-- Collinearity check on standardised factors --")
names = list(std.keys())
stack = np.stack([std[n][land] for n in names], axis=1)
stack = stack[np.all(np.isfinite(stack), axis=1)]
rng = np.random.default_rng(RANDOM_SEED)
if len(stack) > SAMPLE_N:
    stack = stack[rng.choice(len(stack), SAMPLE_N, replace=False)]
varying = stack.std(axis=0) > 1e-9
names_v = [n for n, v in zip(names, varying) if v]
for n, v in zip(names, varying):
    if not v:
        log("WARNING: %s is constant over the sample and is excluded from the correlation check." % n)
R = np.corrcoef(stack[:, varying], rowvar=False)
with open(os.path.join(OUT_DIR, "factor_correlation.csv"), "w") as f:
    f.write("," + ",".join(names_v) + "\n")
    for i, n in enumerate(names_v):
        f.write(n + "," + ",".join("%.3f" % x for x in R[i]) + "\n")
log("Pairs with |r| >= 0.70:")
flag = False
for i in range(len(names_v)):
    for j in range(i + 1, len(names_v)):
        if abs(R[i, j]) >= 0.70:
            flag = True
            log("  %s vs %s: r = %.2f" % (names_v[i], names_v[j], R[i, j]))
if not flag:
    log("  none")
try:
    vif = np.diag(np.linalg.inv(R))
    log("Variance inflation factors: " + ", ".join("%s %.1f" % (n, v) for n, v in zip(names_v, vif)))
    log("(values above about 5 to 10 suggest a factor is largely explained by the others)")
except np.linalg.LinAlgError:
    log("Correlation matrix is singular; VIF not computed.")

# ------------------------------ quick look -----------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for ax, n in zip(axes.ravel(), std.keys()):
        im = ax.imshow(std[n], cmap="viridis", vmin=0, vmax=1)
        ax.set_title(n)
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.6, label="standardised value (1 = more susceptible)")
    fig.savefig(os.path.join(OUT_DIR, "factors_overview.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
except Exception as e:  # plotting is optional
    log("Quick-look figure skipped: %s" % e)

with open(os.path.join(OUT_DIR, "phase2_log.json"), "w") as f:
    json.dump({
        "dem": DEM_FILE, "mask": MASK_FILE, "water": WATER_FILE,
        "lulc_source": LULC_SOURCE, "perm_water_occ": PERM_WATER_OCC,
        "stream_threshold_km2": STREAM_THRESHOLD_KM2, "drainage_density_radius_m": DRAINAGE_DENSITY_RADIUS_M,
        "rain_resampling": RAIN_RESAMPLING, "slope_gentle_is_high": SLOPE_GENTLE_IS_HIGH,
        "clip_pct": CLIP_PCT, "scaling_bounds": bounds, "cell_size_m": CELL,
        "hydrology_on_buffered_dem": True,
    }, f, indent=2)
log("\nDone. Outputs are in %s" % OUT_DIR)
#!/usr/bin/env python3
"""
PHASE 4c (EXPLORATORY): Height Above Nearest Drainage (HAND) check.
Lokoja LGA flood susceptibility project.

Purpose: Phase 4 showed that elevation alone (AUC about 0.96) discriminates the observed flooding
better than the full AHP index (AUC about 0.85). HAND measures how far each cell sits above the
drainage it flows to. Two versions are computed and compared on the SAME cells:
  - HAND to the full stream network (streams.tif), and
  - HAND to the main rivers only (perm_water.tif: the Niger and Benue channels).
The first run of this script (stream network) found HAND weaker than absolute elevation. The likely
explanation is that the flooding is controlled by the main rivers, so absolute elevation already acts as
"height above the Niger and Benue". The main-river version tests that explanation.

IMPORTANT: this is an exploratory, validation-informed analysis. It does NOT replace the pre-specified
AHP index. Report the AHP result as the primary result and this as a labelled extra.

Input  : data/phase2/raw_elevation.tif, streams.tif, std_*.tif, raw_slope_deg.tif
         data/phase3/fsi_scaled.tif, phase3_log.json
         data/phase4/flood_2018.tif, flood_2022.tif
Output : data/phase4_results/hand/
           raw_hand_streams_m.tif, raw_hand_main_rivers_m.tif, std_hand_streams.tif,
           std_hand_main_rivers.tif, hand_validation_summary.csv, hand_bins_<event>.csv

Run    : python phase4c_hand_check.py     (same environment as Phase 2: needs pysheds)
"""
import json
import os
import sys

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.stats import rankdata

# pysheds 0.5 calls np.in1d, which NumPy 2.4 removed. This shim keeps it working.
if not hasattr(np, "in1d"):
    np.in1d = np.isin

# ------------------------------ CONFIG ---------------------------------
PHASE2_DIR = "data/phase2"
PHASE3_DIR = "data/phase3"
FLOOD_DIR = "data/phase4"
OUT_DIR = "data/phase4_results/hand"

EVENTS = {"2018": "flood_2018.tif", "2022": "flood_2022.tif"}
FLOOD_FRACTION_MIN = 0.5
SLOPE_MAX_DEG = 5               # same validated domain as phase4_validation.py
CLIP_PCT = (1, 99)              # standardisation of HAND (same trimming as Phase 2)
BLOCK_CELLS = 10
N_BOOT = 300
RANDOM_SEED = 42
HAND_BINS = [0, 2, 5, 10, 20, 50, 1e9]
FACTORS = ["elevation", "slope", "rainfall", "dist_drainage",
           "flow_acc", "drainage_density", "lulc", "ndvi"]
# -----------------------------------------------------------------------

NODATA_F = -9999.0
NODATA_U = 255


def log(msg=""):
    print(msg, flush=True)


def need(p):
    if not os.path.exists(p):
        sys.exit("Missing file: %s" % p)
    return p


os.makedirs(OUT_DIR, exist_ok=True)
rng = np.random.default_rng(RANDOM_SEED)


def read(path, dtype="float64", nodata=None):
    with rasterio.open(need(path)) as src:
        a = src.read(1).astype(dtype)
        nd = src.nodata if nodata is None else nodata
        if nd is not None and np.issubdtype(a.dtype, np.floating):
            a[a == nd] = np.nan
        info = {"profile": src.profile.copy(), "transform": src.transform, "crs": src.crs}
        return a, info


# ---------------------------- load inputs ------------------------------
dem, info0 = read(os.path.join(PHASE2_DIR, "raw_elevation.tif"))
PROFILE, TRANSFORM, CRS = info0["profile"], info0["transform"], info0["crs"]
SHAPE = dem.shape
CELL_KM2 = abs(TRANSFORM.a * TRANSFORM.e) / 1e6
streams, _ = read(os.path.join(PHASE2_DIR, "streams.tif"), dtype="float64")
streams = np.nan_to_num(streams, nan=0.0)
streams[streams == NODATA_U] = 0
streams = streams == 1
perm, _ = read(os.path.join(PHASE2_DIR, "perm_water.tif"), dtype="float64")
perm = np.nan_to_num(perm, nan=0.0)
perm[perm == NODATA_U] = 0
main_rivers = perm == 1
fsi, _ = read(os.path.join(PHASE3_DIR, "fsi_scaled.tif"))
slope, _ = read(os.path.join(PHASE2_DIR, "raw_slope_deg.tif"))
with open(need(os.path.join(PHASE3_DIR, "phase3_log.json"))) as f:
    weights = json.load(f)["weights"]
std = {}
for name in FACTORS:
    std[name], _ = read(os.path.join(PHASE2_DIR, "std_%s.tif" % name))

valid = np.isfinite(dem)
land = np.isfinite(fsi)
domain_base = land & np.isfinite(slope) & (slope <= SLOPE_MAX_DEG)

# ------------------------------- HAND ----------------------------------
log("Computing flow directions and HAND (can take a few minutes) ...")
from pysheds.grid import Grid  # noqa: E402

tmp = os.path.join(OUT_DIR, "_dem_tmp.tif")
prof = PROFILE.copy()
prof.update(dtype="float32", count=1, nodata=NODATA_F, compress="deflate")
with rasterio.open(tmp, "w", **prof) as dst:
    dst.write(np.where(valid, dem, NODATA_F).astype("float32"), 1)
grid = Grid.from_raster(tmp)
dem_r = grid.read_raster(tmp)
pit_filled = grid.fill_pits(dem_r)
flooded = grid.fill_depressions(pit_filled)      # heights kept; depressions filled
inflated = grid.resolve_flats(flooded)
fdir = grid.flowdir(inflated)
os.remove(tmp)


def hand_to(mask_bool, tag):
    """Height above the nearest cell of mask_bool along the flow path."""
    tmp_m = os.path.join(OUT_DIR, "_mask_%s.tif" % tag)
    sprof = PROFILE.copy()
    sprof.update(dtype="uint8", count=1, nodata=NODATA_U, compress="deflate")
    with rasterio.open(tmp_m, "w", **sprof) as dst:
        dst.write(mask_bool.astype("uint8"), 1)
    m = grid.read_raster(tmp_m) == 1
    h_ = np.asarray(grid.compute_hand(fdir, flooded, m, nodata_out=np.nan), dtype="float64")
    os.remove(tmp_m)
    h_[~valid] = np.nan
    h_[h_ < 0] = 0.0
    return h_


hand_streams = hand_to(streams, "streams")
hand_main = hand_to(main_rivers, "main")

for label, h_ in (("stream network", hand_streams), ("main rivers only", hand_main)):
    cov = domain_base & np.isfinite(h_)
    log("HAND to %s: computed for %d of %d domain cells (%.1f%%)" % (
        label, cov.sum(), domain_base.sum(), 100.0 * cov.sum() / domain_base.sum()))
# All comparisons use the SAME cells: those that drain to a main river inside the LGA.
reach = domain_base & np.isfinite(hand_main) & np.isfinite(hand_streams)
log("Cells used for every comparison below: %d (%.1f%% of the validated domain)" % (
    reach.sum(), 100.0 * reach.sum() / domain_base.sum()))
log("Cells left out drain out of the LGA before reaching a main river; check their location in QGIS if the share is large.")
for label, h_ in (("stream network", hand_streams), ("main rivers only", hand_main)):
    v = h_[reach]
    log("HAND to %s (m): min %.1f, median %.1f, mean %.1f, 95th pct %.1f, max %.1f" % (
        label, v.min(), np.median(v), v.mean(), np.percentile(v, 95), v.max()))


def standardise(h_):
    lo, hi = np.percentile(h_[reach], CLIP_PCT)
    z = 1.0 - np.clip((h_ - lo) / (hi - lo), 0, 1)     # lower HAND = more flood-prone
    z[~np.isfinite(h_) | ~land] = np.nan
    return z


std_hand_streams = standardise(hand_streams)
std_hand_main = standardise(hand_main)


def write(name, arr, nodata=NODATA_F, dtype="float32"):
    p = PROFILE.copy()
    p.update(dtype=dtype, count=1, nodata=nodata, compress="deflate")
    out = np.where(np.isfinite(arr), arr, nodata).astype(dtype)
    with rasterio.open(os.path.join(OUT_DIR, name), "w", **p) as dst:
        dst.write(out, 1)


write("raw_hand_streams_m.tif", np.where(land, hand_streams, np.nan))
write("raw_hand_main_rivers_m.tif", np.where(land, hand_main, np.nan))
write("std_hand_streams.tif", std_hand_streams)
write("std_hand_main_rivers.tif", std_hand_main)

# AHP index with main-river HAND in place of elevation (same weights)
w = {k: weights[k] for k in FACTORS}
variant = np.zeros(SHAPE)
for k in FACTORS:
    variant += w[k] * (std_hand_main if k == "elevation" else std[k])
variant[~np.isfinite(std_hand_main)] = np.nan


# ------------------------------ helpers --------------------------------
def align_flood(fname):
    dst = np.full(SHAPE, -1.0, dtype="float32")
    with rasterio.open(need(os.path.join(FLOOD_DIR, fname))) as src:
        reproject(source=rasterio.band(src, 1), destination=dst,
                  src_transform=src.transform, src_crs=src.crs, src_nodata=NODATA_U,
                  dst_transform=TRANSFORM, dst_crs=CRS, dst_nodata=-1.0,
                  resampling=Resampling.average)
    observed = dst >= 0
    return observed & (dst >= FLOOD_FRACTION_MIN), observed


def auc_rank(y, s):
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def thinned(rows, cols, flag):
    blocks = (rows // BLOCK_CELLS) * 100000 + (cols // BLOCK_CELLS)
    chosen = []
    for f_ in (True, False):
        idx = np.where(flag == f_)[0]
        idx = idx[rng.permutation(len(idx))]
        _, first = np.unique(blocks[idx], return_index=True)
        chosen.append(idx[first])
    return np.concatenate(chosen), blocks


def boot(y, s, blocks):
    ub, inv = np.unique(blocks, return_inverse=True)
    groups = [np.where(inv == k)[0] for k in range(len(ub))]
    out = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[k] for k in pick])
        a = auc_rank(y[idx], s[idx])
        if np.isfinite(a):
            out.append(a)
    return np.percentile(out, [2.5, 97.5]) if out else (np.nan, np.nan)


# ------------------------------ validation -----------------------------
events = {}
for ev, fname in EVENTS.items():
    if os.path.exists(os.path.join(FLOOD_DIR, fname)):
        events[ev] = align_flood(fname)
if not events:
    sys.exit("No flood rasters found in %s." % FLOOD_DIR)
if len(events) > 1:
    fl = np.zeros(SHAPE, bool)
    ob = np.ones(SHAPE, bool)
    for f_, o_ in events.values():
        fl |= f_
        ob &= o_
    events["either"] = (fl, ob)

candidates = [
    ("HAND to stream network (exploratory)", std_hand_streams),
    ("HAND to main rivers only (exploratory)", std_hand_main),
    ("Elevation alone", std["elevation"]),
    ("AHP index (primary, pre-specified)", fsi),
    ("AHP index with main-river HAND instead of elevation (exploratory)", variant),
]
summary = ["event,index,cells,auc_all_cells,auc_thinned,ci_low,ci_high"]
for ev, (flooded, observed) in events.items():
    dom = reach & observed
    n = int(dom.sum())
    y = flooded[dom]
    if y.sum() < 50:
        log("\nEvent %s: too few flooded cells, skipped." % ev)
        continue
    log("\n================ Event: %s (%d cells, %.2f%% flooded) ================" % (ev, n, 100.0 * y.mean()))
    rows_i, cols_i = np.where(dom)
    sel, blocks = thinned(rows_i, cols_i, y)
    log("%-68s AUC all   AUC thinned (95%% CI)" % "index")
    for label, layer in candidates:
        s = layer[dom]
        a_all = auc_rank(y, s)
        a_t = auc_rank(y[sel], s[sel])
        lo_, hi_ = boot(y[sel], s[sel], blocks[sel])
        log("%-68s %.3f     %.3f (%.3f to %.3f)" % (label, a_all, a_t, lo_, hi_))
        summary.append("%s,%s,%d,%.4f,%.4f,%.4f,%.4f" % (ev, label, n, a_all, a_t, lo_, hi_))

    # flooded share by HAND bin: the physically readable result
    hv = hand_main[dom]
    log("\nShare of cells flooded by height above the nearest MAIN RIVER:")
    log("  HAND-main (m)  area %    flooded share of all flooding %    prevalence %")
    brow = ["hand_min_m,hand_max_m,area_percent,share_of_flooding_percent,prevalence_percent"]
    for a, b in zip(HAND_BINS[:-1], HAND_BINS[1:]):
        m = (hv >= a) & (hv < b)
        if not m.any():
            continue
        name = "%g to %g" % (a, b) if b < 1e8 else ">= %g" % a
        log("  %-14s %-9.1f %-34.1f %.2f" % (name, 100.0 * m.mean(), 100.0 * y[m].sum() / y.sum(), 100.0 * y[m].mean()))
        brow.append("%g,%g,%.2f,%.2f,%.3f" % (a, min(b, 9999), 100.0 * m.mean(), 100.0 * y[m].sum() / y.sum(), 100.0 * y[m].mean()))
    with open(os.path.join(OUT_DIR, "hand_bins_%s.csv" % ev), "w") as f:
        f.write("\n".join(brow) + "\n")

with open(os.path.join(OUT_DIR, "hand_validation_summary.csv"), "w") as f:
    f.write("\n".join(summary) + "\n")

log("\nHow to read this:")
log(" - The pre-specified AHP index stays the primary result. The HAND rows are exploratory and validation-informed.")
log(" - If HAND to the main rivers matches or beats elevation, absolute elevation was acting as height above the")
log("   Niger and Benue, and the earlier weaker stream-network HAND came from small tributaries diluting the signal.")
log(" - If main-river HAND is still clearly below elevation, something else in elevation matters (for example")
log("   regional position or the DEM treatment of the river surface); report that as unexplained.")
log(" - Do not swap HAND into the main index on the strength of this result alone. Test it on an independent event")
log("   (for example the 2012 flood) first.")
log("\nDone. Outputs are in %s" % OUT_DIR)
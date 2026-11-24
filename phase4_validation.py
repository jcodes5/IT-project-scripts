#!/usr/bin/env python3
"""
PHASE 4b: Validate the flood susceptibility index against observed flood extent.
Lokoja LGA flood susceptibility project.

Input  : flood_2018.tif, flood_2022.tif from phase4_flood_extent.js (10 m, 1 flooded, 0 not, 255 nodata)
         fsi_scaled.tif, fsi_class.tif, ahp weights (phase3_log.json) from phase3_susceptibility.py
         std_*.tif, raw_slope_deg.tif from phase2_factors.py
Output : in OUT_DIR
           validation_summary.csv         AUC per event (all cells and spatially thinned, with 95% CI)
           class_table_<event>.csv        flooded share and frequency ratio per susceptibility class
           threshold_metrics_<event>.csv  hit rate, false alarm rate, precision at class cut-offs
           factor_auc_<event>.csv         AUC of each single factor and of the index without each factor
           roc_curves.png, flood_aligned_<event>.tif, validation_log.json

Install : pip install rasterio scipy numpy matplotlib
Run     : python phase4_validation.py
"""
import json
import os
import sys

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.stats import rankdata

# ------------------------------ CONFIG ---------------------------------
PHASE2_DIR = "data/phase2"
PHASE3_DIR = "data/phase3"
FLOOD_DIR = "data/phase4"
OUT_DIR = "data/phase4_results"

EVENTS = {"2018": "flood_2018.tif", "2022": "flood_2022.tif"}
FLOOD_FRACTION_MIN = 0.5        # a 30 m cell is flooded if at least this share of its 10 m pixels are
SLOPE_MAX_DEG = 5               # must match the Earth Engine script: steeper cells are not validated
BLOCK_CELLS = 10                # thinning block (10 cells = 300 m): one point per class per block
N_BOOT = 500                    # cluster bootstrap repetitions
HIGH_CLASS_MIN = 4              # "predicted flood-prone" = this class or higher
RANDOM_SEED = 42
# -----------------------------------------------------------------------

NODATA_U = 255
FACTORS = ["elevation", "slope", "rainfall", "dist_drainage",
           "flow_acc", "drainage_density", "lulc", "ndvi"]


def log(msg=""):
    print(msg, flush=True)


def need(p):
    if not os.path.exists(p):
        sys.exit("Missing file: %s" % p)
    return p


os.makedirs(OUT_DIR, exist_ok=True)
rng = np.random.default_rng(RANDOM_SEED)

# ------------------------------ load Phase 3 ---------------------------
with rasterio.open(need(os.path.join(PHASE3_DIR, "fsi_scaled.tif"))) as src:
    prof = src.profile.copy()
    M_TRANSFORM, M_CRS = src.transform, src.crs
    SHAPE = (src.height, src.width)
    fsi = src.read(1).astype("float64")
    fsi[fsi == src.nodata] = np.nan
with rasterio.open(need(os.path.join(PHASE3_DIR, "fsi_class.tif"))) as src:
    cls = src.read(1).astype("int16")
    cls[cls == NODATA_U] = -1
with rasterio.open(need(os.path.join(PHASE2_DIR, "raw_slope_deg.tif"))) as src:
    slope = src.read(1).astype("float64")
    slope[slope == src.nodata] = np.nan
with open(need(os.path.join(PHASE3_DIR, "phase3_log.json"))) as f:
    weights = json.load(f)["weights"]
n_classes = int(cls.max())
CELL_KM2 = abs(M_TRANSFORM.a * M_TRANSFORM.e) / 1e6

domain_base = np.isfinite(fsi) & np.isfinite(slope) & (slope <= SLOPE_MAX_DEG)
log("Scored cells: %d | validated domain (slope <= %g deg): %d (%.0f km2)" % (
    np.isfinite(fsi).sum(), SLOPE_MAX_DEG, domain_base.sum(), domain_base.sum() * CELL_KM2))

std = {}
for name in FACTORS:
    with rasterio.open(need(os.path.join(PHASE2_DIR, "std_%s.tif" % name))) as src:
        a = src.read(1).astype("float64")
        a[a == src.nodata] = np.nan
        std[name] = a


# ------------------------------ helpers --------------------------------
def align_flood(fname):
    """Aggregate the 10 m flood raster to the 30 m grid: returns (flooded, observed)."""
    dst = np.full(SHAPE, -1.0, dtype="float32")
    with rasterio.open(need(os.path.join(FLOOD_DIR, fname))) as src:
        reproject(source=rasterio.band(src, 1), destination=dst,
                  src_transform=src.transform, src_crs=src.crs, src_nodata=NODATA_U,
                  dst_transform=M_TRANSFORM, dst_crs=M_CRS, dst_nodata=-1.0,
                  resampling=Resampling.average)
    observed = dst >= 0
    flooded = observed & (dst >= FLOOD_FRACTION_MIN)
    return flooded, observed


def auc_rank(y, s):
    """AUC by the Mann-Whitney rank method. y is boolean, s the score."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def roc_points(y, s, max_points=400):
    order = np.argsort(-s, kind="mergesort")
    ys = y[order].astype("float64")
    tps = np.cumsum(ys)
    fps = np.cumsum(1 - ys)
    distinct = np.r_[np.where(np.diff(s[order]))[0], len(s) - 1]
    tpr = np.r_[0, tps[distinct] / tps[-1]]
    fpr = np.r_[0, fps[distinct] / fps[-1]]
    if len(tpr) > max_points:
        keep = np.linspace(0, len(tpr) - 1, max_points).astype(int)
        tpr, fpr = tpr[keep], fpr[keep]
    return fpr, tpr


def band(a):
    if a >= 0.9:
        return "excellent"
    if a >= 0.8:
        return "good"
    if a >= 0.7:
        return "acceptable"
    if a >= 0.6:
        return "weak"
    return "poor"


def thinned_sample(rows, cols, flooded_flag):
    """One random cell per class per block. Returns indices into the arrays passed in."""
    blocks = (rows // BLOCK_CELLS) * 100000 + (cols // BLOCK_CELLS)
    chosen = []
    for flag in (True, False):
        idx = np.where(flooded_flag == flag)[0]
        idx = idx[rng.permutation(len(idx))]
        _, first = np.unique(blocks[idx], return_index=True)
        chosen.append(idx[first])
    return np.concatenate(chosen), blocks


def cluster_bootstrap(y, s, blocks, n_boot):
    ub, inv = np.unique(blocks, return_inverse=True)
    groups = [np.where(inv == k)[0] for k in range(len(ub))]
    aucs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[k] for k in pick])
        a = auc_rank(y[idx], s[idx])
        if np.isfinite(a):
            aucs.append(a)
    return np.percentile(aucs, [2.5, 97.5]) if aucs else (np.nan, np.nan)


# ------------------------------ validation -----------------------------
events = {}
for ev, fname in EVENTS.items():
    if os.path.exists(os.path.join(FLOOD_DIR, fname)):
        events[ev] = align_flood(fname)
    else:
        log("Skipping %s: %s not found." % (ev, os.path.join(FLOOD_DIR, fname)))
if not events:
    sys.exit("No flood rasters found in %s." % FLOOD_DIR)
if len(events) > 1:
    fl = np.zeros(SHAPE, bool)
    ob = np.ones(SHAPE, bool)
    for f_, o_ in events.values():
        fl |= f_
        ob &= o_
    events["either"] = (fl, ob)

summary = ["event,domain_cells,flooded_cells,flooded_percent,auc_all_cells,quality_band,"
           "auc_thinned,ci_low,ci_high,thinned_flooded,thinned_not_flooded"]
roc_store = {}
for ev, (flooded, observed) in events.items():
    dom = domain_base & observed
    n_dom = int(dom.sum())
    log("\n================ Event: %s ================" % ev)
    if n_dom == 0:
        log("No observed cells inside the validated domain. Skipping.")
        continue
    cover = n_dom / float(domain_base.sum())
    y_full = flooded[dom]
    s_full = fsi[dom]
    n_fl = int(y_full.sum())
    log("Observed domain: %d cells (%.0f%% of the validated area); flooded: %d cells (%.2f%%)" % (
        n_dom, 100 * cover, n_fl, 100.0 * n_fl / n_dom))
    if cover < 0.9:
        log("WARNING: the flood raster covers under 90%% of the domain; check coverage for this event.")
    if n_fl < 50:
        log("Too few flooded cells to validate. Check the flood extent for this event.")
        continue

    out_prof = prof.copy()
    out_prof.update(dtype="uint8", nodata=NODATA_U, count=1, compress="deflate")
    arr = np.full(SHAPE, NODATA_U, dtype="uint8")
    arr[observed] = flooded[observed]
    with rasterio.open(os.path.join(OUT_DIR, "flood_aligned_%s.tif" % ev), "w", **out_prof) as dst:
        dst.write(arr, 1)

    auc_all = auc_rank(y_full, s_full)
    log("AUC (all cells): %.3f  [%s]" % (auc_all, band(auc_all)))

    rows_i, cols_i = np.where(dom)
    sel, blocks = thinned_sample(rows_i, cols_i, y_full)
    y_t, s_t = y_full[sel], s_full[sel]
    auc_t = auc_rank(y_t, s_t)
    lo, hi = cluster_bootstrap(y_t, s_t, blocks[sel], N_BOOT)
    log("AUC (spatially thinned, one cell per %d m block and class): %.3f  95%% CI %.3f to %.3f" % (
        BLOCK_CELLS * 30, auc_t, lo, hi))
    summary.append("%s,%d,%d,%.4f,%.4f,%s,%.4f,%.4f,%.4f,%d,%d" % (
        ev, n_dom, n_fl, 100.0 * n_fl / n_dom, auc_all, band(auc_all), auc_t, lo, hi,
        int(y_t.sum()), int((~y_t).sum())))
    roc_store[ev] = roc_points(y_full, s_full)

    # ----- class table: where does the observed flooding fall? -----
    c_full = cls[dom]
    rows = ["class,area_percent,flooded_share_percent,frequency_ratio,flood_prevalence_percent"]
    log("\nFlooded area by susceptibility class (frequency ratio = flooded share / area share):")
    log("  class  area %   flooded share %   FR     prevalence %")
    for c in range(1, n_classes + 1):
        m = c_full == c
        a_pct = 100.0 * m.sum() / n_dom
        f_pct = 100.0 * y_full[m].sum() / n_fl
        fr = (f_pct / a_pct) if a_pct > 0 else float("nan")
        prev = 100.0 * y_full[m].mean() if m.any() else float("nan")
        rows.append("%d,%.2f,%.2f,%.3f,%.3f" % (c, a_pct, f_pct, fr, prev))
        log("  %-6d %-8.1f %-16.1f %-6.2f %.2f" % (c, a_pct, f_pct, fr, prev))
    with open(os.path.join(OUT_DIR, "class_table_%s.csv" % ev), "w") as f:
        f.write("\n".join(rows) + "\n")

    # ----- threshold metrics -----
    rows = ["min_class,share_of_area_predicted_percent,hit_rate_percent,false_alarm_rate_percent,precision_percent"]
    log("\nIf cells in class >= k are called flood-prone:")
    log("  k   area flagged %   hit rate %   false alarm rate %   precision %")
    for k in range(2, n_classes + 1):
        pred = c_full >= k
        tp = float((pred & y_full).sum())
        fp = float((pred & ~y_full).sum())
        tpr = 100 * tp / n_fl
        fpr = 100 * fp / (n_dom - n_fl)
        prec = 100 * tp / max(tp + fp, 1)
        rows.append("%d,%.2f,%.2f,%.2f,%.2f" % (k, 100 * pred.mean(), tpr, fpr, prec))
        log("  %-3d %-15.1f %-12.1f %-20.1f %.1f" % (k, 100 * pred.mean(), tpr, fpr, prec))
    with open(os.path.join(OUT_DIR, "threshold_metrics_%s.csv" % ev), "w") as f:
        f.write("\n".join(rows) + "\n")

    # ----- which factors carry the discrimination? -----
    log("\nSingle-factor AUC and AUC of the index without each factor:")
    log("  %-17s single-factor   index without factor   change" % "factor")
    rows = ["factor,single_factor_auc,index_without_factor_auc,change_in_auc"]
    for name in FACTORS:
        single = auc_rank(y_full, std[name][dom])
        w = np.array([weights[k] for k in FACTORS if k != name])
        w = w / w.sum()
        sc = np.zeros(n_dom)
        for wk, k in zip(w, [k for k in FACTORS if k != name]):
            sc += wk * std[k][dom]
        loo = auc_rank(y_full, sc)
        rows.append("%s,%.4f,%.4f,%.4f" % (name, single, loo, loo - auc_all))
        log("  %-17s %-15.3f %-21.3f %+.3f" % (name, single, loo, loo - auc_all))
    with open(os.path.join(OUT_DIR, "factor_auc_%s.csv" % ev), "w") as f:
        f.write("\n".join(rows) + "\n")

with open(os.path.join(OUT_DIR, "validation_summary.csv"), "w") as f:
    f.write("\n".join(summary) + "\n")

# ------------------------------ ROC figure -----------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    for ev, (fpr, tpr) in roc_store.items():
        ax.plot(fpr, tpr, label="%s" % ev)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves: flood susceptibility index")
    ax.legend()
    fig.savefig(os.path.join(OUT_DIR, "roc_curves.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
except Exception as e:
    log("ROC figure skipped: %s" % e)

with open(os.path.join(OUT_DIR, "validation_log.json"), "w") as f:
    json.dump({"slope_max_deg": SLOPE_MAX_DEG, "flood_fraction_min": FLOOD_FRACTION_MIN,
               "block_cells": BLOCK_CELLS, "n_boot": N_BOOT, "high_class_min": HIGH_CLASS_MIN}, f, indent=2)

log("\nReading guide:")
log(" - AUC 0.5 means no skill; 0.7 to 0.8 acceptable, 0.8 to 0.9 good, above 0.9 excellent.")
log(" - The thinned AUC and its interval are the more honest numbers, because neighbouring cells are not independent.")
log(" - A single-factor AUC below 0.5 means that factor points the wrong way for the observed flooding. Do not flip it")
log("   to improve the score; report it and discuss why (for example radar shadow or flood extent only on gentle ground).")
log(" - If one factor (often distance to drainage or elevation) carries most of the AUC, say so; the index then")
log("   mainly re-describes the river floodplain that the flood image also shows.")
log("\nDone. Outputs are in %s" % OUT_DIR)
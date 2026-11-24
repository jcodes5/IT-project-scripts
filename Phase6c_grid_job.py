#!/usr/bin/env python3
"""
PHASE 6c: Grid scoring job. Daily rainfall rasters + soil moisture + climatology tables -> DTI and FRI maps.
Lokoja LGA flood susceptibility project.

What it does
  1. Reads the percentile lookup tables exported by phase6a_climatology.js (7 bands: p10 ... p99).
  2. Reads daily rainfall rasters (mm per day) for the 15 days before the issue date T0 through T0+2:
     days up to T0 are observed (for example IMERG Early), later days are forecast (for example ECMWF).
  3. Scores rainfall and soil moisture on the lookup grid, combines them into the DTI for the valid days
     T0, T0+1, T0+2 (current, +24 h, +48 h), and multiplies by the susceptibility index to give the FRI
     on the 30 m grid.

Input   data/phase6/rain48h_pctl_chirps.tif, sm_pctl_smap.tif, sm_pctl_era5land.tif   (phase6a exports)
        data/phase6_inputs/rain_Y YYYMMDD.tif   daily rainfall, mm/day, any CRS and grid
        data/phase6_inputs/sm_latest.tif       latest soil moisture (m3/m3), same product as SOIL_PRODUCT
        data/phase3/fsi_scaled.tif             susceptibility index from Phase 3
Output  data/phase6_outputs/<T0>/
        dti_lead0.tif, dti_lead1.tif, dti_lead2.tif, fri_lead*.tif, dti_category_lead*.tif,
        dti_native_lead*.tif (on the coarse grid), forecast_summary.json

Install : pip install rasterio numpy
Run     :      (dti_core.py must sit in the same folder)

Fetching the rainfall and soil-moisture rasters is a separate step. The soil moisture is the latest
available day, so it can be several days old (SMAP L4 runs about 3 to 5 days behind).
"""
import argparse
import json
import os
import sys
from datetime import date, timedelta

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

import dti_core as core

# ------------------------------ CONFIG ---------------------------------
LOOKUP_DIR = "data/phase6"
INPUT_DIR = "data/phase6_inputs"
PHASE3_DIR = "data/phase3"
OUT_ROOT = "data/phase6_outputs"

SOIL_PRODUCT = "smap"                 # "smap" or "era5land": must match sm_latest.tif
W_RAIN, W_SOIL = 0.75, 0.25
LEADS = [0, 1, 2]                     # days after T0: current, +24 h, +48 h
BACK_DAYS = core.LOOKBACK_DAYS + core.WINDOW_DAYS - 1     # 15 days of history are needed
RAIN_INPUT_RESAMPLING = Resampling.bilinear
DTI_TO_30M_RESAMPLING = Resampling.nearest   # nearest keeps the true coarse pixels; bilinear looks smoother
# -----------------------------------------------------------------------

ap = argparse.ArgumentParser()
ap.add_argument("--t0", required=True, help="issue date, YYYY-MM-DD")
args = ap.parse_args()
T0 = date.fromisoformat(args.t0)
OUT_DIR = os.path.join(OUT_ROOT, T0.isoformat())
os.makedirs(OUT_DIR, exist_ok=True)


def log(msg=""):
    print(msg, flush=True)


def need(p):
    if not os.path.exists(p):
        sys.exit("Missing file: %s" % p)
    return p


# --------------------------- working grid (lookup) ----------------------
with rasterio.open(need(os.path.join(LOOKUP_DIR, "rain48h_pctl_chirps.tif"))) as src:
    W_TRANSFORM, W_CRS, W_SHAPE = src.transform, src.crs, (src.height, src.width)
    if src.count != len(core.PCTL):
        sys.exit("rain lookup must have %d bands (p10..p99), found %d" % (len(core.PCTL), src.count))
    rain_tab = src.read().astype("float64")
    if src.nodata is not None:
        rain_tab[rain_tab == src.nodata] = np.nan


def to_working(path, resampling, band=1, src_nodata=None):
    """Resample one band of a raster onto the working grid."""
    dst = np.full(W_SHAPE, np.nan, dtype="float64")
    with rasterio.open(path) as src:
        nd = src.nodata if src_nodata is None else src_nodata
        reproject(source=rasterio.band(src, band), destination=dst,
                  src_transform=src.transform, src_crs=src.crs, src_nodata=nd,
                  dst_transform=W_TRANSFORM, dst_crs=W_CRS, dst_nodata=np.nan, resampling=resampling)
    return dst


sm_lookup = "sm_pctl_smap.tif" if SOIL_PRODUCT == "smap" else "sm_pctl_era5land.tif"
sm_path = need(os.path.join(LOOKUP_DIR, sm_lookup))
sm_p10 = to_working(sm_path, Resampling.bilinear, band=1)       # p10 is band 1
sm_p99 = to_working(sm_path, Resampling.bilinear, band=len(core.PCTL))   # p99 is band 7
sm_now = to_working(need(os.path.join(INPUT_DIR, "sm_latest.tif")), Resampling.bilinear)

# ------------------------------ rainfall stack --------------------------
days = [T0 + timedelta(days=k) for k in range(-BACK_DAYS, max(LEADS) + 1)]
stack = []
for d in days:
    p = need(os.path.join(INPUT_DIR, "rain_%s.tif" % d.strftime("%Y%m%d")))
    stack.append(to_working(p, RAIN_INPUT_RESAMPLING))
stack = np.array(stack)                                          # (days, H, W)
log("Rainfall stack: %d days (%s to %s), observed up to %s, forecast after" % (
    len(days), days[0], days[-1], T0))
totals = core.rolling_total(stack, core.WINDOW_DAYS)

# ------------------------- fsi on the 30 m master grid ------------------
with rasterio.open(need(os.path.join(PHASE3_DIR, "fsi_scaled.tif"))) as src:
    M_PROFILE = src.profile.copy()
    M_TRANSFORM, M_CRS, M_SHAPE = src.transform, src.crs, (src.height, src.width)
    fsi = src.read(1).astype("float64")
    fsi[fsi == src.nodata] = np.nan


def write(name, arr, transform, crs, nodata=-9999.0, dtype="float32", shape=None):
    h, w = arr.shape
    out = np.where(np.isfinite(arr), arr, nodata).astype(dtype) if dtype == "float32" else arr.astype(dtype)
    with rasterio.open(os.path.join(OUT_DIR, name), "w", driver="GTiff", height=h, width=w, count=1,
                       dtype=dtype, crs=crs, transform=transform, nodata=nodata, compress="deflate") as dst:
        dst.write(out, 1)


summary = {"issue_date": T0.isoformat(), "soil_product": SOIL_PRODUCT, "weights": {"rain": W_RAIN, "soil": W_SOIL},
           "soil_moisture_note": "latest available day, same value for all lead times", "leads": []}
log("\nLead   valid date    rain score   soil score   DTI (mean over LGA area)   FRI mean / max")
for lead in LEADS:
    idx = days.index(T0 + timedelta(days=lead))
    peak48 = core.max_over_previous(totals, idx, core.LOOKBACK_DAYS)
    r = core.rain_score(peak48, rain_tab)
    s = core.soil_score(sm_now, sm_p10, sm_p99)
    d = core.dti(r, s, W_RAIN, W_SOIL)
    write("dti_native_lead%d.tif" % lead, d, W_TRANSFORM, W_CRS)

    d30 = np.full(M_SHAPE, np.nan, dtype="float64")
    reproject(source=d, destination=d30, src_transform=W_TRANSFORM, src_crs=W_CRS, src_nodata=np.nan,
              dst_transform=M_TRANSFORM, dst_crs=M_CRS, dst_nodata=np.nan, resampling=DTI_TO_30M_RESAMPLING)
    d30[~np.isfinite(fsi)] = np.nan
    fri = fsi * d30
    write("dti_lead%d.tif" % lead, d30, M_TRANSFORM, M_CRS)
    write("fri_lead%d.tif" % lead, fri, M_TRANSFORM, M_CRS)
    write("dti_category_lead%d.tif" % lead, core.categorise(d30), M_TRANSFORM, M_CRS, nodata=0, dtype="uint8")

    valid = np.isfinite(fri)
    row = {"lead_days": lead, "valid_date": (T0 + timedelta(days=lead)).isoformat(),
           "rain_score_mean": float(np.nanmean(r)), "soil_score_mean": float(np.nanmean(s)),
           "dti_mean": float(np.nanmean(d30)), "fri_mean": float(np.nanmean(fri)), "fri_max": float(np.nanmax(fri)),
           "max48_prev14_mm_mean": float(np.nanmean(peak48))}
    summary["leads"].append(row)
    log("%-6s %-13s %-12.2f %-12.2f %-26.2f %.3f / %.3f" % (
        "+%dh" % (24 * lead) if lead else "now", row["valid_date"], row["rain_score_mean"],
        row["soil_score_mean"], row["dti_mean"], row["fri_mean"], row["fri_max"]))

with open(os.path.join(OUT_DIR, "forecast_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
log("\nDone. Outputs are in %s" % OUT_DIR)
log("Reminder: the DTI is a local, rainfall-driven indicator. It does not represent upstream Niger and Benue flow,")
log("and the soil-moisture term uses the latest available day for every lead time.")
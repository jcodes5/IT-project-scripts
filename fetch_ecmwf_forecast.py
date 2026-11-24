#!/usr/bin/env python3
"""
LIVE-DATA STEP 2: forecast rainfall for the two days after the issue date, from ECMWF open data (IFS).
Lokoja LGA flood susceptibility project.

Why it is built this way
  - IMERG (step 1) gives observed rain up to the issue date T0 (the last complete UTC day).
  - The grid job needs rain for T0+1 and T0+2 as well. Those two days come from the ECMWF forecast.
  - ECMWF "tp" is total precipitation ACCUMULATED from the start of the run, in metres. The rain of one
    UTC day is therefore tp at the end of that day minus tp at its start (times 1000 for mm).

What it does
  1. Reads T0 from data/phase6_inputs/t0.txt (written by fetch_imerg_early.py).
  2. Tries these forecast runs in order until one is available:
       the 00 UTC run of T0+1, then the 12 UTC run of T0, then the 00 UTC run of T0.
     (A run appears about 7 to 9 hours after its start time.)
  3. Downloads tp at the steps it needs, converts to daily rain in mm, cuts to the study region, and writes
       data/phase6_inputs/rain_<T0+1>.tif and rain_<T0+2>.tif   (0.25 degree grid)
  4. Writes data/phase6_inputs/forecast_run.txt with the run that was used.

Install : conda install -c conda-forge ecmwf-opendata cfgrib eccodes     (try conda first; pip is the fallback)
Run     : python fetch_ecmwf_forecast.py
Options : --t0 2026-10-11       override the issue date
          --source ecmwf        data mirror: ecmwf (default), aws, azure or google
          --inspect FILE        print the structure of a downloaded forecast file and stop
          --local FILE --s0 N   skip the download and process this file; N = hours from the run start to 00 UTC of T0+1
"""
import argparse
import os
import sys
from datetime import date, datetime, timedelta

import numpy as np
import rasterio
from rasterio.transform import from_origin

# ------------------------------ CONFIG ---------------------------------
LOOKUP = "data/phase6/rain48h_pctl_chirps.tif"
INPUT_DIR = "data/phase6_inputs"
RAW_DIR = "data/ecmwf_raw"
MARGIN_DEG = 0.5          # forecast cells are 0.25 degree, so keep a wider margin than for IMERG
# -----------------------------------------------------------------------

NODATA = -9999.0


def log(msg=""):
    print(msg, flush=True)


def region():
    if not os.path.exists(LOOKUP):
        sys.exit("Missing %s (made by phase6a). It defines the study region." % LOOKUP)
    with rasterio.open(LOOKUP) as src:
        if src.crs is None or src.crs.to_epsg() != 4326:
            sys.exit("%s must be in EPSG:4326." % LOOKUP)
        b = src.bounds
    return b.left - MARGIN_DEG, b.bottom - MARGIN_DEG, b.right + MARGIN_DEG, b.top + MARGIN_DEG


def open_tp(path):
    """Open a forecast file and return tp as a dict {step in hours: 2-D array (lat north to south, lon west to east)} in metres."""
    import xarray as xr
    if path.lower().endswith((".grib", ".grib2", ".grb", ".grb2", ".bin")):
        ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    else:
        ds = xr.open_dataset(path)
    if "tp" not in ds.variables:
        sys.exit("No 'tp' variable in %s. Variables: %s. Run with --inspect and send me the output." % (path, list(ds.data_vars)))
    da = ds["tp"]
    lat = [d for d in da.dims if "lat" in d.lower()]
    lon = [d for d in da.dims if "lon" in d.lower()]
    if len(lat) != 1 or len(lon) != 1:
        sys.exit("Unexpected dimensions %s. Run with --inspect and send me the output." % (da.dims,))
    da = da.rename({lat[0]: "lat", lon[0]: "lon"})
    if "step" not in da.dims:
        da = da.expand_dims("step")
    lons = da["lon"].values
    if lons.max() > 180:      # 0..360 grid: convert to -180..180
        da = da.assign_coords(lon=((da["lon"] + 180) % 360) - 180)
    da = da.sortby("lat", ascending=False).sortby("lon", ascending=True)
    steps = da["step"].values
    hours = steps / np.timedelta64(1, "h") if np.issubdtype(steps.dtype, np.timedelta64) else np.asarray(steps, dtype=float)
    out = {}
    for i, h in enumerate(np.atleast_1d(hours)):
        out[int(round(float(h)))] = da.isel(step=i).transpose("lat", "lon")
    return out


def inspect(path):
    import xarray as xr
    log("File: %s" % path)
    tp = open_tp(path)
    first = tp[sorted(tp)[0]]
    log("tp steps (hours): %s" % sorted(tp))
    log("grid: %d lat x %d lon, lat %.2f to %.2f, lon %.2f to %.2f" % (
        first.sizes["lat"], first.sizes["lon"], first["lat"].values[0], first["lat"].values[-1],
        first["lon"].values[0], first["lon"].values[-1]))
    for h in sorted(tp):
        v = tp[h].values
        log("  step %3d h: tp min %.5f, mean %.5f, max %.5f m" % (h, np.nanmin(v), np.nanmean(v), np.nanmax(v)))


def write_day(daily_mm, lat, lon, bounds, out_path):
    """daily_mm is a DataArray (lat, lon) in mm."""
    w, s, e, n = bounds
    sub = daily_mm.sel(lon=slice(w, e), lat=slice(n, s))
    if sub.sizes["lat"] < 2 or sub.sizes["lon"] < 2:
        sys.exit("The clipped forecast grid is empty. Check that the region %s lies inside the data." % (bounds,))
    lons, lats = sub["lon"].values, sub["lat"].values
    res = abs(float(lons[1] - lons[0]))
    transform = from_origin(float(lons[0]) - res / 2, float(lats[0]) + res / 2, res, res)
    arr = np.where(np.isfinite(sub.values), sub.values, NODATA).astype("float32")
    with rasterio.open(out_path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype="float32", crs="EPSG:4326", transform=transform, nodata=NODATA, compress="deflate") as dst:
        dst.write(arr, 1)
    return float(np.nanmean(sub.values))


def process(path, s0, d1, d2):
    tp = open_tp(path)
    need = ([s0] if s0 > 0 else []) + [s0 + 24, s0 + 48]
    missing = [h for h in need if h not in tp]
    if missing:
        sys.exit("The file has steps %s but I need %s. Run with --inspect and send me the output." % (sorted(tp), need))
    zero = tp[s0 + 24] * 0.0
    at = lambda h: tp[h] if h in tp and h > 0 else zero
    day1 = (at(s0 + 24) - (at(s0) if s0 > 0 else zero)).clip(min=0) * 1000.0
    day2 = (at(s0 + 48) - at(s0 + 24)).clip(min=0) * 1000.0
    bounds = region()
    os.makedirs(INPUT_DIR, exist_ok=True)
    m1 = write_day(day1, None, None, bounds, os.path.join(INPUT_DIR, "rain_%s.tif" % d1.strftime("%Y%m%d")))
    m2 = write_day(day2, None, None, bounds, os.path.join(INPUT_DIR, "rain_%s.tif" % d2.strftime("%Y%m%d")))
    log("Forecast rain written: %s mean over the region %.1f mm, %s mean %.1f mm" % (d1, m1, d2, m2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0", default=None)
    ap.add_argument("--source", default="ecmwf")
    ap.add_argument("--inspect", default=None)
    ap.add_argument("--local", default=None)
    ap.add_argument("--s0", type=int, default=None)
    args = ap.parse_args()

    if args.inspect:
        inspect(args.inspect)
        return

    if args.t0:
        t0 = date.fromisoformat(args.t0)
    else:
        p = os.path.join(INPUT_DIR, "t0.txt")
        if not os.path.exists(p):
            sys.exit("Missing %s. Run fetch_imerg_early.py first, or pass --t0." % p)
        t0 = date.fromisoformat(open(p).read().strip())
    d1, d2 = t0 + timedelta(days=1), t0 + timedelta(days=2)
    log("Issue date T0 = %s; forecast days needed: %s and %s" % (t0, d1, d2))

    if args.local:
        if args.s0 is None:
            sys.exit("With --local you must also give --s0 (hours from the run start to 00 UTC of %s)." % d1)
        process(args.local, args.s0, d1, d2)
        return

    from ecmwf.opendata import Client
    client = Client(source=args.source)
    os.makedirs(RAW_DIR, exist_ok=True)
    d1_midnight = datetime(d1.year, d1.month, d1.day)
    candidates = [(d1, 0), (t0, 12), (t0, 0)]
    for run_date, hour in candidates:
        run_start = datetime(run_date.year, run_date.month, run_date.day, hour)
        s0 = int((d1_midnight - run_start).total_seconds() // 3600)
        steps = ([s0] if s0 > 0 else []) + [s0 + 24, s0 + 48]
        target = os.path.join(RAW_DIR, "ifs_tp_%s_%02dz.grib2" % (run_date.strftime("%Y%m%d"), hour))
        log("Trying the %02d UTC run of %s (steps %s) ..." % (hour, run_date, steps))
        try:
            if not os.path.exists(target):
                client.retrieve(date=int(run_date.strftime("%Y%m%d")), time=hour, type="fc", stream="oper",
                                param="tp", step=steps, target=target)
        except Exception as exc:       # run not published yet, or a network problem
            log("  not available: %s" % exc)
            if os.path.exists(target):
                os.remove(target)
            continue
        process(target, s0, d1, d2)
        with open(os.path.join(INPUT_DIR, "forecast_run.txt"), "w") as f:
            f.write("%s %02dz\n" % (run_date.isoformat(), hour))
        log("Used the %02d UTC run of %s. Next: the soil moisture (SMAP)." % (hour, run_date))
        return
    sys.exit("None of the candidate forecast runs could be downloaded. Check your internet connection and the messages above.")


if __name__ == "__main__":
    main()
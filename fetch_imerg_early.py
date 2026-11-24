#!/usr/bin/env python3
"""
LIVE-DATA STEP 1: observed rainfall from NASA IMERG Early (daily, 0.1 degree, mm per day).
Lokoja LGA flood susceptibility project.

What it does
  1. Logs in to NASA Earthdata with earthaccess (uses the login you saved earlier).
  2. Finds the newest IMERG Early daily file (GPM_3IMERGDE, version 07) and downloads the 16 days
     ending on that date (the issue date T0) into data/imerg_raw/. Files already downloaded are not repeated.
  3. Cuts each day to the study region and writes data/phase6_inputs/rain_YYYYMMDD.tif (mm per day).
  4. Writes data/phase6_inputs/t0.txt with the issue date, which the later steps read.

The forecast days after T0 (ECMWF) and the soil moisture (SMAP) come from the other fetch steps.
The region is taken from the lookup table made by phase6a (data/phase6/rain48h_pctl_chirps.tif).

Install : pip install earthaccess xarray netcdf4 rasterio numpy
Run     : python fetch_imerg_early.py
Options : --t0 2026-10-11          use this issue date (or the newest available if it is not out yet)
          --local-only             do not download; use the files already in data/imerg_raw
          --inspect PATH           print the structure of one downloaded file (or the first one in a folder)
                                   and stop. Use this if the script cannot find the rainfall variable.
"""
import argparse
import glob
import os
import re
import sys
from datetime import date, datetime, timedelta

import numpy as np
import rasterio
from rasterio.transform import from_origin

# ------------------------------ CONFIG ---------------------------------
LOOKUP = "data/phase6/rain48h_pctl_chirps.tif"
INPUT_DIR = "data/phase6_inputs"
RAW_DIR = "data/imerg_raw"
SHORT_NAME = "GPM_3IMERGDE"
VERSION = "07"
BACK_DAYS = 15            # observed days before T0 needed by the 14-day look-back and the 48 h window
MARGIN_DEG = 0.3          # extra margin around the lookup-table extent
SEARCH_DAYS = 12          # how far back to look for the newest available file
# -----------------------------------------------------------------------

NODATA = -9999.0
DATE_RE = re.compile(r"(\d{8})-S\d{6}")


def log(msg=""):
    print(msg, flush=True)


def file_date(path):
    m = DATE_RE.search(os.path.basename(path))
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d").date()


def list_local(raw_dir):
    out = {}
    for p in glob.glob(os.path.join(raw_dir, "*")):
        if p.lower().endswith((".nc4", ".nc", ".hdf5", ".h5")):
            d = file_date(p)
            if d:
                out[d] = p
    return out


def open_ds(path):
    import xarray as xr
    return xr.open_dataset(path, decode_times=False)


def find_rain(ds):
    """Return the rainfall DataArray as (lat, lon), north to south, with fill values removed."""
    name = "precipitation" if "precipitation" in ds.variables else None
    if name is None:
        for cand in ds.data_vars:
            if "precip" in cand.lower() and "cnt" not in cand.lower() and "error" not in cand.lower() \
                    and "prob" not in cand.lower() and "cal" not in cand.lower():
                name = cand
                break
    if name is None:
        sys.exit("Could not find a rainfall variable. Variables: %s. Run with --inspect and send me the output." % list(ds.data_vars))
    da = ds[name]
    for d in list(da.dims):
        if d.lower() in ("time", "nv", "bnds"):
            da = da.isel({d: 0}, drop=True)
    lat = [d for d in da.dims if "lat" in d.lower()]
    lon = [d for d in da.dims if "lon" in d.lower()]
    if len(lat) != 1 or len(lon) != 1:
        sys.exit("Unexpected dimensions %s for %s. Run with --inspect and send me the output." % (da.dims, name))
    da = da.transpose(lat[0], lon[0])
    da = da.rename({lat[0]: "lat", lon[0]: "lon"})
    da = da.sortby("lat", ascending=False).sortby("lon", ascending=True)
    da = da.where((da > -9000) & np.isfinite(da))
    return da


def region():
    if not os.path.exists(LOOKUP):
        sys.exit("Missing %s (made by phase6a). It defines the study region." % LOOKUP)
    with rasterio.open(LOOKUP) as src:
        if src.crs is None or src.crs.to_epsg() != 4326:
            sys.exit("%s must be in EPSG:4326." % LOOKUP)
        b = src.bounds
    return b.left - MARGIN_DEG, b.bottom - MARGIN_DEG, b.right + MARGIN_DEG, b.top + MARGIN_DEG


def write_day(da, bounds, out_path):
    w, s, e, n = bounds
    sub = da.sel(lon=slice(w, e), lat=slice(n, s))
    if sub.sizes["lat"] < 2 or sub.sizes["lon"] < 2:
        sys.exit("The clipped rainfall grid is empty. Check that the region (%s) lies inside the data." % (bounds,))
    lon = sub["lon"].values
    lat = sub["lat"].values
    res = abs(float(lon[1] - lon[0]))
    transform = from_origin(float(lon[0]) - res / 2, float(lat[0]) + res / 2, res, res)
    arr = np.where(np.isfinite(sub.values), sub.values, NODATA).astype("float32")
    with rasterio.open(out_path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype="float32", crs="EPSG:4326", transform=transform, nodata=NODATA,
                       compress="deflate") as dst:
        dst.write(arr, 1)


def inspect(path):
    if os.path.isdir(path):
        files = sorted(list_local(path).values())
        if not files:
            sys.exit("No IMERG files found in %s" % path)
        path = files[0]
    ds = open_ds(path)
    log("File: %s" % path)
    log(ds)
    da = find_rain(ds)
    log("\nRainfall variable after tidy-up: dims %s, shape %s" % (da.dims, da.shape))
    v = da.values[np.isfinite(da.values)]
    log("Valid cells %d, min %.2f, mean %.2f, max %.2f mm/day" % (v.size, v.min(), v.mean(), v.max()))
    log("lat from %.2f to %.2f, lon from %.2f to %.2f" % (da.lat.values[0], da.lat.values[-1], da.lon.values[0], da.lon.values[-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0", default=None)
    ap.add_argument("--local-only", action="store_true")
    ap.add_argument("--inspect", default=None)
    args = ap.parse_args()

    if args.inspect:
        inspect(args.inspect)
        return

    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(INPUT_DIR, exist_ok=True)

    if not args.local_only:
        import earthaccess
        earthaccess.login()
        today = date.today()
        start = today - timedelta(days=SEARCH_DAYS + BACK_DAYS)
        log("Searching %s v%s from %s to %s ..." % (SHORT_NAME, VERSION, start, today))
        results = earthaccess.search_data(short_name=SHORT_NAME, version=VERSION,
                                          temporal=(start.isoformat(), today.isoformat()))
        log("Found %d daily files" % len(results))
        if not results:
            sys.exit("No files found. Check the Earthdata login and that 'NASA GESDISC DATA ARCHIVE' is approved.")
        earthaccess.download(results, RAW_DIR)

    local = list_local(RAW_DIR)
    if not local:
        sys.exit("No IMERG files in %s" % RAW_DIR)
    latest = max(local)
    t0 = date.fromisoformat(args.t0) if args.t0 else latest
    if t0 > latest:
        log("Requested T0 %s is not available yet; using the newest available day %s" % (t0, latest))
        t0 = latest
    need = [t0 - timedelta(days=k) for k in range(BACK_DAYS, -1, -1)]
    missing = [d for d in need if d not in local]
    if missing:
        sys.exit("Missing IMERG days: %s. Download them (run without --local-only) or choose another --t0."
                 % ", ".join(str(d) for d in missing))

    bounds = region()
    log("Region (west, south, east, north): %s" % (tuple(round(v, 2) for v in bounds),))
    for d in need:
        da = find_rain(open_ds(local[d]))
        out = os.path.join(INPUT_DIR, "rain_%s.tif" % d.strftime("%Y%m%d"))
        write_day(da, bounds, out)
    with open(os.path.join(INPUT_DIR, "t0.txt"), "w") as f:
        f.write(t0.isoformat() + "\n")
    log("Wrote %d daily rain files (%s to %s) into %s" % (len(need), need[0], need[-1], INPUT_DIR))
    log("Issue date T0 = %s (written to %s)" % (t0, os.path.join(INPUT_DIR, "t0.txt")))
    log("Next: the forecast days (ECMWF) and the soil moisture (SMAP).")


if __name__ == "__main__":
    main()
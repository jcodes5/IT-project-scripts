#!/usr/bin/env python3
"""
LIVE-DATA STEP 3: latest SMAP L4 surface soil moisture, through the Earth Engine Python API.
Lokoja LGA flood susceptibility project.

What it does
  1. Signs in to Earth Engine (the first run opens a browser window for you to approve).
  2. Finds the newest SMAP L4 day (NASA/SMAP/SPL4SMGP/008, band sm_surface) that has all 8 three-hourly images.
  3. Averages that day, cuts it to the study region and saves data/phase6_inputs/sm_latest.tif (m3/m3).
  4. Writes data/phase6_inputs/sm_date.txt with the day used (SMAP runs about 4 to 5 days behind).

The region comes from the lookup table made by phase6a (data/phase6/rain48h_pctl_chirps.tif).
This must be the same product as SOIL_PRODUCT = "smap" in phase6c_grid_job.py.

Install : conda install -c conda-forge earthengine-api requests      (pip install earthengine-api is the fallback)
First use: you need your Earth Engine Cloud project ID (see below). Run:
            python fetch_smap_latest.py --project YOUR_PROJECT_ID
          A browser opens once so you can approve access. Later runs reuse the saved sign-in.
Run     : python fetch_smap_latest.py --project YOUR_PROJECT_ID     (or set the EE_PROJECT environment variable)

Where to find the project ID: open the Earth Engine Code Editor; the Cloud project is shown in your account menu at
the top right. It is also listed at console.cloud.google.com. It often looks like ee-yourname.
"""
import argparse
import io
import os
import sys
import zipfile
from datetime import date, datetime, timedelta, timezone

import rasterio

# ------------------------------ CONFIG ---------------------------------
LOOKUP = "data/phase6/rain48h_pctl_chirps.tif"
INPUT_DIR = "data/phase6_inputs"
DATASET = "NASA/SMAP/SPL4SMGP/008"
BAND = "sm_surface"
IMAGES_PER_DAY = 8
MARGIN_DEG = 0.3
# -----------------------------------------------------------------------


def log(msg=""):
    print(msg, flush=True)


def region():
    if not os.path.exists(LOOKUP):
        sys.exit("Missing %s (made by phase6a). It defines the study region." % LOOKUP)
    with rasterio.open(LOOKUP) as src:
        b = src.bounds
    return [b.left - MARGIN_DEG, b.bottom - MARGIN_DEG, b.right + MARGIN_DEG, b.top + MARGIN_DEG]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=os.environ.get("EE_PROJECT"))
    args = ap.parse_args()
    if not args.project:
        sys.exit("Give your Earth Engine Cloud project with --project YOUR_PROJECT_ID (or set EE_PROJECT).")

    import ee
    import requests

    try:
        ee.Initialize(project=args.project)
    except Exception:
        log("Earth Engine sign-in needed. A browser window will open ...")
        ee.Authenticate()
        ee.Initialize(project=args.project)

    col = ee.ImageCollection(DATASET).select(BAND)
    last_ms = col.aggregate_max("system:time_start").getInfo()
    last = datetime.fromtimestamp(last_ms / 1000.0, tz=timezone.utc)
    log("Newest SMAP L4 image: %s UTC" % last.strftime("%Y-%m-%d %H:%M"))

    def count(day):
        start = ee.Date(day.isoformat())
        return col.filterDate(start, start.advance(1, "day")).size().getInfo()

    day = last.date()
    n = count(day)
    if n < IMAGES_PER_DAY:
        log("%s has only %d of %d images; using the day before." % (day, n, IMAGES_PER_DAY))
        day = day - timedelta(days=1)
        n = count(day)
    if n < IMAGES_PER_DAY:
        sys.exit("Day %s has only %d images. Something is unusual with the dataset; stopping." % (day, n))

    start = ee.Date(day.isoformat())
    img = col.filterDate(start, start.advance(1, "day")).mean().rename(BAND)
    w, s, e, nn = region()
    proj = col.first().projection().getInfo()
    url = img.getDownloadURL({
        "region": ee.Geometry.Rectangle([w, s, e, nn], None, False),
        "crs": proj["crs"],
        "crs_transform": proj["transform"],
        "format": "GEO_TIFF",
    })
    log("Downloading soil moisture for %s ..." % day)
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    data = r.content
    if data[:2] == b"PK":                       # a zip with the tif inside
        z = zipfile.ZipFile(io.BytesIO(data))
        name = [n_ for n_ in z.namelist() if n_.lower().endswith(".tif")][0]
        data = z.read(name)
    os.makedirs(INPUT_DIR, exist_ok=True)
    out = os.path.join(INPUT_DIR, "sm_latest.tif")
    with open(out, "wb") as f:
        f.write(data)
    with rasterio.open(out) as src:
        a = src.read(1)
        vals = a[(a == a) & (a > -9000)]
        log("Saved %s: %d x %d cells, CRS %s, soil moisture min %.3f, mean %.3f, max %.3f" % (
            out, src.width, src.height, src.crs, vals.min(), vals.mean(), vals.max()))
    with open(os.path.join(INPUT_DIR, "sm_date.txt"), "w") as f:
        f.write(day.isoformat() + "\n")
    t0p = os.path.join(INPUT_DIR, "t0.txt")
    if os.path.exists(t0p):
        t0 = date.fromisoformat(open(t0p).read().strip())
        log("Soil moisture is from %s, which is %d days before the issue date %s." % (day, (t0 - day).days, t0))
    log("Next: score everything with phase6c_grid_job.py.")


if __name__ == "__main__":
    main()
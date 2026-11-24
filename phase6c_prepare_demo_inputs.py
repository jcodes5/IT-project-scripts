#!/usr/bin/env python3
"""
Prepare the demo inputs for phase6c_grid_job.py.

Input  (download these two files from your Google Drive folder, made by phase6c_demo_inputs.js, into data/phase6_demo_raw/)
         rain_stack.tif   18 bands, one per day, band descriptions dYYYYMMDD
         sm_latest.tif    soil moisture
Output data/phase6_inputs/rain_YYYYMMDD.tif   one file per day
       data/phase6_inputs/sm_latest.tif

Then run:  python phase6c_grid_job.py --t0 <the issue date printed below>

Install : pip install rasterio numpy
Run     : python phase6c_prepare_demo_inputs.py
"""
import os
import re
import shutil
import sys
from datetime import datetime, timedelta

import rasterio

RAW_DIR = "data/phase6_demo_raw"
OUT_DIR = "data/phase6_inputs"
FWD_DAYS = 2            # must match phase6c_demo_inputs.js

stack = os.path.join(RAW_DIR, "rain_stack.tif")
sm = os.path.join(RAW_DIR, "sm_latest.tif")
for p in (stack, sm):
    if not os.path.exists(p):
        sys.exit("Missing file: %s" % p)
os.makedirs(OUT_DIR, exist_ok=True)

dates = []
with rasterio.open(stack) as src:
    desc = list(src.descriptions)
    for i in range(src.count):
        m = re.search(r"d?(\d{8})", desc[i] or "")
        if not m:
            sys.exit("Band %d has no date in its name (%r). Re-export with phase6c_demo_inputs.js." % (i + 1, desc[i]))
        dates.append(datetime.strptime(m.group(1), "%Y%m%d").date())
    profile = src.profile.copy()
    profile.update(count=1)
    for i, d in enumerate(dates):
        out = os.path.join(OUT_DIR, "rain_%s.tif" % d.strftime("%Y%m%d"))
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(src.read(i + 1), 1)
shutil.copyfile(sm, os.path.join(OUT_DIR, "sm_latest.tif"))

t0 = dates[-1] - timedelta(days=FWD_DAYS)
print("Wrote %d daily rain files (%s to %s) and sm_latest.tif into %s" % (len(dates), dates[0], dates[-1], OUT_DIR))
print("Issue date T0 = %s (the last %d days stand in for the forecast)" % (t0, FWD_DAYS))
print("Now run:  python phase6c_grid_job.py --t0 %s" % t0)
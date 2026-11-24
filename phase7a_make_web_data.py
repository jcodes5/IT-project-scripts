#!/usr/bin/env python3
"""
PHASE 7a: Build the data the web map needs.
Lokoja LGA flood susceptibility project.

Turns the result rasters into static map tiles (XYZ, PNG, Web Mercator) and writes two small JSON files:
  layers.json   one entry per layer: title, group, legend, tile URL template, zoom range, bounds, note
  summary.json  the tally tables and validation numbers, so the web app can show them without a server

No GDAL command-line tools are needed: tiles are cut with rasterio and Pillow.

Input   (any file that does not exist is skipped)
        data/phase3/fsi_class.tif, fsi_scaled.tif
        data/phase5_results/risk_class.tif, risk_index.tif, exposure_index.tif and the tally CSVs
        data/phase4_results/validation_summary.csv
        data/phase6_outputs/<issue date>/dti_lead*.tif, fri_lead*.tif, forecast_summary.json
Output  web_data/tiles/<layer id>/<z>/<x>/<y>.png, web_data/layers.json, web_data/summary.json
        Copy the web_data folder into the web app's public folder.

Install : pip install rasterio numpy pillow
Run     : python phase7a_make_web_data.py            (options: --out web_data --minzoom 8 --maxzoom 12)
"""
import argparse
import csv
import glob
import json
import math
import os
import shutil
import sys

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform_bounds

# ------------------------------ CONFIG ---------------------------------
PHASE3_DIR = "data/phase3"
PHASE4_DIR = "data/phase4_results"
PHASE5_DIR = "data/phase5_results"
PHASE6_OUT = "data/phase6_outputs"

CLASS_COLORS = ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]
SUSC_LABELS = ["Very low", "Low", "Moderate", "High", "Very high"]
RAMP = [(0.00, "#ffffcc"), (0.25, "#fed976"), (0.50, "#fd8d3c"), (0.75, "#e31a1c"), (1.00, "#800026")]
DTI_NOTE = ("Local rainfall and soil-wetness trigger. It does not include river flow from the Niger and Benue, "
            "so it is not a forecast of river flooding.")
# -----------------------------------------------------------------------

ORIGIN = 20037508.342789244
TILE = 256


def log(msg=""):
    print(msg, flush=True)


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def ramp_rgba(values, vmin, vmax):
    """Colour a float array with RAMP; NaN becomes transparent."""
    t = np.nan_to_num(np.clip((values - vmin) / max(vmax - vmin, 1e-12), 0, 1), nan=0.0)
    stops = np.array([s for s, _ in RAMP])
    cols = np.array([hex_rgb(c) for _, c in RAMP], dtype="float64")
    out = np.zeros(values.shape + (4,), dtype="uint8")
    for ch in range(3):
        out[..., ch] = np.interp(t, stops, cols[:, ch]).astype("uint8")
    out[..., 3] = np.where(np.isfinite(values), 255, 0)
    return out


def class_rgba(arr, colors, nodata):
    out = np.zeros(arr.shape + (4,), dtype="uint8")
    for k, c in enumerate(colors, start=1):
        m = arr == k
        out[m, :3] = hex_rgb(c)
        out[m, 3] = 255
    return out


def build_layers(args):
    layers = []

    def add(layer_id, title, group, path, kind, legend, note="", vmin=0.0, vmax=1.0, colors=None):
        if os.path.exists(path):
            layers.append({"id": layer_id, "title": title, "group": group, "path": path, "kind": kind,
                           "legend": legend, "note": note, "vmin": vmin, "vmax": vmax, "colors": colors})
        else:
            log("  skipped (file not found): %s" % path)

    ramp_legend = [{"label": "%.2f" % s, "color": c} for s, c in RAMP]
    class_legend = [{"label": l, "color": c} for l, c in zip(SUSC_LABELS, CLASS_COLORS)]

    add("susceptibility_class", "Flood susceptibility (classes)", "Static map", os.path.join(PHASE3_DIR, "fsi_class.tif"),
        "class", class_legend, "Relative susceptibility from terrain, drainage and surface factors (AHP). Not a flood depth or probability.",
        colors=CLASS_COLORS)
    add("susceptibility_index", "Flood susceptibility index (0-1)", "Static map", os.path.join(PHASE3_DIR, "fsi_scaled.tif"),
        "continuous", ramp_legend, "Continuous susceptibility index rescaled to 0-1 over Lokoja LGA.")
    add("exposure_index", "Exposure index (0-1)", "Static map", os.path.join(PHASE5_DIR, "exposure_index.tif"),
        "continuous", ramp_legend, "Population, buildings and roads within 500 m, combined with AHP weights.")
    add("risk_class", "Environmental flood risk (classes)", "Static map", os.path.join(PHASE5_DIR, "risk_class.tif"),
        "class", class_legend, "Susceptibility x exposure. A relative ranking inside Lokoja LGA, not an expected loss.",
        colors=CLASS_COLORS)
    add("risk_index", "Environmental flood risk index (0-1)", "Static map", os.path.join(PHASE5_DIR, "risk_index.tif"),
        "continuous", ramp_legend, "Susceptibility x exposure, continuous.")

    dirs = sorted(d for d in glob.glob(os.path.join(PHASE6_OUT, "*")) if os.path.isdir(d))
    if dirs:
        latest = dirs[-1]
        issue = os.path.basename(latest)
        names = {0: "now", 1: "+24 h", 2: "+48 h"}
        for lead in (0, 1, 2):
            add("dti_lead%d" % lead, "Rain and soil-wetness trigger, %s" % names[lead], "Dynamic (issued %s)" % issue,
                os.path.join(latest, "dti_lead%d.tif" % lead), "continuous", ramp_legend, DTI_NOTE)
            add("fri_lead%d" % lead, "Flood-risk index (susceptibility x trigger), %s" % names[lead],
                "Dynamic (issued %s)" % issue, os.path.join(latest, "fri_lead%d.tif" % lead), "continuous",
                ramp_legend, DTI_NOTE)
    else:
        log("  no dynamic outputs found in %s" % PHASE6_OUT)
    return layers


def tile_range(bounds, z):
    minx, miny, maxx, maxy = bounds
    n = 2 ** z
    size = 2 * ORIGIN / n
    x0 = int(math.floor((minx + ORIGIN) / size))
    x1 = int(math.floor((maxx + ORIGIN) / size))
    y0 = int(math.floor((ORIGIN - maxy) / size))
    y1 = int(math.floor((ORIGIN - miny) / size))
    return size, max(x0, 0), min(x1, n - 1), max(y0, 0), min(y1, n - 1)


def make_tiles(layer, out_root, minzoom, maxzoom):
    with rasterio.open(layer["path"]) as src:
        bounds3857 = transform_bounds(src.crs, "EPSG:3857", *src.bounds)
        bounds4326 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        nodata = src.nodata
        categorical = layer["kind"] == "class"
        resampling = Resampling.nearest if categorical else Resampling.bilinear
        written = 0
        for z in range(minzoom, maxzoom + 1):
            size, x0, x1, y0, y1 = tile_range(bounds3857, z)
            for x in range(x0, x1 + 1):
                for y in range(y0, y1 + 1):
                    west = -ORIGIN + x * size
                    north = ORIGIN - y * size
                    dst_transform = from_bounds(west, north - size, west + size, north, TILE, TILE)
                    if categorical:
                        dst = np.full((TILE, TILE), 255, dtype="uint8")
                        reproject(source=rasterio.band(src, 1), destination=dst, src_nodata=nodata, dst_nodata=255,
                                  dst_transform=dst_transform, dst_crs="EPSG:3857", resampling=resampling)
                        rgba = class_rgba(dst, layer["colors"], 255)
                    else:
                        dst = np.full((TILE, TILE), np.nan, dtype="float32")
                        reproject(source=rasterio.band(src, 1), destination=dst, src_nodata=nodata, dst_nodata=np.nan,
                                  dst_transform=dst_transform, dst_crs="EPSG:3857", resampling=resampling)
                        rgba = ramp_rgba(dst, layer["vmin"], layer["vmax"])
                    if rgba[..., 3].max() == 0:
                        continue
                    folder = os.path.join(out_root, layer["id"], str(z), str(x))
                    os.makedirs(folder, exist_ok=True)
                    Image.fromarray(rgba, "RGBA").save(os.path.join(folder, "%d.png" % y), optimize=True)
                    written += 1
    return written, bounds4326


def read_csv(path):
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="web_data")
    ap.add_argument("--minzoom", type=int, default=8)
    ap.add_argument("--maxzoom", type=int, default=12)
    args = ap.parse_args()

    log("Looking for layers ...")
    layers = build_layers(args)
    if not layers:
        sys.exit("No result rasters found. Run the earlier phases first.")
    tiles_root = os.path.join(args.out, "tiles")
    if os.path.exists(tiles_root):
        shutil.rmtree(tiles_root)
    os.makedirs(tiles_root, exist_ok=True)

    manifest = []
    for layer in layers:
        n, b = make_tiles(layer, tiles_root, args.minzoom, args.maxzoom)
        log("  %-22s %4d tiles" % (layer["id"], n))
        manifest.append({
            "id": layer["id"], "title": layer["title"], "group": layer["group"], "kind": layer["kind"],
            "legend": layer["legend"], "note": layer["note"],
            "tiles": "tiles/%s/{z}/{x}/{y}.png" % layer["id"],
            "minzoom": args.minzoom, "maxzoom": args.maxzoom,
            "bounds_wgs84": [round(v, 6) for v in b],
        })
    allb = np.array([m["bounds_wgs84"] for m in manifest])
    bounds = [float(allb[:, 0].min()), float(allb[:, 1].min()), float(allb[:, 2].max()), float(allb[:, 3].max())]
    with open(os.path.join(args.out, "layers.json"), "w") as f:
        json.dump({"bounds_wgs84": bounds, "center": [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2],
                   "layers": manifest}, f, indent=2)

    summary = {
        "susceptibility_exposure": read_csv(os.path.join(PHASE5_DIR, "exposure_by_susceptibility_class.csv")),
        "risk_exposure": read_csv(os.path.join(PHASE5_DIR, "exposure_by_risk_class.csv")),
        "observed_floods_exposure": read_csv(os.path.join(PHASE5_DIR, "exposure_in_observed_floods.csv")),
        "validation": read_csv(os.path.join(PHASE4_DIR, "validation_summary.csv")),
        "forecast": None,
        "caveats": [
            "Susceptibility is a relative index of terrain and surface conditions; it is not a flood depth or probability.",
            "Risk is a relative ranking inside Lokoja LGA, not an expected loss.",
            "Flood extents used for validation come from radar, which misses flooding among buildings.",
            DTI_NOTE,
        ],
    }
    dirs = sorted(d for d in glob.glob(os.path.join(PHASE6_OUT, "*")) if os.path.isdir(d))
    if dirs and os.path.exists(os.path.join(dirs[-1], "forecast_summary.json")):
        with open(os.path.join(dirs[-1], "forecast_summary.json")) as f:
            summary["forecast"] = json.load(f)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log("\nDone. Copy %s into the web app's public folder." % args.out)


if __name__ == "__main__":
    main()
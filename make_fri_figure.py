#!/usr/bin/env python3
"""
Create a publication-quality map of the flood-risk index (FRI) for one forecast lead.

Examples:
  python make_fri_figure.py --issue 2026-10-17 --lead 1
  python make_fri_figure.py --lead 0

The default run uses the strongest rainy-season run present in data/phase6_outputs.
"""

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LinearSegmentedColormap

PHASE2_DIR = "data/phase2"
PHASE6_OUT = "data/phase6_outputs"
WATER_FILE = os.path.join(PHASE2_DIR, "perm_water.tif")
DEFAULT_ISSUE = "2026-10-17"
DPI = 300
MARGIN_PX = 10
SCALE_BAR_KM = 10
WATER_COLOUR = "#08306b"

FRI_RAMP = LinearSegmentedColormap.from_list(
    "fri", ["#ffffcc", "#fed976", "#fd8d3c", "#e31a1c", "#800026"], N=256
)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 9,
})


def read_raster(path, nodata=None):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        nd = src.nodata if src.nodata is not None else nodata
        if nd is not None:
            arr[arr == nd] = np.nan
        return arr, src.transform, src.crs


def latest_issue_dir():
    dirs = sorted(d for d in glob.glob(os.path.join(PHASE6_OUT, "*")) if os.path.isdir(d))
    if not dirs:
        sys.exit("No Phase 6 output directories found")
    return dirs[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue", default=DEFAULT_ISSUE,
                        help="Issue date directory, YYYY-MM-DD (default: 2026-10-17)")
    parser.add_argument("--lead", type=int, choices=(0, 1, 2), default=1,
                        help="Forecast lead: 0=now, 1=+24 h, 2=+48 h")
    parser.add_argument("--out", default=None, help="Optional output PNG path")
    args = parser.parse_args()

    run_dir = os.path.join(PHASE6_OUT, args.issue)
    fri_file = os.path.join(run_dir, "fri_lead%d.tif" % args.lead)
    if not os.path.exists(fri_file):
        sys.exit("Missing %s" % fri_file)

    fri, transform, crs = read_raster(fri_file)
    pixel_m = abs(transform.a)

    water = None
    if os.path.exists(WATER_FILE):
        raw, _, _ = read_raster(WATER_FILE, 255.0)
        if raw.shape == fri.shape:
            water = np.where(raw == 1, 1.0, np.nan)

    extent = np.isfinite(fri)
    if water is not None:
        extent = extent | np.isfinite(water)
    rows = np.where(extent.any(axis=1))[0]
    cols = np.where(extent.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        sys.exit("No valid mapped cells found in %s" % fri_file)

    r0 = max(rows.min() - MARGIN_PX, 0)
    r1 = min(rows.max() + MARGIN_PX + 1, fri.shape[0])
    c0 = max(cols.min() - MARGIN_PX, 0)
    c1 = min(cols.max() + MARGIN_PX + 1, fri.shape[1])
    fri = fri[r0:r1, c0:c1]
    if water is not None:
        water = water[r0:r1, c0:c1]

    lead_name = {0: "now", 1: "+24 h", 2: "+48 h"}[args.lead]
    valid_date = None
    summary_file = os.path.join(run_dir, "forecast_summary.json")
    if os.path.exists(summary_file):
        with open(summary_file, "r") as f:
            summary = json.load(f)
        row = next((x for x in summary["leads"] if x["lead_days"] == args.lead), None)
        if row:
            valid_date = row["valid_date"]

    fig = plt.figure(figsize=(7.2, 6.3), dpi=DPI)
    gs = fig.add_gridspec(1, 2, left=0.02, right=0.98, top=0.87, bottom=0.06,
                          width_ratios=[5.7, 1.15], wspace=0.03)
    ax = fig.add_subplot(gs[0, 0])
    key = fig.add_subplot(gs[0, 1])

    im = ax.imshow(fri, cmap=FRI_RAMP, vmin=0.0, vmax=1.0,
                   interpolation="nearest", aspect="equal")
    if water is not None:
        ax.imshow(water, cmap=matplotlib.colors.ListedColormap([WATER_COLOUR]),
                  vmin=0.0, vmax=1.0, interpolation="nearest", aspect="equal")

    fig.text(0.02, 0.955, "Flood-risk index (FRI)", fontsize=12,
             fontweight="bold", ha="left", va="top")
    fig.text(0.02, 0.925, "Lead: %s  |  Issued: %s%s" % (
        lead_name, args.issue,
        "  |  Valid: %s" % valid_date if valid_date else ""),
        fontsize=8.5, ha="left", va="top")
    ax.axis("off")

    key.axis("off")
    cb = fig.colorbar(im, ax=key, fraction=0.34, pad=0.02, shrink=0.72,
                      aspect=18, location="left")
    cb.set_label("FRI (0-1)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    key.text(0.02, 0.92, "Water", transform=key.transAxes, fontsize=8.5,
             color=WATER_COLOUR, fontweight="bold")
    key.plot([0.02, 0.12], [0.88, 0.88], transform=key.transAxes,
             color=WATER_COLOUR, lw=5, solid_capstyle="butt")

    key.annotate("", xy=(0.72, 0.25), xytext=(0.72, 0.12), xycoords="axes fraction",
                 arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6))
    key.text(0.72, 0.27, "N", transform=key.transAxes, ha="center", va="bottom",
             fontsize=11, fontweight="bold")

    if crs is not None and crs.is_projected:
        fig.canvas.draw()
        pos = ax.get_position()
        n_rows, n_cols = fri.shape
        ipp = min(pos.width * fig.get_size_inches()[0] / n_cols,
                  pos.height * fig.get_size_inches()[1] / n_rows)
        bar_in = (SCALE_BAR_KM * 1000.0 / pixel_m) * ipp
        key_width = key.get_position().width * fig.get_size_inches()[0]
        frac = min(bar_in / key_width, 0.85)
        y = 0.055
        key.plot([0.02, 0.02 + frac], [y, y], transform=key.transAxes,
                 color="black", lw=2.2, solid_capstyle="butt")
        key.text(0.02 + frac / 2, y - 0.025, "%d km" % SCALE_BAR_KM,
                 transform=key.transAxes, ha="center", va="top", fontsize=8)

    out_file = args.out or os.path.join(run_dir,
                                        "fri_lead%d_map.png" % args.lead)
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    fig.savefig(out_file, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_file)


if __name__ == "__main__":
    main()

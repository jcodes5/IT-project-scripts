#!/usr/bin/env python3
"""
Create a publication-quality map of environmental flood risk in five classes.

Inputs:
  data/phase5_results/risk_class.tif
  data/phase2/perm_water.tif (optional)

Output:
  data/phase5_results/environmental_risk_map.png

Run:
  python make_risk_figure.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch


PHASE2_DIR = "data/phase2"
PHASE5_DIR = "data/phase5_results"
RISK_FILE = os.path.join(PHASE5_DIR, "risk_class.tif")
WATER_FILE = os.path.join(PHASE2_DIR, "perm_water.tif")
OUT_FILE = os.path.join(PHASE5_DIR, "environmental_risk_map.png")

DPI = 300
FIG_SIZE = (7.2, 6.3)
MARGIN_PX = 10
SCALE_BAR_KM = 10
WATER_COLOUR = "#08306b"
CLASS_COLOURS = ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]
CLASS_LABELS = ["Very low", "Low", "Moderate", "High", "Very high"]


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 9,
})


def need(path):
    if not os.path.exists(path):
        sys.exit("Missing required file: %s" % path)
    return path


def read_raster(path, nodata=None):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        nd = src.nodata if src.nodata is not None else nodata
        if nd is not None:
            arr[arr == nd] = np.nan
        return arr, src


risk, risk_src = read_raster(need(RISK_FILE), 255.0)
transform = risk_src.transform
crs = risk_src.crs
pixel_m = abs(transform.a)

water = None
if os.path.exists(WATER_FILE):
    water_raw, _ = read_raster(WATER_FILE, 255.0)
    if water_raw.shape == risk.shape:
        water = np.where(water_raw == 1, 1.0, np.nan)
    else:
        print("Permanent-water raster does not match the risk grid; water will not be drawn.")

valid = np.isfinite(risk)
if water is not None:
    valid = valid | np.isfinite(water)

rows = np.where(valid.any(axis=1))[0]
cols = np.where(valid.any(axis=0))[0]
if len(rows) == 0 or len(cols) == 0:
    sys.exit("No valid mapped cells were found in %s" % RISK_FILE)

r0 = max(rows.min() - MARGIN_PX, 0)
r1 = min(rows.max() + MARGIN_PX + 1, risk.shape[0])
c0 = max(cols.min() - MARGIN_PX, 0)
c1 = min(cols.max() + MARGIN_PX + 1, risk.shape[1])
risk = risk[r0:r1, c0:c1]
if water is not None:
    water = water[r0:r1, c0:c1]

n_rows, n_cols = risk.shape

class_cmap = ListedColormap(CLASS_COLOURS)
class_norm = BoundaryNorm(np.arange(0.5, 6.5, 1.0), class_cmap.N)
water_cmap = ListedColormap([WATER_COLOUR])

fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)
gs = fig.add_gridspec(1, 2, left=0.02, right=0.98, top=0.93, bottom=0.06,
                      width_ratios=[5.7, 1.15], wspace=0.03)
ax = fig.add_subplot(gs[0, 0])
key = fig.add_subplot(gs[0, 1])

ax.imshow(risk, cmap=class_cmap, norm=class_norm, interpolation="nearest", aspect="equal")
if water is not None:
    ax.imshow(water, cmap=water_cmap, vmin=0.0, vmax=1.0,
              interpolation="nearest", aspect="equal")

ax.set_title("Environmental flood risk", fontsize=12, loc="left", fontweight="bold")
ax.axis("off")

key.axis("off")
key.set_title("Risk class", fontsize=10, loc="left", fontweight="bold")
handles = [
    Patch(facecolor=CLASS_COLOURS[i], edgecolor="#555555", linewidth=0.5,
          label="%d  %s" % (i + 1, CLASS_LABELS[i]))
    for i in range(5)
]
if water is not None:
    handles.append(Patch(facecolor=WATER_COLOUR, edgecolor="#555555", linewidth=0.5,
                         label="Permanent water"))
key.legend(handles=handles, loc="upper left", frameon=False, fontsize=8,
           handlelength=1.5, borderaxespad=0.0, labelspacing=0.55)

key.annotate("", xy=(0.72, 0.27), xytext=(0.72, 0.11), xycoords="axes fraction",
             arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6))
key.text(0.72, 0.29, "N", transform=key.transAxes, ha="center", va="bottom",
         fontsize=11, fontweight="bold")

if crs is not None and crs.is_projected:
    fig.canvas.draw()
    pos = ax.get_position()
    ipp = min(pos.width * FIG_SIZE[0] / n_cols, pos.height * FIG_SIZE[1] / n_rows)
    bar_in = (SCALE_BAR_KM * 1000.0 / pixel_m) * ipp
    key_width = key.get_position().width * FIG_SIZE[0]
    frac = min(bar_in / key_width, 0.85)
    y = 0.055
    key.plot([0.02, 0.02 + frac], [y, y], transform=key.transAxes,
             color="black", lw=2.2, solid_capstyle="butt")
    key.text(0.02 + frac / 2, y - 0.025, "%d km" % SCALE_BAR_KM,
             transform=key.transAxes, ha="center", va="top", fontsize=8)

os.makedirs(PHASE5_DIR, exist_ok=True)
fig.savefig(OUT_FILE, dpi=DPI, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("Saved", OUT_FILE)

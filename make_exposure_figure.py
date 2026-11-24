#!/usr/bin/env python3
"""
Create a publication-quality map of the Phase 5 exposure index.

Inputs:
  data/phase5_results/exposure_index.tif
  data/phase2/perm_water.tif (optional)
  data/phase2/lga_valid.tif (optional)

Output:
  data/phase5_results/exposure_map.png

Run:
  python make_exposure_figure.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import Patch, Rectangle


PHASE2_DIR = "data/phase2"
PHASE5_DIR = "data/phase5_results"
EXPOSURE_FILE = os.path.join(PHASE5_DIR, "exposure_index.tif")
WATER_FILE = os.path.join(PHASE2_DIR, "perm_water.tif")
LGA_FILE = os.path.join(PHASE2_DIR, "lga_valid.tif")
OUT_FILE = os.path.join(PHASE5_DIR, "exposure_map.png")

DPI = 300
FIG_SIZE = (7.2, 6.3)
MARGIN_PX = 10
SCALE_BAR_KM = 10
WATER_COLOUR = "#08306b"


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


exposure, ex_src = read_raster(need(EXPOSURE_FILE), -9999.0)
transform = ex_src.transform
crs = ex_src.crs
pixel_m = abs(transform.a)

water = None
if os.path.exists(WATER_FILE):
    water_raw, _ = read_raster(WATER_FILE, 255.0)
    if water_raw.shape == exposure.shape:
        water = np.where(water_raw == 1, 1.0, np.nan)
    else:
        print("Permanent-water raster does not match the exposure grid; water will not be drawn.")

lga = None
if os.path.exists(LGA_FILE):
    lga_raw, _ = read_raster(LGA_FILE, 255.0)
    if lga_raw.shape == exposure.shape:
        lga = lga_raw

valid = np.isfinite(exposure)
if water is not None:
    valid = valid | np.isfinite(water)
if lga is not None:
    valid = valid | (lga == 1)

rows = np.where(valid.any(axis=1))[0]
cols = np.where(valid.any(axis=0))[0]
r0 = max(rows.min() - MARGIN_PX, 0)
r1 = min(rows.max() + MARGIN_PX + 1, exposure.shape[0])
c0 = max(cols.min() - MARGIN_PX, 0)
c1 = min(cols.max() + MARGIN_PX + 1, exposure.shape[1])

exposure = exposure[r0:r1, c0:c1]
if water is not None:
    water = water[r0:r1, c0:c1]
if lga is not None:
    lga = lga[r0:r1, c0:c1]

n_rows, n_cols = exposure.shape

fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)
gs = fig.add_gridspec(1, 2, left=0.02, right=0.98, top=0.93, bottom=0.06,
                      width_ratios=[5.7, 1.15], wspace=0.03)
ax = fig.add_subplot(gs[0, 0])
key = fig.add_subplot(gs[0, 1])

im = ax.imshow(exposure, cmap="magma", vmin=0.0, vmax=1.0,
               interpolation="nearest", aspect="equal")

if lga is not None:
    from rasterio.features import shapes
    for geom, value in shapes(lga.astype("uint8"), mask=(lga == 1), transform=transform):
        pass

if water is not None:
    ax.imshow(water, cmap=matplotlib.colors.ListedColormap([WATER_COLOUR]),
              vmin=0.0, vmax=1.0, interpolation="nearest", aspect="equal")

ax.set_title("Exposure index", fontsize=12, loc="left", fontweight="bold")
ax.axis("off")

# Vertical colour legend for the continuous 0-1 exposure index.
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.025, shrink=0.76)
cb.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
cb.set_ticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])
cb.ax.tick_params(labelsize=8)
cb.set_label("Relative exposure (0-1)", fontsize=9, labelpad=8)

# Legend/key panel.
key.axis("off")
key.set_title("Legend", fontsize=10, loc="left", fontweight="bold")
handles = [
    Patch(facecolor="#fcfdbf", edgecolor="#555555", linewidth=0.5, label="Very low"),
    Patch(facecolor="#fc8961", edgecolor="#555555", linewidth=0.5, label="Low"),
    Patch(facecolor="#b73779", edgecolor="#555555", linewidth=0.5, label="Moderate"),
    Patch(facecolor="#57106e", edgecolor="#555555", linewidth=0.5, label="High"),
    Patch(facecolor="#000004", edgecolor="#555555", linewidth=0.5, label="Very high"),
]
if water is not None:
    handles.append(Patch(facecolor=WATER_COLOUR, edgecolor="#555555",
                         linewidth=0.5, label="Permanent water"))
key.legend(handles=handles, loc="upper left", frameon=False, fontsize=8,
           handlelength=1.5, borderaxespad=0.0, labelspacing=0.55)

# North arrow.
key.annotate("", xy=(0.72, 0.27), xytext=(0.72, 0.11), xycoords="axes fraction",
             arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6))
key.text(0.72, 0.29, "N", transform=key.transAxes, ha="center", va="bottom",
         fontsize=11, fontweight="bold")

# Scale bar, measured correctly for a projected UTM raster.
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

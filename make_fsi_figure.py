#!/usr/bin/env python3
"""
Redraw the Phase 3 overview figure for the thesis (Figure 4.2).

Why a new script: the original quick-look figure puts three panels side by side
in one wide image, so the titles become tiny once it is fitted to a Word page.
This script reads the rasters that phase3_susceptibility.py has already written,
so nothing is recomputed, and draws a 2 x 2 layout that fits a portrait page:

    (a) FSI rescaled 0-1        (b) susceptibility classes
    (c) class stability         (d) key: class colours, permanent water,
                                    scale bar and north arrow

It also saves each map panel on its own, so you can use them separately.

Input  (already produced by earlier phases):
    data/phase3/fsi_scaled.tif, fsi_class.tif, class_stability.tif
    data/phase2/perm_water.tif   (optional; drawn in dark blue)
Output (in data/phase3):
    fsi_overview.png             2 x 2 figure, 300 dpi, about 6.3 x 8.4 inches
    fsi_panel_a.png, fsi_panel_b.png, fsi_panel_c.png   single panels, 300 dpi

Install : pip install rasterio numpy matplotlib
Run     : python make_fsi_figure.py
"""
import os
import sys

import numpy as np
import rasterio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# ------------------------------ CONFIG ---------------------------------
PHASE2_DIR = "data/phase2"
PHASE3_DIR = "data/phase3"
OUT_DIR = PHASE3_DIR

N_CLASSES = 5
CLASS_LABELS = ["Very low", "Low", "Moderate", "High", "Very high"]
CLASS_COLOURS = ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]   # same as phase 3
WATER_COLOUR = "#08306b"        # permanent water, as in phase 3
SCALE_BAR_KM = 10               # length of the scale bar
DPI = 300
FIG_W, FIG_H = 6.3, 8.4         # inches: fits a portrait A4 page with 1.5 in and 1 in margins
MARGIN_PX = 10                  # blank border kept around the study area
# -----------------------------------------------------------------------

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]


def read(path, nodata):
    with rasterio.open(path) as src:
        a = src.read(1).astype("float32")
        nd = src.nodata if src.nodata is not None else nodata
        a[a == nd] = np.nan
        info = {"transform": src.transform, "crs": src.crs, "shape": a.shape}
    return a, info


def need(path):
    if not os.path.exists(path):
        sys.exit("Missing %s. Run phase3_susceptibility.py first." % path)
    return path


fsi, info = read(need(os.path.join(PHASE3_DIR, "fsi_scaled.tif")), -9999.0)
cls, _ = read(need(os.path.join(PHASE3_DIR, "fsi_class.tif")), 255.0)
stab, _ = read(need(os.path.join(PHASE3_DIR, "class_stability.tif")), -9999.0)

water = None
wp = os.path.join(PHASE2_DIR, "perm_water.tif")
if os.path.exists(wp):
    w, winfo = read(wp, 255.0)
    if w.shape == fsi.shape:
        water = np.where(w == 1, 1.0, np.nan)
    else:
        print("perm_water.tif has a different grid from the FSI, so water is not drawn.")

# crop to the study area so the panels are not mostly white space
extent = np.isfinite(fsi)
if water is not None:
    extent = extent | np.isfinite(water)
rows = np.where(extent.any(axis=1))[0]
cols = np.where(extent.any(axis=0))[0]
r0, r1 = max(rows[0] - MARGIN_PX, 0), min(rows[-1] + MARGIN_PX, fsi.shape[0] - 1)
c0, c1 = max(cols[0] - MARGIN_PX, 0), min(cols[-1] + MARGIN_PX, fsi.shape[1] - 1)
crop = (slice(r0, r1 + 1), slice(c0, c1 + 1))
fsi, cls, stab = fsi[crop], cls[crop], stab[crop]
if water is not None:
    water = water[crop]
n_rows, n_cols = fsi.shape
pixel_m = abs(info["transform"].a)
projected = bool(info["crs"] is not None and info["crs"].is_projected)

class_cmap = ListedColormap(CLASS_COLOURS[:N_CLASSES])
water_cmap = ListedColormap([WATER_COLOUR])

PANELS = {
    "a": ("(a) Flood susceptibility index, rescaled 0 to 1", fsi,
          dict(cmap="RdYlBu_r", vmin=0, vmax=1), "FSI"),
    "b": ("(b) Susceptibility classes", cls,
          dict(cmap=class_cmap, vmin=1, vmax=N_CLASSES, interpolation="nearest"), None),
    "c": ("(c) Class stability under weight uncertainty", stab,
          dict(cmap="viridis", vmin=0, vmax=1), "Stability"),
}


def draw_map(fig, ax, key, title_size=9, tick_size=7.5):
    title, arr, kw, cbar_label = PANELS[key]
    im = ax.imshow(arr, **kw)
    if water is not None:
        ax.imshow(water, cmap=water_cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=title_size, loc="left", fontweight="bold")
    ax.axis("off")
    if cbar_label:
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, shrink=0.8)
        cb.ax.tick_params(labelsize=tick_size)
        cb.set_label(cbar_label, fontsize=tick_size + 0.5)
    return im


def class_handles():
    h = [Patch(facecolor=CLASS_COLOURS[i], edgecolor="#555555", linewidth=0.5,
               label="%d  %s" % (i + 1, CLASS_LABELS[i])) for i in range(N_CLASSES)]
    if water is not None:
        h.append(Patch(facecolor=WATER_COLOUR, edgecolor="#555555", linewidth=0.5,
                       label="Permanent water (not scored)"))
    return h


def inches_per_pixel(ax):
    """Printed size of one raster pixel in a map panel (the maps use equal aspect)."""
    pos = ax.get_position()
    w_in, h_in = pos.width * FIG_W, pos.height * FIG_H
    return min(w_in / n_cols, h_in / n_rows)


# ----------------------------- 2 x 2 figure ----------------------------
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
gs = fig.add_gridspec(2, 2, left=0.02, right=0.98, top=0.95, bottom=0.03, wspace=0.14, hspace=0.14)
axes = {k: fig.add_subplot(gs[i // 2, i % 2]) for i, k in enumerate("abcd")}
for k in "abc":
    draw_map(fig, axes[k], k)
fig.canvas.draw()                       # lets matplotlib apply the equal aspect before we measure

kd = axes["d"]
kd.axis("off")
kd.set_title("(d) Key", fontsize=9, loc="left", fontweight="bold")
kd.legend(handles=class_handles(), loc="upper left", fontsize=8, frameon=False,
          title="Susceptibility class", title_fontsize=8.5, handlelength=1.6, borderaxespad=0.2)

if projected:                           # scale bar, valid for panels (a) to (c)
    ipp = inches_per_pixel(axes["a"])
    bar_in = (SCALE_BAR_KM * 1000.0 / pixel_m) * ipp
    d_in = kd.get_position().width * FIG_W
    frac = min(bar_in / d_in, 0.9)
    y = 0.16
    kd.plot([0.02, 0.02 + frac], [y, y], transform=kd.transAxes, color="black", lw=2.5,
            solid_capstyle="butt", clip_on=False)
    kd.text(0.02 + frac / 2, y - 0.035, "%d km" % SCALE_BAR_KM, transform=kd.transAxes,
            ha="center", va="top", fontsize=8)
# north arrow
kd.annotate("", xy=(0.88, 0.30), xytext=(0.88, 0.10), xycoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))
kd.text(0.88, 0.32, "N", transform=kd.transAxes, ha="center", va="bottom", fontsize=10, fontweight="bold")

os.makedirs(OUT_DIR, exist_ok=True)
out_main = os.path.join(OUT_DIR, "fsi_overview.png")
fig.savefig(out_main, dpi=DPI, facecolor="white")
plt.close(fig)
print("Saved", out_main)

# --------------------------- single panels -----------------------------
for k in "abc":
    f = plt.figure(figsize=(3.4, 4.6), dpi=DPI)
    a = f.add_axes([0.02, 0.02, 0.96, 0.88])
    draw_map(f, a, k, title_size=9, tick_size=7.5)
    if k == "b":
        a.legend(handles=class_handles(), loc="lower left", fontsize=7, frameon=True,
                 framealpha=0.9, borderaxespad=0.3)
    p = os.path.join(OUT_DIR, "fsi_panel_%s.png" % k)
    f.savefig(p, dpi=DPI, facecolor="white")
    plt.close(f)
    print("Saved", p)
#!/usr/bin/env python3
"""
Draw the eight flood-conditioning factors as one thesis figure (Figure 4.1).

Why a new script: the quick-look figure at the end of phase2_factors.py shows the
STANDARDISED layers (0-1) in one wide 2 x 4 image with the same colour ramp for
every factor, so a reader cannot tell what the values mean. This script reads the
rasters that phase2_factors.py has already written, so nothing is recomputed, and
draws each factor in its own units with its own colour bar:

    (a) elevation (m)              (b) slope (degrees)         (c) mean annual rainfall (mm)
    (d) distance to drainage (m)   (e) flow accumulation       (f) drainage density
        (km2, log scale)               (km per km2)
    (g) land cover (classes)       (h) NDVI                    (i) key: permanent water,
                                                                   land-cover classes,
                                                                   scale bar, north arrow

3 x 3 panels fit a portrait page with each map large enough to read.

Input  (from phase2_factors.py, in data/phase2):
    raw_elevation.tif, raw_slope_deg.tif, raw_rainfall_mm.tif, raw_dist_drainage_m.tif,
    raw_flow_acc_km2.tif, raw_drainage_density_km_km2.tif, raw_ndvi.tif,
    lulc_class_aligned.tif, perm_water.tif (optional)
Output (in data/phase2):
    factors_thesis_overview.png   about 5.8 x 8.8 inches, 300 dpi

Install : pip install rasterio numpy matplotlib
Run     : python make_factors_figure.py
"""
import os
import sys

import numpy as np
import rasterio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ------------------------------ CONFIG ---------------------------------
PHASE2_DIR = "data/phase2"
OUT_FILE = os.path.join(PHASE2_DIR, "factors_thesis_overview.png")
DPI = 300
FIG_W, FIG_H = 5.8, 8.8          # inches: fits the text block of the Word template
MARGIN_PX = 10                   # blank border kept around the study area
SCALE_BAR_KM = 10
WATER_COLOUR = "#08306b"         # permanent water, same colour as the phase 3 figure
CLIP = (1, 99)                   # percentiles used to set the colour range

# Dynamic World class colours and names (the classes phase2_factors.py scores)
DW_COLOURS = {0: "#419BDF", 1: "#397D49", 2: "#88B053", 3: "#7A87C6", 4: "#E49635",
              5: "#DFC35A", 6: "#C4281B", 7: "#A59B8F", 8: "#B39FE1"}
DW_NAMES = {0: "Water", 1: "Trees", 2: "Grass", 3: "Flooded vegetation", 4: "Crops",
            5: "Shrub and scrub", 6: "Built area", 7: "Bare ground", 8: "Snow and ice"}
# -----------------------------------------------------------------------

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]


def need(name):
    p = os.path.join(PHASE2_DIR, name)
    if not os.path.exists(p):
        sys.exit("Missing %s. Run phase2_factors.py first." % p)
    return p


def read(name, nodata=-9999.0):
    with rasterio.open(need(name)) as src:
        a = src.read(1).astype("float32")
        nd = src.nodata if src.nodata is not None else nodata
        a[a == nd] = np.nan
        return a, abs(src.transform.a), src.crs


layers = {}
pixel_m, crs = None, None
for key, fname in [("elev", "raw_elevation.tif"), ("slope", "raw_slope_deg.tif"),
                   ("rain", "raw_rainfall_mm.tif"), ("dist", "raw_dist_drainage_m.tif"),
                   ("acc", "raw_flow_acc_km2.tif"), ("dens", "raw_drainage_density_km_km2.tif"),
                   ("ndvi", "raw_ndvi.tif")]:
    layers[key], pixel_m, crs = read(fname)
lulc, _, _ = read("lulc_class_aligned.tif", 255.0)

water = None
wp = os.path.join(PHASE2_DIR, "perm_water.tif")
if os.path.exists(wp):
    w, _, _ = read("perm_water.tif", 255.0)
    water = np.where(w == 1, 1.0, np.nan)

# crop to the study area
extent = np.isfinite(layers["elev"])
if water is not None:
    extent = extent | np.isfinite(water)
rows = np.where(extent.any(axis=1))[0]
cols = np.where(extent.any(axis=0))[0]
r0, r1 = max(rows[0] - MARGIN_PX, 0), min(rows[-1] + MARGIN_PX, extent.shape[0] - 1)
c0, c1 = max(cols[0] - MARGIN_PX, 0), min(cols[-1] + MARGIN_PX, extent.shape[1] - 1)
crop = (slice(r0, r1 + 1), slice(c0, c1 + 1))
for k in layers:
    layers[k] = layers[k][crop]
lulc = lulc[crop]
if water is not None:
    water = water[crop]
n_rows, n_cols = layers["elev"].shape
projected = bool(crs is not None and crs.is_projected)


def vrange(a, lo=CLIP[0], hi=CLIP[1]):
    v = a[np.isfinite(a)]
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


# log10 of flow accumulation (km2); zeros and tiny values are masked out
acc = layers["acc"].copy()
acc[~(acc > 0)] = np.nan
acc_log = np.log10(acc)

# panel definitions: key, title, array, imshow options, colour-bar label
PANELS = [
    ("elev", "(a) Elevation", layers["elev"], dict(cmap="terrain"), "metres"),
    ("slope", "(b) Slope", layers["slope"], dict(cmap="YlOrBr"), "degrees"),
    ("rain", "(c) Mean annual rainfall", layers["rain"], dict(cmap="Blues"), "millimetres"),
    ("dist", "(d) Distance to drainage", layers["dist"], dict(cmap="viridis_r"), "metres"),
    ("acc", "(e) Flow accumulation", acc_log, dict(cmap="PuBu"), "log10 of km\u00b2"),
    ("dens", "(f) Drainage density", layers["dens"], dict(cmap="Purples"), "km per km\u00b2"),
    ("lulc", "(g) Land cover", lulc, None, None),
    ("ndvi", "(h) NDVI", layers["ndvi"], dict(cmap="RdYlGn", vmin=0.0, vmax=0.9), "NDVI"),
]
water_cmap = ListedColormap([WATER_COLOUR])
present = sorted(int(v) for v in np.unique(lulc[np.isfinite(lulc)]) if int(v) in DW_COLOURS)
lulc_cmap = ListedColormap([DW_COLOURS[k] for k in sorted(DW_COLOURS)])
lulc_norm = BoundaryNorm(np.arange(-0.5, 9.5, 1), lulc_cmap.N)


def draw(ax, key, title, arr, opts, cbar_label):
    if key == "lulc":
        ax.imshow(arr, cmap=lulc_cmap, norm=lulc_norm, interpolation="nearest")
    else:
        kw = dict(opts)
        if "vmin" not in kw:
            kw["vmin"], kw["vmax"] = vrange(arr)
        im = ax.imshow(arr, **kw)
    if water is not None and key != "lulc":
        ax.imshow(water, cmap=water_cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=8, loc="left", fontweight="bold", pad=3)
    ax.axis("off")
    if key != "lulc":
        cax = make_axes_locatable(ax).append_axes("bottom", size="3.5%", pad=0.06)
        cb = plt.colorbar(im, cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=6, length=2)
        cb.set_label(cbar_label, fontsize=6.5, labelpad=1)


fig, axs = plt.subplots(3, 3, figsize=(FIG_W, FIG_H), dpi=DPI)
fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.02, wspace=0.08, hspace=0.20)
flat = axs.ravel()
for ax, (key, title, arr, opts, lab) in zip(flat[:8], PANELS):
    draw(ax, key, title, arr, opts, lab)

# key panel
kd = flat[8]
kd.axis("off")
kd.set_title("(i) Key", fontsize=8, loc="left", fontweight="bold", pad=3)
handles = [Patch(facecolor=DW_COLOURS[k], edgecolor="#555555", linewidth=0.4, label=DW_NAMES[k]) for k in present]
if water is not None:
    handles.append(Patch(facecolor=WATER_COLOUR, edgecolor="#555555", linewidth=0.4,
                         label="Permanent water (panels a to f and h)"))
kd.legend(handles=handles, loc="upper left", fontsize=6.3, frameon=False,
          title="Land cover (panel g)", title_fontsize=6.8, handlelength=1.3, borderaxespad=0.0,
          labelspacing=0.35)

fig.canvas.draw()                                 # applies the equal aspect before we measure
if projected:
    pos = flat[0].get_position()
    ipp = min(pos.width * FIG_W / n_cols, pos.height * FIG_H / n_rows)   # inches per pixel
    bar_in = (SCALE_BAR_KM * 1000.0 / pixel_m) * ipp
    d_in = kd.get_position().width * FIG_W
    frac = min(bar_in / d_in, 0.9)
    y = 0.07
    kd.plot([0.02, 0.02 + frac], [y, y], transform=kd.transAxes, color="black", lw=2, solid_capstyle="butt", clip_on=False)
    kd.text(0.02 + frac / 2, y - 0.03, "%d km" % SCALE_BAR_KM, transform=kd.transAxes, ha="center", va="top", fontsize=6.5)
kd.annotate("", xy=(0.90, 0.19), xytext=(0.90, 0.05), xycoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2))
kd.text(0.90, 0.20, "N", transform=kd.transAxes, ha="center", va="bottom", fontsize=8, fontweight="bold")

os.makedirs(PHASE2_DIR, exist_ok=True)
fig.savefig(OUT_FILE, dpi=DPI, facecolor="white")
plt.close(fig)
print("Saved", OUT_FILE)
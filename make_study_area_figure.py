"""
Generate a high-resolution study-area map for the Lokoja flood thesis.

Outputs:
  data/phase2/study_area.png

The main map uses the Copernicus DEM, Lokoja LGA mask, permanent water,
and derived stream network. Optional GeoPackage/GeoJSON/Shapefile paths can
provide Nigeria, Kogi State, LGA and major-road context layers.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import FancyArrow, Rectangle
from rasterio.features import shapes
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False


# ---------------------------- CONFIGURATION ----------------------------
DATA_PHASE1 = "data/phase1/Lokoja_flood_phase1"
DATA_PHASE2 = "data/phase2"
OUTPUT = os.path.join(DATA_PHASE2, "study_area.png")

DEM_FILE = os.path.join(DATA_PHASE1, "dem_cop30_30m_buf.tif")
LGA_MASK_FILE = os.path.join(DATA_PHASE1, "lga_mask_buf.tif")
WATER_FILE = os.path.join(DATA_PHASE2, "perm_water.tif")
STREAM_FILE = os.path.join(DATA_PHASE2, "streams.tif")

# Optional context vectors. Set these only if the boundary files are available.
NIGERIA_BOUNDARY = None
KOGI_BOUNDARY = None
LOKOJA_BOUNDARY = None
ROADS_FILE = None

DPI = 600
FIG_SIZE = (8.3, 7.2)
MARGIN_PX = 12
MAP_CRS = "EPSG:32632"


# ------------------------------- HELPERS -------------------------------
def need(path):
    if not os.path.exists(path):
        sys.exit("Missing required file: %s" % path)
    return path


def read_raster(path):
    with rasterio.open(need(path)) as src:
        array = src.read(1).astype("float32")
        nodata = src.nodata
        if nodata is not None:
            array[array == nodata] = np.nan
        return array, src.transform, src.crs, src.bounds


def hillshade(dem, azimuth=315.0, altitude=45.0):
    """Create a shaded-relief image from a 30 m DEM."""
    dem_filled = dem.copy()
    finite = np.isfinite(dem_filled)
    if not finite.all():
        ys, xs = np.where(finite)
        from scipy.interpolate import LinearNDInterpolator
        points = np.column_stack((xs, ys))
        values = dem_filled[ys, xs]
        interpolator = LinearNDInterpolator(points, values)
        yy, xx = np.indices(dem.shape)
        dem_filled = interpolator(xx, yy)

    pixel = 30.0
    dzdx = (np.roll(dem_filled, -1, axis=1) - np.roll(dem_filled, 1, axis=1)) / (2 * pixel)
    dzdy = (np.roll(dem_filled, -1, axis=0) - np.roll(dem_filled, 1, axis=0)) / (2 * pixel)

    az = np.radians(360.0 - azimuth)
    alt = np.radians(altitude)
    shaded = np.sin(alt) + np.cos(alt) * (
        np.cos(np.arctan(np.hypot(dzdx, dzdy)))
        * np.cos(az) * dzdx + np.cos(az - np.pi / 2) * dzdy
    )
    return np.clip(shaded, 0, 1)


def mask_polygon(mask, transform, value=1):
    """Convert raster boundary pixels into Shapely-like Matplotlib polygons."""
    return [
        (np.array(poly["coordinates"])[0, :, 0],
         np.array(poly["coordinates"])[0, :, 1])
        for geom, val in shapes(mask.astype("uint8"), mask=(mask == value), transform=transform)
        if val == value
        for poly in [geom]
    ]


def plot_mask_boundary(ax, mask, transform, edgecolor="#000000", linewidth=1.1):
    for xs, ys in mask_polygon(mask, transform, value=1):
        ax.plot(xs, ys, color=edgecolor, linewidth=linewidth, zorder=8)


def add_north_arrow(ax, x=0.95, y=0.08):
    ax.add_patch(FancyArrow(
        x, y - 0.06, 0, 0.055,
        width=0.006, head_width=0.026, head_length=0.018,
        length_includes_head=True, transform=ax.transAxes,
        facecolor="black", edgecolor="black", zorder=20
    ))
    ax.text(x, y + 0.01, "N", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=10, fontweight="bold", zorder=20)


def add_scale_bar(ax, length_km=10):
    """Draw a 10 km scale bar along the projected UTM x-axis."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    bar_m = length_km * 1000
    bx = x0 + 0.06 * (x1 - x0)
    by = y0 + 0.055 * (y1 - y0)
    ax.plot([bx, bx + bar_m], [by, by], color="black", linewidth=3.0, zorder=20)
    ax.plot([bx, bx], [by - 0.001 * (y1 - y0), by + 0.001 * (y1 - y0)],
            color="black", linewidth=1.2, zorder=20)
    ax.plot([bx + bar_m, bx + bar_m], [by - 0.001 * (y1 - y0), by + 0.001 * (y1 - y0)],
            color="black", linewidth=1.2, zorder=20)
    ax.text(bx + bar_m / 2, by - 0.006 * (y1 - y0), "%d km" % length_km,
            ha="center", va="top", fontsize=8, fontweight="bold", zorder=20,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))


def read_optional_vector(path, target_crs):
    if not path or not HAS_GEOPANDAS:
        return None
    if not os.path.exists(path):
        print("Optional vector file not found: %s" % path)
        return None
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(target_crs)
    else:
        gdf = gdf.to_crs(target_crs)
    return gdf


# ----------------------------- MAIN SCRIPT -----------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 9,
})

lga, lga_transform, lga_crs, lga_bounds = read_raster(LGA_MASK_FILE)
dem, dem_transform, dem_crs, dem_bounds = read_raster(DEM_FILE)
water, water_transform, water_crs, water_bounds = read_raster(WATER_FILE)
streams, stream_transform, stream_crs, stream_bounds = read_raster(STREAM_FILE)

if lga.shape != dem.shape:
    sys.exit("DEM and LGA mask have different shapes. Run the Phase 2 grid script first.")

# Mask the DEM outside Lokoja LGA.
dem_lga = np.where(lga == 1, dem, np.nan)
shade = hillshade(dem_lga)

# Map extent, cropped to the LGA with a modest buffer.
ys, xs = np.where(lga == 1)
r0, r1 = max(ys.min() - MARGIN_PX, 0), min(ys.max() + MARGIN_PX + 1, lga.shape[0])
c0, c1 = max(xs.min() - MARGIN_PX, 0), min(xs.max() + MARGIN_PX + 1, lga.shape[1])
extent_bounds = array_bounds(r1 - r0, c1 - c0, rasterio.Affine(
    lga_transform.a,
    lga_transform.b,
    lga_transform.c,
    lga_transform.d,
    lga_transform.e,
    lga_transform.f
) * rasterio.Affine.translation(c0, r0))
left, bottom, right, top = extent_bounds

fig = plt.figure(figsize=FIG_SIZE)
grid = fig.add_gridspec(1, 6, left=0.07, right=0.98, top=0.92, bottom=0.07,
                        wspace=0.05)
ax_main = fig.add_subplot(grid[:, 0:5])
ax_context = fig.add_subplot(grid[:, 5])

# ---------------------------- Main study area ----------------------------
dem_for_plot = dem_lga[r0:r1, c0:c1]
shade_for_plot = shade[r0:r1, c0:c1]

# Elevation with subtle relief.
im = ax_main.imshow(dem_for_plot, cmap="gist_earth_r", alpha=0.88,
                    vmin=np.nanpercentile(dem_for_plot, 2),
                    vmax=np.nanpercentile(dem_for_plot, 98),
                    extent=(left, right, bottom, top), origin="upper",
                    zorder=1)
ax_main.imshow(shade_for_plot, cmap="gray", alpha=0.18,
               extent=(left, right, bottom, top), origin="upper", zorder=2)

# Permanent water and stream network.
water_crop = np.where(water[r0:r1, c0:c1] == 1, 1, np.nan)
stream_crop = np.where(streams[r0:r1, c0:c1] == 1, 1, np.nan)
ax_main.imshow(water_crop, cmap=plt.matplotlib.colors.ListedColormap(["#2463b3"]),
               alpha=0.95, extent=(left, right, bottom, top), origin="upper", zorder=4)
ax_main.imshow(stream_crop, cmap=plt.matplotlib.colors.ListedColormap(["#56b4e9"]),
               alpha=0.75, extent=(left, right, bottom, top), origin="upper", zorder=5)

plot_mask_boundary(ax_main, lga[r0:r1, c0:c1],
                   lga_transform * rasterio.Affine.translation(c0, r0),
                   edgecolor="#202020", linewidth=1.25)

# Optional roads.
roads = read_optional_vector(ROADS_FILE, str(lga_crs))
if roads is not None:
    try:
        roads_clip = gpd.clip(roads, gpd.GeoSeries.from_wkt([]) if False else roads)
        roads_clip.plot(ax=ax_main, color="#d3d3d3", linewidth=0.35,
                        alpha=0.6, zorder=6)
    except Exception as exc:
        print("Road overlay skipped:", exc)

ax_main.set_aspect("equal")
ax_main.set_xlim(left, right)
ax_main.set_ylim(bottom, top)
ax_main.set_xlabel("Easting (m, UTM Zone 32N)")
ax_main.set_ylabel("Northing (m, UTM Zone 32N)")
ax_main.set_title("(a) Study Area: Lokoja LGA, Kogi State, Nigeria", pad=8,
                  loc="left", fontsize=11, fontweight="bold")
ax_main.grid(True, linestyle="--", linewidth=0.35, color="white", alpha=0.55, zorder=7)

add_north_arrow(ax_main)
add_scale_bar(ax_main, 10)

cb = fig.colorbar(im, ax=ax_main, fraction=0.033, pad=0.018)
cb.set_label("Elevation (m above mean sea level)", fontsize=8)
cb.ax.tick_params(labelsize=7)

# ------------------------------- Locator ---------------------------------
# This inset shows the position of the LGA within the buffered project grid.
# It is intentionally not a substitute for an official state or national boundary.
ctx_lga = lga
ctx_transform = lga_transform
ctx_bounds = (lga_transform.c, lga_transform.f + lga.shape[0] * lga_transform.e,
              lga_transform.c + lga.shape[1] * lga_transform.a, lga_transform.f)
ax_context.imshow(np.where(ctx_lga == 1, 1, np.nan),
                  cmap=plt.matplotlib.colors.ListedColormap(["#c62828"]),
                  extent=ctx_bounds, origin="upper", zorder=3)
ax_context.imshow(np.isfinite(dem).astype(float), cmap="Greys",
                  alpha=0.12, extent=(dem_bounds.left, dem_bounds.right,
                                      dem_bounds.bottom, dem_bounds.top),
                  origin="upper", zorder=1)

plot_mask_boundary(ax_context, ctx_lga, ctx_transform,
                   edgecolor="#7f0000", linewidth=0.8)

for name, path in [("Nigeria", NIGERIA_BOUNDARY), ("Kogi", KOGI_BOUNDARY),
                   ("Lokoja", LOKOJA_BOUNDARY)]:
    gdf = read_optional_vector(path, str(lga_crs))
    if gdf is not None:
        gdf.boundary.plot(ax=ax_context, color="#111111" if name != "Lokoja" else "#7f0000",
                          linewidth=0.8 if name != "Lokoja" else 1.3,
                          zorder=5)

ax_context.set_title("(b) Location Context", loc="left", fontsize=9, fontweight="bold")
ax_context.text(0.02, 0.05,
                "LOKOJA LGA\nKogi State\nNorth-central Nigeria\n7.8° N, 6.7° E",
                transform=ax_context.transAxes, fontsize=7,
                va="bottom", ha="left", zorder=20,
                bbox=dict(facecolor="white", edgecolor="#777777", alpha=0.9, boxstyle="round,pad=0.35"))
ax_context.set_aspect("equal")
ax_context.axis("off")

os.makedirs(DATA_PHASE2, exist_ok=True)
fig.savefig(OUTPUT, dpi=DPI, bbox_inches="tight", facecolor="white")
plt.close(fig)

wgs84_bounds = transform_bounds(lga_crs, "EPSG:4326", *lga_bounds)
print("Saved study-area map: %s" % OUTPUT)
print("Lokoja mask WGS84 bounds: west %.6f, south %.6f, east %.6f, north %.6f" % wgs84_bounds)

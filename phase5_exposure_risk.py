#!/usr/bin/env python3
"""
PHASE 5b: Exposure index and environmental flood-risk index.
Lokoja LGA flood susceptibility project.

Risk = flood susceptibility index (FSI, Phase 3) x exposure index (EXI, this script).

Exposure layers (each optional; missing files are skipped and the weights renormalised):
  population  people per km2     (GRID3 or WorldPop raster)
  buildings   buildings per km2  (Open Buildings centroids from phase5a_buildings.js)
  roads       km of road per km2 (GRID3 or OpenStreetMap lines)
  facilities  critical facilities per km2 (health facilities, schools: points or polygons)

Each layer is turned into a density within a circular window (RADIUS_M), standardised to 0-1
(log scale by default because densities are very skewed), and combined with EXPOSURE_WEIGHTS.

Input  : data/phase5/   building_centroids.csv, population.tif, roads.gpkg, health_facilities.gpkg, schools.gpkg
         data/phase2/lga_valid.tif, data/phase3/fsi_scaled.tif, fsi_class.tif
         data/phase4_results/flood_aligned_*.tif (optional, from phase4_validation.py)
Output : data/phase5_results/
           exposure_index.tif, risk_index.tif, risk_class.tif, std_exp_<layer>.tif, *_per_cell.tif
           exposure_by_susceptibility_class.csv, exposure_by_risk_class.csv, exposure_in_observed_floods.csv
           phase5_log.json, risk_overview.png

Install : pip install rasterio scipy numpy pyproj shapely geopandas pyogrio matplotlib
Run     : python phase5_exposure_risk.py
"""
import json
import math
import os
import sys

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.signal import fftconvolve

# ------------------------------ CONFIG ---------------------------------
PHASE2_DIR = "data/phase2"
PHASE3_DIR = "data/phase3"
PHASE4_DIR = "data/phase4_results"
DATA_DIR = "data/phase5"
OUT_DIR = "data/phase5_results"

BUILDINGS_FILE = "building_centroids.csv"     # columns: lon, lat, area_m2, confidence
BUILDING_CONF_MIN = 0.70
POPULATION_FILE = "NGA_population_v3_0_gridded.tif"
ROADS_FILE = "roads.gpkg"                      # any vector format geopandas can read
FACILITY_FILES = {"health": "health_facilities.gpkg"}

# Weights of the exposure layers. EDIT to match your methodology. Equal weights are only a placeholder.
EXPOSURE_WEIGHTS = {"population": 0.540, "buildings": 0.297, "roads": 0.163, "facilities": 0.0}

RADIUS_M = 500                    # circular window for densities
LOG_TRANSFORM = True              # log1p before scaling (recommended for skewed densities)
CLIP_PCT = (1, 99)                # trimming before min-max scaling
N_CLASSES = 5
CLASS_METHOD = "quantile"   # "equal_interval" or "quantile" for risk_class.tif
CLASS_LABELS = ["Very low", "Low", "Moderate", "High", "Very high"]
ROAD_SPACING_M = 10               # roads are sampled every 10 m to estimate length per cell
# -----------------------------------------------------------------------

NODATA_F = -9999.0
NODATA_U = 255


def log(msg=""):
    print(msg, flush=True)


def need(p):
    if not os.path.exists(p):
        sys.exit("Missing file: %s" % p)
    return p


os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------- master grid -----------------------------
with rasterio.open(need(os.path.join(PHASE3_DIR, "fsi_scaled.tif"))) as src:
    PROFILE = src.profile.copy()
    TRANSFORM, CRS = src.transform, src.crs
    SHAPE = (src.height, src.width)
    fsi = src.read(1).astype("float64")
    fsi[fsi == src.nodata] = np.nan
with rasterio.open(need(os.path.join(PHASE3_DIR, "fsi_class.tif"))) as src:
    susc_class = src.read(1).astype("int16")
    susc_class[susc_class == NODATA_U] = 0
with rasterio.open(need(os.path.join(PHASE2_DIR, "lga_valid.tif"))) as src:
    valid = src.read(1) == 1
    if (src.height, src.width) != SHAPE or src.transform != TRANSFORM:
        sys.exit("Grid mismatch: Phase 2 lga_valid.tif and Phase 3 fsi_scaled.tif are on different grids. "
                 "Rerun Phase 3 (and Phase 4) after the latest Phase 2 run.")
if susc_class.shape != SHAPE:
    sys.exit("Grid mismatch: Phase 3 fsi_class.tif does not match fsi_scaled.tif. Rerun Phase 3.")
CELL = abs(TRANSFORM.a)
CELL_KM2 = (CELL / 1000.0) ** 2
scored = np.isfinite(fsi)
log("Grid %d x %d cells at %.0f m | LGA cells %d (%.0f km2) | scored land cells %d (%.0f km2)" % (
    SHAPE[1], SHAPE[0], CELL, valid.sum(), valid.sum() * CELL_KM2, scored.sum(), scored.sum() * CELL_KM2))


def write(name, arr, nodata=NODATA_F, dtype="float32"):
    p = PROFILE.copy()
    p.update(dtype=dtype, count=1, nodata=nodata, compress="deflate")
    out = arr.copy()
    if np.issubdtype(out.dtype, np.floating):
        out = np.where(np.isfinite(out), out, nodata)
    with rasterio.open(os.path.join(OUT_DIR, name), "w", **p) as dst:
        dst.write(out.astype(dtype), 1)


def cells_with_mask(x, y):
    """Map projected coordinates to (row, col, inside) on the master grid; `inside` flags points on the grid."""
    col = np.floor((x - TRANSFORM.c) / TRANSFORM.a).astype(int)
    row = np.floor((y - TRANSFORM.f) / TRANSFORM.e).astype(int)
    ok = (row >= 0) & (row < SHAPE[0]) & (col >= 0) & (col < SHAPE[1])
    return row, col, ok


def to_cells(x, y):
    """Map projected coordinates to (row, col) on the master grid; returns only points inside it."""
    row, col, ok = cells_with_mask(x, y)
    return row[ok], col[ok]


def count_grid(x, y):
    """Count points per master-grid cell (points outside the grid are ignored)."""
    r, c = to_cells(x, y)
    out = np.zeros(SHAPE, dtype="float64")
    np.add.at(out, (r, c), 1.0)
    return out


def read_vector(path):
    import geopandas as gpd
    g = gpd.read_file(path)
    if g.crs is None:
        sys.exit("%s has no CRS defined; set one in QGIS first." % path)
    return g.to_crs(CRS)


# --------------------------- exposure layers ---------------------------
counts = {}        # layer name -> per-cell totals (people, buildings, km of road, facilities)
extra_counts = {}  # facility counts by type, for the tallies

# buildings
bpath = os.path.join(DATA_DIR, BUILDINGS_FILE)
if os.path.exists(bpath):
    import csv
    from pyproj import Transformer
    lon, lat, conf = [], [], []
    with open(bpath, newline="") as f:
        for r_ in csv.DictReader(f):
            try:
                lon.append(float(r_["lon"]))
                lat.append(float(r_["lat"]))
                conf.append(float(r_.get("confidence", 1.0)))
            except (KeyError, ValueError):
                continue
    lon, lat, conf = np.array(lon), np.array(lat), np.array(conf)
    keep = conf >= BUILDING_CONF_MIN
    tr = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    bx, by = tr.transform(lon[keep], lat[keep])
    counts["buildings"] = count_grid(np.asarray(bx), np.asarray(by))
    log("Buildings: %d read, %d kept (confidence >= %.2f), %d on the grid" % (
        len(lon), keep.sum(), BUILDING_CONF_MIN, int(counts["buildings"].sum())))
else:
    log("Buildings: %s not found, layer skipped." % bpath)

# population (people per cell, preserving totals by area scaling)
ppath = os.path.join(DATA_DIR, POPULATION_FILE)
if os.path.exists(ppath):
    with rasterio.open(ppath) as src:
        pop = np.zeros(SHAPE, dtype="float32")
        reproject(source=rasterio.band(src, 1), destination=pop,
                  src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
                  dst_transform=TRANSFORM, dst_crs=CRS, dst_nodata=0.0,
                  resampling=Resampling.nearest)
        if src.crs.is_projected:
            src_area = abs(src.transform.a * src.transform.e)
        else:
            lat0 = 0.5 * (src.bounds.top + src.bounds.bottom)
            src_area = (abs(src.transform.a) * 111320.0 * math.cos(math.radians(lat0))) * (abs(src.transform.e) * 110574.0)
    pop = np.clip(pop, 0, None).astype("float64") * (CELL ** 2 / src_area)
    counts["population"] = np.where(valid, pop, 0.0)
    log("Population: %.0f people on the grid inside the LGA (source cell area about %.0f m2)" % (
        counts["population"].sum(), src_area))
else:
    log("Population: %s not found, layer skipped." % ppath)

# roads (km per cell, estimated by sampling every ROAD_SPACING_M metres)
rpath = os.path.join(DATA_DIR, ROADS_FILE)
if os.path.exists(rpath):
    import shapely
    roads = read_vector(rpath)
    geoms = roads.geometry[~roads.geometry.is_empty & roads.geometry.notna()].values
    # Split long segments so no piece is longer than ROAD_SPACING_M, explode multi-part lines,
    # then give each piece's REAL length (km) to the cell holding its midpoint. (Counting vertices
    # instead would overstate road length wherever the lines are digitised with closely spaced vertices.)
    seg = shapely.get_parts(shapely.segmentize(geoms, ROAD_SPACING_M))
    xy, gid = shapely.get_coordinates(seg, return_index=True)
    same = gid[1:] == gid[:-1]
    x0, y0, x1, y1 = xy[:-1, 0][same], xy[:-1, 1][same], xy[1:, 0][same], xy[1:, 1][same]
    seg_km = np.hypot(x1 - x0, y1 - y0) / 1000.0
    rr, cc, inside = cells_with_mask((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    roadgrid = np.zeros(SHAPE, dtype="float64")
    np.add.at(roadgrid, (rr[inside], cc[inside]), seg_km[inside])
    counts["roads"] = roadgrid
    log("Roads: %d features, about %.0f km on the grid (true segment lengths)" % (len(geoms), counts["roads"].sum()))
else:
    log("Roads: %s not found, layer skipped." % rpath)

# facilities
fac_total = np.zeros(SHAPE, dtype="float64")
fac_found = False
for name, fname in FACILITY_FILES.items():
    fpath = os.path.join(DATA_DIR, fname)
    if not os.path.exists(fpath):
        log("Facilities (%s): %s not found, skipped." % (name, fpath))
        continue
    import shapely
    g = read_vector(fpath)
    pts = shapely.get_coordinates(shapely.point_on_surface(g.geometry[g.geometry.notna()].values))
    cnt = count_grid(pts[:, 0], pts[:, 1])
    extra_counts[name] = cnt
    fac_total += cnt
    fac_found = True
    log("Facilities (%s): %d features, %d on the grid" % (name, len(g), int(cnt.sum())))
if fac_found:
    counts["facilities"] = fac_total

if not counts:
    sys.exit("No exposure data found in %s. Add at least one layer." % DATA_DIR)

# ------------------------ densities and standardisation -----------------
r = int(math.ceil(RADIUS_M / CELL))
yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
kernel = ((xx ** 2 + yy ** 2) <= r ** 2).astype("float32")
window_area = fftconvolve(valid.astype("float32"), kernel, mode="same") * CELL_KM2


def density(cnt):
    s = fftconvolve((cnt * valid).astype("float32"), kernel, mode="same")
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(window_area > 1e-6, s / window_area, np.nan)
    d = np.clip(d, 0, None)
    d[~valid] = np.nan
    return d


def standardise(d):
    x = np.log1p(d) if LOG_TRANSFORM else d
    ok = scored & np.isfinite(x)
    lo, hi = np.percentile(x[ok], CLIP_PCT)
    if hi <= lo:
        hi = lo + 1e-9
    z = np.clip((x - lo) / (hi - lo), 0, 1)
    z[~ok] = np.nan
    return z.astype("float32"), float(lo), float(hi)


units = {"population": "people/km2", "buildings": "buildings/km2", "roads": "km/km2", "facilities": "facilities/km2"}
std_exp, bounds = {}, {}
log("\n-- Exposure layers (density within %d m) --" % RADIUS_M)
for name, cnt in counts.items():
    d = density(cnt)
    write("%s_per_cell.tif" % name, np.where(valid, cnt, np.nan))
    write("density_%s.tif" % name, d)
    z, lo, hi = standardise(d)
    std_exp[name] = z
    bounds[name] = (lo, hi)
    dv = d[scored & np.isfinite(d)]
    log("%-11s density (%s): mean %.2f, 95th pct %.2f, max %.2f" % (name, units[name], dv.mean(), np.percentile(dv, 95), dv.max()))
    log("%-11s %.1f%% of scored cells have density > 0; scaling bounds (log) %.3f to %.3f" % (
        "", 100.0 * np.mean(dv > 1e-6), lo, hi))
    if np.mean(dv > 1e-6) < 0.05:
        log("%-11s WARNING: very sparse layer. The 1-99 percent clip may turn it into near 0/1 'near a feature' values." % "")
    write("std_exp_%s.tif" % name, z)

w = {k: EXPOSURE_WEIGHTS.get(k, 0.0) for k in std_exp}
tot = sum(w.values())
if tot <= 0:
    sys.exit("EXPOSURE_WEIGHTS has no weight for the layers found: %s" % list(std_exp))
w = {k: v / tot for k, v in w.items()}
log("Exposure weights used (renormalised): " + ", ".join("%s %.3f" % (k, v) for k, v in w.items()))
exi = np.zeros(SHAPE)
for k, z in std_exp.items():
    exi += w[k] * np.nan_to_num(z, nan=0.0)
exi[~scored] = np.nan
risk = fsi * exi
write("exposure_index.tif", exi)
write("risk_index.tif", risk)


# ------------------------------ risk classes ----------------------------
def classes(vals, method):
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    s = (vals - lo) / (hi - lo) if hi > lo else np.zeros_like(vals)
    if method == "quantile":
        edges = np.quantile(s, np.linspace(0, 1, N_CLASSES + 1))[1:-1]
    else:
        edges = np.linspace(0, 1, N_CLASSES + 1)[1:-1]
    return (np.digitize(s, edges) + 1).astype("uint8")


rv = risk[scored]
log("\n-- Risk index (FSI x EXI): mean %.3f, 95th pct %.3f, max %.3f --" % (rv.mean(), np.percentile(rv, 95), rv.max()))
for m in ("equal_interval", "quantile"):
    c = classes(rv, m)
    shares = ", ".join("%d: %.1f%%" % (k, 100.0 * np.mean(c == k)) for k in range(1, N_CLASSES + 1))
    log("Class shares by %-14s %s%s" % (m, shares, "   <- used for risk_class.tif" if m == CLASS_METHOD else ""))
risk_cls = np.zeros(SHAPE, dtype="uint8")
risk_cls[scored] = classes(rv, CLASS_METHOD)
write("risk_class.tif", np.where(scored, risk_cls, NODATA_U), nodata=NODATA_U, dtype="uint8")

# ------------------------------- tallies --------------------------------
people = counts.get("population")
builds = counts.get("buildings")
roadkm = counts.get("roads")


def tally(zone_defs, filename, title):
    header = ["zone", "area_km2", "area_percent"]
    if people is not None:
        header += ["people", "people_percent"]
    if builds is not None:
        header += ["buildings", "buildings_percent"]
    if roadkm is not None:
        header += ["road_km", "road_percent"]
    for k in extra_counts:
        header += ["facilities_" + k]
    rows = [",".join(header)]
    totals = {"area": valid.sum() * CELL_KM2,
              "people": people[valid].sum() if people is not None else 0,
              "buildings": builds[valid].sum() if builds is not None else 0,
              "road": roadkm[valid].sum() if roadkm is not None else 0}
    log("\n-- %s --" % title)
    log("%-22s %-10s %-12s %-12s %-10s" % ("zone", "area km2", "people", "buildings", "road km"))
    for label, mask in zone_defs:
        m = mask & valid
        area = m.sum() * CELL_KM2
        vals = [label, "%.2f" % area, "%.2f" % (100.0 * area / totals["area"])]
        line = [label, "%.1f" % area]
        if people is not None:
            p = people[m].sum()
            vals += ["%.0f" % p, "%.2f" % (100.0 * p / max(totals["people"], 1e-9))]
            line.append("%.0f (%.1f%%)" % (p, 100.0 * p / max(totals["people"], 1e-9)))
        else:
            line.append("-")
        if builds is not None:
            b = builds[m].sum()
            vals += ["%.0f" % b, "%.2f" % (100.0 * b / max(totals["buildings"], 1e-9))]
            line.append("%.0f (%.1f%%)" % (b, 100.0 * b / max(totals["buildings"], 1e-9)))
        else:
            line.append("-")
        if roadkm is not None:
            rk = roadkm[m].sum()
            vals += ["%.1f" % rk, "%.2f" % (100.0 * rk / max(totals["road"], 1e-9))]
            line.append("%.0f (%.1f%%)" % (rk, 100.0 * rk / max(totals["road"], 1e-9)))
        else:
            line.append("-")
        for k, cnt in extra_counts.items():
            vals.append("%.0f" % cnt[m].sum())
        rows.append(",".join(vals))
        log("%-22s %-10s %-12s %-12s %-10s" % tuple(line[:1] + line[1:2] + line[2:5]))
    with open(os.path.join(OUT_DIR, filename), "w") as f:
        f.write("\n".join(rows) + "\n")


zones = [("%d %s" % (k, CLASS_LABELS[k - 1] if N_CLASSES == len(CLASS_LABELS) else k), susc_class == k)
         for k in range(1, N_CLASSES + 1)]
zones.append(("permanent water / unscored", valid & ~scored))
tally(zones, "exposure_by_susceptibility_class.csv", "Exposure by flood susceptibility class (Phase 3)")

zones = [("%d %s" % (k, CLASS_LABELS[k - 1] if N_CLASSES == len(CLASS_LABELS) else k), (risk_cls == k) & scored)
         for k in range(1, N_CLASSES + 1)]
zones.append(("permanent water / unscored", valid & ~scored))
tally(zones, "exposure_by_risk_class.csv", "Exposure by environmental risk class (%s breaks)" % CLASS_METHOD)

flood_zones = []
fl_any, ob_all = np.zeros(SHAPE, bool), np.ones(SHAPE, bool)
found = 0
for ev in ("2018", "2022"):
    p = os.path.join(PHASE4_DIR, "flood_aligned_%s.tif" % ev)
    if os.path.exists(p):
        with rasterio.open(p) as src:
            a = src.read(1)
        if a.shape != SHAPE:
            log("flood_aligned_%s.tif is on a different grid (%s vs %s); skipped. Rerun Phase 4." % (ev, a.shape, SHAPE))
            continue
        flood_zones.append(("observed flood %s" % ev, a == 1))
        fl_any |= a == 1
        ob_all &= a != NODATA_U
        found += 1
if found:
    if found > 1:
        flood_zones.append(("observed flood either", fl_any & ob_all))
    flood_zones.append(("whole LGA", valid))
    tally(flood_zones, "exposure_in_observed_floods.csv",
          "Exposure inside the observed flood extents (Sentinel-1; flooded on gentle ground only)")
else:
    log("\nNo flood_aligned_*.tif found in %s; observed-flood tally skipped." % PHASE4_DIR)

# ------------------------------ quick look ------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (arr, title, cmap) in zip(axes, [(fsi, "Flood susceptibility (FSI)", "RdYlBu_r"),
                                            (exi, "Exposure index (EXI)", "magma"),
                                            (risk, "Risk = FSI x EXI", "inferno")]):
        im = ax.imshow(arr, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.savefig(os.path.join(OUT_DIR, "risk_overview.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
except Exception as e:
    log("Quick-look figure skipped: %s" % e)

with open(os.path.join(OUT_DIR, "phase5_log.json"), "w") as f:
    json.dump({"radius_m": RADIUS_M, "log_transform": LOG_TRANSFORM, "clip_pct": CLIP_PCT,
               "weights": w, "layers": list(std_exp), "class_method": CLASS_METHOD,
               "scaling_bounds": bounds, "building_conf_min": BUILDING_CONF_MIN}, f, indent=2)
log("\nReading guide:")
log(" - Exposure weights are a placeholder until you set them from your methodology; the risk map depends on them.")
log(" - Risk is relative (0-1 within this LGA). It ranks places; it is not an expected loss.")
log(" - Population and buildings come from modelled datasets. Check totals against what you know of Lokoja.")
log(" - Radar misses flooding among buildings, so exposure counted inside observed flood extents is a minimum.")
log("\nDone. Outputs are in %s" % OUT_DIR)
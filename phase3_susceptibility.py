#!/usr/bin/env python3
"""
PHASE 3: AHP weights, flood susceptibility index (FSI), classes and sensitivity.
Lokoja LGA flood susceptibility project.

Input  : std_*.tif (eight factors scaled 0-1, 1 = more susceptible) from phase2_factors.py,
         plus perm_water.tif (optional, for the map).
Output : in OUT_DIR
           fsi.tif              weighted sum of the eight factors (0-1)
           fsi_scaled.tif       FSI rescaled to 0-1 over the study area (min-max)
           fsi_class.tif        classes 1..N_CLASSES (1 = very low), 255 = nodata
           class_stability.tif  share of Monte Carlo weight draws that keep the baseline class
           ahp_weights.csv, class_areas.csv, sensitivity.csv, phase3_log.json, fsi_overview.png

Install : pip install rasterio scipy numpy matplotlib
Run     : python phase3_susceptibility.py
"""
import json
import os
import sys

import numpy as np
import rasterio
from scipy.stats import spearmanr

# ------------------------------ CONFIG ---------------------------------
PHASE2_DIR = "data/phase2"
OUT_DIR = "data/phase3"

# Order must match the rows and columns of MATRIX.
FACTORS = ["elevation", "slope", "rainfall", "dist_drainage",
           "flow_acc", "drainage_density", "lulc", "ndvi"]

# Pairwise comparison matrix (Saaty 1-9 scale). Row i vs column j: how much more
# important factor i is than factor j. Check it against your methodology document.
MATRIX = np.array([
    [1,   2,   1,   2,   1,   2,   2,   4],
    [1/2, 1,   1/2, 1,   1/2, 1/2, 1,   3],
    [1,   2,   1,   1,   1,   2,   2,   3],
    [1/2, 1,   1,   1,   2,   2,   2,   4],
    [1,   2,   1,   1/2, 1,   2,   2,   4],
    [1/2, 2,   1/2, 1/2, 1/2, 1,   1,   3],
    [1/2, 1,   1/2, 1/2, 1/2, 1,   1,   2],
    [1/4, 1/3, 1/3, 1/4, 1/4, 1/3, 1/2, 1],
], dtype="float64")

N_CLASSES = 5
CLASS_METHOD = "equal_interval"     # "equal_interval" or "quantile", applied to the scaled FSI
CLASS_LABELS = ["Very low", "Low", "Moderate", "High", "Very high"]

SENS_PCT = 0.20                     # one-at-a-time weight change (+/- 20%)
MC_DRAWS = 100                      # Monte Carlo weight draws for class stability
MC_CONCENTRATION = 200              # higher = draws stay closer to the AHP weights
SAMPLE_N = 200000                   # cells used for rank correlations
RANDOM_SEED = 42
# -----------------------------------------------------------------------

RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
NODATA_F = -9999.0
NODATA_U = 255


def log(msg=""):
    print(msg, flush=True)


os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------ AHP weights ----------------------------
n = len(FACTORS)
if MATRIX.shape != (n, n):
    sys.exit("MATRIX must be %d x %d to match FACTORS." % (n, n))
if not np.allclose(MATRIX, 1.0 / MATRIX.T):
    sys.exit("MATRIX is not reciprocal (a_ij must equal 1 / a_ji).")

vals, vecs = np.linalg.eig(MATRIX)
k = int(np.argmax(vals.real))
lam = float(vals[k].real)
w = np.abs(vecs[:, k].real)
w = w / w.sum()
CI = (lam - n) / (n - 1)
CR = CI / RI[n] if RI[n] > 0 else 0.0
gm = np.prod(MATRIX, axis=1) ** (1.0 / n)
gm = gm / gm.sum()

log("-- AHP --")
log("lambda_max = %.4f, CI = %.4f, RI = %.2f, CR = %.4f (%.2f%%)" % (lam, CI, RI[n], CR, 100 * CR))
log("Consistency %s (threshold 0.10)" % ("ACCEPTABLE" if CR < 0.10 else "NOT ACCEPTABLE: revise the judgements"))
order = np.argsort(-w)
for rank, i in enumerate(order, 1):
    log("  %d. %-17s weight %.4f   (geometric-mean check %.4f)" % (rank, FACTORS[i], w[i], gm[i]))
with open(os.path.join(OUT_DIR, "ahp_weights.csv"), "w") as f:
    f.write("factor,weight,rank,geometric_mean_weight\n")
    for rank, i in enumerate(order, 1):
        f.write("%s,%.6f,%d,%.6f\n" % (FACTORS[i], w[i], rank, gm[i]))
    f.write("lambda_max,%.6f,,\nCI,%.6f,,\nCR,%.6f,,\n" % (lam, CI, CR))

# ---------------------------- read the factors -------------------------
layers = []
profile = None
for name in FACTORS:
    p = os.path.join(PHASE2_DIR, "std_%s.tif" % name)
    if not os.path.exists(p):
        sys.exit("Missing %s. Run phase2_factors.py first." % p)
    with rasterio.open(p) as src:
        a = src.read(1).astype("float32")
        a[a == src.nodata] = np.nan
        if profile is None:
            profile = src.profile.copy()
            transform = src.transform
            shape = (src.height, src.width)
        elif (src.height, src.width) != shape:
            sys.exit("%s has a different grid from the other layers." % p)
    layers.append(a)
stack = np.stack(layers)            # (n, H, W)
del layers
valid = np.all(np.isfinite(stack), axis=0)
CELL_KM2 = abs(transform.a * transform.e) / 1e6
log("\nCells scored: %d (%.0f km2)" % (valid.sum(), valid.sum() * CELL_KM2))
X = stack[:, valid]                 # (n, n_valid)
del stack


def write(name, arr, nodata, dtype):
    prof = profile.copy()
    prof.update(dtype=dtype, count=1, nodata=nodata, compress="deflate")
    out = np.full(shape, nodata, dtype=dtype)
    out[valid] = arr
    with rasterio.open(os.path.join(OUT_DIR, name), "w", **prof) as dst:
        dst.write(out, 1)


# -------------------------- classification helper ----------------------
def scale01(f):
    lo, hi = f.min(), f.max()
    return (f - lo) / (hi - lo) if hi > lo else np.zeros_like(f)


def classify(f):
    """Return classes 1..N_CLASSES for FSI values f, using the configured method."""
    s = scale01(f)
    if CLASS_METHOD == "quantile":
        edges = np.quantile(s, np.linspace(0, 1, N_CLASSES + 1))[1:-1]
    else:
        edges = np.linspace(0, 1, N_CLASSES + 1)[1:-1]
    return (np.digitize(s, edges) + 1).astype("uint8")


# ----------------------------- baseline FSI ----------------------------
w32 = w.astype("float32")
fsi = (w32 @ X).astype("float32")
fsi_scaled = scale01(fsi).astype("float32")
cls = classify(fsi)
write("fsi.tif", fsi, NODATA_F, "float32")
write("fsi_scaled.tif", fsi_scaled, NODATA_F, "float32")
write("fsi_class.tif", cls, NODATA_U, "uint8")

log("\n-- FSI --")
log("FSI range %.3f to %.3f, mean %.3f, std %.3f" % (fsi.min(), fsi.max(), fsi.mean(), fsi.std()))
log("Classes by %s on the rescaled FSI:" % CLASS_METHOD)
rows = ["class,label,cells,area_km2,percent"]
for c in range(1, N_CLASSES + 1):
    cnt = int(np.sum(cls == c))
    area = cnt * CELL_KM2
    pct = 100.0 * cnt / len(cls)
    label = CLASS_LABELS[c - 1] if N_CLASSES == len(CLASS_LABELS) else str(c)
    rows.append("%d,%s,%d,%.2f,%.2f" % (c, label, cnt, area, pct))
    log("  %d %-10s %8.1f km2  %5.1f%%" % (c, label, area, pct))
with open(os.path.join(OUT_DIR, "class_areas.csv"), "w") as f:
    f.write("\n".join(rows) + "\n")

# ------------------------------ sensitivity ----------------------------
rng = np.random.default_rng(RANDOM_SEED)
idx = rng.choice(len(fsi), min(SAMPLE_N, len(fsi)), replace=False)


def compare(wv):
    f = (wv.astype("float32") @ X)
    c = classify(f)
    rho = spearmanr(fsi[idx], f[idx])[0]
    return rho, float(np.mean(c != cls)), float(np.mean(np.abs(c.astype(int) - cls.astype(int))))


log("\n-- Sensitivity: one factor weight changed by +/-%.0f%% (others rescaled) --" % (100 * SENS_PCT))
log("%-17s %-6s %-10s %-14s" % ("factor", "change", "rank corr", "cells re-classed"))
srows = ["test,change,spearman_rho,share_cells_changed_class,mean_class_shift"]
for i, name in enumerate(FACTORS):
    for sign in (-1, 1):
        w2 = w.copy()
        w2[i] *= 1 + sign * SENS_PCT
        w2 /= w2.sum()
        rho, changed, shift = compare(w2)
        tag = "%+.0f%%" % (100 * sign * SENS_PCT)
        log("%-17s %-6s %-10.4f %5.1f%%" % (name, tag, rho, 100 * changed))
        srows.append("%s,%s,%.5f,%.5f,%.5f" % (name, tag, rho, changed, shift))
rho, changed, shift = compare(np.full(n, 1.0 / n))
log("%-17s %-6s %-10.4f %5.1f%%" % ("EQUAL WEIGHTS", "", rho, 100 * changed))
srows.append("equal_weights,,%.5f,%.5f,%.5f" % (rho, changed, shift))

# Monte Carlo: how stable is each cell's class if the weights are uncertain?
log("\n-- Monte Carlo class stability (%d weight draws, concentration %d) --" % (MC_DRAWS, MC_CONCENTRATION))
draws = rng.dirichlet(w * MC_CONCENTRATION, size=MC_DRAWS)
same = np.zeros(len(fsi), dtype="uint16")
share_same = []
for wv in draws:
    c = classify((wv.astype("float32") @ X))
    same += (c == cls)
    share_same.append(float(np.mean(c == cls)))
stability = (same / float(MC_DRAWS)).astype("float32")
write("class_stability.tif", stability, NODATA_F, "float32")
log("Share of cells keeping their baseline class per draw: mean %.1f%%, 5th percentile %.1f%%" % (
    100 * np.mean(share_same), 100 * np.percentile(share_same, 5)))
log("Cells that keep their class in at least 80%% of draws: %.1f%%" % (100 * np.mean(stability >= 0.8)))
srows.append("monte_carlo_mean_same_class,,,%.5f," % np.mean(share_same))
with open(os.path.join(OUT_DIR, "sensitivity.csv"), "w") as f:
    f.write("\n".join(srows) + "\n")

# ------------------------------ quick look -----------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    def full(arr, fill=np.nan):
        o = np.full(shape, fill, dtype="float32")
        o[valid] = arr
        return o

    pw = None
    pw_path = os.path.join(PHASE2_DIR, "perm_water.tif")
    if os.path.exists(pw_path):
        with rasterio.open(pw_path) as s:
            pw = np.where(s.read(1) == 1, 1.0, np.nan)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    im0 = axes[0].imshow(full(fsi_scaled), cmap="RdYlBu_r", vmin=0, vmax=1)
    axes[0].set_title("FSI (rescaled 0-1)")
    fig.colorbar(im0, ax=axes[0], shrink=0.7)
    cmap = ListedColormap(["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"][:N_CLASSES])
    axes[1].imshow(full(cls.astype("float32")), cmap=cmap, vmin=1, vmax=N_CLASSES)
    axes[1].set_title("Susceptibility classes (1 very low to %d very high)" % N_CLASSES)
    im2 = axes[2].imshow(full(stability), cmap="viridis", vmin=0, vmax=1)
    axes[2].set_title("Class stability under weight uncertainty")
    fig.colorbar(im2, ax=axes[2], shrink=0.7)
    for ax in axes:
        if pw is not None:
            ax.imshow(pw, cmap=ListedColormap(["#08306b"]), vmin=0, vmax=1)
        ax.axis("off")
    fig.savefig(os.path.join(OUT_DIR, "fsi_overview.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
except Exception as e:
    log("Quick-look figure skipped: %s" % e)

with open(os.path.join(OUT_DIR, "phase3_log.json"), "w") as f:
    json.dump({
        "factors": FACTORS, "weights": {FACTORS[i]: float(w[i]) for i in range(n)},
        "lambda_max": lam, "CI": CI, "CR": CR, "class_method": CLASS_METHOD, "n_classes": N_CLASSES,
        "sens_pct": SENS_PCT, "mc_draws": MC_DRAWS, "mc_concentration": MC_CONCENTRATION,
    }, f, indent=2)
log("\nDone. Outputs are in %s" % OUT_DIR)
"""
dti_core.py: shared scoring functions for the dynamic trigger index (DTI).
Lokoja LGA flood susceptibility project.

Pre-specified design (agreed before any evaluation against the floods):

  rainfall score  R = percentile of the LARGEST rolling 48 h rainfall total in the 14 days ending at the
                      valid time (the valid day included), judged against the pixel's own wet-season history of
                      48 h totals of at least 1 mm. Totals below 1 mm score 0.
  soil score      S = linear scaling of surface soil moisture between the 10th and 99th percentile of that
                      product's own wet-season history, clipped to 0-1.
  DTI = W_RAIN * R + W_SOIL * S         (weights come from a two-criterion AHP, see CONFIG of the callers)
  FRI = FSI_scaled * DTI

Used by phase6b_retro_eval.py (LGA-mean daily series) and phase6c_grid_job.py (rasters).
Every function works on scalars, 1-D series or 2-D rasters.
"""
import numpy as np

PCTL = np.array([10, 25, 50, 75, 90, 95, 99], dtype="float64")
MIN_RAIN_MM = 1.0
LOOKBACK_DAYS = 14
WINDOW_DAYS = 2            # 48 h


def rolling_total(daily, window=WINDOW_DAYS):
    """Rolling sums along axis 0. The first window-1 entries are NaN."""
    d = np.asarray(daily, dtype="float64")
    out = np.full(d.shape, np.nan)
    for i in range(window - 1, d.shape[0]):
        out[i] = np.sum(d[i - window + 1:i + 1], axis=0)
    return out


def max_over_previous(totals, index, days=LOOKBACK_DAYS):
    """Maximum of totals[index-days+1 .. index] along axis 0 (NaNs ignored)."""
    lo = max(0, index - days + 1)
    with np.errstate(all="ignore"):
        return np.nanmax(totals[lo:index + 1], axis=0)


def rain_table(values):
    """Percentile table (p10..p99) from 1-D totals, using only rainy windows (>= MIN_RAIN_MM)."""
    v = np.asarray(values, dtype="float64")
    v = v[np.isfinite(v) & (v >= MIN_RAIN_MM)]
    return np.percentile(v, PCTL)


def rain_score(x, table):
    """
    Percentile rank (0-1) of x among rainy windows, by piecewise-linear interpolation through
    (MIN_RAIN_MM, 0), (p10, .10), ... (p99, .99). Above p99 the score creeps up to 1.
    table has shape (7,) or (7, ...) matching x.
    """
    x = np.asarray(x, dtype="float64")
    table = np.asarray(table, dtype="float64")
    shape = x.shape
    tab = [np.broadcast_to(table[i], shape) for i in range(len(PCTL))]
    xs = [np.full(shape, MIN_RAIN_MM)] + tab
    ys = [0.0] + list(PCTL / 100.0)
    score = np.zeros(shape)
    for i in range(len(xs) - 1):
        lo, hi = xs[i], xs[i + 1]
        m = (x >= lo) & (x < hi)
        t = (x - lo) / np.maximum(hi - lo, 1e-9)
        score = np.where(m, ys[i] + t * (ys[i + 1] - ys[i]), score)
    top = tab[-1]
    score = np.where(x >= top, np.minimum(1.0, 0.99 + 0.01 * (x - top) / np.maximum(top, 1e-9)), score)
    score = np.where(np.isfinite(x) & np.isfinite(top), score, np.nan)
    return score


def soil_score(sm, p10, p99):
    """Linear scaling of soil moisture between the product's own p10 and p99, clipped to 0-1."""
    sm = np.asarray(sm, dtype="float64")
    span = np.maximum(np.asarray(p99, dtype="float64") - np.asarray(p10, dtype="float64"), 1e-9)
    return np.clip((sm - p10) / span, 0.0, 1.0)


def dti(rain, soil, w_rain, w_soil):
    """Weighted sum of the two scores (weights are renormalised to sum to 1)."""
    tot = float(w_rain) + float(w_soil)
    return (w_rain * np.asarray(rain) + w_soil * np.asarray(soil)) / tot


def categorise(d, thresholds=(0.25, 0.50, 0.75)):
    """DTI categories 1 (low) .. 4 (very high) with fixed thresholds; NaN stays 0."""
    d = np.asarray(d, dtype="float64")
    out = np.digitize(d, thresholds) + 1
    return np.where(np.isfinite(d), out, 0).astype("uint8")
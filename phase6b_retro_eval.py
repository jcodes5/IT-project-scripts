#!/usr/bin/env python3
"""
PHASE 6b: Retrospective evaluation of the dynamic trigger index (DTI) at LGA level.
Lokoja LGA flood susceptibility project.

Question: do the rainfall and soil-moisture scores rise before the periods when Sentinel-1 shows
more of the LGA flooded? This is a NOWCAST test (observed rainfall, no forecast error). With only two
seasons (about 30 Sentinel-1 dates) the numbers are indicative, not conclusive.

Input  (data/phase6/, from phase6b_season_series.js):
         daily_rain_chirps_lga.csv, daily_sm_smap_lga.csv, daily_sm_era5land_lga.csv  (columns: date, value)
         s1_flood_series.csv  (columns: year, date, domain_km2, flooded_km2, flooded_pct)
Output (data/phase6_results/):
         daily_scores.csv, lag_correlations.csv, peak_date_ranks.csv, retro_eval.png

Design is pre-specified in dti_core.py. The lag table is exploratory: report all lags, not only the best one.

Install : pip install numpy scipy pandas matplotlib
Run     : python phase6b_retro_eval.py        (dti_core.py must sit in the same folder)
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import dti_core as core

# ------------------------------ CONFIG ---------------------------------
DATA_DIR = "data/phase6"
OUT_DIR = "data/phase6_results"

W_RAIN, W_SOIL = 0.75, 0.25            # two-criterion AHP, rainfall moderately more important (3)
SOIL_PRODUCT = "smap"                  # "smap" or "era5land"

CLIM_MONTHS = (4, 10)                  # April to October
RAIN_CLIM_YEARS = (2001, 2025)
SMAP_CLIM_YEARS = (2016, 2024)
ERA_CLIM_YEARS = (2001, 2025)

EVAL_YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
SEASON_START, SEASON_END = (6, 1), (11, 30)
EVENT_DATES = {2018: "2018-09-21", 2022: "2022-10-13"}
LAGS = [0, 3, 7, 10, 14, 21]

# Sentinel-1 dates: each relative orbit may cover only part of the LGA, so dates are filtered by coverage.
ORBIT = None                           # None = use the relative orbit with the best coverage; or set a number
MIN_COVERAGE = 0.80                    # keep dates whose observed domain is at least this share of the largest
# -----------------------------------------------------------------------


def log(msg=""):
    print(msg, flush=True)


def load_series(pattern):
    """Read one or more CSVs (date, value). Several rows per day (SMAP 3-hourly) are averaged to a daily mean."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, pattern)))
    if not files:
        sys.exit("Missing file(s): %s" % os.path.join(DATA_DIR, pattern))
    df = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in files])
    s = df.groupby("date")["value"].mean().astype(float).sort_index()
    return s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))


os.makedirs(OUT_DIR, exist_ok=True)
rain = load_series("daily_rain_chirps_lga*.csv")
sm_smap = load_series("daily_sm_smap_lga*.csv")
sm_era = load_series("daily_sm_era5land_lga*.csv")
sm = sm_smap if SOIL_PRODUCT == "smap" else sm_era
sm_years = SMAP_CLIM_YEARS if SOIL_PRODUCT == "smap" else ERA_CLIM_YEARS
log("Rain series %s to %s | SMAP %s to %s | ERA5-Land %s to %s" % (
    rain.index.min().date(), rain.index.max().date(), sm_smap.index.min().date(), sm_smap.index.max().date(),
    sm_era.index.min().date(), sm_era.index.max().date()))


def in_clim(s, years):
    m = (s.index.month >= CLIM_MONTHS[0]) & (s.index.month <= CLIM_MONTHS[1])
    y = (s.index.year >= years[0]) & (s.index.year <= years[1])
    return s[m & y]


# --------------------------- climatology tables -------------------------
totals = pd.Series(core.rolling_total(rain.values, core.WINDOW_DAYS), index=rain.index)
table = core.rain_table(in_clim(totals, RAIN_CLIM_YEARS).values)
log("\nLGA-mean 48 h rainfall percentiles among rainy windows (mm): " +
    ", ".join("p%d %.1f" % (p, v) for p, v in zip(core.PCTL, table)))
sm_clim = in_clim(sm, sm_years).dropna()
p10, p99 = np.percentile(sm_clim, [10, 99])
log("Soil moisture (%s) p10 %.3f, p99 %.3f" % (SOIL_PRODUCT, p10, p99))

# ------------------------------ daily scores ----------------------------
rain_v = rain.values
tot_v = totals.values


def score_days(days):
    """Rain, soil and DTI scores for each day in days (same design for every day)."""
    rows = []
    for day in days:
        if day not in rain.index:
            continue
        i = rain.index.get_loc(day)
        r48 = tot_v[i]
        rows.append({
            "date": day, "year": day.year,
            "rain48_mm": r48,
            "max48_prev14_mm": core.max_over_previous(tot_v, i, 14),
            "score_48h": core.rain_score(r48, table),
            "score_max7d": core.rain_score(core.max_over_previous(tot_v, i, 7), table),
            "score_max14d": core.rain_score(core.max_over_previous(tot_v, i, 14), table),
            "score_max30d": core.rain_score(core.max_over_previous(tot_v, i, 30), table),
            "soil_moisture": sm.get(day, np.nan),
            "score_soil": core.soil_score(sm.get(day, np.nan), p10, p99),
        })
    df = pd.DataFrame(rows).set_index("date")
    df["dti"] = core.dti(df["score_max14d"], df["score_soil"], W_RAIN, W_SOIL)
    df["dti_48h_only"] = core.dti(df["score_48h"], df["score_soil"], W_RAIN, W_SOIL)
    return df


eval_days = []
for year in EVAL_YEARS:
    eval_days += list(pd.date_range(pd.Timestamp(year, *SEASON_START), pd.Timestamp(year, *SEASON_END), freq="D"))
scores = score_days(eval_days)
scores.to_csv(os.path.join(OUT_DIR, "daily_scores.csv"))
log("Daily scores written for %d days. Missing soil-moisture days: %d" % (len(scores), scores["score_soil"].isna().sum()))

# ------------------------------ flood series ----------------------------
ffiles = sorted(glob.glob(os.path.join(DATA_DIR, "s1_flood_series*.csv")))
if not ffiles:
    sys.exit("Missing file(s): %s" % os.path.join(DATA_DIR, "s1_flood_series*.csv"))
fl = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in ffiles]).dropna(subset=["flooded_pct"])
fl = fl[fl["year"].isin(EVAL_YEARS)].sort_values("date")
log("\nSentinel-1 dates in the files: %d" % len(fl))
if "orbit" in fl.columns:
    cov = fl.groupby("orbit")["domain_km2"].agg(["count", "median", "max"])
    log("Coverage by relative orbit (observed domain, km2):")
    log(cov.round(0).to_string())
    orbit = ORBIT if ORBIT is not None else cov["median"].idxmax()
    fl = fl[fl["orbit"] == orbit]
    log("Using relative orbit %s (%d dates)" % (orbit, len(fl)))
full = fl["domain_km2"].max()
dropped = int((fl["domain_km2"] < MIN_COVERAGE * full).sum())
fl = fl[fl["domain_km2"] >= MIN_COVERAGE * full]
log("Dropped %d dates with under %.0f%% of the largest coverage (%.0f km2)" % (dropped, 100 * MIN_COVERAGE, full))
log("\nSentinel-1 dates used: %d (%s)" % (len(fl), ", ".join("%d: %d" % (y, n) for y, n in fl.groupby("year").size().items())))
for y, g in fl.groupby("year"):
    log("  %d flooded %% of domain: min %.1f, median %.1f, max %.1f (peak on %s)" % (
        y, g["flooded_pct"].min(), g["flooded_pct"].median(), g["flooded_pct"].max(),
        g.loc[g["flooded_pct"].idxmax(), "date"].date()))
if len(fl) < 8:
    log("WARNING: fewer than 8 Sentinel-1 dates; correlations below are not meaningful.")

# --------------------------- lagged correlations ------------------------
indicators = ["score_48h", "score_max7d", "score_max14d", "score_max30d", "score_soil", "dti", "dti_48h_only"]
rows = []
for ind in indicators:
    for lag in LAGS:
        xs, ys = [], []
        for _, r in fl.iterrows():
            t = r["date"] - pd.Timedelta(days=lag)
            if t in scores.index and np.isfinite(scores.at[t, ind]):
                xs.append(scores.at[t, ind])
                ys.append(r["flooded_pct"])
        if len(xs) >= 5 and np.std(xs) > 0:
            rho = spearmanr(xs, ys)[0]
        else:
            rho = np.nan
        rows.append({"indicator": ind, "lag_days": lag, "n": len(xs), "spearman_rho": rho})
lagdf = pd.DataFrame(rows)
lagdf.to_csv(os.path.join(OUT_DIR, "lag_correlations.csv"), index=False)
log("\nSpearman correlation between flooded % of domain and the indicator LAG days earlier")
piv = lagdf.pivot(index="indicator", columns="lag_days", values="spearman_rho").reindex(indicators)
log(piv.round(2).to_string())
log("(n = %s Sentinel-1 dates per cell; pooled over %s)" % (int(lagdf["n"].max()), EVAL_YEARS))

# ----------------------------- peak-date ranks --------------------------
rows = []
for year, ds in EVENT_DATES.items():
    d = pd.Timestamp(ds)
    season = scores[scores["year"] == year]
    if d not in season.index:
        continue
    for ind in indicators:
        v = season[ind].dropna()
        rank = float((v <= season.at[d, ind]).mean()) if np.isfinite(season.at[d, ind]) else np.nan
        rows.append({"year": year, "date": ds, "indicator": ind, "value": season.at[d, ind], "season_percentile": rank})
rk = pd.DataFrame(rows)
rk.to_csv(os.path.join(OUT_DIR, "peak_date_ranks.csv"), index=False)
log("\nWhere the flood-peak dates sit within their own Jun-Nov distribution (1.00 = highest day of the season)")
log(rk.pivot(index="indicator", columns="year", values="season_percentile").reindex(indicators).round(2).to_string())

# ------------------------ seasonality control (exploratory) --------------
# Soil moisture and the flooded fraction both peak late in the wet season, so a high raw correlation can come
# from the shared seasonal cycle alone. Two checks: (1) how well a pure calendar baseline does, and
# (2) correlations of ANOMALIES, i.e. the indicator minus its normal value for that date (the normal is built
# from the other years of the SMAP record, with the evaluation years left out).
base_years = list(range(SMAP_CLIM_YEARS[0], SMAP_CLIM_YEARS[1] + 1))
base_days = []
for y in base_years:
    base_days += list(pd.date_range(pd.Timestamp(y, *SEASON_START), pd.Timestamp(y, *SEASON_END), freq="D"))
base = score_days(base_days)
base["doy"] = base.index.dayofyear
# Leave-one-year-out normals: the normal for a date in year Y is built from all OTHER baseline years.
climday_parts, anom_parts = [], []
for y in sorted(scores["year"].unique()):
    other = base[base["year"] != y]
    clim_y = other.groupby("doy")[indicators].mean().rolling(21, center=True, min_periods=5).mean()
    sy = scores[scores["year"] == y]
    cd = clim_y.reindex(sy.index.dayofyear)
    cd.index = sy.index
    climday_parts.append(cd)
    anom_parts.append(sy[indicators] - cd[indicators])
climdays = pd.concat(climday_parts)
anom = pd.concat(anom_parts)

log("\nSeasonality control (exploratory): leave-one-year-out normals from %d baseline years (%s)" % (len(base_years), base_years))
cal = []
for ind in ("score_soil", "dti"):
    xs = [climdays.at[r["date"], ind] for _, r in fl.iterrows() if np.isfinite(climdays.at[r["date"], ind])]
    ys = [r["flooded_pct"] for _, r in fl.iterrows() if np.isfinite(climdays.at[r["date"], ind])]
    cal.append((ind, spearmanr(xs, ys)[0] if len(xs) >= 5 else np.nan))
log("Calendar-only baseline (the normal value of the indicator for that date, no weather information):")
for ind, rho in cal:
    log("  %-12s rho with flooded %% = %.2f" % (ind, rho))

rows = []
for ind in indicators:
    for lag in LAGS:
        xs, ys = [], []
        for _, r in fl.iterrows():
            t = r["date"] - pd.Timedelta(days=lag)
            if t in anom.index and np.isfinite(anom.at[t, ind]):
                xs.append(anom.at[t, ind])
                ys.append(r["flooded_pct"])
        rows.append({"indicator": ind, "lag_days": lag, "n": len(xs),
                     "spearman_rho": spearmanr(xs, ys)[0] if len(xs) >= 5 and np.std(xs) > 0 else np.nan})
adf = pd.DataFrame(rows)
adf.to_csv(os.path.join(OUT_DIR, "anomaly_correlations.csv"), index=False)
log("\nSpearman correlation between flooded % and the indicator ANOMALY (above or below normal for the date), LAG days earlier")
log(adf.pivot(index="indicator", columns="lag_days", values="spearman_rho").reindex(indicators).round(2).to_string())

late = fl[(fl["date"].dt.month >= 9) & ((fl["date"].dt.month < 11) | ((fl["date"].dt.month == 11) & (fl["date"].dt.day <= 15)))]
rows = []
for ind in indicators:
    for lag in (0, 7, 14):
        xs, ys = [], []
        for _, r in late.iterrows():
            t = r["date"] - pd.Timedelta(days=lag)
            if t in scores.index and np.isfinite(scores.at[t, ind]):
                xs.append(scores.at[t, ind])
                ys.append(r["flooded_pct"])
        rows.append({"indicator": ind, "lag_days": lag, "n": len(xs),
                     "spearman_rho": spearmanr(xs, ys)[0] if len(xs) >= 5 and np.std(xs) > 0 else np.nan})
ldf = pd.DataFrame(rows)
ldf.to_csv(os.path.join(OUT_DIR, "late_season_correlations.csv"), index=False)
log("\nLate-season only (1 September to 15 November), raw indicators, n = %d Sentinel-1 dates" % len(late))
log(ldf.pivot(index="indicator", columns="lag_days", values="spearman_rho").reindex(indicators).round(2).to_string())

# --------------------------- year-level summary --------------------------
# One row per season: does a bigger flood season go with a higher DTI, wetter soil or more rain?
yrows = []
for y, g in fl.groupby("year"):
    sy = scores[scores["year"] == y]
    aso = sy[(sy.index.month >= 8) & (sy.index.month <= 10)]
    season_rain = rain[(rain.index >= pd.Timestamp(y, *SEASON_START)) & (rain.index <= pd.Timestamp(y, *SEASON_END))].sum()
    yrows.append({"year": y, "peak_flooded_pct": g["flooded_pct"].max(),
                  "peak_date": g.loc[g["flooded_pct"].idxmax(), "date"].date(),
                  "mean_soil_score_AugOct": aso["score_soil"].mean(),
                  "max_dti_AugOct": aso["dti"].max(), "mean_dti_AugOct": aso["dti"].mean(),
                  "season_rain_mm": season_rain, "max_48h_mm": sy["rain48_mm"].max()})
ydf = pd.DataFrame(yrows).set_index("year").sort_values("peak_flooded_pct", ascending=False)
ydf.to_csv(os.path.join(OUT_DIR, "year_summary.csv"))
log("\nYear-level summary, sorted by the season's peak flooded share (Jun-Nov; soil and DTI over Aug-Oct)")
log(ydf.round(2).to_string())
if len(ydf) >= 5:
    log("Spearman across seasons (n = %d) between the peak flooded share and:" % len(ydf))
    for col in ["mean_soil_score_AugOct", "max_dti_AugOct", "mean_dti_AugOct", "season_rain_mm", "max_48h_mm"]:
        log("  %-24s %.2f" % (col, spearmanr(ydf["peak_flooded_pct"], ydf[col])[0]))
    log("With only a handful of seasons these values are very uncertain; look at the table, not only the numbers.")

# --------------------------------- figure -------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(EVAL_YEARS), 1, figsize=(11, 4.5 * len(EVAL_YEARS)))
    axes = np.atleast_1d(axes)
    for ax, year in zip(axes, EVAL_YEARS):
        s = scores[scores["year"] == year]
        ax.plot(s.index, s["score_max14d"], label="rain score (max 48 h, 14 d)", color="tab:blue")
        ax.plot(s.index, s["score_48h"], label="rain score (48 h)", color="tab:blue", alpha=0.35)
        ax.plot(s.index, s["score_soil"], label="soil score", color="tab:green")
        ax.plot(s.index, s["dti"], label="DTI", color="black", linewidth=2)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("score (0-1)")
        ax2 = ax.twinx()
        g = fl[fl["year"] == year]
        ax2.bar(g["date"], g["flooded_pct"], width=3, color="tab:red", alpha=0.45, label="Sentinel-1 flooded %")
        ax2.set_ylabel("flooded % of observed domain")
        if year in EVENT_DATES:
            ax.axvline(pd.Timestamp(EVENT_DATES[year]), color="grey", linestyle="--")
        ax.set_title("%d season" % year)
        ax.legend(loc="upper left", fontsize=8)
    fig.savefig(os.path.join(OUT_DIR, "retro_eval.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
except Exception as e:
    log("Figure skipped: %s" % e)

log("\nHow to read this:")
log(" - A weak or negative correlation at lag 0 with a stronger one at longer lags fits riverine flooding that")
log("   follows rainfall by days to weeks. A weak correlation at every lag means local rainfall does not track")
log("   the observed flooding, which is a limitation to state, not a failure to fix by changing the design.")
log(" - The lag table is exploratory. Report the whole table. Do not change windows or weights after seeing it.")
log(" - Radar sees little flooding among buildings, and the Sentinel-1 dates are 6 to 12 days apart.")
log("\nDone. Outputs are in %s" % OUT_DIR)
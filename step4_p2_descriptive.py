"""
CIVI 6666 - Assignment 1 - P2: Descriptive climate characteristics
Yellowknife A (2204100) | 1961-1990 | Station, NCEP-NCAR R1, CanESM5 (NEX-GDDP)

For each dataset, builds yearly series of
  - mean temperature  (mean of daily Tmean)        [degC]
  - precipitation total (sum of daily precip)      [mm]
for 12 months, 4 seasons (JFM, AMJ, JAS, OND) and the calendar year,
then computes descriptive statistics over the 30 years.

Completeness: a month / season / year value is computed only if >= 90 % of its
daily values are present (otherwise NaN, never filled with 0).

Input :  output/all_daily_1961-1990.csv   (from Step 3)
Outputs (output/):
  P2_series_monthly.csv, P2_series_seasonal.csv, P2_series_annual.csv
  P2_stats_monthly.csv, P2_stats_seasonal_annual.csv
  P2_table_seasonal_annual.csv   compact table for the report
  P2_fig1_annual.png             annual time series (3 datasets)
  P2_fig2_seasonal.png           seasonal time series (4 seasons x 2 variables)

Run:  python3 step4_p2_descriptive.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = "output"
Y0, Y1 = 1961, 1990
DATASETS = ["Station", "NCEP", "CanESM5"]
COLORS = {"Station": "black", "NCEP": "#1f5fbf", "CanESM5": "#c0392b"}
SEASON_ORDER = ["JFM", "AMJ", "JAS", "OND"]
SEASON_NAME = {"JFM": "JFM (winter)", "AMJ": "AMJ (spring)",
               "JAS": "JAS (summer)", "OND": "OND (fall)"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

d = pd.read_csv(os.path.join(OUT, f"all_daily_{Y0}-{Y1}.csv"), parse_dates=["date"])


# ------------------------------------------------ 1. aggregate with 90 % rule
def aggregate(group_cols):
    g = d.groupby(["dataset"] + group_cols)
    out = pd.DataFrame({
        "n_days": g.size(),
        "pct_tmean": g["tmean"].apply(lambda s: s.notna().mean() * 100),
        "pct_precip": g["precip"].apply(lambda s: s.notna().mean() * 100),
        "tmean": g["tmean"].mean(),
        "precip_total": g["precip"].sum(min_count=1),
    })
    out.loc[out.pct_tmean < 90, "tmean"] = np.nan
    out.loc[out.pct_precip < 90, "precip_total"] = np.nan
    return out.reset_index()


mon = aggregate(["year", "month"])
sea = aggregate(["year", "season"])
ann = aggregate(["year"])
mon.to_csv(os.path.join(OUT, "P2_series_monthly.csv"), index=False)
sea.to_csv(os.path.join(OUT, "P2_series_seasonal.csv"), index=False)
ann.to_csv(os.path.join(OUT, "P2_series_annual.csv"), index=False)

excluded = {
    "months": int(((mon.pct_tmean < 90) | (mon.pct_precip < 90)).sum()),
    "seasons": int(((sea.pct_tmean < 90) | (sea.pct_precip < 90)).sum()),
    "years": int(((ann.pct_tmean < 90) | (ann.pct_precip < 90)).sum()),
}


# ------------------------------------------------ 2. descriptive statistics
def describe(x):
    x = x.dropna()
    q25, q75 = x.quantile([0.25, 0.75])
    return pd.Series({
        "n_years": len(x),
        "mean": x.mean(),
        "sd": x.std(ddof=1),
        "cv_pct": x.std(ddof=1) / x.mean() * 100 if x.mean() != 0 else np.nan,
        "min": x.min(),
        "q25": q25,
        "median": x.median(),
        "q75": q75,
        "max": x.max(),
        "range": x.max() - x.min(),
        "iqr": q75 - q25,
        "skewness": x.skew(),
    })


def stats_table(df, period_col):
    rows = []
    for var, label in [("tmean", "Mean temperature (degC)"),
                       ("precip_total", "Precipitation total (mm)")]:
        s = df.groupby(["dataset", period_col])[var].apply(describe).unstack()
        s["variable"] = label
        rows.append(s.reset_index())
    t = pd.concat(rows)
    # CV is not meaningful for temperature in degC (mean near/below zero)
    t.loc[t.variable.str.startswith("Mean temp"), "cv_pct"] = np.nan
    return t


mon_stats = stats_table(mon, "month")
mon_stats["period"] = mon_stats["month"].map(lambda m: MONTHS[int(m) - 1])
sea_stats = stats_table(sea, "season").rename(columns={"season": "period"})
ann2 = ann.assign(period="Annual")
ann_stats = stats_table(ann2, "period")
sa_stats = pd.concat([sea_stats, ann_stats])

cols = ["variable", "dataset", "period", "n_years", "mean", "sd", "cv_pct", "min",
        "q25", "median", "q75", "max", "range", "iqr", "skewness"]
order_p = {p: i for i, p in enumerate(MONTHS + SEASON_ORDER + ["Annual"])}
order_d = {k: i for i, k in enumerate(DATASETS)}


def tidy(t):
    t = t[cols].copy()
    t["_p"] = t.period.map(order_p)
    t["_d"] = t.dataset.map(order_d)
    t = t.sort_values(["variable", "_p", "_d"]).drop(columns=["_p", "_d"])
    return t.round(2)


tidy(mon_stats).to_csv(os.path.join(OUT, "P2_stats_monthly.csv"), index=False)
sa = tidy(sa_stats)
sa.to_csv(os.path.join(OUT, "P2_stats_seasonal_annual.csv"), index=False)

# Compact report table: mean (SD) [min - max] per period x dataset
rep = sa.copy()
rep["value"] = rep.apply(lambda r: f"{r['mean']:.1f} ({r['sd']:.1f}) [{r['min']:.1f} to {r['max']:.1f}]", axis=1)
report = rep.pivot_table(index=["variable", "period"], columns="dataset",
                         values="value", aggfunc="first")[DATASETS]
report = report.reindex(sorted(report.index, key=lambda i: (i[0], order_p[i[1]])))
report.to_csv(os.path.join(OUT, "P2_table_seasonal_annual.csv"))

# ------------------------------------------------------------ 3. figures
# Fig 1: annual series
fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
for ds in DATASETS:
    a = ann[ann.dataset == ds]
    axes[0].plot(a.year, a.tmean, "-o", ms=3, color=COLORS[ds], label=ds)
    axes[1].plot(a.year, a.precip_total, "-o", ms=3, color=COLORS[ds], label=ds)
axes[0].set_ylabel("Annual mean temperature (°C)")
axes[1].set_ylabel("Annual precipitation total (mm)")
axes[1].set_xlabel("Year")
axes[0].set_title("(a) Annual mean temperature")
axes[1].set_title("(b) Annual precipitation total")
for ax in axes:
    ax.grid(alpha=0.3)
axes[0].legend(ncol=3, fontsize=9)
fig.suptitle(f"Yellowknife, {Y0}-{Y1}: annual series from station (ECCC 2204100), "
             "NCEP-NCAR R1 and CanESM5 (NEX-GDDP)", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "P2_fig1_annual.png"), dpi=200)

# Fig 2: seasonal series (2 rows x 4 seasons)
fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharex=True)
for j, s in enumerate(SEASON_ORDER):
    for ds in DATASETS:
        x = sea[(sea.dataset == ds) & (sea.season == s)]
        axes[0, j].plot(x.year, x.tmean, "-", lw=1.3, color=COLORS[ds], label=ds)
        axes[1, j].plot(x.year, x.precip_total, "-", lw=1.3, color=COLORS[ds], label=ds)
    axes[0, j].set_title(SEASON_NAME[s], fontsize=10)
    axes[1, j].set_xlabel("Year")
    for i in range(2):
        axes[i, j].grid(alpha=0.3)
axes[0, 0].set_ylabel("Seasonal mean temperature (°C)")
axes[1, 0].set_ylabel("Seasonal precipitation total (mm)")
axes[0, 0].legend(fontsize=8)
fig.suptitle(f"Yellowknife, {Y0}-{Y1}: seasonal series (complete seasons only)", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "P2_fig2_seasonal.png"), dpi=200)

# ------------------------------------------- 4. comparison summary (terminal)
print("Excluded for < 90 % completeness:", excluded)

for var, label in [("Mean temperature (degC)", "T"), ("Precipitation total (mm)", "P")]:
    t = sa[sa.variable == var].set_index(["period", "dataset"])
    print(f"\n{var}: mean and SD over 1961-1990")
    rows = []
    for p in SEASON_ORDER + ["Annual"]:
        st = t.loc[(p, "Station")]
        row = {"period": p, "Station mean": st["mean"], "Station SD": st["sd"]}
        for ds in ["NCEP", "CanESM5"]:
            o = t.loc[(p, ds)]
            if label == "T":
                row[f"{ds} - Station (degC)"] = o["mean"] - st["mean"]
            else:
                row[f"{ds} / Station (%)"] = o["mean"] / st["mean"] * 100
            row[f"{ds} SD / Station SD"] = o["sd"] / st["sd"]
        rows.append(row)
    print(pd.DataFrame(rows).round(2).to_string(index=False))

print("\nSaved P2 tables and figures in ./output/")

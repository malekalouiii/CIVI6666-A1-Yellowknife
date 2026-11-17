"""
CIVI 6666 - Assignment 1 - P4: Trend detection
Yellowknife A (2204100) | 1961-1990 | Station, NCEP-NCAR R1, CanESM5 (NEX-GDDP)

7 periods (Feb, May, Aug, Oct, JFM, JAS, Annual) x 2 variables x 3 datasets = 42 series
  (i)  mean temperature of the period, each year            [degC]
       -> computed only if >= 90 % of daily values are present
  (ii) maximum daily precipitation within the period, each year [mm]
       -> computed only if >= 95 % of daily values are present

For each series:
  - Mann-Kendall test (two-sided, normal approximation with tie correction)
  - Sen's slope (median of all pairwise slopes), reported per decade
  - direction, p-value, significance at 0.05 and 0.10
  - lag-1 autocorrelation (r1) as a check of the independence assumption

Input : output/all_daily_1961-1990.csv
Outputs (output/):
  P4_series.csv              the 42 yearly series
  P4_results_all.csv         full results
  P4_table_temperature.csv   compact table (degC/decade)
  P4_table_maxprecip.csv     compact table (mm/decade)
  P4_fig_timeseries.png      illustrative series with Sen trend lines
  P4_fig_summary.png         overview of all 42 slopes with significance

Run:  python3 step6_p4_trends.py
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = "output"
Y0, Y1 = 1961, 1990
DATASETS = ["Station", "NCEP", "CanESM5"]
COLORS = {"Station": "black", "NCEP": "#1f5fbf", "CanESM5": "#c0392b"}
PERIODS = [("Feb", [2]), ("May", [5]), ("Aug", [8]), ("Oct", [10]),
           ("JFM", [1, 2, 3]), ("JAS", [7, 8, 9]), ("Annual", list(range(1, 13)))]
VARS = {"tmean": ("Mean temperature", "degC/decade", 90),
        "pmax": ("Maximum daily precipitation", "mm/decade", 95)}

d = pd.read_csv(os.path.join(OUT, f"all_daily_{Y0}-{Y1}.csv"), parse_dates=["date"])


# --------------------------------------------------------- 1. build series
rows = []
for ds in DATASETS:
    dd = d[d.dataset == ds]
    for pname, months in PERIODS:
        sub = dd[dd.month.isin(months)]
        for y, g in sub.groupby("year"):
            pct_t = g.tmean.notna().mean() * 100
            pct_p = g.precip.notna().mean() * 100
            rows.append({
                "dataset": ds, "period": pname, "year": y,
                "tmean": g.tmean.mean() if pct_t >= 90 else np.nan,
                "pmax": g.precip.max() if pct_p >= 95 else np.nan,
                "pct_complete_t": round(pct_t, 1), "pct_complete_p": round(pct_p, 1),
            })
series = pd.DataFrame(rows)
series.to_csv(os.path.join(OUT, "P4_series.csv"), index=False)


# ------------------------------------------- 2. Mann-Kendall and Sen's slope
def norm_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def mann_kendall(t, x):
    """Two-sided Mann-Kendall test with tie correction. Returns S, Z, p, tau."""
    n = len(x)
    s = 0
    for i in range(n - 1):
        s += np.sign(x[i + 1:] - x[i]).sum()
    _, counts = np.unique(x, return_counts=True)
    ties = counts[counts > 1]
    var_s = (n * (n - 1) * (2 * n + 5) - (ties * (ties - 1) * (2 * ties + 5)).sum()) / 18
    if s > 0:
        z = (s - 1) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / math.sqrt(var_s)
    else:
        z = 0.0
    p = 2 * (1 - norm_cdf(abs(z)))
    tau = s / (n * (n - 1) / 2)
    return int(s), z, p, tau


def sen_slope(t, x):
    """Median of pairwise slopes (per year) and the matching intercept."""
    slopes = [(x[j] - x[i]) / (t[j] - t[i])
              for i in range(len(x) - 1) for j in range(i + 1, len(x)) if t[j] != t[i]]
    b = float(np.median(slopes))
    a = float(np.median(x - b * t))
    return b, a


def lag1(x):
    x = x - x.mean()
    return float((x[:-1] * x[1:]).sum() / (x * x).sum())


res = []
for (ds, pname), g in series.groupby(["dataset", "period"]):
    for var, (vname, unit, _) in VARS.items():
        s = g[["year", var]].dropna()
        t, x = s.year.values.astype(float), s[var].values.astype(float)
        S, Z, p, tau = mann_kendall(t, x)
        b, a = sen_slope(t, x)
        res.append({
            "variable": vname, "unit": unit, "dataset": ds, "period": pname,
            "n_years": len(x), "mean": x.mean(),
            "sen_slope_per_decade": b * 10, "sen_intercept": a,
            "change_over_30yr": b * (t.max() - t.min()),
            "direction": "increasing" if S > 0 else ("decreasing" if S < 0 else "no trend"),
            "MK_S": S, "MK_Z": Z, "MK_p": p, "kendall_tau": tau,
            "sig_0.05": p < 0.05, "sig_0.10": p < 0.10, "lag1_r": lag1(x),
        })
res = pd.DataFrame(res)

# optional cross-check against pymannkendall if installed
try:
    import pymannkendall as pmk
    diffs = []
    for _, r in res.iterrows():
        col = "tmean" if r.variable.startswith("Mean") else "pmax"
        x = series[(series.dataset == r.dataset) & (series.period == r.period)][col].dropna().values
        o = pmk.original_test(x)
        diffs.append(abs(o.p - r.MK_p))
    print(f"Cross-check with pymannkendall: max |p difference| = {max(diffs):.2e}")
except ImportError:
    pass

order_p = {p: i for i, (p, _) in enumerate(PERIODS)}
order_d = {k: i for i, k in enumerate(DATASETS)}
res["_p"], res["_d"] = res.period.map(order_p), res.dataset.map(order_d)
res = res.sort_values(["variable", "_p", "_d"]).drop(columns=["_p", "_d"])
res.round(4).to_csv(os.path.join(OUT, "P4_results_all.csv"), index=False)


# ------------------------------------------------------- 3. compact tables
def mark(p):
    return "**" if p < 0.05 else ("*" if p < 0.10 else "")


def compact(vname, fmt):
    t = res[res.variable == vname].copy()
    t["cell"] = t.apply(lambda r: f"{r.sen_slope_per_decade:+{fmt}} (p={r.MK_p:.2f}){mark(r.MK_p)}", axis=1)
    out = t.pivot(index="period", columns="dataset", values="cell")[DATASETS]
    return out.loc[[p for p, _ in PERIODS]]


tt = compact("Mean temperature", ".2f")
tp = compact("Maximum daily precipitation", ".1f")
tt.to_csv(os.path.join(OUT, "P4_table_temperature.csv"))
tp.to_csv(os.path.join(OUT, "P4_table_maxprecip.csv"))


# ---------------------------------------------------------- 4. figures
def plot_series(ax, period, var, title, ylabel):
    for ds in DATASETS:
        s = series[(series.dataset == ds) & (series.period == period)].dropna(subset=[var])
        r = res[(res.dataset == ds) & (res.period == period) &
                (res.variable == VARS[var][0])].iloc[0]
        lab = f"{ds}: {r.sen_slope_per_decade:+.2f} {VARS[var][1].split('/')[0]}/dec (p={r.MK_p:.2f})"
        ax.plot(s.year, s[var], "-o", ms=3, lw=1, color=COLORS[ds], alpha=0.8)
        yy = np.array([s.year.min(), s.year.max()], dtype=float)
        ax.plot(yy, r.sen_intercept + r.sen_slope_per_decade / 10 * yy, "--",
                lw=2, color=COLORS[ds], label=lab)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="best")


fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
plot_series(axes[0, 0], "JFM", "tmean", "(a) Winter (JFM) mean temperature", "°C")
plot_series(axes[0, 1], "Annual", "tmean", "(b) Annual mean temperature", "°C")
plot_series(axes[1, 0], "JAS", "pmax", "(c) Summer (JAS) maximum daily precipitation", "mm")
plot_series(axes[1, 1], "Annual", "pmax", "(d) Annual maximum daily precipitation", "mm")
for ax in axes[1]:
    ax.set_xlabel("Year")
fig.suptitle(f"Yellowknife, {Y0}-{Y1}: illustrative series with Sen's slope trend lines "
             "(dashed); p = Mann-Kendall p-value", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "P4_fig_timeseries.png"), dpi=200)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
for ax, (vname, unit, cmap) in zip(axes, [("Mean temperature", "°C/decade", "RdBu_r"),
                                         ("Maximum daily precipitation", "mm/decade", "BrBG")]):
    t = res[res.variable == vname]
    M = t.pivot(index="period", columns="dataset", values="sen_slope_per_decade").loc[
        [p for p, _ in PERIODS], DATASETS]
    P = t.pivot(index="period", columns="dataset", values="MK_p").loc[M.index, DATASETS]
    lim = np.nanmax(np.abs(M.values))
    im = ax.imshow(M.values, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M.values[i, j]:+.2f}{mark(P.values[i, j])}",
                    ha="center", va="center", fontsize=9)
    ax.set_xticks(range(3), DATASETS)
    ax.set_yticks(range(len(M.index)), M.index)
    ax.set_title(f"{vname} trend ({unit})", fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.85)
fig.suptitle(f"Sen's slope of all 42 series, {Y0}-{Y1} (** p < 0.05, * p < 0.10, Mann-Kendall)",
             fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "P4_fig_summary.png"), dpi=200)

# ----------------------------------------------------------- 5. terminal
pd.set_option("display.width", 200)
print("\nMEAN TEMPERATURE trends, Sen's slope degC/decade (MK p-value)  ** p<0.05  * p<0.10")
print(tt.to_string())
print("\nMAXIMUM DAILY PRECIPITATION trends, Sen's slope mm/decade (MK p-value)")
print(tp.to_string())
print("\nCount of significant series (out of 7 per row):")
cnt = res.groupby(["variable", "dataset"]).agg(
    sig_005=("sig_0.05", "sum"), sig_010=("sig_0.10", "sum"),
    increasing=("direction", lambda s: (s == "increasing").sum()))
print(cnt.to_string())
high_r1 = res[res.lag1_r.abs() > 0.3][["variable", "dataset", "period", "lag1_r", "MK_p"]]
print("\nSeries with |lag-1 autocorrelation| > 0.3 (MK significance may be overstated):")
print(high_r1.round(2).to_string(index=False) if len(high_r1) else "  none")
print("\nSaved P4 tables and figures in ./output/")

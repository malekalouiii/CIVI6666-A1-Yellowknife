"""
CIVI 6666 - Assignment 1 - P8: Engineering indicators
Yellowknife A (2204100) | 1961-1990 | Station, NCEP-NCAR R1, CanESM5 (NEX-GDDP)

Application A - stormwater drainage / culverts
  Rx1day (annual maximum daily precipitation) -> Gumbel (method of moments)
  return levels for 2, 10, 25, 50, 100 years; station also with the Hershfield
  factor 1.13 (fixed observation day -> true 24-hour maximum).

Application B - foundations on permafrost / frost-susceptible ground
  FDD  freezing degree days per freezing season (Jul-Jun), degC-days
  TDD  thawing degree days per calendar year, degC-days
  HDD  heating degree days (base 18 degC) per calendar year (bonus)
  FT   freeze-thaw days (Tmax > 0 and Tmin < 0) per calendar year (bonus)
  For each: mean, SD, Sen's slope per decade + Mann-Kendall p, first vs last
  decade change, and implied change in frost / thaw depth via the Stefan
  relation (depth proportional to sqrt(index)).

Input : output/all_daily_1961-1990.csv, output/P4_series.csv
Outputs (output/):
  P8_indices_series.csv, P8_indices_summary.csv, P8_rx1day_return_levels.csv,
  P8_fig_engineering.png
Run:  python3 step9_p8_engineering.py
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = "output"
DS = ["Station", "NCEP", "CanESM5"]
COLORS = {"Station": "black", "NCEP": "#1f5fbf", "CanESM5": "#c0392b"}
RP = [2, 5, 10, 25, 50, 100]
HERSHFIELD = 1.13

d = pd.read_csv(os.path.join(OUT, "all_daily_1961-1990.csv"), parse_dates=["date"])
s4 = pd.read_csv(os.path.join(OUT, "P4_series.csv"))


# ----------------------------------------------------- trend tools (as in P4)
def mann_kendall_p(x):
    n = len(x)
    s = sum(np.sign(x[i + 1:] - x[i]).sum() for i in range(n - 1))
    _, c = np.unique(x, return_counts=True)
    t = c[c > 1]
    var = (n * (n - 1) * (2 * n + 5) - (t * (t - 1) * (2 * t + 5)).sum()) / 18
    z = 0.0 if s == 0 else (s - np.sign(s)) / math.sqrt(var)
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def sen(t, x):
    return float(np.median([(x[j] - x[i]) / (t[j] - t[i])
                            for i in range(len(x) - 1) for j in range(i + 1, len(x))]))


# ----------------------------------------------------------- 1. indices
rows = []
for ds in DS:
    x = d[d.dataset == ds].copy()
    x["fseason"] = np.where(x.month >= 7, x.year, x.year - 1)   # Jul-Jun season, labelled by start year
    fdd = (-x.tmean.clip(upper=0)).groupby(x.fseason).sum()
    fdd = fdd.loc[1961:1989]                                     # complete seasons Jul 1961 - Jun 1990
    tdd = x.tmean.clip(lower=0).groupby(x.year).sum()
    hdd = (18 - x.tmean).clip(lower=0).groupby(x.year).sum()
    ft = ((x.tmax > 0) & (x.tmin < 0)).groupby(x.year).sum()
    for name, s in [("FDD", fdd), ("TDD", tdd), ("HDD", hdd), ("FT days", ft)]:
        for yr, v in s.items():
            rows.append({"dataset": ds, "index": name, "year": int(yr), "value": float(v)})
ser = pd.DataFrame(rows)
ser.to_csv(os.path.join(OUT, "P8_indices_series.csv"), index=False)

UNITS = {"FDD": "degC-days", "TDD": "degC-days", "HDD": "degC-days", "FT days": "days"}
summ = []
for (ds, idx), g in ser.groupby(["dataset", "index"]):
    g = g.sort_values("year")
    t, v = g.year.values.astype(float), g.value.values
    first, last = v[:10].mean(), v[-10:].mean()
    r = {"index": idx, "unit": UNITS[idx], "dataset": ds, "n": len(v),
         "mean": v.mean(), "sd": v.std(ddof=1),
         "sen_slope_per_decade": sen(t, v) * 10, "MK_p": mann_kendall_p(v),
         "first_decade_mean": first, "last_decade_mean": last,
         "decade_change": last - first, "decade_change_pct": (last / first - 1) * 100}
    if idx in ("FDD", "TDD"):
        r["implied_depth_change_pct (sqrt rule)"] = (math.sqrt(last / first) - 1) * 100
    summ.append(r)
summ = pd.DataFrame(summ)
st_mean = summ[summ.dataset == "Station"].set_index("index")["mean"]
summ["diff_vs_station_pct"] = summ.apply(lambda r: (r["mean"] / st_mean[r["index"]] - 1) * 100, axis=1)
order = {"FDD": 0, "TDD": 1, "HDD": 2, "FT days": 3}
summ = summ.sort_values(["index", "dataset"], key=lambda c: c.map(order) if c.name == "index"
                        else c.map({k: i for i, k in enumerate(DS)}))
summ.round(3).to_csv(os.path.join(OUT, "P8_indices_summary.csv"), index=False)


# ------------------------------------------- 2. Rx1day Gumbel return levels
def gumbel_mom(x):
    beta = math.sqrt(6) * np.std(x, ddof=1) / math.pi
    mu = np.mean(x) - 0.5772 * beta
    return mu, beta


def return_level(mu, beta, T):
    return mu - beta * math.log(-math.log(1 - 1 / T))


rl_rows = []
rx = s4[s4.period == "Annual"]
fits = {}
for ds in DS:
    x = rx[rx.dataset == ds].sort_values("year").pmax.values
    mu, beta = gumbel_mom(x)
    fits[ds] = (mu, beta, x)
    r = {"dataset": ds, "mean": x.mean(), "sd": x.std(ddof=1), "max": x.max(),
         "mean 1961-1975": x[:15].mean(), "mean 1976-1990": x[15:].mean()}
    for T in RP:
        r[f"{T}-yr"] = return_level(mu, beta, T)
    rl_rows.append(r)
st_mu, st_beta, _ = fits["Station"]
r = {"dataset": "Station x 1.13 (24-h)"}
for T in RP:
    r[f"{T}-yr"] = return_level(st_mu, st_beta, T) * HERSHFIELD
rl_rows.append(r)
rl = pd.DataFrame(rl_rows)
for T in [10, 25, 100]:
    rl[f"{T}-yr vs station (%)"] = (rl[f"{T}-yr"] / rl.loc[0, f"{T}-yr"] - 1) * 100
rl.round(1).to_csv(os.path.join(OUT, "P8_rx1day_return_levels.csv"), index=False)

# ------------------------------------------------------------------ figure
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, idx, ttl in [(axes[0], "FDD", "(a) Freezing index (Jul-Jun season)"),
                     (axes[1], "TDD", "(b) Thawing index (calendar year)")]:
    for ds in DS:
        g = ser[(ser.dataset == ds) & (ser["index"] == idx)].sort_values("year")
        s = summ[(summ.dataset == ds) & (summ["index"] == idx)].iloc[0]
        ax.plot(g.year, g.value, "-o", ms=3, lw=1, color=COLORS[ds],
                label=f"{ds}: {s.sen_slope_per_decade:+.0f} °C-d/decade (p={s.MK_p:.2f})")
        t = g.year.values.astype(float)
        b = s.sen_slope_per_decade / 10
        a = np.median(g.value.values - b * t)
        ax.plot([t.min(), t.max()], [a + b * t.min(), a + b * t.max()], "--", color=COLORS[ds], lw=1.8)
    ax.set_xlabel("Year (season start for FDD)")
    ax.set_ylabel("°C-days")
    ax.set_title(ttl, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5)

ax = axes[2]
Ts = np.linspace(1.05, 200, 400)
for ds in DS:
    mu, beta, x = fits[ds]
    ax.plot(Ts, [return_level(mu, beta, T) for T in Ts], color=COLORS[ds], label=f"{ds} (Gumbel)")
    xs = np.sort(x)
    n = len(xs)
    Tp = 1 / (1 - (np.arange(1, n + 1) - 0.44) / (n + 0.12))     # Gringorten plotting positions
    ax.plot(Tp, xs, "o", ms=3.5, color=COLORS[ds], alpha=0.7)
ax.plot(Ts, [return_level(st_mu, st_beta, T) * HERSHFIELD for T in Ts], ":", color="black",
        label="Station × 1.13 (24-h)")
ax.set_xscale("log")
ax.set_xticks([2, 5, 10, 25, 50, 100], ["2", "5", "10", "25", "50", "100"])
ax.set_xlabel("Return period (years)")
ax.set_ylabel("Annual maximum daily precipitation (mm)")
ax.set_title("(c) Rx1day return levels, 1961-1990", fontsize=10)
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8)
fig.suptitle("Yellowknife, 1961-1990: engineering climate indicators from the three datasets", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "P8_fig_engineering.png"), dpi=200)

# ---------------------------------------------------------------- terminal
pd.set_option("display.width", 220)
print("ENGINEERING INDICES (mean, trend, decade change, difference vs station)")
cols = ["index", "dataset", "mean", "sd", "sen_slope_per_decade", "MK_p", "first_decade_mean",
        "last_decade_mean", "decade_change_pct", "implied_depth_change_pct (sqrt rule)",
        "diff_vs_station_pct"]
print(summ[cols].round(2).to_string(index=False))
print("\nRx1day GUMBEL RETURN LEVELS (mm)")
print(rl.round(1).to_string(index=False))
print("\nSaved P8 tables and figure in ./output/")

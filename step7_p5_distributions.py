"""
ENGR 6666 - Assignment 1 - P5: Changes in seasonality and distributions
Yellowknife A (2204100) | Station, NCEP-NCAR R1, CanESM5 (NEX-GDDP)

First decade  = years 1-10  = 1961-1970
Last decade   = years 21-30 = 1981-1990
Change        = last - first

A. Seasonal cycle: 12 monthly mean temperatures and mean monthly precipitation totals
   per decade, their change, and the change in seasonal precipitation fractions.
B. Daily distributions: empirical quantiles Q5, Q10, Q25, Q50, Q75, Q90, Q95 of
   daily Tmean and daily precipitation, from the POOLED daily values of each decade,
   separately for Annual, JFM and JAS (numpy linear interpolation).
C. Wet days (precip >= 1 mm): frequency, mean wet-day intensity and wet-day
   quantiles, to interpret the all-day precipitation quantiles.

Input : output/all_daily_1961-1990.csv, output/P2_series_monthly.csv
Outputs (output/):
  P5_table_cycle_changes.csv, P5_fraction_changes.csv
  P5_quantiles_all.csv, P5_table_quantiles_T.csv, P5_table_quantiles_P.csv
  P5_wetdays.csv
  P5_fig_cycles.png, P5_fig_quantile_changes.png

Run:  python3 step7_p5_distributions.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = "output"
FIRST, LAST = (1961, 1970), (1981, 1990)
DATASETS = ["Station", "NCEP", "CanESM5"]
COLORS = {"Station": "black", "NCEP": "#1f5fbf", "CanESM5": "#c0392b"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SEASONS = ["JFM", "AMJ", "JAS", "OND"]
PERIODS = {"Annual": list(range(1, 13)), "JFM": [1, 2, 3], "JAS": [7, 8, 9]}
QS = [5, 10, 25, 50, 75, 90, 95]
WET = 1.0  # mm

d = pd.read_csv(os.path.join(OUT, "all_daily_1961-1990.csv"), parse_dates=["date"])
mon = pd.read_csv(os.path.join(OUT, "P2_series_monthly.csv"))


def decade(y):
    if FIRST[0] <= y <= FIRST[1]:
        return "first"
    if LAST[0] <= y <= LAST[1]:
        return "last"
    return None


d["decade"] = d.year.map(decade)
mon["decade"] = mon.year.map(decade)
d = d[d.decade.notna()]
mon = mon[mon.decade.notna()]

# ------------------------------------------------- A. seasonal cycle changes
cyc = (mon.groupby(["dataset", "decade", "month"])
       .agg(tmean=("tmean", "mean"), precip=("precip_total", "mean")).reset_index())
w = cyc.pivot_table(index=["dataset", "month"], columns="decade",
                    values=["tmean", "precip"])
w.columns = [("T_" if v == "tmean" else "P_") + dec for v, dec in w.columns]
w = w.reset_index()
w["dT"] = w.T_last - w.T_first
w["dP_mm"] = w.P_last - w.P_first
w["dP_pct"] = w.dP_mm / w.P_first * 100
w["month_name"] = w.month.map(lambda m: MONTHS[m - 1])

cyc_tab = pd.DataFrame({"month": MONTHS})
for ds in DATASETS:
    x = w[w.dataset == ds].sort_values("month")
    cyc_tab[f"{ds} dT (degC)"] = x.dT.round(1).values
for ds in DATASETS:
    x = w[w.dataset == ds].sort_values("month")
    cyc_tab[f"{ds} dP (mm)"] = x.dP_mm.round(1).values
ann_row = {"month": "Annual"}
for ds in DATASETS:
    x = w[w.dataset == ds]
    ann_row[f"{ds} dT (degC)"] = round(x.T_last.mean() - x.T_first.mean(), 1)
    ann_row[f"{ds} dP (mm)"] = round(x.P_last.sum() - x.P_first.sum(), 1)
cyc_tab = pd.concat([cyc_tab, pd.DataFrame([ann_row])], ignore_index=True)
cyc_tab.to_csv(os.path.join(OUT, "P5_table_cycle_changes.csv"), index=False)

# seasonal fractions per decade (mean seasonal total / mean annual total)
SEAS_OF = {m: s for s, ms in zip(SEASONS, [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]) for m in ms}
cyc["season"] = cyc.month.map(SEAS_OF)
sp = cyc.groupby(["dataset", "decade", "season"])["precip"].sum().unstack()[SEASONS]
fr = sp.div(sp.sum(axis=1), axis=0) * 100
frows = []
for ds in DATASETS:
    for s in SEASONS:
        f1, f2 = fr.loc[(ds, "first"), s], fr.loc[(ds, "last"), s]
        frows.append({"dataset": ds, "season": s,
                      "first_pct": round(f1, 1), "last_pct": round(f2, 1),
                      "change_pct_points": round(f2 - f1, 1),
                      "first_total_mm": round(sp.loc[(ds, "first"), s], 1),
                      "last_total_mm": round(sp.loc[(ds, "last"), s], 1)})
fr_tab = pd.DataFrame(frows)
fr_tab.to_csv(os.path.join(OUT, "P5_fraction_changes.csv"), index=False)

# ------------------------------------------------- B. quantile changes
qrows = []
for ds in DATASETS:
    for pname, months in PERIODS.items():
        for var in ["tmean", "precip"]:
            vals = {}
            for dec in ["first", "last"]:
                x = d[(d.dataset == ds) & (d.decade == dec) & d.month.isin(months)][var].dropna()
                vals[dec] = np.percentile(x.values, QS)
            for q, a, b in zip(QS, vals["first"], vals["last"]):
                qrows.append({"dataset": ds, "period": pname, "variable": var,
                              "quantile": f"Q{q}", "q": q, "first": a, "last": b,
                              "change": b - a})
qa = pd.DataFrame(qrows)
qa.round(3).to_csv(os.path.join(OUT, "P5_quantiles_all.csv"), index=False)

# ------------------------------------------------- C. wet days
wrows = []
for ds in DATASETS:
    for pname, months in PERIODS.items():
        r = {"dataset": ds, "period": pname}
        for dec in ["first", "last"]:
            x = d[(d.dataset == ds) & (d.decade == dec) & d.month.isin(months)].precip.dropna()
            wet = x[x >= WET]
            r[f"wetfreq_{dec}_pct"] = (x >= WET).mean() * 100
            r[f"zero_days_{dec}_pct"] = (x == 0).mean() * 100
            r[f"wet_days_per_year_{dec}"] = len(wet) / 10
            r[f"intensity_{dec}_mm"] = wet.mean()
            r[f"wetQ90_{dec}"] = np.percentile(wet, 90)
            r[f"wetQ95_{dec}"] = np.percentile(wet, 95)
        r["wetfreq_change_pct_points"] = r["wetfreq_last_pct"] - r["wetfreq_first_pct"]
        r["intensity_change_mm"] = r["intensity_last_mm"] - r["intensity_first_mm"]
        r["wetQ95_change_mm"] = r["wetQ95_last"] - r["wetQ95_first"]
        wrows.append(r)
wd = pd.DataFrame(wrows)
wd.round(2).to_csv(os.path.join(OUT, "P5_wetdays.csv"), index=False)

# compact report tables
order_p = {p: i for i, p in enumerate(PERIODS)}
order_d = {k: i for i, k in enumerate(DATASETS)}


def qtable(var, fmt):
    t = qa[qa.variable == var].pivot_table(index=["dataset", "period"], columns="quantile",
                                           values="change")[[f"Q{q}" for q in QS]]
    t = t.reindex(sorted(t.index, key=lambda i: (order_d[i[0]], order_p[i[1]])))
    return t.round(fmt)


tq = qtable("tmean", 1)
pq = qtable("precip", 2)
wdi = wd.set_index(["dataset", "period"])
pq["wet-day freq first (%)"] = wdi["wetfreq_first_pct"].round(1)
pq["wet-day freq last (%)"] = wdi["wetfreq_last_pct"].round(1)
pq["wet-day freq change (pp)"] = wdi["wetfreq_change_pct_points"].round(1)
pq["wet-day intensity change (mm)"] = wdi["intensity_change_mm"].round(2)
tq.to_csv(os.path.join(OUT, "P5_table_quantiles_T.csv"))
pq.to_csv(os.path.join(OUT, "P5_table_quantiles_P.csv"))

# ------------------------------------------------------------ figures
# Fig 1: seasonal cycles first vs last decade (rows: T, P; cols: datasets)
fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharex=True)
x = np.arange(1, 13)
for j, ds in enumerate(DATASETS):
    c = w[w.dataset == ds].sort_values("month")
    a = axes[0, j]
    a.plot(x, c.T_first, "-o", ms=3, color="#7f8c8d", label="1961-1970")
    a.plot(x, c.T_last, "-o", ms=3, color=COLORS[ds] if ds != "Station" else "#d35400",
           label="1981-1990")
    a2 = a.twinx()
    a2.bar(x, c.dT, color="#f5b041", alpha=0.35, width=0.6)
    a2.axhline(0, color="grey", lw=0.6)
    a2.set_ylim(-4, 4)
    if j == 2:
        a2.set_ylabel("ΔT last − first (°C, bars)")
    a.set_title(f"{ds}: mean temperature", fontsize=10)
    a.grid(alpha=0.3)
    if j == 0:
        a.set_ylabel("Monthly mean temperature (°C)")
    a.legend(fontsize=8, loc="upper left")

    b = axes[1, j]
    b.bar(x - 0.2, c.P_first, width=0.4, color="#95a5a6", label="1961-1970")
    b.bar(x + 0.2, c.P_last, width=0.4, color=COLORS[ds] if ds != "Station" else "#d35400",
          label="1981-1990")
    b.set_title(f"{ds}: mean monthly precipitation", fontsize=10)
    b.set_xticks(x, MONTHS, rotation=45)
    b.grid(alpha=0.3, axis="y")
    if j == 0:
        b.set_ylabel("Mean monthly total (mm)")
    b.legend(fontsize=8, loc="upper left")
fig.suptitle("Yellowknife: seasonal cycle in the first (1961-1970) and last (1981-1990) decade",
             fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "P5_fig_cycles.png"), dpi=200)

# Fig 2: quantile changes (rows: T, P; cols: Annual, JFM, JAS) + wet-day freq annotation
fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
for j, pname in enumerate(PERIODS):
    for i, (var, ylab) in enumerate([("tmean", "Change in daily Tmean quantile (°C)"),
                                     ("precip", "Change in daily precip quantile (mm)")]):
        ax = axes[i, j]
        for ds in DATASETS:
            t = qa[(qa.dataset == ds) & (qa.period == pname) & (qa.variable == var)]
            lab = ds
            if var == "precip":
                r = wd[(wd.dataset == ds) & (wd.period == pname)].iloc[0]
                lab = (f"{ds} (wet days {r.wetfreq_first_pct:.0f}%→"
                       f"{r.wetfreq_last_pct:.0f}%)")
            ax.plot(t.q, t.change, "-o", ms=4, color=COLORS[ds], label=lab)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xticks(QS, [f"Q{q}" for q in QS])
        ax.grid(alpha=0.3)
        ax.set_title(f"{pname}: {'daily mean temperature' if var == 'tmean' else 'daily precipitation (all days)'}",
                     fontsize=10)
        if j == 0:
            ax.set_ylabel(ylab)
        ax.legend(fontsize=7.5)
fig.suptitle("Change in empirical quantiles of daily values, 1981-1990 minus 1961-1970 "
             "(pooled daily values per decade)", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "P5_fig_quantile_changes.png"), dpi=200)

# ------------------------------------------------------------ terminal
pd.set_option("display.width", 220)
print("A. Monthly cycle change (last - first decade):")
print(cyc_tab.to_string(index=False))
print("\nSeasonal precipitation fractions (%), first -> last decade:")
print(fr_tab.pivot(index="dataset", columns="season",
                   values="change_pct_points")[SEASONS].loc[DATASETS].to_string())
print(fr_tab.to_string(index=False))
print("\nB. Daily Tmean quantile change (degC), last - first:")
print(tq.to_string())
print("\nB. Daily precipitation quantile change (mm, all days) + wet-day frequency:")
print(pq.to_string())
print("\nC. Wet days (>= 1 mm): frequency, intensity, wet-day Q95:")
cols = ["dataset", "period", "wetfreq_first_pct", "wetfreq_last_pct", "zero_days_first_pct",
        "intensity_first_mm", "intensity_last_mm", "wetQ95_first", "wetQ95_last", "wetQ95_change_mm"]
print(wd[cols].round(2).to_string(index=False))
print("\nSaved P5 tables and figures in ./output/")

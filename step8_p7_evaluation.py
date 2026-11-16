"""
ENGR 6666 - Assignment 1 - P7: Differences among station, reanalysis and GCM
Yellowknife A (2204100) | 1961-1990 | reference = station

1. Mean climate and variability: bias (dataset - station) and SD ratio
   for seasonal and annual mean temperature and precipitation totals
2. Seasonality: correlation / RMSE of the 12-month cycle, amplitude,
   wettest month, seasonal-fraction dissimilarity
3. Distributions (pooled daily values 1961-1990): quantile differences,
   lower tail (Q5, Q10) vs upper tail (Q90, Q95); dry / light / wet days,
   wet-day intensity, wet-day Q95, annual maximum daily precipitation (Rx1day)
4. Trends: direction and magnitude agreement with the station (from P4)
5. Time-aligned evaluation (meaningful for NCEP only): daily and monthly
   anomaly correlations, RMSE, wet-day hit rate / false alarms, lag test
   (-1, 0, +1 day) for the UTC day definition. The same statistics are computed
   for CanESM5 to show that a free-running GCM is not synchronized with reality.

Inputs : output/all_daily_1961-1990.csv, P2_series_*.csv, P3_seasonal_fractions.csv,
         P4_results_all.csv, P4_series.csv
Outputs: output/P7_*.csv, output/P7_fig_evaluation.png
Run:     python3 step8_p7_evaluation.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

OUT = "output"
DS = ["Station", "NCEP", "CanESM5"]
GR = ["NCEP", "CanESM5"]
COLORS = {"Station": "black", "NCEP": "#1f5fbf", "CanESM5": "#c0392b"}
SEASONS = ["JFM", "AMJ", "JAS", "OND"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
PERQ = {"Annual": list(range(1, 13)), "JFM": [1, 2, 3], "JAS": [7, 8, 9]}
QS = [5, 10, 25, 50, 75, 90, 95]

d = pd.read_csv(os.path.join(OUT, "all_daily_1961-1990.csv"), parse_dates=["date"])
mon = pd.read_csv(os.path.join(OUT, "P2_series_monthly.csv"))
sea = pd.read_csv(os.path.join(OUT, "P2_series_seasonal.csv"))
ann = pd.read_csv(os.path.join(OUT, "P2_series_annual.csv"))
fr3 = pd.read_csv(os.path.join(OUT, "P3_seasonal_fractions.csv"), index_col=0)
r4 = pd.read_csv(os.path.join(OUT, "P4_results_all.csv"))
s4 = pd.read_csv(os.path.join(OUT, "P4_series.csv"))
pd.set_option("display.width", 220)


# ------------------------------------------- 1. mean climate and variability
def period_series(ds, p, var):
    if p == "Annual":
        return ann[ann.dataset == ds][var]
    return sea[(sea.dataset == ds) & (sea.season == p)][var]


rows_t, rows_p = [], []
for p in SEASONS + ["Annual"]:
    st_t, st_p = period_series("Station", p, "tmean"), period_series("Station", p, "precip_total")
    rt = {"period": p, "Station mean (degC)": st_t.mean(), "Station SD": st_t.std(ddof=1)}
    rp = {"period": p, "Station mean (mm)": st_p.mean(), "Station SD": st_p.std(ddof=1)}
    for g in GR:
        gt, gp = period_series(g, p, "tmean"), period_series(g, p, "precip_total")
        rt[f"{g} bias (degC)"] = gt.mean() - st_t.mean()
        rt[f"{g} SD ratio"] = gt.std(ddof=1) / st_t.std(ddof=1)
        rp[f"{g} bias (%)"] = (gp.mean() / st_p.mean() - 1) * 100
        rp[f"{g} SD ratio"] = gp.std(ddof=1) / st_p.std(ddof=1)
    rows_t.append(rt)
    rows_p.append(rp)
mv_t, mv_p = pd.DataFrame(rows_t).round(2), pd.DataFrame(rows_p).round(2)
mv_t.to_csv(os.path.join(OUT, "P7_mean_variability_T.csv"), index=False)
mv_p.to_csv(os.path.join(OUT, "P7_mean_variability_P.csv"), index=False)

# ------------------------------------------------------------- 2. seasonality
clim = mon.groupby(["dataset", "month"])[["tmean", "precip_total"]].mean()
srows = []
st_c = clim.loc["Station"]
for g in GR:
    c = clim.loc[g]
    srows.append({
        "dataset": g,
        "T cycle correlation": stats.pearsonr(st_c.tmean, c.tmean)[0],
        "T cycle RMSE (degC)": np.sqrt(((c.tmean - st_c.tmean) ** 2).mean()),
        "T amplitude: dataset - station (degC)": (c.tmean.max() - c.tmean.min())
                                               - (st_c.tmean.max() - st_c.tmean.min()),
        "P cycle correlation": stats.pearsonr(st_c.precip_total, c.precip_total)[0],
        "P cycle RMSE (mm/month)": np.sqrt(((c.precip_total - st_c.precip_total) ** 2).mean()),
        "wettest month (station: " + MONTHS[st_c.precip_total.idxmax() - 1] + ")":
            MONTHS[c.precip_total.idxmax() - 1],
        "seasonal fraction dissimilarity (pp)": 0.5 * (fr3.loc[g, SEASONS] - fr3.loc["Station", SEASONS]).abs().sum(),
    })
seas = pd.DataFrame(srows).round(2)
seas.to_csv(os.path.join(OUT, "P7_seasonality.csv"), index=False)

# ------------------------------------------------------------ 3. distributions
qrows = []
for p, months in PERQ.items():
    for var in ["tmean", "precip"]:
        ref = np.percentile(d[(d.dataset == "Station") & d.month.isin(months)][var], QS)
        for ds in DS:
            q = np.percentile(d[(d.dataset == ds) & d.month.isin(months)][var], QS)
            r = {"period": p, "variable": var, "dataset": ds}
            for k, qq in enumerate(QS):
                r[f"Q{qq}"] = q[k] if ds == "Station" else q[k] - ref[k]
            qrows.append(r)
qd = pd.DataFrame(qrows)
qd["values"] = np.where(qd.dataset == "Station", "station value", "difference vs station")
qd.round(2).to_csv(os.path.join(OUT, "P7_distribution_quantiles.csv"), index=False)

wrows = []
rx = s4[s4.period == "Annual"]
for p, months in PERQ.items():
    for ds in DS:
        x = d[(d.dataset == ds) & d.month.isin(months)].precip
        wet = x[x >= 1]
        r = {"period": p, "dataset": ds,
             "dry days P=0 (%)": (x == 0).mean() * 100,
             "light days 0<P<1 mm (%)": ((x > 0) & (x < 1)).mean() * 100,
             "wet days P>=1 mm (%)": (x >= 1).mean() * 100,
             "wet days per year": (x >= 1).sum() / 30,
             "wet-day intensity (mm)": wet.mean(),
             "wet-day Q50 (mm)": np.percentile(wet, 50),
             "wet-day Q95 (mm)": np.percentile(wet, 95),
             "all-day Q99 (mm)": np.percentile(x, 99)}
        if p == "Annual":
            z = rx[rx.dataset == ds].pmax
            r["Rx1day mean (mm)"] = z.mean()
            r["Rx1day max (mm)"] = z.max()
        wrows.append(r)
wd = pd.DataFrame(wrows).round(2)
wd.to_csv(os.path.join(OUT, "P7_wetday_stats.csv"), index=False)

# ------------------------------------------------------------------- 4. trends
trows = []
for var in r4.variable.unique():
    st = r4[(r4.dataset == "Station") & (r4.variable == var)].set_index("period")
    for g in GR:
        x = r4[(r4.dataset == g) & (r4.variable == var)].set_index("period").loc[st.index]
        same = (np.sign(x.sen_slope_per_decade) == np.sign(st.sen_slope_per_decade)).sum()
        sig_agree = ((x.MK_p < 0.10) == (st.MK_p < 0.10)).sum()
        trows.append({"variable": var, "dataset": g,
                      "same direction as station (of 7)": int(same),
                      "same significance at 0.10 (of 7)": int(sig_agree),
                      "mean |slope difference| per decade": (x.sen_slope_per_decade
                                                             - st.sen_slope_per_decade).abs().mean(),
                      "significant at 0.05 (of 7)": int((x.MK_p < 0.05).sum()),
                      "station significant at 0.05 (of 7)": int((st.MK_p < 0.05).sum())})
tr = pd.DataFrame(trows).round(2)
tr.to_csv(os.path.join(OUT, "P7_trend_agreement.csv"), index=False)


# ----------------------------------------------------- 5. time-aligned evaluation
def daily_anom(x):
    key = [x.index.month, x.index.day]
    return x - x.groupby(key).transform("mean")


full = pd.date_range("1961-01-01", "1990-12-31", freq="D")
st = d[d.dataset == "Station"].set_index("date").reindex(full)
st_ta = daily_anom(st.tmean)
mon_a = mon.copy()
for v in ["tmean", "precip_total"]:
    mon_a[v + "_anom"] = mon_a[v] - mon_a.groupby(["dataset", "month"])[v].transform("mean")
ma = mon_a.set_index(["dataset", "year", "month"])

arows = []
for g in GR:
    x = d[d.dataset == g].set_index("date").reindex(full)       # CanESM5: Feb 29 -> NaN
    ok = x.tmean.notna() & st.tmean.notna()
    r = {"dataset": g, "evaluation type": "time-aligned (valid)" if g == "NCEP"
         else "time-aligned (NOT meaningful: free-running GCM)"}
    r["daily Tmean anomaly r"] = stats.pearsonr(st_ta[ok], daily_anom(x.tmean)[ok])[0]
    r["daily Tmean RMSE (degC)"] = np.sqrt(((x.tmean - st.tmean)[ok] ** 2).mean())
    r["daily Tmax RMSE (degC)"] = np.sqrt(((x.tmax - st.tmax)[ok] ** 2).mean())
    r["daily Tmin RMSE (degC)"] = np.sqrt(((x.tmin - st.tmin)[ok] ** 2).mean())
    best = None
    for lag in [-1, 0, 1]:
        xp = x.precip.shift(lag)          # lag +1: dataset day d compared with station day d+1
        okp = xp.notna() & st.precip.notna()
        rho = stats.spearmanr(st.precip[okp], xp[okp])[0]
        r[f"daily precip Spearman r (lag {lag:+d})"] = rho
        if best is None or rho > best[1]:
            best = (lag, rho)
    r["best lag (days)"] = best[0]
    for lag in sorted({0, best[0]}):
        xp = x.precip.shift(lag)
        okp = xp.notna() & st.precip.notna()
        sw, gw = st.precip[okp] >= 1, xp[okp] >= 1
        hits, misses, fa = (sw & gw).sum(), (sw & ~gw).sum(), (~sw & gw).sum()
        r[f"wet-day POD (lag {lag:+d})"] = hits / (hits + misses)
        r[f"wet-day false alarm ratio (lag {lag:+d})"] = fa / (hits + fa)
    a, b = ma.loc["Station"], ma.loc[g]
    idx = a.index.intersection(b.index)
    r["monthly T anomaly r"] = stats.pearsonr(a.loc[idx, "tmean_anom"], b.loc[idx, "tmean_anom"])[0]
    r["monthly P anomaly r"] = stats.pearsonr(a.loc[idx, "precip_total_anom"],
                                              b.loc[idx, "precip_total_anom"])[0]
    aa, bb = ann[ann.dataset == "Station"].set_index("year"), ann[ann.dataset == g].set_index("year")
    r["annual T r"] = stats.pearsonr(aa.tmean, bb.loc[aa.index].tmean)[0]
    r["annual P r"] = stats.pearsonr(aa.precip_total, bb.loc[aa.index].precip_total)[0]
    arows.append(r)
ta = pd.DataFrame(arows).round(3)
ta.to_csv(os.path.join(OUT, "P7_time_aligned.csv"), index=False)

# ------------------------------------------------------------------- figure
fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
pc = np.arange(1, 100)
q_st_t = np.percentile(d[d.dataset == "Station"].tmean, pc)
ax = axes[0, 0]
for g in GR:
    ax.plot(q_st_t, np.percentile(d[d.dataset == g].tmean, pc), "o", ms=3, color=COLORS[g], label=g)
lim = [q_st_t.min() - 3, q_st_t.max() + 3]
ax.plot(lim, lim, "k--", lw=1, label="1:1")
ax.set_xlabel("Station daily Tmean quantile (°C)")
ax.set_ylabel("Gridded daily Tmean quantile (°C)")
ax.set_title("(a) Q-Q plot, daily mean temperature (Q1-Q99)", fontsize=10)
ax.grid(alpha=0.3); ax.legend(fontsize=8)

pp = np.array([70, 75, 80, 85, 90, 92.5, 95, 97, 98, 99, 99.5, 99.9])
q_st_p = np.percentile(d[d.dataset == "Station"].precip, pp)
ax = axes[0, 1]
for g in GR:
    qg = np.percentile(d[d.dataset == g].precip, pp)
    ax.plot(q_st_p, qg, "o-", ms=4, color=COLORS[g], label=g)
    for k in [4, 6, 9, 11]:
        ax.annotate(f"Q{pp[k]:g}", (q_st_p[k], qg[k]), fontsize=7, xytext=(3, 3),
                    textcoords="offset points", color=COLORS[g])
m = max(q_st_p.max(), 1) * 1.1
ax.plot([0, m], [0, m], "k--", lw=1, label="1:1")
ax.set_xlabel("Station daily precipitation quantile (mm)")
ax.set_ylabel("Gridded daily precipitation quantile (mm)")
ax.set_title("(b) Q-Q plot, daily precipitation, all days (Q70-Q99.9)", fontsize=10)
ax.grid(alpha=0.3); ax.legend(fontsize=8)

ax = axes[0, 2]
w = wd[wd.period == "Annual"].set_index("dataset").loc[DS]
b1 = w["dry days P=0 (%)"].values
b2 = w["light days 0<P<1 mm (%)"].values
b3 = w["wet days P>=1 mm (%)"].values
ax.bar(DS, b1, color="#f0e6d2", label="dry (0 mm)")
ax.bar(DS, b2, bottom=b1, color="#a9cce3", label="light (0-1 mm)")
ax.bar(DS, b3, bottom=b1 + b2, color="#1f618d", label="wet (≥1 mm)")
for j, ds in enumerate(DS):
    ax.text(j, 102, f"{w.loc[ds, 'wet-day intensity (mm)']:.1f} mm/wet day", ha="center", fontsize=8)
ax.set_ylim(0, 110)
ax.set_ylabel("Share of days (%)")
ax.set_title("(c) Dry, light and wet days (annual) and wet-day intensity", fontsize=10)
ax.legend(fontsize=8, loc="lower right")

for ax, v, lab, ttl in [(axes[1, 0], "tmean_anom", "°C", "(d) Monthly temperature anomalies"),
                        (axes[1, 1], "precip_total_anom", "mm", "(e) Monthly precipitation anomalies")]:
    a = ma.loc["Station"]
    for g in GR:
        b = ma.loc[g]
        idx = a.index.intersection(b.index)
        rr = stats.pearsonr(a.loc[idx, v], b.loc[idx, v])[0]
        ax.plot(a.loc[idx, v], b.loc[idx, v], "o", ms=2.5, alpha=0.5, color=COLORS[g],
                label=f"{g} (r = {rr:.2f})")
    ax.axhline(0, color="grey", lw=0.6); ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlabel(f"Station anomaly ({lab})"); ax.set_ylabel(f"Gridded anomaly ({lab})")
    ax.set_title(ttl + ", same month and year", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

ax = axes[1, 2]
per = ["Feb", "May", "Aug", "Oct", "JFM", "JAS", "Annual"]
xx = np.arange(len(per))
for k, ds in enumerate(DS):
    t = r4[(r4.dataset == ds) & r4.variable.str.startswith("Mean")].set_index("period").loc[per]
    bars = ax.bar(xx + (k - 1) * 0.27, t.sen_slope_per_decade, width=0.27, color=COLORS[ds], label=ds)
    for bx, pv in zip(bars, t.MK_p):
        if pv < 0.05:
            ax.text(bx.get_x() + bx.get_width() / 2, bx.get_height() + (0.05 if bx.get_height() >= 0 else -0.25),
                    "**", ha="center", fontsize=8)
ax.axhline(0, color="grey", lw=0.8)
ax.set_xticks(xx, per)
ax.set_ylabel("Sen's slope (°C/decade)")
ax.set_title("(f) Temperature trends 1961-1990 (** p < 0.05)", fontsize=10)
ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)

fig.suptitle("Yellowknife 1961-1990: evaluation of NCEP-NCAR R1 and CanESM5 (NEX-GDDP) against "
             "the station (ECCC 2204100)", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "P7_fig_evaluation.png"), dpi=200)

# ------------------------------------------------------------------ terminal
print("1. MEAN AND VARIABILITY - temperature")
print(mv_t.to_string(index=False))
print("\n1. MEAN AND VARIABILITY - precipitation")
print(mv_p.to_string(index=False))
print("\n2. SEASONALITY")
print(seas.to_string(index=False))
print("\n3. QUANTILES (station value, then dataset minus station)")
for var in ["tmean", "precip"]:
    print(f"\n  {var}:")
    print(qd[qd.variable == var].drop(columns=["variable", "values"]).round(2).to_string(index=False))
print("\n3. DRY / LIGHT / WET DAYS AND INTENSITY")
print(wd.to_string(index=False))
print("\n4. TREND AGREEMENT WITH STATION")
print(tr.to_string(index=False))
print("\n5. TIME-ALIGNED EVALUATION")
print(ta.T.to_string())
print("\nSaved P7 tables and figure in ./output/")

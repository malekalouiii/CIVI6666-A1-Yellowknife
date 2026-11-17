"""
CIVI 6666 - Assignment 1 - P3: Climatological seasonality
Yellowknife A (2204100) | 1961-1990 | Station, NCEP-NCAR R1, CanESM5 (NEX-GDDP)

  - 12 monthly climatological mean temperatures (mean over 30 years of monthly means)
  - 12 mean monthly precipitation totals (mean over 30 years of monthly totals)
  - warmest / coldest and wettest / driest months
  - fraction of mean annual precipitation falling in JFM, AMJ, JAS, OND
    (= mean seasonal total / mean annual total x 100; the four add up to 100 %)

Input : output/P2_series_monthly.csv, output/P2_series_seasonal.csv,
        output/P2_series_annual.csv   (from the P2 script; 90 % rule already applied)
Outputs (output/):
  P3_monthly_climatology.csv
  P3_seasonal_fractions.csv
  P3_fig_annual_cycle.png

Run:  python3 step5_p3_seasonality.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = "output"
DATASETS = ["Station", "NCEP", "CanESM5"]
COLORS = {"Station": "black", "NCEP": "#1f5fbf", "CanESM5": "#c0392b"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SEASONS = ["JFM", "AMJ", "JAS", "OND"]
SEASON_NAME = {"JFM": "JFM (winter)", "AMJ": "AMJ (spring)",
               "JAS": "JAS (summer)", "OND": "OND (fall)"}
SEASON_COL = {"JFM": "#9ecae1", "AMJ": "#a1d99b", "JAS": "#fdae6b", "OND": "#bcbddc"}

mon = pd.read_csv(os.path.join(OUT, "P2_series_monthly.csv"))
sea = pd.read_csv(os.path.join(OUT, "P2_series_seasonal.csv"))
ann = pd.read_csv(os.path.join(OUT, "P2_series_annual.csv"))

# ------------------------------------------------- 1. monthly climatology
clim = (mon.groupby(["dataset", "month"])
        .agg(tmean=("tmean", "mean"), precip=("precip_total", "mean"),
             n_years_t=("tmean", "count"), n_years_p=("precip_total", "count"))
        .reset_index())

tab = pd.DataFrame({"month": MONTHS})
for ds in DATASETS:
    c = clim[clim.dataset == ds].sort_values("month")
    tab[f"{ds} Tmean (degC)"] = c["tmean"].round(1).values
for ds in DATASETS:
    c = clim[clim.dataset == ds].sort_values("month")
    tab[f"{ds} precip (mm)"] = c["precip"].round(1).values
# annual row: mean of monthly temperatures / sum of monthly precipitation
annual_row = {"month": "Annual"}
for ds in DATASETS:
    c = clim[clim.dataset == ds]
    annual_row[f"{ds} Tmean (degC)"] = round(c["tmean"].mean(), 1)
    annual_row[f"{ds} precip (mm)"] = round(c["precip"].sum(), 1)
tab = pd.concat([tab, pd.DataFrame([annual_row])], ignore_index=True)
tab.to_csv(os.path.join(OUT, "P3_monthly_climatology.csv"), index=False)

# ------------------------------------------------- 2. extremes of the cycle
ext_rows = []
for ds in DATASETS:
    c = clim[clim.dataset == ds].set_index("month")
    ext_rows.append({
        "dataset": ds,
        "warmest": f"{MONTHS[c.tmean.idxmax()-1]} ({c.tmean.max():.1f} degC)",
        "coldest": f"{MONTHS[c.tmean.idxmin()-1]} ({c.tmean.min():.1f} degC)",
        "annual T amplitude (degC)": round(c.tmean.max() - c.tmean.min(), 1),
        "wettest": f"{MONTHS[c.precip.idxmax()-1]} ({c.precip.max():.1f} mm)",
        "driest": f"{MONTHS[c.precip.idxmin()-1]} ({c.precip.min():.1f} mm)",
        "months below 0 degC": int((c.tmean < 0).sum()),
    })
ext = pd.DataFrame(ext_rows)

# ------------------------------------------------- 3. seasonal fractions
seas_mean = sea.groupby(["dataset", "season"])["precip_total"].mean().unstack()[SEASONS]
annual_mean = ann.groupby("dataset")["precip_total"].mean()
frac = seas_mean.div(seas_mean.sum(axis=1), axis=0) * 100   # sums to exactly 100 %
frac = frac.loc[DATASETS]
frac_out = frac.round(1).copy()
frac_out["mean annual total (mm)"] = annual_mean.loc[DATASETS].round(1)
for s in SEASONS:
    frac_out[f"{s} mean total (mm)"] = seas_mean.loc[DATASETS, s].round(1)
frac_out.index.name = "dataset"
frac_out.to_csv(os.path.join(OUT, "P3_seasonal_fractions.csv"))

# ------------------------------------------------------------ 4. figure
fig = plt.figure(figsize=(15, 5.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.25, 0.8])
ax1, ax2, ax3 = (fig.add_subplot(gs[0, i]) for i in range(3))
x = np.arange(1, 13)

for ds in DATASETS:
    c = clim[clim.dataset == ds].sort_values("month")
    ax1.plot(x, c.tmean, "-o", ms=4, color=COLORS[ds], label=ds)
ax1.axhline(0, color="grey", lw=0.8, ls="--")
ax1.set_xticks(x, MONTHS, rotation=45)
ax1.set_ylabel("Climatological mean temperature (°C)")
ax1.set_title("(a) Monthly mean temperature")
ax1.grid(alpha=0.3)
ax1.legend(fontsize=9)

w = 0.27
for i, ds in enumerate(DATASETS):
    c = clim[clim.dataset == ds].sort_values("month")
    ax2.bar(x + (i - 1) * w, c.precip, width=w, color=COLORS[ds], label=ds,
            alpha=0.85 if ds != "Station" else 1.0)
ax2.set_xticks(x, MONTHS, rotation=45)
ax2.set_ylabel("Mean monthly precipitation total (mm)")
ax2.set_title("(b) Mean monthly precipitation")
ax2.grid(alpha=0.3, axis="y")
ax2.legend(fontsize=9)

bottom = np.zeros(len(DATASETS))
for s in SEASONS:
    vals = frac.loc[DATASETS, s].values
    ax3.bar(DATASETS, vals, bottom=bottom, color=SEASON_COL[s], edgecolor="white",
            label=SEASON_NAME[s])
    for j, v in enumerate(vals):
        ax3.text(j, bottom[j] + v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=9)
    bottom += vals
ax3.set_ylim(0, 100)
ax3.set_ylabel("Share of mean annual precipitation (%)")
ax3.set_title("(c) Seasonal precipitation fractions")
ax3.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)

fig.suptitle("Yellowknife, 1961-1990: climatological annual cycle from station (ECCC 2204100), "
             "NCEP-NCAR R1 and CanESM5 (NEX-GDDP)", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "P3_fig_annual_cycle.png"), dpi=200, bbox_inches="tight")

# ------------------------------------------------------------ 5. terminal
pd.set_option("display.width", 200)
print("Monthly climatology 1961-1990:")
print(tab.to_string(index=False))
print("\nWarmest / coldest / wettest / driest months:")
print(ext.to_string(index=False))
print("\nSeasonal fractions of mean annual precipitation (%):")
print(frac_out.to_string())
print("\nSaved P3 tables and figure in ./output/")

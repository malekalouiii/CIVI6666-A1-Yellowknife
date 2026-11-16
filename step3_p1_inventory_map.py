"""
ENGR 6666 - Assignment 1 - Step 3  (completes P1)
Yellowknife A (ECCC 2204100) | window 1961-1990

Inputs (from Steps 1 and 2, in ./output/):
  station_daily_2204100.csv
  ncep_daily_1961-1990.csv
  canesm5_daily_1961-1990.csv
  grid_cells.csv

Outputs (in ./output/):
  all_daily_1961-1990.csv          the three datasets in ONE tidy file (used by P2-P5)
  data_inventory.csv               sources, variables, units, conversions, grid, % missing
  missing_by_year_1961-1990.csv    missing days per year, variable and dataset
  P1_map.png                       station + NCEP cell + CanESM5 cell over a map

Requirements:  pip3 install pandas numpy matplotlib cartopy
Run:           python3 step3_p1_inventory_map.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = "output"
Y0, Y1 = 1961, 1990
STN_LAT, STN_LON = 62.46, -114.44
SEASONS = {1: "JFM", 2: "JFM", 3: "JFM", 4: "AMJ", 5: "AMJ", 6: "AMJ",
           7: "JAS", 8: "JAS", 9: "JAS", 10: "OND", 11: "OND", 12: "OND"}
VARS = ["tmax", "tmin", "tmean", "precip"]


# ------------------------------------------------------------- 1. load + merge
def load(fname, name):
    df = pd.read_csv(os.path.join(OUT, fname), index_col=0, parse_dates=True)
    df = df.loc[f"{Y0}-01-01":f"{Y1}-12-31", VARS].copy()
    df["dataset"] = name
    return df


st = load("station_daily_2204100.csv", "Station")
nc = load(f"ncep_daily_{Y0}-{Y1}.csv", "NCEP")
cm = load(f"canesm5_daily_{Y0}-{Y1}.csv", "CanESM5")

# Station: guarantee every calendar day is present (absent day = missing, not 0)
full = pd.date_range(f"{Y0}-01-01", f"{Y1}-12-31", freq="D")
st = st.reindex(full)
st["dataset"] = "Station"

alld = pd.concat([st, nc, cm])
alld.index.name = "date"
alld["year"] = alld.index.year
alld["month"] = alld.index.month
alld["season"] = alld["month"].map(SEASONS)
alld = alld.reset_index()[["dataset", "date", "year", "month", "season"] + VARS]
alld.to_csv(os.path.join(OUT, f"all_daily_{Y0}-{Y1}.csv"), index=False)

# ------------------------------------------------- 2. missing days by year
miss = (alld.groupby(["dataset", "year"])[["tmean", "precip"]]
        .apply(lambda g: g.isna().sum())
        .unstack("dataset"))
miss.columns = [f"{d}_{v}_missing" for v, d in miss.columns]
miss = miss[sorted(miss.columns)]
miss.to_csv(os.path.join(OUT, f"missing_by_year_{Y0}-{Y1}.csv"))

# --------------------------------------------------------- 3. data inventory
cells = pd.read_csv(os.path.join(OUT, "grid_cells.csv"))
ncep_cell = cells[cells.dataset.str.contains("NCEP")].iloc[0]
cm_cell = cells[cells.dataset.str.contains("CanESM5")].iloc[0]

expected_days = {"Station": len(full), "NCEP": len(full),
                 "CanESM5": len(full) - int(((full.month == 2) & (full.day == 29)).sum())}

meta = {
    "Station": dict(
        source="ECCC historical climate archive (climate.weather.gc.ca), via MSC GeoMet API",
        variables="MAX_TEMPERATURE, MIN_TEMPERATURE, TOTAL_PRECIPITATION",
        original_units="degC, degC, mm/day",
        conversion="none; Tmean=(Tmax+Tmin)/2; trace (T) = 0 mm (valid)",
        calendar="standard (Gregorian)",
        lat=STN_LAT, lon=STN_LON, resolution="point", distance_km=0.0),
    "NCEP": dict(
        source="NCEP-NCAR Reanalysis 1, daily, NOAA PSL THREDDS (OPeNDAP)",
        variables="tmax.2m, tmin.2m, prate.sfc (Gaussian grid)",
        original_units="K, K, kg m-2 s-1",
        conversion="T-273.15; precip x86400 -> mm/day; Tmean=(Tmax+Tmin)/2",
        calendar="standard; days in UTC (local = UTC-7)",
        lat=ncep_cell.lat, lon=ncep_cell.lon,
        resolution="T62 Gaussian ~1.9 deg lat x 1.875 deg lon (~210 x 100 km)",
        distance_km=ncep_cell.distance_km),
    "CanESM5": dict(
        source="NEX-GDDP-CMIP6 CanESM5 historical r1i1p1f1, NASA NCCS THREDDS (OPeNDAP)",
        variables="tasmax, tasmin, pr (newest file version)",
        original_units="K, K, kg m-2 s-1",
        conversion="T-273.15; precip x86400 -> mm/day; Tmean=(Tmax+Tmin)/2",
        calendar="365-day (no Feb 29)",
        lat=cm_cell.lat, lon=cm_cell.lon,
        resolution="0.25 x 0.25 deg (~28 x 13 km)",
        distance_km=cm_cell.distance_km),
}

inv = []
for name, m in meta.items():
    d = alld[alld.dataset == name]
    inv.append({"dataset": name, **m, "period": f"{Y0}-{Y1}",
                "days_present": int(len(d)), "days_expected": expected_days[name],
                "missing_tmean_days": int(d.tmean.isna().sum()),
                "missing_precip_days": int(d.precip.isna().sum()),
                "pct_missing_tmean": round(d.tmean.isna().mean() * 100, 2),
                "pct_missing_precip": round(d.precip.isna().mean() * 100, 2)})
inv = pd.DataFrame(inv)
inv.to_csv(os.path.join(OUT, "data_inventory.csv"), index=False)

# ------------------------------------------------------------------ 4. map
def ncep_bounds(clat, clon):
    # Gaussian T62: neighbour latitudes around 61.9033 are 63.8079 (N) and 59.9986 (S)
    north = (63.8079 + clat) / 2 if abs(clat - 61.9033) < 0.01 else clat + 0.95
    south = (clat + 59.9986) / 2 if abs(clat - 61.9033) < 0.01 else clat - 0.95
    return south, north, clon - 0.9375, clon + 0.9375


def nex_bounds(clat, clon):
    return clat - 0.125, clat + 0.125, clon - 0.125, clon + 0.125


nb = ncep_bounds(ncep_cell.lat, ncep_cell.lon)
cb = nex_bounds(cm_cell.lat, cm_cell.lon)

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAVE_CARTOPY = True
except ImportError:
    HAVE_CARTOPY = False
    print("cartopy not installed: drawing the map without a background.")

fig = plt.figure(figsize=(13, 6))
panels = [("(a) Regional view", (-116.6, -112.2, 60.4, 63.3)),
          ("(b) Zoom on CanESM5 cell", (-114.62, -114.13, 62.20, 62.55))]

for i, (title, ext) in enumerate(panels, start=1):
    if HAVE_CARTOPY:
        ax = fig.add_subplot(1, 2, i, projection=ccrs.PlateCarree())
        ax.set_extent(ext, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.NaturalEarthFeature("physical", "land", "10m"),
                       facecolor="#f2efe6", edgecolor="none")
        ax.add_feature(cfeature.NaturalEarthFeature("physical", "lakes", "10m"),
                       facecolor="#cfe3f2", edgecolor="#6b9cc2", linewidth=0.5)
        ax.add_feature(cfeature.NaturalEarthFeature("physical", "rivers_lake_centerlines", "10m"),
                       facecolor="none", edgecolor="#6b9cc2", linewidth=0.4)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="grey", alpha=0.6)
        gl.top_labels = gl.right_labels = False
        tr = dict(transform=ccrs.PlateCarree())
    else:
        ax = fig.add_subplot(1, 2, i)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_xlabel("Longitude (deg E)"); ax.set_ylabel("Latitude (deg N)")
        ax.grid(alpha=0.3)
        tr = {}

    # zorder > map layers, otherwise the land/lake fills are painted over the squares
    ax.add_patch(Rectangle((nb[2], nb[0]), nb[3] - nb[2], nb[1] - nb[0],
                           fill=False, edgecolor="#1f5fbf", linewidth=2.5, zorder=10,
                           label=f"NCEP cell (~1.9 deg; centre {ncep_cell.distance_km:.0f} km away)", **tr))
    ax.add_patch(Rectangle((cb[2], cb[0]), cb[3] - cb[2], cb[1] - cb[0],
                           fill=False, edgecolor="#c0392b", linewidth=2.5, zorder=11,
                           label=f"CanESM5 NEX-GDDP cell (0.25 deg; centre {cm_cell.distance_km:.0f} km away)", **tr))
    ax.plot(ncep_cell.lon, ncep_cell.lat, "x", color="#1f5fbf", ms=9, mew=2, zorder=12, **tr)
    ax.plot(cm_cell.lon, cm_cell.lat, "x", color="#c0392b", ms=9, mew=2, zorder=12, **tr)
    ax.plot(STN_LON, STN_LAT, "*", color="black", ms=15, zorder=13,
            label="ECCC station 2204100 (Yellowknife A)", **tr)
    ax.set_title(title, fontsize=11)
    if i == 1:
        ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

fig.suptitle("Yellowknife A (62.46 N, 114.44 W): station location and representative grid cells "
             f"({Y0}-{Y1})", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "P1_map.png"), dpi=200)

# ---------------------------------------------------- 5. quick sanity summary
def summary(d):
    annual_p = d.groupby("year")["precip"].sum(min_count=int(0.9 * 365))
    return pd.Series({
        "mean Tmean (degC)": d.tmean.mean(),
        "mean Tmax (degC)": d.tmax.mean(),
        "mean Tmin (degC)": d.tmin.mean(),
        "mean annual precip (mm)": annual_p.mean(),
        "wet days >=1 mm (% of days)": (d.precip >= 1).mean() * 100,
        "days with precip = 0 (% of days)": (d.precip == 0).mean() * 100,
        "max daily precip (mm)": d.precip.max(),
    })


quick = alld.groupby("dataset").apply(summary).T[["Station", "NCEP", "CanESM5"]].round(1)
print("\nData inventory:")
print(inv[["dataset", "lat", "lon", "resolution", "distance_km", "days_present",
           "days_expected", "pct_missing_tmean", "pct_missing_precip"]].to_string(index=False))
print("\nQuick comparison 1961-1990 (sanity check, not the P2 analysis):")
print(quick.to_string())
print(f"\nSaved: all_daily_{Y0}-{Y1}.csv, data_inventory.csv, "
      f"missing_by_year_{Y0}-{Y1}.csv, P1_map.png  (in ./{OUT}/)")

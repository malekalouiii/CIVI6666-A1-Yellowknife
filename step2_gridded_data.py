"""
CIVI 6666 - Assignment 1 - Step 2
Gridded data for Yellowknife A (ECCC 2204100, 62.46 N, -114.44 E)

  (A) NCEP-NCAR Reanalysis 1, daily, T62 Gaussian grid (~1.875 deg)
      NOAA PSL THREDDS: Datasets/ncep.reanalysis/Dailies/surface_gauss/
      variables: tmax.2m, tmin.2m (K), prate.sfc (kg m-2 s-1)
  (B) NEX-GDDP-CMIP6, CanESM5, historical, r1i1p1f1, 0.25 deg
      NASA NCCS THREDDS: AMES/NEX/GDDP-CMIP6/CanESM5/historical/r1i1p1f1/
      variables: tasmax, tasmin (K), pr (kg m-2 s-1)

Only the nearest grid cell is requested (OPeNDAP point subset) - no global files
are downloaded.

Unit conversions
  K -> degC              : T - 273.15
  kg m-2 s-1 -> mm/day   : x 86400   (1 kg of water on 1 m2 = 1 mm; 86400 s/day)
Tmean = (Tmax + Tmin) / 2   (same rule as the station)

Requirements:  pip install xarray netCDF4 cftime pandas numpy requests
Run:           python step2_gridded_data.py
Outputs (in ./output/):
  ncep_daily_1961-1990.csv
  canesm5_daily_1961-1990.csv
  grid_cells.csv            grid-cell coordinates, resolution, distance to station
"""

import os
import re
import math
import requests
import numpy as np
import pandas as pd
import xarray as xr

# ------------------------------------------------------------------ settings
START_YEAR, END_YEAR = 1961, 1990          # <- your 30-year window
STN_LAT, STN_LON = 62.46, -114.44          # Yellowknife A
STN_LON360 = STN_LON % 360                 # 245.56 (grids use 0-360 E)
OUT = "output"
os.makedirs(OUT, exist_ok=True)
YEARS = range(START_YEAR, END_YEAR + 1)

NCEP_DAP = ("https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/"
            "Dailies/surface_gauss/{var}.gauss.{year}.nc")
NEX_ROOT = "AMES/NEX/GDDP-CMIP6/CanESM5/historical/r1i1p1f1"
NEX_CATALOG = "https://ds.nccs.nasa.gov/thredds/catalog/" + NEX_ROOT + "/{var}/catalog.xml"
NEX_DAP = "https://ds.nccs.nasa.gov/thredds/dodsC/" + NEX_ROOT + "/{var}/{fname}"


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def to_date_index(time_values):
    """Works for standard and model (365-day / cftime) calendars."""
    return pd.Index([f"{t.year:04d}-{t.month:02d}-{t.day:02d}" for t in
                     (pd.DatetimeIndex(time_values) if np.issubdtype(
                         np.asarray(time_values).dtype, np.datetime64)
                      else time_values)], name="date")


def point_series(url, varname):
    """Open one remote yearly file and return (series, grid_lat, grid_lon)."""
    with xr.open_dataset(url, decode_times=True) as ds:
        da = ds[varname]
        latn = "lat" if "lat" in da.dims else "latitude"
        lonn = "lon" if "lon" in da.dims else "longitude"
        p = da.sel({latn: STN_LAT, lonn: STN_LON360}, method="nearest").load()
        s = pd.Series(p.values.astype(float), index=to_date_index(p["time"].values))
        return s, float(p[latn]), float(p[lonn])


def fetch(label, years, url_for, nc_varname, cache):
    """Loop over years with a per-variable cache so the run can be resumed."""
    if os.path.exists(cache):
        print(f"  {label}: using cached {cache}")
        c = pd.read_csv(cache, index_col=0)
        return c.iloc[:, 0], c.attrs if hasattr(c, "attrs") else None
    parts, glat, glon = [], None, None
    for y in years:
        url = url_for(y)
        for attempt in range(3):
            try:
                s, glat, glon = point_series(url, nc_varname)
                break
            except Exception as e:
                if attempt == 2:
                    raise RuntimeError(f"Failed {label} {y}: {url}\n{e}")
        parts.append(s)
        print(f"  {label} {y}: {len(s)} days")
    out = pd.concat(parts)
    out = out[~out.index.duplicated()]
    out.to_frame(label).to_csv(cache)
    pd.Series({"lat": glat, "lon": glon}).to_csv(cache + ".cell")
    return out, None


def cell_of(cache):
    c = pd.read_csv(cache + ".cell", index_col=0).iloc[:, 0]
    return float(c["lat"]), float(c["lon"])


# ------------------------------------------------------ (A) NCEP reanalysis
print("NCEP-NCAR Reanalysis 1 ...")
ncep = {}
for v in ["tmax.2m", "tmin.2m", "prate.sfc"]:
    short = v.split(".")[0]                          # tmax / tmin / prate
    cache = os.path.join(OUT, f"_cache_ncep_{short}.csv")
    ncep[short], _ = fetch(f"ncep_{short}", YEARS,
                           lambda y, v=v: NCEP_DAP.format(var=v, year=y),
                           short, cache)
ncep_cell = cell_of(os.path.join(OUT, "_cache_ncep_tmax.csv"))

nc = pd.DataFrame({
    "tmax": ncep["tmax"] - 273.15,
    "tmin": ncep["tmin"] - 273.15,
    "precip": ncep["prate"] * 86400.0,
})
nc["tmean"] = (nc["tmax"] + nc["tmin"]) / 2
nc["precip"] = nc["precip"].clip(lower=0)            # tiny negative packing noise
nc.to_csv(os.path.join(OUT, f"ncep_daily_{START_YEAR}-{END_YEAR}.csv"))


# ------------------------------------------------------ (B) CanESM5 (NEX-GDDP)
def nex_file_for(var, year, _cache={}):
    """Pick the newest file version for a given year from the THREDDS catalog."""
    if var not in _cache:
        xml = requests.get(NEX_CATALOG.format(var=var), timeout=120).text
        _cache[var] = re.findall(r'name="([^"]+\.nc)"', xml)
    pat = re.compile(rf"^{var}_day_CanESM5_historical_r1i1p1f1_gn_{year}(_v[\d.]+)?\.nc$")
    hits = [f for f in _cache[var] if pat.match(f)]
    if not hits:
        raise FileNotFoundError(f"No {var} file for {year} in NCCS catalog")
    return sorted(hits, key=lambda f: (len(f), f))[-1]   # versioned > plain


print("\nCanESM5 historical (NEX-GDDP-CMIP6) ...")
cm = {}
for v in ["tasmax", "tasmin", "pr"]:
    cache = os.path.join(OUT, f"_cache_canesm5_{v}.csv")
    cm[v], _ = fetch(f"canesm5_{v}", YEARS,
                     lambda y, v=v: NEX_DAP.format(var=v, fname=nex_file_for(v, y)),
                     v, cache)
cm_cell = cell_of(os.path.join(OUT, "_cache_canesm5_tasmax.csv"))

gc = pd.DataFrame({
    "tmax": cm["tasmax"] - 273.15,
    "tmin": cm["tasmin"] - 273.15,
    "precip": cm["pr"] * 86400.0,
})
gc["tmean"] = (gc["tmax"] + gc["tmin"]) / 2
gc.to_csv(os.path.join(OUT, f"canesm5_daily_{START_YEAR}-{END_YEAR}.csv"))

# ------------------------------------------------------ grid-cell summary (P1)
def lon180(x):
    return x - 360 if x > 180 else x

cells = pd.DataFrame([
    {"dataset": "ECCC station 2204100", "lat": STN_LAT, "lon": STN_LON,
     "resolution": "point", "distance_km": 0.0},
    {"dataset": "NCEP-NCAR R1 (T62 Gaussian)", "lat": round(ncep_cell[0], 4),
     "lon": round(lon180(ncep_cell[1]), 4), "resolution": "~1.875 deg (~200 km)",
     "distance_km": round(haversine_km(STN_LAT, STN_LON, ncep_cell[0], lon180(ncep_cell[1])), 1)},
    {"dataset": "NEX-GDDP-CMIP6 CanESM5", "lat": round(cm_cell[0], 4),
     "lon": round(lon180(cm_cell[1]), 4), "resolution": "0.25 deg (~25 km)",
     "distance_km": round(haversine_km(STN_LAT, STN_LON, cm_cell[0], lon180(cm_cell[1])), 1)},
])
cells.to_csv(os.path.join(OUT, "grid_cells.csv"), index=False)

# ------------------------------------------------------------------ summary
print("\nGrid cells:")
print(cells.to_string(index=False))
for name, df in [("NCEP", nc), ("CanESM5", gc)]:
    print(f"\n{name}: {len(df)} days, {df.index.min()} to {df.index.max()}, "
          f"missing tmean={df['tmean'].isna().sum()}, precip={df['precip'].isna().sum()}")
    print(df[["tmax", "tmin", "tmean", "precip"]].describe().round(1))

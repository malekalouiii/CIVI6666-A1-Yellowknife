# CIVI 6666 – Assignment 1: Historical climate-change signals at Yellowknife, NT

Station: ECCC Yellowknife A (Climate ID 2204100; 62.46°N, 114.44°W). Analysis window: 1961–1990.

## Data inventory
| Dataset | Source | Variables | Grid cell | Units → conversion |
|---|---|---|---|---|
| Station | ECCC archive via MSC GeoMet API (api.weather.gc.ca, climate-daily) | Tmax, Tmin, total precipitation | Point | °C, mm → none |
| NCEP–NCAR Reanalysis 1 | NOAA PSL THREDDS (OPeNDAP), daily, T62 Gaussian grid | tmax.2m, tmin.2m, prate.sfc | 61.90°N, 114.375°W (index 14, 131); 62 km from station | K → −273.15; kg m⁻² s⁻¹ → ×86,400 |
| CanESM5 historical r1i1p1f1 | NEX-GDDP-CMIP6, NASA NCCS THREDDS (OPeNDAP), v2.0 | tasmax, tasmin, pr | 62.375°N, 114.375°W (index 489, 982); 10 km from station | K → −273.15; kg m⁻² s⁻¹ → ×86,400 |

Daily mean temperature = (Tmax + Tmin)/2 for all datasets. Missing values are never replaced with zero.
1961–1990 has 0% missing data in all three datasets (CanESM5 uses a 365-day calendar).

## How to run
Requires Python 3 and: `pip install pandas numpy requests xarray netCDF4 cftime matplotlib cartopy scipy pymannkendall`

Run in this order (each script reads the outputs of the previous ones from `output/`):
1. `step1_yellowknife_station_qc.py` – station download, completeness, window ranking (P1)
2. `step2_gridded_data.py` – NCEP and CanESM5 extraction, unit conversion (P1)
3. `step3_p1_inventory_map.py` – merged daily file, data inventory, map (P1)
4. `step4_p2_descriptive.py` – descriptive statistics (P2); `p2_minmax.py` – extremes and their years
5. `step5_p3_seasonality.py` – annual cycle, seasonal fractions (P3)
6. `step6_p4_trends.py` – Mann–Kendall and Sen's slope, 42 series (P4)
7. `step7_p5_distributions.py` – first vs last decade, quantiles, wet days (P5)
8. `step8_p7_evaluation.py` – dataset evaluation (P7)
9. `step9_p8_engineering.py` – freezing/thawing indices, Rx1day return levels (P8)

## Outputs
All tables and figures are in `output/`. Full monthly statistics: `output/P2_stats_monthly.csv`.

"""
CIVI 6666 - Assignment 1 - Step 1
Station: Yellowknife A, NT  |  ECCC Climate ID 2204100  |  62.46 N, -114.44 E
Permitted record: 1950-1992

What this script does
  1. Downloads daily Tmax, Tmin, total precipitation from the ECCC/MSC GeoMet API
  2. Computes Tmean = (Tmax + Tmin) / 2  (as required by the assignment)
  3. Counts missing days per year / month / season for each variable
  4. Ranks every candidate 30-year window (1950-1979 ... 1963-1992)
Missing values stay NaN -- they are NEVER replaced with zero.

Requirements:  pip install pandas requests
Run:           python step1_yellowknife_station_qc.py
Outputs (in ./output/):
  station_daily_2204100.csv        tidy daily data (all years)
  missing_by_year.csv              missing days per year and variable
  missing_by_month.csv             % complete per month (90% rule)
  missing_by_season.csv            % complete per season (90% / 95% rules)
  window_ranking.csv               all 30-yr windows ranked
"""

import os
import requests
import pandas as pd
import numpy as np

CLIMATE_ID = "2204100"
START, END = "1950-01-01", "1992-12-31"
WINDOW = 30
OUT = "output"
os.makedirs(OUT, exist_ok=True)

SEASONS = {1: "JFM", 2: "JFM", 3: "JFM", 4: "AMJ", 5: "AMJ", 6: "AMJ",
           7: "JAS", 8: "JAS", 9: "JAS", 10: "OND", 11: "OND", 12: "OND"}


# ---------------------------------------------------------------- 1. download
def download_station(climate_id, start, end):
    """Page through the GeoMet climate-daily collection for one station."""
    url = "https://api.weather.gc.ca/collections/climate-daily/items"
    rows, offset, limit = [], 0, 10000
    while True:
        params = {
            "CLIMATE_IDENTIFIER": climate_id,
            "datetime": f"{start} 00:00:00/{end} 00:00:00",
            "sortby": "LOCAL_DATE",
            "limit": limit,
            "offset": offset,
            "f": "json",
        }
        r = requests.get(url, params=params, timeout=120)
        r.raise_for_status()
        feats = r.json().get("features", [])
        if not feats:
            break
        rows += [f["properties"] for f in feats]
        offset += limit
        print(f"  downloaded {len(rows)} days ...")
        if len(feats) < limit:
            break
    return pd.DataFrame(rows)


cache = os.path.join(OUT, "raw_api_2204100.csv")
if os.path.exists(cache):
    raw = pd.read_csv(cache)
else:
    print("Downloading ECCC daily data for", CLIMATE_ID)
    raw = download_station(CLIMATE_ID, START, END)
    raw.to_csv(cache, index=False)

raw["date"] = pd.to_datetime(raw["LOCAL_DATE"]).dt.normalize()
raw = raw.drop_duplicates("date").set_index("date").sort_index()

# Reindex to a FULL daily calendar so absent days show up as NaN (= missing)
full = pd.date_range(START, END, freq="D")
d = pd.DataFrame(index=full)
d.index.name = "date"
d["tmax"] = pd.to_numeric(raw["MAX_TEMPERATURE"], errors="coerce").reindex(full)
d["tmin"] = pd.to_numeric(raw["MIN_TEMPERATURE"], errors="coerce").reindex(full)
d["precip"] = pd.to_numeric(raw["TOTAL_PRECIPITATION"], errors="coerce").reindex(full)
# Keep the precip flag: 'T' = trace (recorded as 0 mm, a valid value, NOT missing)
if "TOTAL_PRECIPITATION_FLAG" in raw:
    d["precip_flag"] = raw["TOTAL_PRECIPITATION_FLAG"].reindex(full)

# Tmean from Tmax and Tmin; missing if either is missing
d["tmean"] = (d["tmax"] + d["tmin"]) / 2
d["year"], d["month"] = d.index.year, d.index.month
d["season"] = d["month"].map(SEASONS)
d.to_csv(os.path.join(OUT, f"station_daily_{CLIMATE_ID}.csv"))

VARS = ["tmean", "precip"]

# ------------------------------------------------- 2. missing-data accounting
by_year = d.groupby("year")[VARS].apply(lambda g: g.isna().sum())
by_year.columns = [f"missing_{v}" for v in VARS]
by_year["days_in_year"] = d.groupby("year").size()
by_year["year_ok"] = (by_year[[f"missing_{v}" for v in VARS]] <= 30).all(axis=1)
by_year.to_csv(os.path.join(OUT, "missing_by_year.csv"))


def pct_complete(group_cols):
    t = d.groupby(group_cols)[VARS].apply(lambda g: g.notna().mean() * 100)
    t.columns = [f"pct_{v}" for v in VARS]
    t["tmean_ok_90"] = t["pct_tmean"] >= 90
    t["precip_ok_90"] = t["pct_precip"] >= 90   # totals / seasonality
    t["precip_ok_95"] = t["pct_precip"] >= 95   # max daily precipitation
    return t


by_month = pct_complete(["year", "month"])
by_season = pct_complete(["year", "season"])
by_month.to_csv(os.path.join(OUT, "missing_by_month.csv"))
by_season.to_csv(os.path.join(OUT, "missing_by_season.csv"))

# ------------------------------------------------ 3. rank 30-year windows
first, last = d["year"].min(), d["year"].max()
rows = []
for y0 in range(first, last - WINDOW + 2):
    y1 = y0 + WINDOW - 1
    w = d[(d.year >= y0) & (d.year <= y1)]
    m = by_month.loc[y0:y1]
    s = by_season.loc[y0:y1]
    rows.append({
        "window": f"{y0}-{y1}",
        "all_years_pass_30day_rule": bool(by_year.loc[y0:y1, "year_ok"].all()),
        "years_failing": int((~by_year.loc[y0:y1, "year_ok"]).sum()),
        "pct_missing_tmean": round(w["tmean"].isna().mean() * 100, 2),
        "pct_missing_precip": round(w["precip"].isna().mean() * 100, 2),
        "months_fail_90_tmean": int((~m["tmean_ok_90"]).sum()),
        "months_fail_95_precip": int((~m["precip_ok_95"]).sum()),
        "seasons_fail_90_tmean": int((~s["tmean_ok_90"]).sum()),
        "seasons_fail_95_precip": int((~s["precip_ok_95"]).sum()),
    })

rank = pd.DataFrame(rows)
rank["total_pct_missing"] = rank["pct_missing_tmean"] + rank["pct_missing_precip"]
rank = rank.sort_values(
    ["all_years_pass_30day_rule", "years_failing", "total_pct_missing",
     "months_fail_95_precip"],
    ascending=[False, True, True, True])
rank.to_csv(os.path.join(OUT, "window_ranking.csv"), index=False)

# ------------------------------------------------------------- 4. summary
print("\nMissing days per year (worst 10):")
print(by_year.sort_values(["missing_tmean", "missing_precip"],
                          ascending=False).head(10))
print("\nTop 5 candidate 30-year windows:")
print(rank.head(5).to_string(index=False))
best = rank.iloc[0]["window"]
print(f"\nSuggested window: {best}  (check window_ranking.csv before deciding)")

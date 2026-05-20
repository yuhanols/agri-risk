"""
Build weekly analysis panel by merging:
  1. Truck shipments (weekly, by district)
  2. FOB prices (daily -> weekly avg, by district)
  3. PRISM weather (daily -> weekly avg, by location/district)

Output: data/processed/weekly_panel_2010_2026.csv
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"

# ============================================================
# 1. Load truck data (already weekly)
# ============================================================
truck = pd.read_csv(PROC / "truck_lettuce_weekly_2010_2026.csv", parse_dates=["week_ending"])

# Standardize district names to a common set
DISTRICT_MAP_TRUCK = {
    "Western Arizona": "Western Arizona",
    "Central Arizona": "Western Arizona",  # merge into one AZ district
    "Salinas-Watsonville": "Salinas-Watsonville",
    "Salinas Valley": "Salinas-Watsonville",
    "Santa Maria": "Santa Maria",
    "Imperial Valley": "Imperial Valley",
    "Coachella Valley": "Imperial Valley",  # merge desert valleys
    "Palo Verde Valley": "Imperial Valley",
    "Oxnard": "Oxnard",
    "Central San Joaquin Valley": "San Joaquin Valley",
    "San Joaquin Valley": "San Joaquin Valley",
    "Kern": "San Joaquin Valley",
}
truck["district_std"] = truck["district"].map(DISTRICT_MAP_TRUCK)
truck = truck.dropna(subset=["district_std"])

# Focus on main 5 lettuce types
MAIN_LETTUCE = ["Lettuce, Iceberg", "Lettuce, Romaine", "Lettuce, Green Leaf",
                "Lettuce, Red Leaf", "Lettuce, Boston"]
truck = truck[truck["commodity"].isin(MAIN_LETTUCE)]

# Aggregate: total weekly volume by district
truck_weekly = (truck.groupby(["week_ending", "district_std"])
                .agg(total_volume=("volumes", "sum"),
                     n_commodities=("commodity", "nunique"))
                .reset_index()
                .rename(columns={"district_std": "district"}))

# Also get volume by commodity type
truck_by_type = (truck.groupby(["week_ending", "district_std", "commodity"])["volumes"]
                 .sum().unstack(fill_value=0).reset_index()
                 .rename(columns={"district_std": "district"}))
# Clean column names
truck_by_type.columns = [c.replace("Lettuce, ", "vol_").replace(" ", "_").lower()
                          if "Lettuce" in c else c for c in truck_by_type.columns]

truck_weekly = truck_weekly.merge(truck_by_type, on=["week_ending", "district"], how="left")

# ============================================================
# 2. Load price data (daily -> weekly)
# ============================================================
price = pd.read_csv(PROC / "price_lettuce_daily_2010_2026.csv")
price["report_date"] = pd.to_datetime(price["report_date"])

# Standardize district names
DISTRICT_MAP_PRICE = {
    "WESTERN ARIZONA": "Western Arizona",
    "SALINAS-WATSONVILLE CALIFORNIA": "Salinas-Watsonville",
    "SANTA MARIA CALIFORNIA": "Santa Maria",
    "IMPERIAL VALLEY CALIFORNIA": "Imperial Valley",
    "IMPERIAL, COACHELLA AND PALO VERDE VALLEYS CALIFORNIA": "Imperial Valley",
    "IMPERIAL AND COACHELLA VALLEYS CALIFORNIA": "Imperial Valley",
    "IMPERIAL AND PALO VERDE VALLEYS CALIFORNIA": "Imperial Valley",
    "COACHELLA VALLEY CALIFORNIA": "Imperial Valley",
    "OXNARD DISTRICT CALIFORNIA": "Oxnard",
    "SOUTH AND CENTRAL DISTRICT CALIFORNIA": "San Joaquin Valley",
    "CENTRAL SAN JOAQUIN VALLEY CALIFORNIA": "San Joaquin Valley",
    "SAN JOAQUIN VALLEY CALIFORNIA": "San Joaquin Valley",
    "KERN DISTRICT CALIFORNIA": "San Joaquin Valley",
}
price["district"] = price["district"].map(DISTRICT_MAP_PRICE)
price = price.dropna(subset=["district"])

# Only non-organic
price = price[price["organic"] == "N"]

# Compute midpoint of mostly price (more representative than low/high range)
price["mostly_mid"] = pd.to_numeric(price["mostly_low_price"], errors="coerce").add(
    pd.to_numeric(price["mostly_high_price"], errors="coerce")) / 2
price["price_mid"] = pd.to_numeric(price["low_price"], errors="coerce").add(
    pd.to_numeric(price["high_price"], errors="coerce")) / 2

# Use mostly_mid if available, else price_mid
price["fob_price"] = price["mostly_mid"].fillna(price["price_mid"])
price = price.dropna(subset=["fob_price"])

# Create week_ending (Tuesday) to match truck data
price["week_ending"] = price["report_date"].dt.to_period("W-TUE").dt.end_time.dt.normalize()

# Weekly avg price by district (across all lettuce types and package sizes)
price_weekly = (price.groupby(["week_ending", "district"])
                .agg(avg_fob_price=("fob_price", "mean"),
                     median_fob_price=("fob_price", "median"),
                     n_price_obs=("fob_price", "count"))
                .reset_index())

# Also get price by commodity
price_by_type = (price.groupby(["week_ending", "district", "commodity"])["fob_price"]
                 .mean().unstack(fill_value=None).reset_index())
price_by_type.columns = [c.replace("Lettuce, ", "price_").replace(" ", "_").lower()
                          if "Lettuce" in c else c for c in price_by_type.columns]

price_weekly = price_weekly.merge(price_by_type, on=["week_ending", "district"], how="left")

# ============================================================
# 3. Load weather data (daily -> weekly)
# ============================================================
weather = pd.read_csv(PROC / "weather_daily_2010_2026.csv", parse_dates=["date"])

# Map weather locations to districts
WEATHER_DISTRICT_MAP = {
    "yuma_az": "Western Arizona",
    "salinas_ca": "Salinas-Watsonville",
    "imperial_ca": "Imperial Valley",
    "oxnard_ca": "Oxnard",
    "santa_maria_ca": "Santa Maria",
    "san_joaquin_ca": "San Joaquin Valley",
}
weather["district"] = weather["location"].map(WEATHER_DISTRICT_MAP)

# Create week_ending (Tuesday)
weather["week_ending"] = weather["date"].dt.to_period("W-TUE").dt.end_time.dt.normalize()

# Degree days (daily, before weekly aggregation)
# DD_heat: degree days above 40C (cumulative excess heat)
# DD_freeze: degree days below 0C (cumulative frost severity)
weather["dd_heat"] = (weather["tmax"] - 40).clip(lower=0)
weather["dd_freeze"] = (0 - weather["tmin"]).clip(lower=0)

# Weekly aggregation
weather_weekly = (weather.groupby(["week_ending", "district"])
                  .agg(
                      tmax_avg=("tmax", "mean"),
                      tmin_avg=("tmin", "mean"),
                      tmax_max=("tmax", "max"),      # hottest day in week
                      tmin_min=("tmin", "min"),       # coldest night in week
                      ppt_total=("ppt", "sum"),       # total weekly precip
                      ppt_days=("ppt", lambda x: (x > 0).sum()),  # rainy days
                      dd_heat=("dd_heat", "sum"),     # weekly degree days >40C
                      dd_freeze=("dd_freeze", "sum"), # weekly degree days <0C
                      n_weather_days=("tmax", "count"),
                  ).reset_index())

# Derived features
weather_weekly["tmean_avg"] = (weather_weekly["tmax_avg"] + weather_weekly["tmin_avg"]) / 2
weather_weekly["diurnal_range"] = weather_weekly["tmax_avg"] - weather_weekly["tmin_avg"]

# Extreme weather indicators
weather_weekly["extreme_heat"] = (weather_weekly["tmax_max"] > 40).astype(int)  # >40C
weather_weekly["freeze_risk"] = (weather_weekly["tmin_min"] < 0).astype(int)     # <0C
weather_weekly["heavy_rain"] = (weather_weekly["ppt_total"] > 25).astype(int)    # >25mm/week

# ============================================================
# 4. Merge all three
# ============================================================
# Start with truck (weekly, by district)
panel = truck_weekly.copy()

# Merge price
panel = panel.merge(price_weekly, on=["week_ending", "district"], how="left")

# Merge weather
panel = panel.merge(weather_weekly, on=["week_ending", "district"], how="left")

# Add time features
panel["year"] = panel["week_ending"].dt.year
panel["month"] = panel["week_ending"].dt.month
panel["week_of_year"] = panel["week_ending"].dt.isocalendar().week.astype(int)

# Classify state
AZ_DISTRICTS = ["Western Arizona"]
panel["state"] = panel["district"].apply(lambda x: "AZ" if x in AZ_DISTRICTS else "CA")

# Sort
panel = panel.sort_values(["week_ending", "district"]).reset_index(drop=True)

# Save
outpath = PROC / "weekly_panel_2010_2026.csv"
panel.to_csv(outpath, index=False)

# ============================================================
# Summary
# ============================================================
print(f"Saved: {outpath}")
print(f"Total rows: {len(panel)}")
print(f"Date range: {panel['week_ending'].min().date()} to {panel['week_ending'].max().date()}")
print(f"Unique weeks: {panel['week_ending'].nunique()}")
print(f"Districts: {sorted(panel['district'].unique())}")
print()

# Coverage
print("=== Data coverage by district ===")
for dist in sorted(panel["district"].unique()):
    d = panel[panel["district"] == dist]
    has_vol = d["total_volume"].notna().sum()
    has_price = d["avg_fob_price"].notna().sum()
    has_weather = d["tmax_avg"].notna().sum()
    print(f"  {dist}: {len(d)} weeks, volume={has_vol}, price={has_price}, weather={has_weather}")

print()
print("=== Sample rows ===")
print(panel[["week_ending", "district", "total_volume", "avg_fob_price",
             "tmax_avg", "tmin_avg", "ppt_total"]].dropna().head(10).to_string(index=False))

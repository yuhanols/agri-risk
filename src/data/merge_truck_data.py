"""
Merge old (2010-2021) and new (2022-2026) USDA AMS truck shipment data for lettuce.
Output: data/processed/truck_lettuce_weekly_2010_2026.csv
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

OLD_FILE = Path("/Users/yuhan/Library/CloudStorage/GoogleDrive-redacted-user/My Drive/UCD/Lettuce/Lettuce_2010_2021.csv")
NEW_FILE = RAW / "usda_ams_truck_lettuce_az_ca_2022_2026.csv"

# --- Load old data (2010-2021) ---
df_old = pd.read_csv(OLD_FILE, encoding="utf-8-sig", usecols=[0, 1, 2, 3, 4, 5],
                     names=["week_ending", "region", "origin", "district", "commodity", "volumes"],
                     header=0)
df_old["week_ending"] = pd.to_datetime(df_old["week_ending"], format="mixed")
df_old["volumes"] = pd.to_numeric(df_old["volumes"].astype(str).str.replace(",", ""), errors="coerce")

# --- Load new data (2022-2026) ---
df_new = pd.read_csv(NEW_FILE)
df_new = df_new.rename(columns={"tuesday_week_ending": "week_ending"})
df_new["week_ending"] = pd.to_datetime(df_new["week_ending"])
df_new["volumes"] = pd.to_numeric(df_new["volumes"], errors="coerce")

# Standardize region/origin casing to match old data
df_new["region"] = df_new["region"].str.title()
df_new["origin"] = df_new["origin"].str.title()

# --- Merge ---
df = pd.concat([df_old, df_new], ignore_index=True)
df = df.dropna(subset=["week_ending", "volumes"])
df = df.sort_values(["week_ending", "region", "district", "commodity"]).reset_index(drop=True)

# Remove duplicates (if any overlap at boundary)
df = df.drop_duplicates(subset=["week_ending", "district", "commodity"], keep="last")

# --- Standardize district names ---
district_map = {
    "Salinas-watsonville": "Salinas-Watsonville",
    "Salinas-watsonville California": "Salinas-Watsonville",
    "Salinas Valley California": "Salinas Valley",
    "Imperial Valley District": "Imperial Valley",
    "Imperial Valley California": "Imperial Valley",
    "Oxnard District": "Oxnard",
    "Oxnard District California": "Oxnard",
    "Santa Maria California": "Santa Maria",
    "San Joaquin Valley District": "San Joaquin Valley",
    "San Joaquin Valley California": "San Joaquin Valley",
    "Central San Joaquin Valley": "Central San Joaquin Valley",
    "Central San Joaquin Valley California": "Central San Joaquin Valley",
    "Coachella Valley": "Coachella Valley",
    "Coachella Valley California": "Coachella Valley",
    "Palo Verde Valley": "Palo Verde Valley",
    "Palo Verde Valley California": "Palo Verde Valley",
    "Kern District": "Kern",
    "Central Arizona": "Central Arizona",
    "Western Arizona": "Western Arizona",
}
df["district"] = df["district"].map(district_map).fillna(df["district"])

# Standardize commodity names
df["commodity"] = df["commodity"].str.strip()

# --- Save ---
outpath = OUT / "truck_lettuce_weekly_2010_2026.csv"
df.to_csv(outpath, index=False)

# --- Summary ---
print(f"Saved: {outpath}")
print(f"Total rows: {len(df)}")
print(f"Date range: {df['week_ending'].min().date()} to {df['week_ending'].max().date()}")
print(f"\nDistricts:\n{df['district'].value_counts().to_string()}")
print(f"\nCommodities:\n{df['commodity'].value_counts().to_string()}")
print(f"\nWeekly obs by year:")
print(df.groupby(df["week_ending"].dt.year).size().to_string())

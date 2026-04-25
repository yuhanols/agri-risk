"""
Build two modeling datasets from the weekly panel:

1. Market-level (aggregated): weekly total Iceberg volume, shipment-weighted
   FOB price, volume-weighted weather, coverage ratio.
   -> data/processed/market_iceberg_weekly.csv

2. District-level (panel): Iceberg only, 3 core districts
   (Salinas-Watsonville, Western Arizona, Imperial Valley).
   -> data/processed/district_iceberg_weekly.csv

Both datasets include lagged features and time features for ML modeling.
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"

# ============================================================
# Load panel
# ============================================================
panel = pd.read_csv(PROC / "weekly_panel_2010_2026.csv", parse_dates=["week_ending"])

# Treat price=0 as missing (no valid FOB price is $0)
panel["price_iceberg"] = panel["price_iceberg"].replace(0, np.nan)

# Mark 2026 as partial year
panel["is_partial_year"] = (panel["week_ending"].dt.year == 2026).astype(int)

# ============================================================
# 1. District-level dataset (Iceberg, 3 core districts)
# ============================================================
CORE_DISTRICTS = ["Salinas-Watsonville", "Western Arizona", "Imperial Valley"]
dist = panel[panel["district"].isin(CORE_DISTRICTS)].copy()

dist = dist[["week_ending", "district", "state",
              "vol_iceberg", "price_iceberg",
              "tmax_avg", "tmin_avg", "tmax_max", "tmin_min",
              "ppt_total", "ppt_days", "tmean_avg", "diurnal_range",
              "extreme_heat", "freeze_risk", "heavy_rain",
              "year", "month", "week_of_year", "is_partial_year"]].copy()

dist = dist.rename(columns={"vol_iceberg": "volume", "price_iceberg": "price"})

# Flag: does this district-week have price?
dist["has_price"] = dist["price"].notna().astype(int)

# Lagged features (by district)
dist = dist.sort_values(["district", "week_ending"])
for lag in [1, 2, 4]:
    dist[f"volume_lag{lag}"] = dist.groupby("district")["volume"].shift(lag)
    dist[f"price_lag{lag}"] = dist.groupby("district")["price"].shift(lag)
    dist[f"tmax_lag{lag}"] = dist.groupby("district")["tmax_avg"].shift(lag)
    dist[f"ppt_lag{lag}"] = dist.groupby("district")["ppt_total"].shift(lag)

# Rolling features (4-week window)
for col in ["volume", "tmax_avg", "ppt_total"]:
    rolled = dist.groupby("district")[col].rolling(4, min_periods=2)
    dist[f"{col}_roll4_mean"] = rolled.mean().reset_index(level=0, drop=True)

# Season indicator (AZ winter Nov-Mar, CA summer Apr-Oct)
dist["az_season"] = ((dist["state"] == "AZ") & (dist["month"].isin([11, 12, 1, 2, 3]))).astype(int)
dist["ca_season"] = ((dist["state"] == "CA") & (dist["month"].isin([4, 5, 6, 7, 8, 9, 10]))).astype(int)
dist["peak_season"] = (dist["az_season"] | dist["ca_season"]).astype(int)

dist = dist.sort_values(["week_ending", "district"]).reset_index(drop=True)

# Save
dist_path = PROC / "district_iceberg_weekly.csv"
dist.to_csv(dist_path, index=False)

# ============================================================
# 2. Market-level dataset (aggregated Iceberg)
# ============================================================
weeks = sorted(panel["week_ending"].unique())
market_rows = []

for wk in weeks:
    wk_data = panel[panel["week_ending"] == wk]

    # Total Iceberg volume across all districts
    total_vol = wk_data["vol_iceberg"].sum()

    # Shipment-weighted price (only districts with price)
    has_price = wk_data[wk_data["price_iceberg"].notna()]
    if len(has_price) > 0 and has_price["vol_iceberg"].sum() > 0:
        weighted_price = (
            (has_price["vol_iceberg"] * has_price["price_iceberg"]).sum()
            / has_price["vol_iceberg"].sum()
        )
        price_vol = has_price["vol_iceberg"].sum()
    else:
        weighted_price = np.nan
        price_vol = 0

    # Coverage ratio
    coverage = price_vol / total_vol if total_vol > 0 else 0

    # Volume-weighted weather
    vol_nonzero = wk_data[wk_data["vol_iceberg"] > 0]
    if len(vol_nonzero) > 0 and vol_nonzero["vol_iceberg"].sum() > 0:
        v = vol_nonzero["vol_iceberg"]
        w_tmax = (v * vol_nonzero["tmax_avg"]).sum() / v.sum()
        w_tmin = (v * vol_nonzero["tmin_avg"]).sum() / v.sum()
        w_ppt = (v * vol_nonzero["ppt_total"]).sum() / v.sum()
        w_tmean = (w_tmax + w_tmin) / 2
        w_diurnal = w_tmax - w_tmin
        extreme_heat = vol_nonzero["extreme_heat"].max()
        freeze_risk = vol_nonzero["freeze_risk"].max()
        heavy_rain = vol_nonzero["heavy_rain"].max()
    else:
        w_tmax = w_tmin = w_ppt = w_tmean = w_diurnal = np.nan
        extreme_heat = freeze_risk = heavy_rain = 0

    # Number of active districts
    n_districts = (wk_data["vol_iceberg"] > 0).sum()

    market_rows.append({
        "week_ending": wk,
        "volume": total_vol,
        "price": weighted_price,
        "coverage": round(coverage, 3),
        "n_districts": n_districts,
        "tmax_avg": round(w_tmax, 2) if not np.isnan(w_tmax) else np.nan,
        "tmin_avg": round(w_tmin, 2) if not np.isnan(w_tmin) else np.nan,
        "tmean_avg": round(w_tmean, 2) if not np.isnan(w_tmean) else np.nan,
        "ppt_total": round(w_ppt, 3) if not np.isnan(w_ppt) else np.nan,
        "diurnal_range": round(w_diurnal, 2) if not np.isnan(w_diurnal) else np.nan,
        "extreme_heat": extreme_heat,
        "freeze_risk": freeze_risk,
        "heavy_rain": heavy_rain,
    })

mkt = pd.DataFrame(market_rows)
mkt["week_ending"] = pd.to_datetime(mkt["week_ending"])

# Time features
mkt["year"] = mkt["week_ending"].dt.year
mkt["month"] = mkt["week_ending"].dt.month
mkt["week_of_year"] = mkt["week_ending"].dt.isocalendar().week.astype(int)
mkt["is_partial_year"] = (mkt["year"] == 2026).astype(int)

# Risk labels
price_roll_mean = mkt["price"].rolling(8, min_periods=4).mean()
price_roll_std = mkt["price"].rolling(8, min_periods=4).std()
mkt["price_spike"] = (mkt["price"] > price_roll_mean + 2 * price_roll_std).astype(int)

vol_roll_mean = mkt["volume"].rolling(8, min_periods=4).mean()
vol_roll_std = mkt["volume"].rolling(8, min_periods=4).std()
mkt["low_supply"] = (mkt["volume"] < vol_roll_mean - 1.5 * vol_roll_std).astype(int)

# Lagged features
mkt = mkt.sort_values("week_ending")
for lag in [1, 2, 4]:
    mkt[f"volume_lag{lag}"] = mkt["volume"].shift(lag)
    mkt[f"price_lag{lag}"] = mkt["price"].shift(lag)
    mkt[f"tmax_lag{lag}"] = mkt["tmax_avg"].shift(lag)
    mkt[f"ppt_lag{lag}"] = mkt["ppt_total"].shift(lag)

# Rolling features
for col in ["volume", "tmax_avg", "ppt_total"]:
    mkt[f"{col}_roll4_mean"] = mkt[col].rolling(4, min_periods=2).mean()

mkt = mkt.reset_index(drop=True)

# Save
mkt_path = PROC / "market_iceberg_weekly.csv"
mkt.to_csv(mkt_path, index=False)

# ============================================================
# Summary
# ============================================================
print("=" * 60)
print("DISTRICT-LEVEL DATASET")
print(f"  Path: {dist_path}")
print(f"  Rows: {len(dist)}")
print(f"  Date range: {dist['week_ending'].min().date()} to {dist['week_ending'].max().date()}")
print(f"  Districts: {sorted(dist['district'].unique())}")
print(f"  Weeks with price: {dist['has_price'].sum()} / {len(dist)} ({dist['has_price'].mean()*100:.1f}%)")
print(f"  Volume: mean={dist['volume'].mean():.0f}, median={dist['volume'].median():.0f}")
print(f"  Price: mean=${dist['price'].mean():.2f}, median=${dist['price'].median():.2f}")

print()
print("=" * 60)
print("MARKET-LEVEL DATASET")
print(f"  Path: {mkt_path}")
print(f"  Rows: {len(mkt)}")
print(f"  Date range: {mkt['week_ending'].min().date()} to {mkt['week_ending'].max().date()}")
print(f"  Weeks with price: {mkt['price'].notna().sum()} / {len(mkt)} ({mkt['price'].notna().mean()*100:.1f}%)")
print(f"  Coverage: mean={mkt['coverage'].mean():.3f}, min={mkt['coverage'].min():.3f}")
print(f"  Volume: mean={mkt['volume'].mean():.0f}, median={mkt['volume'].median():.0f}")
print(f"  Price: mean=${mkt['price'].mean():.2f}, median=${mkt['price'].median():.2f}")

print()
print("=== Market dataset sample ===")
print(mkt[["week_ending", "volume", "price", "coverage", "n_districts",
           "tmax_avg", "ppt_total"]].dropna().head(10).to_string(index=False))

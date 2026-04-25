"""
Model 3: Predict missing district-level Iceberg prices.

Train on district-weeks where price IS observed.
Predict price for district-weeks where price is missing but volume exists.
Features: weather + volume + district + season (no price lags).
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"

panel = pd.read_csv(PROC / "weekly_panel_2010_2026.csv", parse_dates=["week_ending"])

# Treat price=0 as missing
panel["price_iceberg"] = panel["price_iceberg"].replace(0, np.nan)

# ============================================================
# Build district-level features (all 6 districts)
# ============================================================
df = panel[["week_ending", "district", "state",
            "vol_iceberg", "price_iceberg",
            "tmax_avg", "tmin_avg", "tmax_max", "tmin_min",
            "ppt_total", "ppt_days", "tmean_avg", "diurnal_range",
            "extreme_heat", "freeze_risk", "heavy_rain",
            "year", "month", "week_of_year"]].copy()

df = df.rename(columns={"vol_iceberg": "volume", "price_iceberg": "price"})

# District dummies
for dist in df["district"].unique():
    col = "d_" + dist.lower().replace("-", "_").replace(" ", "_")
    df[col] = (df["district"] == dist).astype(int)

# Season
df["q1"] = df["month"].isin([1, 2, 3]).astype(int)
df["q2"] = df["month"].isin([4, 5, 6]).astype(int)
df["q3"] = df["month"].isin([7, 8, 9]).astype(int)

# Volume lags (by district)
df = df.sort_values(["district", "week_ending"])
df["volume_lag1"] = df.groupby("district")["volume"].shift(1)
df["volume_lag2"] = df.groupby("district")["volume"].shift(2)
df["volume_roll4"] = df.groupby("district")["volume"].transform(
    lambda x: x.rolling(4, min_periods=2).mean())

# Weather lags
df["tmax_lag1"] = df.groupby("district")["tmax_avg"].shift(1)
df["ppt_lag1"] = df.groupby("district")["ppt_total"].shift(1)

# ============================================================
# Define features (NO price lags - we're predicting price)
# ============================================================
WEATHER = ["tmax_avg", "tmin_avg", "ppt_total", "diurnal_range",
           "extreme_heat", "freeze_risk", "heavy_rain",
           "tmax_lag1", "ppt_lag1"]

VOLUME = ["volume", "volume_lag1", "volume_lag2", "volume_roll4"]

DISTRICT = [c for c in df.columns if c.startswith("d_")]

SEASON = ["q1", "q2", "q3", "month", "week_of_year", "year"]

FEATURES = WEATHER + VOLUME + DISTRICT + SEASON

# ============================================================
# Split: observed vs missing price
# ============================================================
has_price = df[df["price"].notna() & df["volume"] > 0].copy()
missing_price = df[df["price"].isna() & (df["volume"] > 0)].copy()

# Drop rows with missing features
has_price = has_price.dropna(subset=FEATURES)
missing_price = missing_price.dropna(subset=FEATURES)

print(f"Observed price rows: {len(has_price)}")
print(f"Missing price rows (with volume): {len(missing_price)}")
print(f"Missing price districts:")
for dist in sorted(missing_price["district"].unique()):
    n = len(missing_price[missing_price["district"] == dist])
    print(f"  {dist}: {n}")

# ============================================================
# Cross-validated evaluation (expanding window on observed data)
# ============================================================
print("\n" + "=" * 75)
print("EXPANDING WINDOW VALIDATION (on observed data)")
print("=" * 75)

has_price["year"] = has_price["week_ending"].dt.year
years = sorted(has_price["year"].unique())

XGB_PARAMS = {
    "n_estimators": 200, "max_depth": 4, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
    "reg_alpha": 1.0, "reg_lambda": 3.0, "random_state": 42, "verbosity": 0,
}

results = []
for test_year in years[4:]:
    train = has_price[has_price["year"] < test_year]
    test = has_price[has_price["year"] == test_year]
    if len(test) == 0:
        continue

    # OLS
    ols = LinearRegression()
    ols.fit(train[FEATURES], train["price"])
    ols_pred = ols.predict(test[FEATURES])

    # XGBoost
    xgb = XGBRegressor(**XGB_PARAMS)
    xgb.fit(train[FEATURES], train["price"])
    xgb_pred = xgb.predict(test[FEATURES])

    results.append({
        "year": test_year,
        "n": len(test),
        "ols_rmse": np.sqrt(mean_squared_error(test["price"], ols_pred)),
        "ols_mae": mean_absolute_error(test["price"], ols_pred),
        "xgb_rmse": np.sqrt(mean_squared_error(test["price"], xgb_pred)),
        "xgb_mae": mean_absolute_error(test["price"], xgb_pred),
    })

res = pd.DataFrame(results)
print(f"{'Year':>6} {'N':>4} | {'OLS RMSE':>9} {'OLS MAE':>8} | {'XGB RMSE':>9} {'XGB MAE':>8}")
print("-" * 60)
for _, r in res.iterrows():
    print(f"{r['year']:>6.0f} {r['n']:>4.0f} | ${r['ols_rmse']:>8.2f} ${r['ols_mae']:>7.2f} | "
          f"${r['xgb_rmse']:>8.2f} ${r['xgb_mae']:>7.2f}")
print("-" * 60)
print(f"{'AVG':>6} {'':>4} | ${res['ols_rmse'].mean():>8.2f} ${res['ols_mae'].mean():>7.2f} | "
      f"${res['xgb_rmse'].mean():>8.2f} ${res['xgb_mae'].mean():>7.2f}")

# ============================================================
# Train full model and predict missing prices
# ============================================================
print("\n" + "=" * 75)
print("PREDICTING MISSING PRICES")
print("=" * 75)

xgb_full = XGBRegressor(**XGB_PARAMS)
xgb_full.fit(has_price[FEATURES], has_price["price"])

missing_price["predicted_price"] = xgb_full.predict(missing_price[FEATURES])

# Clip negative predictions
missing_price["predicted_price"] = missing_price["predicted_price"].clip(lower=0)

print(f"\nPredicted {len(missing_price)} missing prices")
print(f"Predicted price range: ${missing_price['predicted_price'].min():.2f} - "
      f"${missing_price['predicted_price'].max():.2f}")
print(f"Predicted price mean: ${missing_price['predicted_price'].mean():.2f}")
print(f"Observed price mean: ${has_price['price'].mean():.2f}")

print(f"\nBy district:")
for dist in sorted(missing_price["district"].unique()):
    sub = missing_price[missing_price["district"] == dist]
    obs = has_price[has_price["district"] == dist]
    obs_mean = f"${obs['price'].mean():.2f}" if len(obs) > 0 else "N/A"
    print(f"  {dist}: {len(sub)} predicted, mean=${sub['predicted_price'].mean():.2f} "
          f"(observed mean={obs_mean})")

# ============================================================
# Feature importance
# ============================================================
print("\n" + "=" * 75)
print("FEATURE IMPORTANCE (top 15)")
print("=" * 75)

imp = pd.Series(xgb_full.feature_importances_, index=FEATURES).sort_values(ascending=False)
for feat, val in imp.head(15).items():
    bar = "█" * int(val * 100)
    print(f"  {feat:<25} {val:.4f} {bar}")

# ============================================================
# Save predictions
# ============================================================
out_cols = ["week_ending", "district", "volume", "predicted_price"]
missing_price[out_cols].to_csv(OUT / "missing_price_predictions.csv", index=False)
print(f"\nSaved: {OUT / 'missing_price_predictions.csv'}")

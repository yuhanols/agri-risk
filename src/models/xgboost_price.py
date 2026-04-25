"""
XGBoost model for market-level Iceberg lettuce price prediction.

Comparison with OLS baseline. Uses expanding window validation
to respect time series structure (no future leakage).
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# ============================================================
# Load data
# ============================================================
mkt = pd.read_csv(PROC / "market_iceberg_weekly.csv", parse_dates=["week_ending"])

FEATURES = [
    # Weather (contemporaneous)
    "tmax_avg", "tmin_avg", "ppt_total", "diurnal_range",
    "extreme_heat", "freeze_risk", "heavy_rain",
    # Lags
    "price_lag1", "price_lag2", "price_lag4",
    "volume_lag1", "volume_lag2",
    "tmax_lag1", "tmax_lag2", "ppt_lag1",
    # Rolling
    "volume_roll4_mean", "tmax_avg_roll4_mean", "ppt_total_roll4_mean",
    # Calendar
    "month", "week_of_year",
    # Market
    "coverage", "n_districts",
]

TARGET = "price"

# Drop rows without target or features
df = mkt[["week_ending", TARGET] + FEATURES].dropna()
print(f"Total usable rows: {len(df)}")
print(f"Date range: {df['week_ending'].min().date()} to {df['week_ending'].max().date()}")

# ============================================================
# Expanding window validation
# ============================================================
# Train on first N years, test on next year, expand forward
min_train_years = 4  # start training with 2010-2013, test on 2014

df["year"] = df["week_ending"].dt.year
years = sorted(df["year"].unique())

results = []
all_preds = []

for test_year in years[min_train_years:]:
    train = df[df["year"] < test_year]
    test = df[df["year"] == test_year]

    if len(test) == 0:
        continue

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    # OLS baseline
    ols = LinearRegression()
    ols.fit(X_train, y_train)
    ols_pred = ols.predict(X_test)

    # XGBoost
    xgb = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=42,
        verbosity=0,
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)

    # Metrics
    ols_rmse = np.sqrt(mean_squared_error(y_test, ols_pred))
    ols_mae = mean_absolute_error(y_test, ols_pred)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    xgb_mae = mean_absolute_error(y_test, xgb_pred)

    results.append({
        "test_year": test_year,
        "n_train": len(train),
        "n_test": len(test),
        "ols_rmse": ols_rmse,
        "ols_mae": ols_mae,
        "xgb_rmse": xgb_rmse,
        "xgb_mae": xgb_mae,
        "improvement": (ols_rmse - xgb_rmse) / ols_rmse * 100,
    })

    for i, (idx, row) in enumerate(test.iterrows()):
        all_preds.append({
            "week_ending": row["week_ending"],
            "actual": y_test.iloc[i],
            "ols_pred": ols_pred[i],
            "xgb_pred": xgb_pred[i],
        })

res = pd.DataFrame(results)

# ============================================================
# Print results
# ============================================================
print("\n" + "=" * 80)
print("EXPANDING WINDOW VALIDATION: OLS vs XGBoost")
print("=" * 80)
print(f"{'Year':>6} {'N_train':>8} {'N_test':>7} | {'OLS RMSE':>9} {'OLS MAE':>8} | {'XGB RMSE':>9} {'XGB MAE':>8} | {'Improv%':>7}")
print("-" * 80)
for _, r in res.iterrows():
    print(f"{r['test_year']:>6.0f} {r['n_train']:>8.0f} {r['n_test']:>7.0f} | "
          f"${r['ols_rmse']:>8.2f} ${r['ols_mae']:>7.2f} | "
          f"${r['xgb_rmse']:>8.2f} ${r['xgb_mae']:>7.2f} | "
          f"{r['improvement']:>6.1f}%")

print("-" * 80)
avg_ols_rmse = res["ols_rmse"].mean()
avg_xgb_rmse = res["xgb_rmse"].mean()
avg_ols_mae = res["ols_mae"].mean()
avg_xgb_mae = res["xgb_mae"].mean()
avg_improv = res["improvement"].mean()
print(f"{'AVG':>6} {'':>8} {'':>7} | ${avg_ols_rmse:>8.2f} ${avg_ols_mae:>7.2f} | "
      f"${avg_xgb_rmse:>8.2f} ${avg_xgb_mae:>7.2f} | {avg_improv:>6.1f}%")

# ============================================================
# Feature importance (from full model)
# ============================================================
print("\n" + "=" * 80)
print("FEATURE IMPORTANCE (full model)")
print("=" * 80)

xgb_full = XGBRegressor(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    random_state=42, verbosity=0,
)
xgb_full.fit(df[FEATURES], df[TARGET])

importance = pd.Series(xgb_full.feature_importances_, index=FEATURES).sort_values(ascending=False)
for feat, imp in importance.items():
    bar = "█" * int(imp * 100)
    print(f"  {feat:<25} {imp:.4f} {bar}")

# ============================================================
# Save predictions
# ============================================================
pred_df = pd.DataFrame(all_preds)
pred_df.to_csv(OUT / "price_predictions_expanding.csv", index=False)

# Save results
res.to_csv(OUT / "model_comparison_expanding.csv", index=False)

print(f"\nPredictions saved: {OUT / 'price_predictions_expanding.csv'}")
print(f"Results saved: {OUT / 'model_comparison_expanding.csv'}")

# ============================================================
# Diagnostics: coverage vs error
# ============================================================
print("\n" + "=" * 80)
print("DIAGNOSTIC: Coverage vs Prediction Error")
print("=" * 80)
pred_with_cov = pred_df.merge(mkt[["week_ending", "coverage"]], on="week_ending")
pred_with_cov["xgb_error"] = (pred_with_cov["actual"] - pred_with_cov["xgb_pred"]).abs()

for label, low, high in [("Low (<0.7)", 0, 0.7), ("Mid (0.7-0.9)", 0.7, 0.9), ("High (>0.9)", 0.9, 1.01)]:
    subset = pred_with_cov[(pred_with_cov["coverage"] >= low) & (pred_with_cov["coverage"] < high)]
    if len(subset) > 0:
        print(f"  {label}: n={len(subset):>4}, MAE=${subset['xgb_error'].mean():.2f}, "
              f"median_error=${subset['xgb_error'].median():.2f}")

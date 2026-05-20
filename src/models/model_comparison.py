"""
Complete model comparison: Level vs Change, OLS vs XGBoost vs Random Forest.

Model matrix:
  - Target: price level vs Δprice vs log Δprice
  - Method: OLS vs XGBoost vs Random Forest (conservative)
  - Features: with/without price lags

Expanding window validation (train on <year, test on year).
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

mkt = pd.read_csv(PROC / "market_iceberg_weekly.csv", parse_dates=["week_ending"])

# ============================================================
# Construct targets
# ============================================================
mkt["dprice"] = mkt["price"] - mkt["price_lag1"]
mkt["log_price"] = np.log(mkt["price"])
mkt["log_price_lag1"] = np.log(mkt["price_lag1"])
mkt["log_dprice"] = mkt["log_price"] - mkt["log_price_lag1"]

# ============================================================
# Feature sets
# ============================================================
WEATHER = [
    "tmax_avg", "tmin_avg", "ppt_total", "diurnal_range",
    "dd_heat", "dd_freeze", "heavy_rain",
    "tmax_lag1", "tmax_lag2", "ppt_lag1",
    "tmax_avg_roll4_mean", "ppt_total_roll4_mean",
]

MARKET = [
    "volume_lag1", "volume_lag2", "volume_roll4_mean",
    "coverage", "n_districts",
    "month", "week_of_year",
]

PRICE_LAGS = ["price_lag1", "price_lag2", "price_lag4"]

# For change models: no price lags (already differenced out)
CHANGE_FEATURES = WEATHER + MARKET

# For level models: full features
LEVEL_FEATURES = WEATHER + MARKET + PRICE_LAGS

# For no-lag model: see weather importance without AR
NOLAGS_FEATURES = WEATHER + MARKET

# ============================================================
# XGBoost params (conservative)
# ============================================================
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "min_child_weight": 10,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "random_state": 42,
    "verbosity": 0,
}

RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "min_samples_leaf": 10,
    "max_features": 0.7,
    "random_state": 42,
    "n_jobs": -1,
}

# ============================================================
# Prepare data
# ============================================================
all_cols = list(set(
    ["week_ending", "price", "price_lag1", "dprice", "log_dprice"]
    + LEVEL_FEATURES + CHANGE_FEATURES
))
df = mkt[all_cols].dropna()
df["year"] = df["week_ending"].dt.year
years = sorted(df["year"].unique())
MIN_TRAIN = 4

print(f"Rows: {len(df)}, Years: {years[0]}-{years[-1]}")


# ============================================================
# Expanding window engine
# ============================================================
def expanding_eval(df, features, target, model_cls, model_kwargs,
                   convert_to_level=None):
    """
    convert_to_level: if target is delta, function to convert pred back to level.
      e.g., lambda pred, row: row["price_lag1"] + pred
    """
    yearly = []
    for test_year in years[MIN_TRAIN:]:
        train = df[df["year"] < test_year]
        test = df[df["year"] == test_year]
        if len(test) == 0:
            continue

        X_tr, y_tr = train[features], train[target]
        X_te, y_te = test[features], test[target]

        m = model_cls(**model_kwargs)
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)

        if convert_to_level is not None:
            pred_level = np.array([convert_to_level(p, test.iloc[i])
                                   for i, p in enumerate(pred)])
            actual_level = test["price"].values
        else:
            pred_level = pred
            actual_level = test["price"].values

        yearly.append({
            "year": test_year,
            "n": len(test),
            "rmse": np.sqrt(mean_squared_error(actual_level, pred_level)),
            "mae": mean_absolute_error(actual_level, pred_level),
        })

    return pd.DataFrame(yearly)


def get_importance(df, features, target, model_cls, model_kwargs):
    m = model_cls(**model_kwargs)
    m.fit(df[features], df[target])
    if hasattr(m, "feature_importances_"):
        return pd.Series(m.feature_importances_, index=features).sort_values(ascending=False)
    else:
        return pd.Series(np.abs(m.coef_), index=features).sort_values(ascending=False)


# ============================================================
# Run all models
# ============================================================
models = {
    # Level models
    "OLS Level (full)": (LEVEL_FEATURES, "price", LinearRegression, {}, None),
    "OLS Level (no lags)": (NOLAGS_FEATURES, "price", LinearRegression, {}, None),
    "XGB Level (full)": (LEVEL_FEATURES, "price", XGBRegressor, XGB_PARAMS, None),
    "XGB Level (no lags)": (NOLAGS_FEATURES, "price", XGBRegressor, XGB_PARAMS, None),

    # Random Forest level models
    "RF Level (full)": (LEVEL_FEATURES, "price", RandomForestRegressor, RF_PARAMS, None),
    "RF Level (no lags)": (NOLAGS_FEATURES, "price", RandomForestRegressor, RF_PARAMS, None),

    # Change models (convert back to level for comparable RMSE)
    "OLS Δprice": (CHANGE_FEATURES, "dprice", LinearRegression, {},
                    lambda p, r: r["price_lag1"] + p),
    "XGB Δprice": (CHANGE_FEATURES, "dprice", XGBRegressor, XGB_PARAMS,
                    lambda p, r: r["price_lag1"] + p),
    "RF Δprice": (CHANGE_FEATURES, "dprice", RandomForestRegressor, RF_PARAMS,
                   lambda p, r: r["price_lag1"] + p),
    "OLS log Δprice": (CHANGE_FEATURES, "log_dprice", LinearRegression, {},
                        lambda p, r: np.exp(np.log(r["price_lag1"]) + p)),
    "XGB log Δprice": (CHANGE_FEATURES, "log_dprice", XGBRegressor, XGB_PARAMS,
                        lambda p, r: np.exp(np.log(r["price_lag1"]) + p)),
}

# Naive baseline
naive_res = []
for test_year in years[MIN_TRAIN:]:
    test = df[df["year"] == test_year]
    if len(test) == 0:
        continue
    naive_res.append({
        "year": test_year,
        "rmse": np.sqrt(mean_squared_error(test["price"], test["price_lag1"])),
    })
naive_avg = pd.DataFrame(naive_res)["rmse"].mean()

all_results = {}
for name, (feats, target, cls, kwargs, conv) in models.items():
    res = expanding_eval(df, feats, target, cls, kwargs, conv)
    all_results[name] = res

# ============================================================
# Print comparison table
# ============================================================
print("\n" + "=" * 80)
print("MODEL COMPARISON: Average RMSE and MAE (expanding window, 2014-2026)")
print("=" * 80)
print(f"{'Model':<30} {'Avg RMSE':>10} {'Avg MAE':>10} {'vs Naive':>10}")
print("-" * 62)
print(f"{'Naive (price=lag1)':<30} ${naive_avg:>9.2f} {'':>10} {'baseline':>10}")

for name, res in all_results.items():
    avg_rmse = res["rmse"].mean()
    avg_mae = res["mae"].mean()
    vs_naive = (naive_avg - avg_rmse) / naive_avg * 100
    print(f"{name:<30} ${avg_rmse:>9.2f} ${avg_mae:>9.2f} {vs_naive:>+9.1f}%")

# ============================================================
# Year-by-year for top models
# ============================================================
top_models = ["OLS Level (full)", "XGB Level (full)", "RF Level (full)", "OLS Δprice", "XGB Δprice", "RF Δprice"]

print("\n" + "=" * 90)
print("YEAR-BY-YEAR RMSE: Top models")
print("=" * 90)
header = f"{'Year':>6}"
for name in top_models:
    short = name.replace("Level ", "Lv").replace("(full)", "F")
    header += f" {short:>16}"
print(header)
print("-" * 90)

for test_year in years[MIN_TRAIN:]:
    row = f"{test_year:>6}"
    for name in top_models:
        res = all_results[name]
        yr = res[res["year"] == test_year]
        if len(yr) > 0:
            row += f" ${yr.iloc[0]['rmse']:>14.2f}"
        else:
            row += f" {'N/A':>15}"
    print(row)

# ============================================================
# Feature importance comparison
# ============================================================
print("\n" + "=" * 80)
print("FEATURE IMPORTANCE: XGB Level (full) vs XGB Δprice")
print("=" * 80)

imp_level = get_importance(df, LEVEL_FEATURES, "price", XGBRegressor, XGB_PARAMS)
imp_delta = get_importance(df, CHANGE_FEATURES, "dprice", XGBRegressor, XGB_PARAMS)

print(f"\n{'Feature':<25} {'Level':>10} {'Δprice':>10}")
print("-" * 47)

# Combine and show all
all_feats = sorted(set(imp_level.index) | set(imp_delta.index),
                   key=lambda f: imp_level.get(f, 0) + imp_delta.get(f, 0),
                   reverse=True)
for feat in all_feats[:15]:
    lv = imp_level.get(feat, 0)
    dv = imp_delta.get(feat, 0)
    print(f"  {feat:<23} {lv:>10.4f} {dv:>10.4f}")

# ============================================================
# Save
# ============================================================
summary_rows = []
for name, res in all_results.items():
    summary_rows.append({
        "model": name,
        "avg_rmse": res["rmse"].mean(),
        "avg_mae": res["mae"].mean(),
    })
pd.DataFrame(summary_rows).to_csv(OUT / "model_comparison_all.csv", index=False)
print(f"\nSaved: {OUT / 'model_comparison_all.csv'}")

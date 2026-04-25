"""
Market-level volume prediction for Iceberg lettuce.

Dataset A (aggregated). Two targets:
  1. Volume level
  2. Δvolume (volume change)

Methods: OLS vs XGBoost, expanding window validation.
No price lags (causality: supply → price, not reverse).
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"

mkt = pd.read_csv(PROC / "market_iceberg_weekly.csv", parse_dates=["week_ending"])

# ============================================================
# Construct targets
# ============================================================
mkt["dvolume"] = mkt["volume"] - mkt["volume_lag1"]

# ============================================================
# Feature sets (no price lags)
# ============================================================
WEATHER = [
    "tmax_avg", "tmin_avg", "ppt_total", "diurnal_range",
    "extreme_heat", "freeze_risk", "heavy_rain",
]

WEATHER_LAGS = [
    "tmax_lag1", "tmax_lag2", "ppt_lag1",
    "tmax_avg_roll4_mean", "ppt_total_roll4_mean",
]

VOLUME_LAGS = ["volume_lag1", "volume_lag2", "volume_roll4_mean"]

CALENDAR = ["month", "week_of_year"]

OTHER = ["coverage", "n_districts"]

LEVEL_FEATURES = WEATHER + WEATHER_LAGS + VOLUME_LAGS + CALENDAR + OTHER
CHANGE_FEATURES = WEATHER + WEATHER_LAGS + CALENDAR + OTHER  # no volume lags

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

# ============================================================
# OLS: Volume level (in-sample)
# ============================================================
print("=" * 75)
print("OLS: Volume ~ weather + volume lags + calendar")
print("=" * 75)

df_ols = mkt[["volume"] + LEVEL_FEATURES].dropna()
X = sm.add_constant(df_ols[LEVEL_FEATURES])
y = df_ols["volume"]
m_level = sm.OLS(y, X).fit(cov_type="HC1")
print(m_level.summary2().tables[1].to_string())
print(f"\nN={m_level.nobs:.0f}, R²={m_level.rsquared:.4f}, RMSE={np.sqrt(m_level.mse_resid):.0f}")

# ============================================================
# OLS: Δvolume (in-sample)
# ============================================================
print("\n" + "=" * 75)
print("OLS: Δvolume ~ weather + calendar (no volume lags)")
print("=" * 75)

df_dv = mkt[["dvolume"] + CHANGE_FEATURES].dropna()
X_dv = sm.add_constant(df_dv[CHANGE_FEATURES])
y_dv = df_dv["dvolume"]
m_change = sm.OLS(y_dv, X_dv).fit(cov_type="HC1")
print(m_change.summary2().tables[1].to_string())
print(f"\nN={m_change.nobs:.0f}, R²={m_change.rsquared:.4f}, RMSE={np.sqrt(m_change.mse_resid):.0f}")

# ============================================================
# Expanding window validation
# ============================================================
_cols = list(dict.fromkeys(["week_ending", "volume", "dvolume", "volume_lag1"] + LEVEL_FEATURES))
df = mkt[_cols].dropna()
df["year"] = df["week_ending"].dt.year
years = sorted(df["year"].unique())
MIN_TRAIN = 4


def expanding_eval(features, target, model_cls, model_kwargs, convert_to_level=None):
    results = []
    for test_year in years[MIN_TRAIN:]:
        train = df[df["year"] < test_year]
        test = df[df["year"] == test_year]
        if len(test) == 0:
            continue

        m = model_cls(**model_kwargs)
        m.fit(train[features], train[target])
        pred = m.predict(test[features])

        if convert_to_level is not None:
            pred_level = test["volume_lag1"].values + pred
            actual = test["volume"].values
        else:
            pred_level = pred
            actual = test["volume"].values

        results.append({
            "year": test_year,
            "n": len(test),
            "rmse": np.sqrt(mean_squared_error(actual, pred_level)),
            "mae": mean_absolute_error(actual, pred_level),
        })
    return pd.DataFrame(results)


models = {
    "Naive (vol=lag1)": None,
    "OLS Level": (LEVEL_FEATURES, "volume", LinearRegression, {}, None),
    "XGB Level": (LEVEL_FEATURES, "volume", XGBRegressor, XGB_PARAMS, None),
    "OLS Δvolume": (CHANGE_FEATURES, "dvolume", LinearRegression, {},
                     lambda p, r: r + p),
    "XGB Δvolume": (CHANGE_FEATURES, "dvolume", XGBRegressor, XGB_PARAMS,
                     lambda p, r: r + p),
}

# Naive baseline
naive_results = []
for test_year in years[MIN_TRAIN:]:
    test = df[df["year"] == test_year]
    if len(test) == 0:
        continue
    naive_results.append({
        "year": test_year,
        "rmse": np.sqrt(mean_squared_error(test["volume"].to_numpy(), test["volume_lag1"].to_numpy())),
        "mae": mean_absolute_error(test["volume"].to_numpy(), test["volume_lag1"].to_numpy()),
    })
naive_df = pd.DataFrame(naive_results)

all_results = {"Naive (vol=lag1)": naive_df}
for name, spec in models.items():
    if spec is None:
        continue
    feats, target, cls, kwargs, conv = spec
    all_results[name] = expanding_eval(feats, target, cls, kwargs, conv)

# ============================================================
# Print comparison
# ============================================================
print("\n" + "=" * 75)
print("EXPANDING WINDOW VALIDATION: Volume prediction")
print("=" * 75)
print(f"{'Model':<25} {'Avg RMSE':>10} {'Avg MAE':>10} {'vs Naive':>10}")
print("-" * 57)

naive_avg = naive_df["rmse"].mean()
for name, res in all_results.items():
    avg_rmse = res["rmse"].mean()
    avg_mae = res["mae"].mean() if "mae" in res.columns else 0
    vs = (naive_avg - avg_rmse) / naive_avg * 100
    print(f"{name:<25} {avg_rmse:>10.0f} {avg_mae:>10.0f} {vs:>+9.1f}%")

# Year-by-year for top models
print("\n" + "=" * 80)
print("YEAR-BY-YEAR RMSE")
print("=" * 80)
top = ["Naive (vol=lag1)", "OLS Level", "XGB Level", "OLS Δvolume", "XGB Δvolume"]
header = f"{'Year':>6}"
for name in top:
    header += f" {name[:12]:>13}"
print(header)
print("-" * 80)
for test_year in years[MIN_TRAIN:]:
    row = f"{test_year:>6}"
    for name in top:
        yr = all_results[name]
        yr_data = yr[yr["year"] == test_year]
        if len(yr_data) > 0:
            row += f" {yr_data.iloc[0]['rmse']:>13.0f}"
        else:
            row += f" {'N/A':>13}"
    print(row)

# ============================================================
# Feature importance: XGB Level
# ============================================================
print("\n" + "=" * 75)
print("FEATURE IMPORTANCE: XGB Volume Level")
print("=" * 75)

xgb_full = XGBRegressor(**XGB_PARAMS)
xgb_full.fit(df[LEVEL_FEATURES], df["volume"])
imp = pd.Series(xgb_full.feature_importances_, index=LEVEL_FEATURES).sort_values(ascending=False)
for feat, val in imp.items():
    bar = "█" * int(val * 100)
    print(f"  {feat:<25} {val:.4f} {bar}")

# ============================================================
# Weather coefficients comparison
# ============================================================
print("\n" + "=" * 75)
print("WEATHER COEFFICIENTS: Level vs Change OLS")
print("=" * 75)
print(f"{'Variable':<25} {'Level Coef':>12} {'Level p':>10} {'ΔVol Coef':>12} {'ΔVol p':>10}")
print("-" * 71)
for var in WEATHER:
    lc = m_level.params.get(var, 0)
    lp = m_level.pvalues.get(var, 1)
    dc = m_change.params.get(var, 0)
    dp = m_change.pvalues.get(var, 1)
    ls = "***" if lp < 0.01 else "**" if lp < 0.05 else "*" if lp < 0.1 else ""
    ds = "***" if dp < 0.01 else "**" if dp < 0.05 else "*" if dp < 0.1 else ""
    print(f"  {var:<23} {lc:>10.1f}{ls:<3} {lp:>10.4f} {dc:>10.1f}{ds:<3} {dp:>10.4f}")

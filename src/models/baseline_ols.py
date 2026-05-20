"""
Baseline OLS regression for market-level Iceberg lettuce price prediction.

Target: price (shipment-weighted FOB price)
Features: weather, lagged price/volume, calendar, coverage
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"

# ============================================================
# Load market dataset
# ============================================================
mkt = pd.read_csv(PROC / "market_iceberg_weekly.csv", parse_dates=["week_ending"])

# ============================================================
# Define features
# ============================================================
WEATHER = ["tmax_avg", "tmin_avg", "ppt_total", "diurnal_range",
           "dd_heat", "dd_freeze", "heavy_rain"]

LAGS = ["price_lag1", "price_lag2", "price_lag4",
        "volume_lag1", "volume_lag2",
        "tmax_lag1", "tmax_lag2", "ppt_lag1"]

ROLLING = ["volume_roll4_mean", "tmax_avg_roll4_mean", "ppt_total_roll4_mean"]

CALENDAR = ["month", "week_of_year"]

OTHER = ["coverage", "n_districts"]

# ============================================================
# Model 1: Simple OLS (weather only)
# ============================================================
print("=" * 70)
print("MODEL 1: Price ~ Weather (contemporaneous)")
print("=" * 70)

features_1 = WEATHER + CALENDAR
df1 = mkt[["price"] + features_1].dropna()
X1 = sm.add_constant(df1[features_1])
y1 = df1["price"]

model1 = sm.OLS(y1, X1).fit(cov_type="HC1")
print(model1.summary2().tables[0].to_string())
print()
print(model1.summary2().tables[1].to_string())
print(f"\nN={model1.nobs:.0f}, R²={model1.rsquared:.4f}, Adj R²={model1.rsquared_adj:.4f}")
print(f"RMSE={np.sqrt(model1.mse_resid):.2f}")

# ============================================================
# Model 2: OLS with lags (autoregressive)
# ============================================================
print("\n" + "=" * 70)
print("MODEL 2: Price ~ Weather + Lags + Calendar")
print("=" * 70)

features_2 = WEATHER + LAGS + ROLLING + CALENDAR + OTHER
df2 = mkt[["price"] + features_2].dropna()
X2 = sm.add_constant(df2[features_2])
y2 = df2["price"]

model2 = sm.OLS(y2, X2).fit(cov_type="HC1")
print(model2.summary2().tables[0].to_string())
print()
print(model2.summary2().tables[1].to_string())
print(f"\nN={model2.nobs:.0f}, R²={model2.rsquared:.4f}, Adj R²={model2.rsquared_adj:.4f}")
print(f"RMSE={np.sqrt(model2.mse_resid):.2f}")

# ============================================================
# Model 3: Log price (better for skewed distribution)
# ============================================================
print("\n" + "=" * 70)
print("MODEL 3: Log(Price) ~ Weather + Lags + Calendar")
print("=" * 70)

df3 = mkt[["price"] + features_2].dropna()
df3 = df3[df3["price"] > 0]
X3 = sm.add_constant(df3[features_2])
y3 = np.log(df3["price"])

model3 = sm.OLS(y3, X3).fit(cov_type="HC1")
print(model3.summary2().tables[0].to_string())
print()
print(model3.summary2().tables[1].to_string())
print(f"\nN={model3.nobs:.0f}, R²={model3.rsquared:.4f}, Adj R²={model3.rsquared_adj:.4f}")
print(f"RMSE (log scale)={np.sqrt(model3.mse_resid):.4f}")

# ============================================================
# Summary comparison
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'Model':<40} {'N':>5} {'R²':>8} {'Adj R²':>8} {'RMSE':>8}")
print("-" * 70)
print(f"{'1. Weather only':<40} {model1.nobs:>5.0f} {model1.rsquared:>8.4f} {model1.rsquared_adj:>8.4f} {np.sqrt(model1.mse_resid):>8.2f}")
print(f"{'2. Weather + Lags + Calendar':<40} {model2.nobs:>5.0f} {model2.rsquared:>8.4f} {model2.rsquared_adj:>8.4f} {np.sqrt(model2.mse_resid):>8.2f}")
print(f"{'3. Log(Price) + Lags + Calendar':<40} {model3.nobs:>5.0f} {model3.rsquared:>8.4f} {model3.rsquared_adj:>8.4f} {np.sqrt(model3.mse_resid):>8.4f}")

"""
District-level panel OLS for Iceberg lettuce price.

Dataset B: 3 core districts (Salinas, Western AZ, Imperial)
Two specifications:
  1. Δprice ~ weather + district FE + season FE
  2. Price level ~ weather + district FE + season FE + lags

Key question: does weather show up more clearly at district level
than in the market-level aggregate?
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"

dist = pd.read_csv(PROC / "district_iceberg_weekly.csv", parse_dates=["week_ending"])

# ============================================================
# Prepare
# ============================================================
# Create Δprice
dist["dprice"] = dist["price"] - dist["price_lag1"]

# District dummies (Salinas as reference)
dist["d_western_az"] = (dist["district"] == "Western Arizona").astype(int)
dist["d_imperial"] = (dist["district"] == "Imperial Valley").astype(int)

# Season dummies (quarter-based)
dist["q1"] = (dist["month"].isin([1, 2, 3])).astype(int)   # winter
dist["q2"] = (dist["month"].isin([4, 5, 6])).astype(int)   # spring
dist["q3"] = (dist["month"].isin([7, 8, 9])).astype(int)   # summer
# Q4 (Oct-Dec) is reference

# District x season interactions
dist["az_winter"] = dist["d_western_az"] * dist["q1"]
dist["az_spring"] = dist["d_western_az"] * dist["q2"]

print(f"Total rows: {len(dist)}")
print(f"With price: {dist['price'].notna().sum()}")
print(f"With Δprice: {dist['dprice'].notna().sum()}")
print(f"Districts: {sorted(dist['district'].unique())}")

# ============================================================
# Model 1: Δprice ~ weather + district FE + season
# ============================================================
print("\n" + "=" * 75)
print("MODEL 1: Δprice ~ weather + district FE + season")
print("=" * 75)

WEATHER = ["tmax_avg", "tmin_avg", "ppt_total", "diurnal_range",
           "dd_heat", "dd_freeze", "heavy_rain"]
DISTRICT_FE = ["d_western_az", "d_imperial"]
SEASON = ["q1", "q2", "q3"]
INTERACT = ["az_winter", "az_spring"]

features_1 = WEATHER + DISTRICT_FE + SEASON + INTERACT
df1 = dist[["dprice"] + features_1].dropna()
X1 = sm.add_constant(df1[features_1])
y1 = df1["dprice"]

m1 = sm.OLS(y1, X1).fit(cov_type="cluster", cov_kwds={"groups": dist.loc[df1.index, "district"]})
print(m1.summary2().tables[1].to_string())
print(f"\nN={m1.nobs:.0f}, R²={m1.rsquared:.4f}, RMSE={np.sqrt(m1.mse_resid):.2f}")

# ============================================================
# Model 2: Δprice ~ weather + district FE + season + volume change
# ============================================================
print("\n" + "=" * 75)
print("MODEL 2: Δprice ~ weather + district FE + season + volume lags")
print("=" * 75)

dist["volume_change"] = dist["volume"] - dist["volume_lag1"]

MARKET_VARS = ["volume_lag1", "volume_change", "volume_roll4_mean"]
features_2 = WEATHER + DISTRICT_FE + SEASON + INTERACT + MARKET_VARS
df2 = dist[["dprice"] + features_2].dropna()
X2 = sm.add_constant(df2[features_2])
y2 = df2["dprice"]

m2 = sm.OLS(y2, X2).fit(cov_type="cluster", cov_kwds={"groups": dist.loc[df2.index, "district"]})
print(m2.summary2().tables[1].to_string())
print(f"\nN={m2.nobs:.0f}, R²={m2.rsquared:.4f}, RMSE={np.sqrt(m2.mse_resid):.2f}")

# ============================================================
# Model 3: Δprice ~ weather + lagged weather + district FE + season
# ============================================================
print("\n" + "=" * 75)
print("MODEL 3: Δprice ~ weather + lagged weather + district FE + season + volume")
print("=" * 75)

WEATHER_LAGS = ["tmax_lag1", "tmax_lag2", "ppt_lag1"]
features_3 = WEATHER + WEATHER_LAGS + DISTRICT_FE + SEASON + INTERACT + MARKET_VARS
df3 = dist[["dprice"] + features_3].dropna()
X3 = sm.add_constant(df3[features_3])
y3 = df3["dprice"]

m3 = sm.OLS(y3, X3).fit(cov_type="cluster", cov_kwds={"groups": dist.loc[df3.index, "district"]})
print(m3.summary2().tables[1].to_string())
print(f"\nN={m3.nobs:.0f}, R²={m3.rsquared:.4f}, RMSE={np.sqrt(m3.mse_resid):.2f}")

# ============================================================
# Model 4: Price level ~ weather + lags + district FE + season
# ============================================================
print("\n" + "=" * 75)
print("MODEL 4: Price level ~ weather + price lags + district FE + season")
print("=" * 75)

PRICE_LAGS = ["price_lag1", "price_lag2", "price_lag4"]
features_4 = WEATHER + WEATHER_LAGS + PRICE_LAGS + DISTRICT_FE + SEASON + INTERACT + MARKET_VARS
df4 = dist[["price"] + features_4].dropna()
X4 = sm.add_constant(df4[features_4])
y4 = df4["price"]

m4 = sm.OLS(y4, X4).fit(cov_type="cluster", cov_kwds={"groups": dist.loc[df4.index, "district"]})
print(m4.summary2().tables[1].to_string())
print(f"\nN={m4.nobs:.0f}, R²={m4.rsquared:.4f}, RMSE={np.sqrt(m4.mse_resid):.2f}")

# ============================================================
# Model 5: Transition week interaction
# ============================================================
print("\n" + "=" * 75)
print("MODEL 5: Δprice ~ weather + transition_week + weather × transition + FE")
print("=" * 75)

dist["transition_week"] = dist["week_of_year"].isin(list(range(15, 21)) + list(range(45, 51))).astype(int)
dist["freeze_x_transition"] = dist["dd_freeze"] * dist["transition_week"]
dist["heat_x_transition"] = dist["dd_heat"] * dist["transition_week"]
dist["ppt_x_transition"] = dist["ppt_total"] * dist["transition_week"]

TRANSITION = ["transition_week", "freeze_x_transition", "heat_x_transition", "ppt_x_transition"]
features_5 = WEATHER + DISTRICT_FE + SEASON + INTERACT + MARKET_VARS + TRANSITION
df5 = dist[["dprice"] + features_5].dropna()
X5 = sm.add_constant(df5[features_5])
y5 = df5["dprice"]

m5 = sm.OLS(y5, X5).fit(cov_type="cluster", cov_kwds={"groups": dist.loc[df5.index, "district"]})
print(m5.summary2().tables[1].to_string())
print(f"\nN={m5.nobs:.0f}, R²={m5.rsquared:.4f}, RMSE={np.sqrt(m5.mse_resid):.2f}")

# Transition week summary
n_trans = dist["transition_week"].sum()
n_total = len(dist)
print(f"\nTransition weeks: {n_trans}/{n_total} ({n_trans/n_total*100:.1f}%)")
print(f"Avg Δprice in transition weeks: ${dist.loc[dist['transition_week']==1, 'dprice'].mean():.2f}")
print(f"Avg Δprice in non-transition weeks: ${dist.loc[dist['transition_week']==0, 'dprice'].mean():.2f}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 75)
print("SUMMARY")
print("=" * 75)
print(f"{'Model':<55} {'N':>5} {'R²':>8} {'RMSE':>8}")
print("-" * 78)
print(f"{'1. Δprice ~ weather + FE + season':<55} {m1.nobs:>5.0f} {m1.rsquared:>8.4f} {np.sqrt(m1.mse_resid):>8.2f}")
print(f"{'2. Δprice ~ weather + FE + season + volume':<55} {m2.nobs:>5.0f} {m2.rsquared:>8.4f} {np.sqrt(m2.mse_resid):>8.2f}")
print(f"{'3. Δprice ~ weather + lagged weather + FE + volume':<55} {m3.nobs:>5.0f} {m3.rsquared:>8.4f} {np.sqrt(m3.mse_resid):>8.2f}")
print(f"{'4. Price level ~ weather + price lags + FE + volume':<55} {m4.nobs:>5.0f} {m4.rsquared:>8.4f} {np.sqrt(m4.mse_resid):>8.2f}")
print(f"{'5. Δprice ~ weather + transition interaction + FE':<55} {m5.nobs:>5.0f} {m5.rsquared:>8.4f} {np.sqrt(m5.mse_resid):>8.2f}")

# ============================================================
# Key weather coefficients comparison
# ============================================================
print("\n" + "=" * 75)
print("WEATHER COEFFICIENTS ACROSS MODELS")
print("=" * 75)
print(f"{'Variable':<20} {'M1 Δp':>10} {'M2 Δp+V':>10} {'M3 +WLag':>10} {'M4 Level':>10}")
print("-" * 62)
for var in WEATHER:
    vals = []
    for m in [m1, m2, m3, m4]:
        if var in m.params:
            coef = m.params[var]
            pval = m.pvalues[var]
            star = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            vals.append(f"{coef:>7.3f}{star}")
        else:
            vals.append(f"{'N/A':>10}")
    print(f"  {var:<18} {'  '.join(vals)}")

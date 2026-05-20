"""
Diagnostics for market-level OLS price model.

1. Rolling vs expanding window validation
2. Coverage vs prediction error
3. 2022 extreme year analysis
4. Feature importance stability across time periods
5. Early years vs later years performance
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

mkt = pd.read_csv(PROC / "market_iceberg_weekly.csv", parse_dates=["week_ending"])

FEATURES = [
    "tmax_avg", "tmin_avg", "ppt_total", "diurnal_range",
    "dd_heat", "dd_freeze", "heavy_rain",
    "tmax_lag1", "tmax_lag2", "ppt_lag1",
    "tmax_avg_roll4_mean", "ppt_total_roll4_mean",
    "price_lag1", "price_lag2", "price_lag4",
    "volume_lag1", "volume_lag2", "volume_roll4_mean",
    "month", "week_of_year",
    "coverage", "n_districts",
]

_cols = list(dict.fromkeys(["week_ending", "price", "coverage"] + FEATURES))
df = mkt[_cols].dropna()
df["year"] = df["week_ending"].dt.year
years = sorted(df["year"].unique())

# ============================================================
# 1. Expanding vs Rolling window validation
# ============================================================
print("=" * 75)
print("1. EXPANDING vs ROLLING WINDOW VALIDATION")
print("=" * 75)

def run_validation(df, years, mode="expanding", window_years=5):
    results = []
    for test_year in years[4:]:  # start from 2014
        if mode == "expanding":
            train = df[df["year"] < test_year]
        else:  # rolling
            train = df[(df["year"] >= test_year - window_years) & (df["year"] < test_year)]

        test = df[df["year"] == test_year]
        if len(train) < 50 or len(test) == 0:
            continue

        m = LinearRegression()
        m.fit(train[FEATURES], train["price"])
        pred = m.predict(test[FEATURES])

        results.append({
            "year": test_year,
            "n_train": len(train),
            "n_test": len(test),
            "rmse": np.sqrt(mean_squared_error(test["price"], pred)),
            "mae": mean_absolute_error(test["price"], pred),
        })
    return pd.DataFrame(results)

exp = run_validation(df, years, "expanding")
roll = run_validation(df, years, "rolling", window_years=5)

print(f"{'Year':>6} | {'Expanding':>15} | {'Rolling(5yr)':>15}")
print(f"{'':>6} | {'RMSE':>7} {'MAE':>7} | {'RMSE':>7} {'MAE':>7}")
print("-" * 50)
for _, er in exp.iterrows():
    rr = roll[roll["year"] == er["year"]]
    if len(rr) > 0:
        rr = rr.iloc[0]
        print(f"{er['year']:>6.0f} | ${er['rmse']:>6.2f} ${er['mae']:>6.2f} | ${rr['rmse']:>6.2f} ${rr['mae']:>6.2f}")
    else:
        print(f"{er['year']:>6.0f} | ${er['rmse']:>6.2f} ${er['mae']:>6.2f} | {'N/A':>15}")

print("-" * 50)
print(f"{'AVG':>6} | ${exp['rmse'].mean():>6.2f} ${exp['mae'].mean():>6.2f} | "
      f"${roll['rmse'].mean():>6.2f} ${roll['mae'].mean():>6.2f}")

# ============================================================
# 2. Coverage vs prediction error
# ============================================================
print("\n" + "=" * 75)
print("2. COVERAGE vs PREDICTION ERROR")
print("=" * 75)

# Get out-of-sample predictions from expanding window
all_preds = []
for test_year in years[4:]:
    train = df[df["year"] < test_year]
    test = df[df["year"] == test_year]
    if len(test) == 0:
        continue
    m = LinearRegression()
    m.fit(train[FEATURES], train["price"])
    pred = m.predict(test[FEATURES])
    for i in range(len(test)):
        all_preds.append({
            "week_ending": test.iloc[i]["week_ending"],
            "actual": test.iloc[i]["price"],
            "predicted": pred[i],
            "coverage": test.iloc[i]["coverage"],
            "year": test_year,
        })

pred_df = pd.DataFrame(all_preds)
pred_df["abs_error"] = (pred_df["actual"] - pred_df["predicted"]).abs()
pred_df["pct_error"] = pred_df["abs_error"] / pred_df["actual"] * 100

bins = [(0, 0.5, "Very low (<0.5)"),
        (0.5, 0.7, "Low (0.5-0.7)"),
        (0.7, 0.85, "Medium (0.7-0.85)"),
        (0.85, 0.95, "High (0.85-0.95)"),
        (0.95, 1.01, "Very high (>0.95)")]

print(f"{'Coverage':<22} {'N':>5} {'MAE($)':>8} {'Med Err($)':>11} {'MAPE(%)':>9}")
print("-" * 57)
for low, high, label in bins:
    sub = pred_df[(pred_df["coverage"] >= low) & (pred_df["coverage"] < high)]
    if len(sub) > 0:
        print(f"{label:<22} {len(sub):>5} ${sub['abs_error'].mean():>7.2f} "
              f"${sub['abs_error'].median():>10.2f} {sub['pct_error'].mean():>8.1f}%")

# ============================================================
# 3. 2022 extreme year analysis
# ============================================================
print("\n" + "=" * 75)
print("3. 2022 EXTREME YEAR ANALYSIS")
print("=" * 75)

yr22 = pred_df[pred_df["year"] == 2022]
other = pred_df[pred_df["year"] != 2022]

print(f"2022:  N={len(yr22)}, MAE=${yr22['abs_error'].mean():.2f}, "
      f"max_error=${yr22['abs_error'].max():.2f}")
print(f"Other: N={len(other)}, MAE=${other['abs_error'].mean():.2f}, "
      f"max_error=${other['abs_error'].max():.2f}")

# What happened in 2022?
mkt22 = mkt[mkt["week_ending"].dt.year == 2022]
print(f"\n2022 price stats:")
print(f"  Mean: ${mkt22['price'].mean():.2f}")
print(f"  Max:  ${mkt22['price'].max():.2f}")
print(f"  Std:  ${mkt22['price'].std():.2f}")
print(f"  Price spikes: {mkt22['price_spike'].sum()}")

all_years_stats = mkt.groupby(mkt["week_ending"].dt.year).agg(
    price_mean=("price", "mean"),
    price_max=("price", "max"),
    price_std=("price", "std"),
).round(2)
print(f"\nPrice volatility by year (std):")
print(all_years_stats.sort_values("price_std", ascending=False).head(5).to_string())

# ============================================================
# 4. Feature importance stability
# ============================================================
print("\n" + "=" * 75)
print("4. FEATURE IMPORTANCE STABILITY ACROSS PERIODS")
print("=" * 75)

periods = [
    ("2010-2015", 2010, 2015),
    ("2016-2020", 2016, 2020),
    ("2021-2026", 2021, 2026),
]

coef_stability = {}
for label, y1, y2 in periods:
    sub = df[(df["year"] >= y1) & (df["year"] <= y2)]
    if len(sub) < 30:
        continue
    m = LinearRegression()
    m.fit(sub[FEATURES], sub["price"])
    coefs = pd.Series(m.coef_, index=FEATURES)
    coef_stability[label] = coefs

if coef_stability:
    cs = pd.DataFrame(coef_stability)
    # Show key weather variables
    key_vars = ["tmax_avg", "tmin_avg", "ppt_total", "dd_freeze",
                "dd_heat", "heavy_rain", "price_lag1"]
    print(f"{'Variable':<20}", end="")
    for label in coef_stability:
        print(f" {label:>12}", end="")
    print()
    print("-" * (20 + 13 * len(coef_stability)))
    for var in key_vars:
        print(f"  {var:<18}", end="")
        for label in coef_stability:
            print(f" {cs.loc[var, label]:>12.4f}", end="")
        print()

# ============================================================
# 5. Early vs late performance
# ============================================================
print("\n" + "=" * 75)
print("5. EARLY vs LATE YEARS PERFORMANCE")
print("=" * 75)

early = pred_df[pred_df["year"] <= 2018]
late = pred_df[pred_df["year"] >= 2019]

print(f"{'Period':<15} {'N':>5} {'RMSE':>8} {'MAE':>8} {'MAPE':>8}")
print("-" * 46)
for label, sub in [("2014-2018", early), ("2019-2026", late), ("All", pred_df)]:
    rmse = np.sqrt((sub["abs_error"] ** 2).mean())
    mae = sub["abs_error"].mean()
    mape = sub["pct_error"].mean()
    print(f"{label:<15} {len(sub):>5} ${rmse:>7.2f} ${mae:>7.2f} {mape:>7.1f}%")

# ============================================================
# 6. Partial year (2026) sensitivity
# ============================================================
print("\n" + "=" * 75)
print("6. PARTIAL YEAR (2026) SENSITIVITY")
print("=" * 75)

yr26 = pred_df[pred_df["year"] == 2026]
if len(yr26) > 0:
    print(f"2026: N={len(yr26)}, MAE=${yr26['abs_error'].mean():.2f}, "
          f"MAPE={yr26['pct_error'].mean():.1f}%")
    no26 = pred_df[pred_df["year"] != 2026]
    rmse_with = np.sqrt((pred_df["abs_error"] ** 2).mean())
    rmse_without = np.sqrt((no26["abs_error"] ** 2).mean())
    print(f"Overall RMSE with 2026: ${rmse_with:.2f}")
    print(f"Overall RMSE without 2026: ${rmse_without:.2f}")
    print(f"Impact: {'negligible' if abs(rmse_with - rmse_without) < 0.1 else 'notable'}")

# ============================================================
# Save predictions for visualization
# ============================================================
pred_df.to_csv(OUT / "diagnostics_predictions.csv", index=False)
print(f"\nSaved: {OUT / 'diagnostics_predictions.csv'}")

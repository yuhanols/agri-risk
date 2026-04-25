"""
Core visualization for AgriRisk project.

Fig 1: Actual vs Predicted Price (market-level OLS)
Fig 2: Feature Importance comparison (level vs change)
Fig 3: Coverage vs Prediction Error
Fig 4: Seasonal pattern (shipment + price by district)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})

# ============================================================
# Load data
# ============================================================
mkt = pd.read_csv(PROC / "market_iceberg_weekly.csv", parse_dates=["week_ending"])
dist = pd.read_csv(PROC / "district_iceberg_weekly.csv", parse_dates=["week_ending"])
pred = pd.read_csv(OUT / "diagnostics_predictions.csv", parse_dates=["week_ending"])

FEATURES_LEVEL = [
    "tmax_avg", "tmin_avg", "ppt_total", "diurnal_range",
    "extreme_heat", "freeze_risk", "heavy_rain",
    "tmax_lag1", "tmax_lag2", "ppt_lag1",
    "tmax_avg_roll4_mean", "ppt_total_roll4_mean",
    "price_lag1", "price_lag2", "price_lag4",
    "volume_lag1", "volume_lag2", "volume_roll4_mean",
    "month", "week_of_year", "coverage", "n_districts",
]

FEATURES_CHANGE = [
    "tmax_avg", "tmin_avg", "ppt_total", "diurnal_range",
    "extreme_heat", "freeze_risk", "heavy_rain",
    "tmax_lag1", "tmax_lag2", "ppt_lag1",
    "tmax_avg_roll4_mean", "ppt_total_roll4_mean",
    "volume_lag1", "volume_lag2", "volume_roll4_mean",
    "coverage", "n_districts", "month", "week_of_year",
]

# ============================================================
# Fig 1: Actual vs Predicted Price
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Panel A: full time series
ax = axes[0]
ax.plot(pred["week_ending"], pred["actual"], color="black", linewidth=0.8,
        label="Actual", alpha=0.9)
ax.plot(pred["week_ending"], pred["predicted"], color="#2196F3", linewidth=0.8,
        label="OLS Predicted", alpha=0.8)
ax.fill_between(pred["week_ending"], pred["actual"], pred["predicted"],
                alpha=0.15, color="#2196F3")
ax.set_ylabel("FOB Price ($/unit)")
ax.set_title("Market-Level Iceberg Lettuce Price: Actual vs OLS Predicted (Out-of-Sample)")
ax.legend(loc="upper left")
ax.set_ylim(0, None)

# Highlight 2022
yr22 = pred[pred["year"] == 2022]
if len(yr22) > 0:
    ax.axvspan(yr22["week_ending"].min(), yr22["week_ending"].max(),
               alpha=0.08, color="red", label="2022 (extreme)")

# Panel B: prediction error
ax2 = axes[1]
error = pred["actual"] - pred["predicted"]
ax2.bar(pred["week_ending"], error, width=5, color=np.where(error > 0, "#F44336", "#4CAF50"),
        alpha=0.6)
ax2.axhline(0, color="black", linewidth=0.5)
ax2.set_ylabel("Prediction Error ($)")
ax2.set_xlabel("Week")
ax2.set_title("Prediction Error (Actual - Predicted)")

ax2.xaxis.set_major_locator(mdates.YearLocator(2))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
plt.savefig(OUT / "fig1_actual_vs_predicted.png", bbox_inches="tight")
plt.close()
print("Saved: fig1_actual_vs_predicted.png")

# ============================================================
# Fig 2: Feature Importance (Level vs Change)
# ============================================================
mkt["dprice"] = mkt["price"] - mkt["price_lag1"]

_cols_l = list(dict.fromkeys(["price"] + FEATURES_LEVEL))
df_l = mkt[_cols_l].dropna()
_cols_c = list(dict.fromkeys(["dprice"] + FEATURES_CHANGE))
df_c = mkt[_cols_c].dropna()

XGB_PARAMS = {
    "n_estimators": 200, "max_depth": 3, "learning_rate": 0.03,
    "subsample": 0.7, "colsample_bytree": 0.7, "min_child_weight": 10,
    "reg_alpha": 1.0, "reg_lambda": 5.0, "random_state": 42, "verbosity": 0,
}

xgb_level = XGBRegressor(**XGB_PARAMS)
xgb_level.fit(df_l[FEATURES_LEVEL], df_l["price"])
imp_level = pd.Series(xgb_level.feature_importances_, index=FEATURES_LEVEL)

xgb_change = XGBRegressor(**XGB_PARAMS)
xgb_change.fit(df_c[FEATURES_CHANGE], df_c["dprice"])
imp_change = pd.Series(xgb_change.feature_importances_, index=FEATURES_CHANGE)

# Combine and sort by level importance
all_feats = sorted(set(imp_level.index) | set(imp_change.index),
                   key=lambda f: imp_level.get(f, 0), reverse=True)
top_feats = all_feats[:15]

fig, ax = plt.subplots(figsize=(10, 7))
y_pos = np.arange(len(top_feats))
width = 0.35

bars1 = ax.barh(y_pos + width/2, [imp_level.get(f, 0) for f in top_feats],
                width, label="Price Level", color="#1976D2", alpha=0.8)
bars2 = ax.barh(y_pos - width/2, [imp_change.get(f, 0) for f in top_feats],
                width, label="Price Change (Δp)", color="#FF7043", alpha=0.8)

ax.set_yticks(y_pos)
ax.set_yticklabels(top_feats)
ax.invert_yaxis()
ax.set_xlabel("Feature Importance")
ax.set_title("XGBoost Feature Importance: Price Level vs Price Change")
ax.legend()

plt.tight_layout()
plt.savefig(OUT / "fig2_feature_importance.png", bbox_inches="tight")
plt.close()
print("Saved: fig2_feature_importance.png")

# ============================================================
# Fig 3: Coverage vs Prediction Error
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(pred["coverage"], (pred["actual"] - pred["predicted"]).abs(),
           alpha=0.3, s=15, color="#1976D2")

# Bin averages
bins = [0, 0.5, 0.7, 0.85, 0.95, 1.01]
labels = ["<0.5", "0.5-0.7", "0.7-0.85", "0.85-0.95", ">0.95"]
pred["cov_bin"] = pd.cut(pred["coverage"], bins=bins, labels=labels)
pred["abs_error"] = (pred["actual"] - pred["predicted"]).abs()
bin_means = pred.groupby("cov_bin", observed=True)["abs_error"].mean()

bin_centers = [0.25, 0.6, 0.775, 0.9, 0.975]
ax.plot(bin_centers[:len(bin_means)], bin_means.values, "ro-", markersize=8,
        linewidth=2, label="Bin Average", zorder=5)

ax.set_xlabel("Coverage Ratio")
ax.set_ylabel("Absolute Prediction Error ($)")
ax.set_title("Price Coverage vs Prediction Error")
ax.legend()

plt.tight_layout()
plt.savefig(OUT / "fig3_coverage_vs_error.png", bbox_inches="tight")
plt.close()
print("Saved: fig3_coverage_vs_error.png")

# ============================================================
# Fig 4: Seasonal Pattern (Shipment + Price by District)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

colors = {"Salinas-Watsonville": "#1976D2", "Western Arizona": "#F44336",
          "Imperial Valley": "#4CAF50"}

# Panel A: Volume by week of year
ax = axes[0]
for district, color in colors.items():
    d = dist[dist["district"] == district]
    seasonal = d.groupby("week_of_year")["volume"].mean()
    ax.plot(seasonal.index, seasonal.values, color=color, linewidth=2,
            label=district, alpha=0.9)

ax.set_xlabel("Week of Year")
ax.set_ylabel("Avg Weekly Iceberg Shipment (tons)")
ax.set_title("Seasonal Shipment Pattern by District")
ax.legend(fontsize=9)
ax.set_xlim(1, 52)

# Panel B: Price by week of year
ax = axes[1]
for district, color in colors.items():
    d = dist[dist["district"] == district]
    seasonal = d.groupby("week_of_year")["price"].mean()
    ax.plot(seasonal.index, seasonal.values, color=color, linewidth=2,
            label=district, alpha=0.9)

ax.set_xlabel("Week of Year")
ax.set_ylabel("Avg Weekly Iceberg FOB Price ($)")
ax.set_title("Seasonal Price Pattern by District")
ax.legend(fontsize=9)
ax.set_xlim(1, 52)

plt.tight_layout()
plt.savefig(OUT / "fig4_seasonal_pattern.png", bbox_inches="tight")
plt.close()
print("Saved: fig4_seasonal_pattern.png")

print("\nAll figures saved to outputs/")

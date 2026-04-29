"""
Paper figures for AgriRisk project.

Figure 1: Seasonal Pattern by District        → Section 3 (Market Background)
Figure 2: Actual vs Predicted Price + Error    → Section 5.1 (Results)
Figure 3: Feature Importance Level vs Change   → Section 5.2 (Results)
Figure 4: Coverage vs Prediction Error         → Section 5.3 (Results)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from xgboost import XGBRegressor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
FIG = ROOT / "paper" / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

# ── Style ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 300,
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1.2,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# ── Load data ─────────────────────────────────────────────────
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

# Nicer feature labels for Fig 3
FEAT_LABELS = {
    "price_lag1": "Price (t-1)",
    "price_lag2": "Price (t-2)",
    "price_lag4": "Price (t-4)",
    "volume_lag1": "Volume (t-1)",
    "volume_lag2": "Volume (t-2)",
    "volume_roll4_mean": "Volume (4-wk avg)",
    "tmax_avg": "Max temp",
    "tmin_avg": "Min temp",
    "ppt_total": "Precipitation",
    "diurnal_range": "Diurnal range",
    "extreme_heat": "Extreme heat",
    "freeze_risk": "Freeze risk",
    "heavy_rain": "Heavy rain",
    "tmax_lag1": "Max temp (t-1)",
    "tmax_lag2": "Max temp (t-2)",
    "ppt_lag1": "Precip (t-1)",
    "tmax_avg_roll4_mean": "Max temp (4-wk avg)",
    "ppt_total_roll4_mean": "Precip (4-wk avg)",
    "month": "Month",
    "week_of_year": "Week of year",
    "coverage": "Coverage ratio",
    "n_districts": "No. districts",
}

# ============================================================
# Figure 1: Seasonal Pattern by District
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

colors = {
    "Salinas-Watsonville": "#1565C0",
    "Western Arizona": "#C62828",
    "Imperial Valley": "#2E7D32",
}

# Panel A: Volume
ax = axes[0]
for district, color in colors.items():
    d = dist[dist["district"] == district]
    seasonal = d.groupby("week_of_year")["volume"].mean()
    ax.plot(seasonal.index, seasonal.values, color=color, label=district)

# Shade transition periods
ax.axvspan(14, 18, alpha=0.08, color="gray")
ax.axvspan(44, 50, alpha=0.08, color="gray")
ax.text(16, ax.get_ylim()[1] * 0.92, "Spring\ntransition", ha="center",
        fontsize=7, color="gray")
ax.text(47, ax.get_ylim()[1] * 0.92, "Fall\ntransition", ha="center",
        fontsize=7, color="gray")

ax.set_xlabel("Week of year")
ax.set_ylabel("Avg weekly shipments (10,000 lbs)")
ax.set_title("(a) Shipment volume")
ax.legend(fontsize=8, loc="upper right")
ax.set_xlim(1, 52)

# Panel B: Price
ax = axes[1]
for district, color in colors.items():
    d = dist[dist["district"] == district]
    seasonal = d.groupby("week_of_year")["price"].mean()
    ax.plot(seasonal.index, seasonal.values, color=color, label=district)

ax.axvspan(14, 18, alpha=0.08, color="gray")
ax.axvspan(44, 50, alpha=0.08, color="gray")

ax.set_xlabel("Week of year")
ax.set_ylabel("Avg FOB price ($/carton)")
ax.set_title("(b) FOB price")
ax.legend(fontsize=8, loc="upper right")
ax.set_xlim(1, 52)

plt.tight_layout()
fig.savefig(FIG / "fig1_seasonal_pattern.pdf")
fig.savefig(OUT / "fig1_seasonal_pattern.png")
plt.close()
print("Saved: fig1_seasonal_pattern")

# ============================================================
# Figure 2: Actual vs Predicted Price + Error
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                         gridspec_kw={"height_ratios": [2.5, 1]})

# Panel A: price series
ax = axes[0]
ax.plot(pred["week_ending"], pred["actual"], color="black", linewidth=0.7,
        label="Actual", alpha=0.9)
ax.plot(pred["week_ending"], pred["predicted"], color="#E53935", linewidth=0.7,
        label="OLS predicted", alpha=0.8)
ax.fill_between(pred["week_ending"], pred["actual"], pred["predicted"],
                alpha=0.12, color="#E53935")

# Highlight 2022
yr22 = pred[pred["year"] == 2022]
if len(yr22) > 0:
    ax.axvspan(yr22["week_ending"].min(), yr22["week_ending"].max(),
               alpha=0.07, color="red")
    ax.annotate("2022", xy=(yr22["week_ending"].median(), pred["actual"].max() * 0.95),
                fontsize=8, color="#C62828", ha="center")

ax.set_ylabel("FOB price ($/carton)")
ax.set_title("(a) Market-level price: actual vs. OLS predicted (out-of-sample)")
ax.legend(loc="upper left", framealpha=0.9)
ax.set_ylim(0, None)

# Panel B: error
ax2 = axes[1]
error = pred["actual"] - pred["predicted"]
colors_err = np.where(error >= 0, "#C62828", "#2E7D32")
ax2.bar(pred["week_ending"], error, width=5, color=colors_err, alpha=0.5)
ax2.axhline(0, color="black", linewidth=0.4)
ax2.set_ylabel("Error ($)")
ax2.set_xlabel("")
ax2.set_title("(b) Prediction error (actual minus predicted)")

ax2.xaxis.set_major_locator(mdates.YearLocator(2))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
fig.savefig(FIG / "fig2_actual_vs_predicted.pdf")
fig.savefig(OUT / "fig2_actual_vs_predicted.png")
plt.close()
print("Saved: fig2_actual_vs_predicted")

# ============================================================
# Figure 3: Feature Importance (Level vs Change)
# ============================================================
mkt["dprice"] = mkt["price"] - mkt["price_lag1"]

df_l = mkt[["price"] + FEATURES_LEVEL].dropna()
df_c = mkt[["dprice"] + FEATURES_CHANGE].dropna()

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

# Show top 12, sorted by level importance
all_feats = sorted(set(imp_level.index) | set(imp_change.index),
                   key=lambda f: imp_level.get(f, 0), reverse=True)
top_feats = all_feats[:12]
top_labels = [FEAT_LABELS.get(f, f) for f in top_feats]

fig, ax = plt.subplots(figsize=(8, 5))
y_pos = np.arange(len(top_feats))
width = 0.35

ax.barh(y_pos + width / 2, [imp_level.get(f, 0) for f in top_feats],
        width, label="Price level", color="#1565C0", alpha=0.85)
ax.barh(y_pos - width / 2, [imp_change.get(f, 0) for f in top_feats],
        width, label="Price change ($\\Delta P$)", color="#FF7043", alpha=0.85)

ax.set_yticks(y_pos)
ax.set_yticklabels(top_labels)
ax.invert_yaxis()
ax.set_xlabel("Feature importance (XGBoost gain)")
ax.set_title("Feature importance: price level vs. price change models")
ax.legend(loc="lower right", framealpha=0.9)

plt.tight_layout()
fig.savefig(FIG / "fig3_feature_importance.pdf")
fig.savefig(OUT / "fig3_feature_importance.png")
plt.close()
print("Saved: fig3_feature_importance")

# ============================================================
# Figure 4: Coverage vs Prediction Error
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5))

abs_error = (pred["actual"] - pred["predicted"]).abs()
ax.scatter(pred["coverage"], abs_error, alpha=0.25, s=12, color="#1565C0",
           edgecolors="none")

# Bin averages
bins = [0, 0.5, 0.7, 0.85, 0.95, 1.01]
labels_bin = ["<0.50", "0.50\u20130.70", "0.70\u20130.85", "0.85\u20130.95", ">0.95"]
pred["cov_bin"] = pd.cut(pred["coverage"], bins=bins, labels=labels_bin)
pred["abs_error"] = abs_error
bin_stats = pred.groupby("cov_bin", observed=True)["abs_error"].agg(["mean", "count"])

bin_centers = [0.25, 0.6, 0.775, 0.9, 0.975]
ax.plot(bin_centers[:len(bin_stats)], bin_stats["mean"].values, "o-",
        color="#C62828", markersize=7, linewidth=1.5, label="Bin average", zorder=5)

# Annotate counts
for x, row in zip(bin_centers[:len(bin_stats)], bin_stats.itertuples()):
    ax.annotate(f"n={row.count}", xy=(x, row.mean), xytext=(0, 8),
                textcoords="offset points", fontsize=7, ha="center", color="#C62828")

ax.set_xlabel("Coverage ratio")
ax.set_ylabel("Absolute prediction error ($/carton)")
ax.set_title("Price coverage vs. prediction error")
ax.legend(loc="upper left", framealpha=0.9)

plt.tight_layout()
fig.savefig(FIG / "fig4_coverage_vs_error.pdf")
fig.savefig(OUT / "fig4_coverage_vs_error.png")
plt.close()
print("Saved: fig4_coverage_vs_error")

print("\nAll figures saved to paper/figures/ (PDF) and outputs/ (PNG)")

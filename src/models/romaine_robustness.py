"""
Romaine robustness check: replicate core Iceberg results with Romaine.

Build market-level dataset for Romaine, run same OLS models,
compare weather coefficients and model fit with Iceberg.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"

panel = pd.read_csv(PROC / "weekly_panel_2010_2026.csv", parse_dates=["week_ending"])

# Treat price=0 as missing
panel["price_romaine"] = panel["price_romaine"].replace(0, np.nan)

# ============================================================
# Build market-level Romaine dataset (same logic as Iceberg)
# ============================================================
weeks = sorted(panel["week_ending"].unique())
rows = []

for wk in weeks:
    wk_data = panel[panel["week_ending"] == wk]
    total_vol = wk_data["vol_romaine"].sum()

    has_price = wk_data[wk_data["price_romaine"].notna()]
    if len(has_price) > 0 and has_price["vol_romaine"].sum() > 0:
        wp = (has_price["vol_romaine"] * has_price["price_romaine"]).sum() / has_price["vol_romaine"].sum()
        price_vol = has_price["vol_romaine"].sum()
    else:
        wp = np.nan
        price_vol = 0

    coverage = price_vol / total_vol if total_vol > 0 else 0

    vol_nz = wk_data[wk_data["vol_romaine"] > 0]
    if len(vol_nz) > 0 and vol_nz["vol_romaine"].sum() > 0:
        v = vol_nz["vol_romaine"]
        w_tmax = (v * vol_nz["tmax_avg"]).sum() / v.sum()
        w_tmin = (v * vol_nz["tmin_avg"]).sum() / v.sum()
        w_ppt = (v * vol_nz["ppt_total"]).sum() / v.sum()
    else:
        w_tmax = w_tmin = w_ppt = np.nan

    rows.append({
        "week_ending": wk, "volume": total_vol, "price": wp,
        "coverage": round(coverage, 3),
        "tmax_avg": w_tmax, "tmin_avg": w_tmin, "ppt_total": w_ppt,
        "diurnal_range": (w_tmax - w_tmin) if not np.isnan(w_tmax) else np.nan,
        "extreme_heat": vol_nz["extreme_heat"].max() if len(vol_nz) > 0 else 0,
        "freeze_risk": vol_nz["freeze_risk"].max() if len(vol_nz) > 0 else 0,
        "heavy_rain": vol_nz["heavy_rain"].max() if len(vol_nz) > 0 else 0,
    })

rom = pd.DataFrame(rows)
rom["week_ending"] = pd.to_datetime(rom["week_ending"])
rom["year"] = rom["week_ending"].dt.year
rom["month"] = rom["week_ending"].dt.month
rom["week_of_year"] = rom["week_ending"].dt.isocalendar().week.astype(int)

# Lags
rom = rom.sort_values("week_ending")
for lag in [1, 2, 4]:
    rom[f"price_lag{lag}"] = rom["price"].shift(lag)
    rom[f"volume_lag{lag}"] = rom["volume"].shift(lag)
    rom[f"tmax_lag{lag}"] = rom["tmax_avg"].shift(lag)
    rom[f"ppt_lag{lag}"] = rom["ppt_total"].shift(lag)

rom["volume_roll4_mean"] = rom["volume"].rolling(4, min_periods=2).mean()
rom["tmax_avg_roll4_mean"] = rom["tmax_avg"].rolling(4, min_periods=2).mean()
rom["ppt_total_roll4_mean"] = rom["ppt_total"].rolling(4, min_periods=2).mean()

rom["dprice"] = rom["price"] - rom["price_lag1"]

print(f"Romaine market dataset: {len(rom)} weeks")
print(f"Price coverage: {rom['price'].notna().sum()} / {len(rom)} ({rom['price'].notna().mean()*100:.1f}%)")
print(f"Price mean: ${rom['price'].mean():.2f}, median: ${rom['price'].median():.2f}")

# ============================================================
# Load Iceberg for comparison
# ============================================================
ice = pd.read_csv(PROC / "market_iceberg_weekly.csv", parse_dates=["week_ending"])
ice["dprice"] = ice["price"] - ice["price_lag1"]

# ============================================================
# Features
# ============================================================
WEATHER = ["tmax_avg", "tmin_avg", "ppt_total", "diurnal_range",
           "extreme_heat", "freeze_risk", "heavy_rain"]
LAGS = ["price_lag1", "price_lag2", "price_lag4",
        "volume_lag1", "volume_lag2",
        "tmax_lag1", "tmax_lag2", "ppt_lag1"]
ROLLING = ["volume_roll4_mean", "tmax_avg_roll4_mean", "ppt_total_roll4_mean"]
CALENDAR = ["month", "week_of_year"]
OTHER = ["coverage"]

LEVEL_FEATURES = WEATHER + LAGS + ROLLING + CALENDAR + OTHER
CHANGE_FEATURES = [f for f in LEVEL_FEATURES if "price_lag" not in f]

# ============================================================
# OLS: Romaine price level
# ============================================================
print("\n" + "=" * 75)
print("ROMAINE: Price Level OLS (in-sample)")
print("=" * 75)

df_rom = rom[["price"] + LEVEL_FEATURES].dropna()
X = sm.add_constant(df_rom[LEVEL_FEATURES])
m_rom = sm.OLS(df_rom["price"], X).fit(cov_type="HC1")
print(f"N={m_rom.nobs:.0f}, R²={m_rom.rsquared:.4f}, RMSE=${np.sqrt(m_rom.mse_resid):.2f}")

# ============================================================
# OLS: Romaine Δprice
# ============================================================
print("\n" + "=" * 75)
print("ROMAINE: Δprice OLS (in-sample)")
print("=" * 75)

df_rom_d = rom[["dprice"] + CHANGE_FEATURES].dropna()
X_d = sm.add_constant(df_rom_d[CHANGE_FEATURES])
m_rom_d = sm.OLS(df_rom_d["dprice"], X_d).fit(cov_type="HC1")
print(f"N={m_rom_d.nobs:.0f}, R²={m_rom_d.rsquared:.4f}, RMSE=${np.sqrt(m_rom_d.mse_resid):.2f}")

# ============================================================
# Expanding window: Romaine vs Iceberg
# ============================================================
print("\n" + "=" * 75)
print("EXPANDING WINDOW: Romaine vs Iceberg OLS Level")
print("=" * 75)

def expanding_ols(data, features, target):
    df = data[[target] + features + ["year"]].dropna() if "year" in data.columns else data
    if "year" not in df.columns:
        df["year"] = pd.to_datetime(data.loc[df.index, "week_ending"]).dt.year
    years = sorted(df["year"].unique())
    results = []
    for test_year in years[4:]:
        train = df[df["year"] < test_year]
        test = df[df["year"] == test_year]
        if len(test) == 0:
            continue
        m = LinearRegression()
        m.fit(train[features], train[target])
        pred = m.predict(test[features])
        results.append({
            "year": test_year,
            "rmse": np.sqrt(mean_squared_error(test[target], pred)),
        })
    return pd.DataFrame(results)

rom["year"] = rom["week_ending"].dt.year
ice["year"] = ice["week_ending"].dt.year

rom_exp = expanding_ols(rom, LEVEL_FEATURES, "price")
ice_exp = expanding_ols(ice, LEVEL_FEATURES, "price")

print(f"{'Year':>6} {'Iceberg RMSE':>14} {'Romaine RMSE':>14}")
print("-" * 36)
for _, ir in ice_exp.iterrows():
    rr = rom_exp[rom_exp["year"] == ir["year"]]
    if len(rr) > 0:
        print(f"{ir['year']:>6.0f} ${ir['rmse']:>12.2f} ${rr.iloc[0]['rmse']:>12.2f}")
print("-" * 36)
print(f"{'AVG':>6} ${ice_exp['rmse'].mean():>12.2f} ${rom_exp['rmse'].mean():>12.2f}")

# ============================================================
# Weather coefficient comparison
# ============================================================
print("\n" + "=" * 75)
print("WEATHER COEFFICIENTS: Iceberg vs Romaine (Level OLS)")
print("=" * 75)

# Iceberg level OLS
_cols_ice = list(dict.fromkeys(["price"] + LEVEL_FEATURES))
df_ice = ice[_cols_ice].dropna()
X_ice = sm.add_constant(df_ice[LEVEL_FEATURES])
m_ice = sm.OLS(df_ice["price"], X_ice).fit(cov_type="HC1")

print(f"{'Variable':<25} {'Iceberg':>12} {'Romaine':>12} {'Same sign?':>12}")
print("-" * 63)
for var in WEATHER + ["price_lag1"]:
    ic = m_ice.params.get(var, 0)
    ip = m_ice.pvalues.get(var, 1)
    rc = m_rom.params.get(var, 0)
    rp = m_rom.pvalues.get(var, 1)
    is_ = "***" if ip < 0.01 else "**" if ip < 0.05 else "*" if ip < 0.1 else ""
    rs_ = "***" if rp < 0.01 else "**" if rp < 0.05 else "*" if rp < 0.1 else ""
    same = "Yes" if (ic * rc > 0) else "No"
    print(f"  {var:<23} {ic:>8.3f}{is_:<3} {rc:>8.3f}{rs_:<3} {same:>10}")

# ============================================================
# Δprice weather comparison
# ============================================================
print("\n" + "=" * 75)
print("WEATHER COEFFICIENTS: Iceberg vs Romaine (Δprice OLS)")
print("=" * 75)

_cols_ice_d = list(dict.fromkeys(["dprice"] + CHANGE_FEATURES))
df_ice_d = ice[_cols_ice_d].dropna()
X_ice_d = sm.add_constant(df_ice_d[CHANGE_FEATURES])
m_ice_d = sm.OLS(df_ice_d["dprice"], X_ice_d).fit(cov_type="HC1")

print(f"{'Variable':<25} {'Iceberg':>12} {'Romaine':>12} {'Same sign?':>12}")
print("-" * 63)
for var in WEATHER:
    ic = m_ice_d.params.get(var, 0)
    ip = m_ice_d.pvalues.get(var, 1)
    rc = m_rom_d.params.get(var, 0)
    rp = m_rom_d.pvalues.get(var, 1)
    is_ = "***" if ip < 0.01 else "**" if ip < 0.05 else "*" if ip < 0.1 else ""
    rs_ = "***" if rp < 0.01 else "**" if rp < 0.05 else "*" if rp < 0.1 else ""
    same = "Yes" if (ic * rc > 0) else "No"
    print(f"  {var:<23} {ic:>8.3f}{is_:<3} {rc:>8.3f}{rs_:<3} {same:>10}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 75)
print("SUMMARY")
print("=" * 75)
print(f"{'Metric':<35} {'Iceberg':>12} {'Romaine':>12}")
print("-" * 61)
print(f"{'In-sample R² (level)':<35} {m_ice.rsquared:>12.4f} {m_rom.rsquared:>12.4f}")
print(f"{'In-sample RMSE (level)':<35} ${np.sqrt(m_ice.mse_resid):>11.2f} ${np.sqrt(m_rom.mse_resid):>11.2f}")
print(f"{'OOS avg RMSE (expanding)':<35} ${ice_exp['rmse'].mean():>11.2f} ${rom_exp['rmse'].mean():>11.2f}")
print(f"{'In-sample R² (Δprice)':<35} {m_ice_d.rsquared:>12.4f} {m_rom_d.rsquared:>12.4f}")
print(f"{'Price mean':<35} ${ice['price'].mean():>11.2f} ${rom['price'].mean():>11.2f}")
print(f"{'Price coverage':<35} {ice['price'].notna().mean()*100:>11.1f}% {rom['price'].notna().mean()*100:>11.1f}%")

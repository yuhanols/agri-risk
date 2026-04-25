# AgriRisk: Climate-Driven Supply and Price Risk in U.S. Lettuce Markets

## Overview

Fresh produce markets lack real-time, in-season indicators of supply and price risk. In the U.S. lettuce market, production shifts seasonally between California (Salinas Valley) and Arizona (Yuma region), yet there is no widely available commercial system that translates weather shocks into actionable supply signals.

This project builds a weekly, data-driven pipeline that integrates USDA shipment data, FOB prices, and weather data to model climate-driven supply dynamics and price risk in the U.S. iceberg lettuce market.

Unlike traditional approaches focused on yield prediction, this project models downstream market outcomes, shipment volume and price, which are directly relevant for buyers, distributors, and supply chain decision-making.

## Key Findings

**1. Simple linear models outperform tree-based ML for price prediction.**

Across all modeling tasks (price, volume, missing price), OLS consistently outperforms XGBoost. Lettuce prices are highly persistent (autoregressive), and the underlying market dynamics are largely linear. The best market-level price model achieves RMSE of $4.36 out-of-sample (expanding window, 2014-2026).

**2. Weather effects are real but masked by price persistence.**

In price level models, lagged price dominates all other features (47% importance). However, when modeling price *changes*, weather variables become significantly more important: temperature, precipitation, and freeze risk all gain explanatory power. At the district level, freeze risk increases weekly prices by ~$2.30 (p < 0.001).

**3. Seasonal production migration is the dominant market structure.**

California (Salinas) and Arizona (Yuma) operate as a sequential, year-round production system. Price spikes concentrate during transition periods (weeks 15-20 and 45-50) when production shifts between regions and supply temporarily contracts.

**4. Results are robust across lettuce varieties.**

All core findings replicate when using Romaine lettuce instead of Iceberg, with consistent coefficient signs and even lower prediction errors (RMSE $2.61 vs $4.36).

## Market Structure: Seasonal Production Migration

The U.S. lettuce market operates as a seasonal, two-region production system:

- **Salinas-Watsonville, CA** (spring/summer/fall, ~weeks 15-50): primary production region
- **Western Arizona / Yuma** (winter, ~weeks 1-15 and 45-52): winter production region
- **Imperial Valley, CA**: supplementary desert region

The California-Arizona production shift is primarily seasonal and calendar-driven. Rather than modeling migration timing directly, we treat it as part of the market structure captured by week-of-year controls and district-level seasonality. Weather shocks matter most when they disrupt supply during transition periods, which is consistent with the observed price spikes around weeks 15-20 and 45-50 (Figure 4).

We tested whether weather effects are amplified during transition periods using interaction terms. The coefficients are not statistically significant, consistent with weather operating through within-season supply disruptions rather than changes in migration timing.

## Data

| Source | Variables | Frequency | Coverage | Access |
|--------|-----------|-----------|----------|--------|
| USDA AMS Truck Rate Reports | Shipment volume by district and commodity | Weekly | 2010-2026 | [Socrata API](https://usda.library.cornell.edu/concern/publications/mg74qm08h) |
| USDA Market News (FOB Shipping Point) | FOB prices by district, variety, package | Daily | 2010-2026 | [USDA Market News](https://marketnews.usda.gov/mnp/fv-home) |
| PRISM Climate Data | Temperature (max/min), precipitation | Daily | 2010-2026 | [PRISM Climate Group](https://prism.oregonstate.edu/) |

**Production districts:** Salinas-Watsonville, Western Arizona (Yuma), Imperial Valley, Santa Maria, Oxnard, San Joaquin Valley.

**Market-level dataset:** 849 weeks, 98.2% price coverage (shipment-weighted aggregation).

**District-level panel:** 1,466 observations (3 core districts x ~849 weeks), 78.4% price coverage.

## Methods

### Dual-layer modeling approach

**Market-level (Dataset A):** All 6 districts aggregated into a single weekly market observation. Shipment-weighted price and weather. Used for prediction and forecasting.

**District-level (Dataset B):** 3 core districts (Salinas, Western AZ, Imperial Valley) as a panel. Used for causal interpretation and heterogeneity analysis.

### Models

| Model | Target | Method | Purpose |
|-------|--------|--------|---------|
| Market price (level) | P_t | OLS, XGBoost, Random Forest | Primary prediction |
| Market price (change) | dP_t | OLS, XGBoost, Random Forest | Shock identification |
| Market volume | V_t | OLS, XGBoost | Supply prediction |
| District panel | P_it, dP_it | OLS + district FE | Mechanism and heterogeneity |
| Missing price | P_it (unobserved) | OLS, XGBoost | Partial label learning |

### Key features

- Lagged price, volume, temperature, precipitation (1, 2, 4 weeks)
- Rolling 4-week averages
- Extreme weather indicators (freeze risk, extreme heat, heavy rain)
- Coverage ratio (price signal reliability)
- District fixed effects and seasonal indicators

## Results

### Model comparison (out-of-sample, expanding window)

| Model | Avg RMSE | vs Naive |
|-------|----------|----------|
| Naive (price = lag1) | $5.12 | baseline |
| **OLS Level (full)** | **$4.36** | **+14.8%** |
| OLS log-diff | $5.00 | +2.4% |
| XGBoost Level | $6.67 | -30.4% |
| Random Forest Level | $6.81 | -33.2% |
| XGBoost diff | $5.12 | +0.0% |
| RF diff | $5.00 | +2.3% |

Both tree-based methods underperform OLS in level models. In change models, all methods converge to similar performance, confirming that the residual weather-price relationship is approximately linear. Tree models are particularly fragile during extreme events: in 2022, XGBoost RMSE exploded to $24.37 and RF to $23.83, while OLS remained at $7.77.

### District-level weather effects (Iceberg, OLS)

| Variable | Coefficient | p-value | Interpretation |
|----------|-------------|---------|----------------|
| freeze_risk | +$2.33 | < 0.001 | Frost events increase price ~$2.3/week |
| tmin_avg | -$0.15 | 0.008 | Higher minimum temperature reduces price |
| ppt_total | +$0.10 | 0.18 | Precipitation tends to increase price |
| heavy_rain | -$0.92 | 0.58 | Heavy rain weeks show lower prices |
| extreme_heat | +$2.16 | 0.008 | Heat waves increase price (level model) |

### Diagnostics

- **Expanding vs rolling window:** Expanding is more stable (RMSE $4.36 vs $4.57)
- **Coverage vs error:** Low coverage weeks have higher prediction error ($4.98 vs $2.83)
- **2022 extreme year:** Price volatility 2-3x normal; model tracks trend but misses peak ($94 spike)
- **Feature stability:** Core coefficients (price_lag1, ppt_total) stable across 2010-2015, 2016-2020, 2021-2026
- **2026 partial year:** Negligible impact on results

## Project Structure

```
agri-risk/
  data/
    raw/               # Raw data (truck, price CSVs, PRISM)
    processed/         # Cleaned datasets
  src/
    data/              # Data pipeline scripts
      merge_truck_data.py
      merge_price_data.py
      fetch_prism.py
      build_weekly_panel.py
      build_model_datasets.py
    models/            # Modeling scripts
      baseline_ols.py
      model_comparison.py
      market_volume.py
      district_panel_ols.py
      romaine_robustness.py
      missing_price.py
      diagnostics.py
    viz/               # Visualization
      plot_results.py
  outputs/             # Figures and result tables
  paper/               # LaTeX paper
```

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Explore the project interactively: see `notebooks/walkthrough.ipynb` for a guided tour of the key results.

## Requirements

- Python 3.12
- See `requirements.txt` for full dependency list

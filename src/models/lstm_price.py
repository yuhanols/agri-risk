"""
LSTM baseline for price prediction using PyTorch.

Same expanding-window framework as model_comparison.py.
Trains an LSTM on sequential windows of features to predict
price level and price change.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

# ── Data ─────────────────────────────────────────────────────
mkt = pd.read_csv(PROC / "market_iceberg_weekly.csv", parse_dates=["week_ending"])
mkt["dprice"] = mkt["price"] - mkt["price_lag1"]

WEATHER = [
    "tmax_avg", "tmin_avg", "ppt_total", "diurnal_range",
    "extreme_heat", "freeze_risk", "heavy_rain",
    "tmax_lag1", "tmax_lag2", "ppt_lag1",
    "tmax_avg_roll4_mean", "ppt_total_roll4_mean",
]

MARKET = [
    "volume_lag1", "volume_lag2", "volume_roll4_mean",
    "coverage", "n_districts",
    "month", "week_of_year",
]

PRICE_LAGS = ["price_lag1", "price_lag2", "price_lag4"]

LEVEL_FEATURES = WEATHER + MARKET + PRICE_LAGS
CHANGE_FEATURES = WEATHER + MARKET

all_cols = list(set(
    ["week_ending", "price", "price_lag1", "dprice"]
    + LEVEL_FEATURES + CHANGE_FEATURES
))
df = mkt[all_cols].dropna().sort_values("week_ending").reset_index(drop=True)
df["year"] = df["week_ending"].dt.year
years = sorted(df["year"].unique())
MIN_TRAIN = 4

# ── LSTM Model ───────────────────────────────────────────────
SEQ_LEN = 8
HIDDEN_SIZE = 32
NUM_LAYERS = 1
EPOCHS = 100
BATCH_SIZE = 32
LR = 0.005
DEVICE = "cpu"


class PriceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)


def create_sequences(X, y, seq_len):
    """Create (seq_len, n_features) windows for LSTM input."""
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def train_lstm(X_train, y_train, input_size, epochs=EPOCHS):
    model = PriceLSTM(input_size, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.MSELoss()

    dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    return model


def predict_lstm(model, X_test):
    model.eval()
    with torch.no_grad():
        pred = model(torch.FloatTensor(X_test).to(DEVICE))
    return pred.cpu().numpy()


# ── Expanding window evaluation ──────────────────────────────
def lstm_expanding_eval(df, features, target, convert_to_level=None):
    yearly = []

    for test_year in years[MIN_TRAIN:]:
        train_df = df[df["year"] < test_year].reset_index(drop=True)
        test_df = df[df["year"] == test_year].reset_index(drop=True)
        if len(test_df) == 0:
            continue

        # Combine for sequential windowing, then split
        combined = pd.concat([train_df, test_df], ignore_index=True)

        scaler = StandardScaler()
        X_all = scaler.fit_transform(combined[features].values)
        y_all = combined[target].values

        X_seq, y_seq = create_sequences(X_all, y_all, SEQ_LEN)

        # Split: sequences ending in train vs test
        n_train = len(train_df)
        # Sequence i uses data[i:i+seq_len], predicts y[i+seq_len]
        # So train sequences: those where target index < n_train
        train_mask = np.arange(SEQ_LEN, len(combined)) < n_train
        test_mask = ~train_mask

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        X_tr, y_tr = X_seq[train_mask], y_seq[train_mask]
        X_te, y_te = X_seq[test_mask], y_seq[test_mask]

        model = train_lstm(X_tr, y_tr, len(features))
        pred = predict_lstm(model, X_te)

        if convert_to_level is not None:
            # Get price_lag1 for test observations
            test_indices = np.where(test_mask)[0]
            actual_indices = test_indices + SEQ_LEN  # index in combined
            price_lag1_vals = combined.iloc[actual_indices]["price_lag1"].values
            pred_level = convert_to_level(pred, price_lag1_vals)
            actual_level = combined.iloc[actual_indices]["price"].values
        else:
            test_indices = np.where(test_mask)[0]
            actual_indices = test_indices + SEQ_LEN
            pred_level = pred
            actual_level = combined.iloc[actual_indices]["price"].values

        yearly.append({
            "year": test_year,
            "n": len(pred),
            "rmse": np.sqrt(mean_squared_error(actual_level, pred_level)),
            "mae": mean_absolute_error(actual_level, pred_level),
        })

    return pd.DataFrame(yearly)


# ── Run ──────────────────────────────────────────────────────
print(f"Data: {len(df)} obs, years {years[0]}-{years[-1]}")
print(f"LSTM: seq_len={SEQ_LEN}, hidden={HIDDEN_SIZE}, epochs={EPOCHS}")
print()

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

# Level model
print("Training LSTM Level (full features)...")
lstm_level = lstm_expanding_eval(df, LEVEL_FEATURES, "price")

# Change model
print("Training LSTM Δprice...")
lstm_change = lstm_expanding_eval(
    df, CHANGE_FEATURES, "dprice",
    convert_to_level=lambda pred, lag1: lag1 + pred,
)

# ── Results ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("LSTM RESULTS (expanding window, 2014-2026)")
print("=" * 70)
print(f"{'Model':<25} {'Avg RMSE':>10} {'Avg MAE':>10} {'vs Naive':>10}")
print("-" * 57)
print(f"{'Naive (P=P_lag1)':<25} ${naive_avg:>9.2f} {'':>10} {'baseline':>10}")

for name, res in [("LSTM Level (full)", lstm_level), ("LSTM Δprice", lstm_change)]:
    avg_rmse = res["rmse"].mean()
    avg_mae = res["mae"].mean()
    vs = (naive_avg - avg_rmse) / naive_avg * 100
    print(f"{name:<25} ${avg_rmse:>9.2f} ${avg_mae:>9.2f} {vs:>+9.1f}%")

print("\nYear-by-year RMSE:")
print(f"{'Year':>6} {'Level':>10} {'ΔPrice':>10}")
print("-" * 28)
for _, row in lstm_level.iterrows():
    yr = int(row["year"])
    lv = row["rmse"]
    ch_row = lstm_change[lstm_change["year"] == yr]
    ch = ch_row.iloc[0]["rmse"] if len(ch_row) > 0 else float("nan")
    print(f"{yr:>6} ${lv:>9.2f} ${ch:>9.2f}")

# Save
summary = []
for name, res in [("LSTM Level (full)", lstm_level), ("LSTM Δprice", lstm_change)]:
    summary.append({
        "model": name,
        "avg_rmse": res["rmse"].mean(),
        "avg_mae": res["mae"].mean(),
    })
pd.DataFrame(summary).to_csv(OUT / "lstm_results.csv", index=False)
print(f"\nSaved: {OUT / 'lstm_results.csv'}")

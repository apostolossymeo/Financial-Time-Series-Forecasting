"""Evaluate lightweight non-neural baselines and save comparison figures.

Baselines:
- persistence: tomorrow's close equals today's close
- SMA20: rolling 20-session mean
- EMA20: exponential 20-session mean
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.eda_report import load_ohlcv, save
from src.evaluate import metrics


def baseline_predictions(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "actual": close,
        "persistence": close.shift(1),
        "sma20": close.rolling(20).mean().shift(1),
        "ema20": close.ewm(span=20, adjust=False).mean().shift(1),
    }).dropna()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/AAPL.csv")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_ohlcv(args.csv)
    preds = baseline_predictions(df["Close"])
    split = int(len(preds) * args.train_ratio)
    test = preds.iloc[split:]

    rows = []
    for name in ["persistence", "sma20", "ema20"]:
        row = metrics(test["actual"].values, test[name].values)
        row["model"] = name
        rows.append(row)
    table = pd.DataFrame(rows).set_index("model")
    table.to_csv(out / "baseline_metrics.csv")
    print(table.round(4))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(table.index, table["RMSE"])
    ax.set_title(f"{args.ticker} Baseline RMSE Comparison")
    ax.set_ylabel("RMSE (USD)")
    save(fig, out / "baseline_rmse.png")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(test.index, test["actual"], lw=1.2, label="Actual")
    for name in ["persistence", "sma20", "ema20"]:
        ax.plot(test.index, test[name], lw=1.0, label=name)
    ax.set_title(f"{args.ticker} Baseline Predictions on Holdout")
    ax.set_ylabel("Close Price (USD)")
    ax.legend()
    save(fig, out / "baseline_predictions.png")


if __name__ == "__main__":
    main()

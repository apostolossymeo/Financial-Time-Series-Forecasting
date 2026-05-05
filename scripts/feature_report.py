"""Generate technical-indicator and feature-correlation figures."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.eda_report import load_ohlcv, save
from src.features import add_features


def plot_technical_indicators(raw: pd.DataFrame, feat: pd.DataFrame, ticker: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(raw.index, raw["Close"], lw=1.0, label="Close")
    ax.plot(feat.index, feat["Close"].rolling(50).mean(), lw=1.0, label="50D MA")
    ax.plot(feat.index, feat["Close"].rolling(200).mean(), lw=1.0, label="200D MA")
    ax.set_title(f"{ticker} Close with Moving Averages")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    save(fig, out / "moving_averages.png")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(feat.index, feat["rsi"], lw=1.0, label="RSI(14)")
    ax.axhline(70, ls="--", lw=1.0)
    ax.axhline(30, ls="--", lw=1.0)
    ax.set_title(f"{ticker} RSI Regime")
    ax.set_ylabel("RSI")
    ax.legend()
    save(fig, out / "rsi_regime.png")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(feat.index, feat["macd_line"], lw=1.0, label="MACD")
    ax.plot(feat.index, feat["macd_sig"], lw=1.0, label="Signal")
    ax.axhline(0, lw=0.8)
    ax.set_title(f"{ticker} MACD")
    ax.legend()
    save(fig, out / "macd.png")


def plot_feature_corr(feat: pd.DataFrame, out: Path) -> None:
    corr = feat.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.columns)), corr.columns)
    ax.set_title("Feature Correlation Matrix")
    fig.colorbar(im, ax=ax, label="Correlation")
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    save(fig, out / "feature_correlation.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/AAPL.csv")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = load_ohlcv(args.csv)
    feat = add_features(raw)
    plot_technical_indicators(raw, feat, args.ticker, out)
    plot_feature_corr(feat, out)


if __name__ == "__main__":
    main()

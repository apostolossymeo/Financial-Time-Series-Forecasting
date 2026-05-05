"""Generate exploratory data-analysis figures for the raw OHLCV dataset.

Usage:
    python scripts/eda_report.py --csv data/AAPL.csv --ticker AAPL --out results
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_ohlcv(csv_path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    raw.columns = raw.columns.str.strip()
    if "Close/Last" in raw.columns:
        for col in ["Close/Last", "Open", "High", "Low"]:
            raw[col] = raw[col].astype(str).str.replace("$", "", regex=False).astype(float)
        raw = raw.rename(columns={"Close/Last": "Close"})
    raw["Date"] = pd.to_datetime(raw["Date"].astype(str).str.strip())
    raw = raw.sort_values("Date").set_index("Date")
    return raw[["Open", "High", "Low", "Close", "Volume"]].dropna()


def save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[eda] saved {path}")


def plot_price_volume(df: pd.DataFrame, ticker: str, out: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(df.index, df["Close"], lw=1.2, label="Close")
    ax1.set_ylabel("Close Price (USD)")
    ax1.set_title(f"{ticker} Price and Volume")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax2 = ax1.twinx()
    ax2.bar(df.index, df["Volume"], alpha=0.18, width=1.0, label="Volume")
    ax2.set_ylabel("Volume")
    save(fig, out / "price_volume.png")


def plot_returns_distribution(df: pd.DataFrame, ticker: str, out: Path) -> None:
    returns = df["Close"].pct_change().dropna()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(returns, bins=80, alpha=0.85)
    ax.axvline(returns.mean(), lw=1.5, ls="--", label=f"Mean: {returns.mean():.3%}")
    ax.set_title(f"{ticker} Daily Return Distribution")
    ax.set_xlabel("Daily return")
    ax.set_ylabel("Frequency")
    ax.legend()
    save(fig, out / "returns_distribution.png")


def plot_rolling_risk(df: pd.DataFrame, ticker: str, out: Path) -> None:
    returns = df["Close"].pct_change()
    vol = returns.rolling(60).std() * np.sqrt(252)
    dd = df["Close"] / df["Close"].cummax() - 1
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(vol.index, vol, lw=1.2, label="60D annualized volatility")
    ax.plot(dd.index, dd.abs(), lw=1.2, label="Absolute drawdown")
    ax.set_title(f"{ticker} Rolling Risk Regime")
    ax.set_ylabel("Risk metric")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    save(fig, out / "rolling_risk.png")


def plot_monthly_heatmap(df: pd.DataFrame, ticker: str, out: Path) -> None:
    monthly = df["Close"].resample("ME").last().pct_change().dropna()
    table = monthly.to_frame("ret")
    table["year"] = table.index.year
    table["month"] = table.index.month
    piv = table.pivot(index="year", columns="month", values="ret")

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(piv.values, aspect="auto")
    ax.set_title(f"{ticker} Monthly Returns Heatmap")
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    ax.set_xticks(range(12), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], rotation=45)
    ax.set_yticks(range(len(piv.index)), piv.index)
    fig.colorbar(im, ax=ax, label="Monthly return")
    save(fig, out / "monthly_returns_heatmap.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/AAPL.csv")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_ohlcv(args.csv)
    plot_price_volume(df, args.ticker, out)
    plot_returns_distribution(df, args.ticker, out)
    plot_rolling_risk(df, args.ticker, out)
    plot_monthly_heatmap(df, args.ticker, out)


if __name__ == "__main__":
    main()

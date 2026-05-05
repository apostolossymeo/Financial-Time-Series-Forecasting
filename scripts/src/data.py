import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def fetch(ticker: str, start: str, end: str | None = None, csv_path: str | None = None) -> pd.DataFrame:
    """
    Load OHLCV data. Priority:
      1. Local CSV at csv_path, or data/<ticker>.csv
         Accepts Nasdaq format (Date, Close/Last, Volume, Open, High, Low with $ prefixes)
         or standard Yahoo format (Date, Open, High, Low, Close, Volume)
      2. yfinance live download — caches result to data/<ticker>.csv

    Manual download:
      Nasdaq: https://www.nasdaq.com/market-activity/stocks/aapl/historical
      Yahoo:  https://finance.yahoo.com/quote/AAPL/history → Download
    Place file at data/AAPL.csv
    """
    local = csv_path or f"data/{ticker}.csv"
    if os.path.exists(local):
        raw = pd.read_csv(local)
        raw.columns = raw.columns.str.strip()

        # Nasdaq format detection
        if "Close/Last" in raw.columns:
            for col in ["Close/Last", "Open", "High", "Low"]:
                raw[col] = raw[col].astype(str).str.replace("$", "", regex=False).astype(float)
            raw = raw.rename(columns={"Close/Last": "Close"})

        raw["Date"] = pd.to_datetime(raw["Date"].astype(str).str.strip())
        raw = raw.sort_values("Date").reset_index(drop=True)
        raw = raw.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]

        if start:
            raw = raw[raw.index >= start]
        if end:
            raw = raw[raw.index <= end]
        if raw.empty:
            raise ValueError(f"No local rows for {ticker} after applying date filters")

        print(f"[data]  {ticker}: {len(raw)} sessions from local CSV "
              f"({raw.index[0].date()} → {raw.index[-1].date()})")
        return raw

    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(
            f"No data for {ticker}. yfinance returned empty.\n"
            f"Manual fallback: download CSV from "
            f"https://www.nasdaq.com/market-activity/stocks/{ticker.lower()}/historical "
            f"and save to data/{ticker}.csv"
        )
    os.makedirs("data", exist_ok=True)
    df[["Open", "High", "Low", "Close", "Volume"]].to_csv(f"data/{ticker}.csv")
    print(f"[data]  {ticker}: {len(df)} sessions "
          f"({df.index[0].date()} → {df.index[-1].date()}) — cached to data/{ticker}.csv")
    return df[["Open", "High", "Low", "Close", "Volume"]]


def make_windows(data: np.ndarray, time_steps: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i : i + time_steps])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)


def split_and_scale(
    df: pd.DataFrame,
    feature_cols: list[str],
    time_steps: int,
    train_ratio: float,
) -> dict:
    values = df[feature_cols].values
    split  = int(len(values) * train_ratio)
    if split <= time_steps or len(values) - split <= 1:
        raise ValueError("Not enough rows for the requested train_ratio/time_steps")

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(values[:split])
    test_scaled  = scaler.transform(values[split - time_steps:])

    X_train, y_train = make_windows(train_scaled, time_steps)
    X_test,  y_test  = make_windows(test_scaled,  time_steps)

    close_scaler = MinMaxScaler()
    close_scaler.fit(values[:split, [0]])

    print(f"[data]  X_train {X_train.shape}  X_test {X_test.shape}")
    return {
        "X_train": X_train, "y_train": y_train,
        "X_test":  X_test,  "y_test":  y_test,
        "scaler": scaler, "close_scaler": close_scaler,
        "split": split,
    }

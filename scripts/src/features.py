import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - 100 / (1 + gain / loss)


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series]:
    line   = close.ewm(span=fast).mean() - close.ewm(span=slow).mean()
    sig    = line.ewm(span=signal).mean()
    return line, sig


def bollinger(
    close: pd.Series,
    period: int = 20,
    n_std: float = 2.0,
) -> tuple[pd.Series, pd.Series]:
    ma    = close.rolling(period).mean()
    sigma = close.rolling(period).std()
    return ma + n_std * sigma, ma - n_std * sigma


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c  = df["Close"]

    df["rsi"]          = rsi(c)
    df["macd_line"], df["macd_sig"] = macd(c)
    df["bb_upper"], df["bb_lower"]  = bollinger(c)
    df["bb_width"]     = (df["bb_upper"] - df["bb_lower"]) / c
    df["return_1d"]    = c.pct_change()
    df["return_5d"]    = c.pct_change(5)
    df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

    df.drop(columns=["Open", "High", "Low", "Volume",
                      "bb_upper", "bb_lower"], inplace=True)
    return df.dropna()

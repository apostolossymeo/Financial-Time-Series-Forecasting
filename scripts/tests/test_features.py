import pandas as pd

from src.features import add_features, rsi, macd, bollinger


def test_add_features_outputs_expected_columns():
    df = pd.DataFrame({
        "Open": range(1, 81),
        "High": range(2, 82),
        "Low": range(0, 80),
        "Close": range(1, 81),
        "Volume": [1000 + i for i in range(80)],
    })
    out = add_features(df)
    assert set(out.columns) == {"Close", "rsi", "macd_line", "macd_sig", "bb_width", "return_1d", "return_5d", "volume_ratio"}
    assert not out.isna().any().any()


def test_indicators_return_series():
    close = pd.Series(range(1, 80), dtype=float)
    assert len(rsi(close)) == len(close)
    assert len(macd(close)[0]) == len(close)
    assert len(bollinger(close)[0]) == len(close)

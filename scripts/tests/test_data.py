import numpy as np
import pandas as pd

from src.data import make_windows, split_and_scale


def test_make_windows_shapes():
    data = np.arange(20).reshape(10, 2)
    X, y = make_windows(data, time_steps=3)
    assert X.shape == (7, 3, 2)
    assert y.shape == (7,)


def test_split_and_scale_shapes():
    df = pd.DataFrame({
        "Close": np.linspace(100, 120, 120),
        "rsi": np.linspace(30, 70, 120),
        "return_1d": np.linspace(-0.01, 0.01, 120),
    })
    out = split_and_scale(df, list(df.columns), time_steps=10, train_ratio=0.8)
    assert out["X_train"].shape[1:] == (10, 3)
    assert out["X_test"].shape[1:] == (10, 3)
    assert out["split"] == 96

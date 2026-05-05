import numpy as np

from src.evaluate import backtest, metrics, moving_average_baseline, naive_last_value


def test_metrics_perfect_prediction():
    y = np.array([10.0, 11.0, 12.0, 11.0])
    m = metrics(y, y)
    assert m["RMSE"] == 0
    assert m["MAE"] == 0
    assert m["MAPE_%"] == 0


def test_baselines_shape():
    y = np.arange(10.0)
    assert naive_last_value(y).shape == y.shape
    assert moving_average_baseline(y, 3).shape == y.shape


def test_backtest_outputs_curves():
    y = np.array([100.0, 101.0, 99.0, 103.0, 104.0])
    p = np.array([100.0, 102.0, 98.0, 104.0, 105.0])
    result = backtest(y, p, transaction_cost=0.0)
    assert len(result["equity_curve"]) == len(y) - 1
    assert "max_drawdown_%" in result

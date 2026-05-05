from __future__ import annotations

import numpy as np


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Regression and directional metrics for price forecasts."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true{y_true.shape} vs y_pred{y_pred.shape}")
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    mape = float(np.mean(np.abs(err / np.clip(np.abs(y_true), 1e-8, None))) * 100)
    if len(y_true) > 1:
        actual_dir = np.sign(np.diff(y_true))
        pred_dir = np.sign(y_pred[1:] - y_true[:-1])
        directional_accuracy = float(np.mean(actual_dir == pred_dir) * 100)
    else:
        directional_accuracy = float("nan")
    return {"RMSE": rmse, "MAE": mae, "MAPE_%": mape, "Directional_Accuracy_%": directional_accuracy}


def naive_last_value(y_true: np.ndarray) -> np.ndarray:
    """Naive baseline: tomorrow's close equals today's close."""
    y_true = np.asarray(y_true, dtype=float)
    if len(y_true) < 2:
        return y_true.copy()
    return np.r_[y_true[0], y_true[:-1]]


def moving_average_baseline(y_true: np.ndarray, window: int = 5) -> np.ndarray:
    """Rolling mean baseline using only prior observed closes."""
    y_true = np.asarray(y_true, dtype=float)
    out = np.empty_like(y_true, dtype=float)
    for i in range(len(y_true)):
        start = max(0, i - window)
        hist = y_true[start:i]
        out[i] = hist.mean() if len(hist) else y_true[i]
    return out


def max_drawdown(equity: np.ndarray) -> float:
    equity = np.asarray(equity, dtype=float)
    peaks = np.maximum.accumulate(equity)
    dd = equity / np.clip(peaks, 1e-8, None) - 1.0
    return float(abs(dd.min()) * 100)


def annualized_sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    returns = np.asarray(returns, dtype=float)
    sigma = returns.std(ddof=1)
    if sigma == 0 or np.isnan(sigma):
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / sigma)


def backtest(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    initial_capital: float = 10_000.0,
    transaction_cost: float = 0.0005,
    allow_short: bool = True,
) -> dict[str, object]:
    """
    Directional strategy backtest.

    Signal at t is derived from forecasted next movement vs today's observed price.
    The position is applied to the realized return from t to t+1. Transaction cost is
    charged whenever the position changes.
    """
    prices = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if len(prices) != len(pred):
        raise ValueError("y_true and y_pred must have equal length")
    if len(prices) < 3:
        raise ValueError("need at least 3 observations for a backtest")

    realized = prices[1:] / prices[:-1] - 1.0
    raw_signal = np.where(pred[1:] > prices[:-1], 1.0, -1.0)
    if not allow_short:
        raw_signal = np.where(raw_signal > 0, 1.0, 0.0)

    position_changes = np.abs(np.diff(np.r_[0.0, raw_signal]))
    strategy_returns = raw_signal * realized - transaction_cost * position_changes
    bh_returns = realized

    equity_curve = initial_capital * np.cumprod(1.0 + strategy_returns)
    bh_curve = initial_capital * np.cumprod(1.0 + bh_returns)

    return {
        "equity_curve": equity_curve,
        "bh_curve": bh_curve,
        "strategy_returns": strategy_returns,
        "bh_returns": bh_returns,
        "signals": raw_signal,
        "total_return_%": float((equity_curve[-1] / initial_capital - 1.0) * 100),
        "buy_hold_return_%": float((bh_curve[-1] / initial_capital - 1.0) * 100),
        "sharpe": annualized_sharpe(strategy_returns),
        "buy_hold_sharpe": annualized_sharpe(bh_returns),
        "max_drawdown_%": max_drawdown(equity_curve),
        "buy_hold_max_drawdown_%": max_drawdown(bh_curve),
        "transaction_cost": transaction_cost,
        "trades": int(np.count_nonzero(position_changes)),
    }

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STYLE = {
    "actual":   "#1f77b4",
    "train":    "#ff7f0e",
    "test":     "#2ca02c",
    "forecast": "#d62728",
    "band":     "#d62728",
    "equity":   "#2ca02c",
    "bh":       "#1f77b4",
}


def forecast_plot(
    dates:         pd.DatetimeIndex,
    actual:        np.ndarray,
    train_pred:    np.ndarray,
    test_pred:     np.ndarray,
    test_std:      np.ndarray,
    future_pred:   np.ndarray,
    future_std:    np.ndarray,
    split:         int,
    time_steps:    int,
    forecast_days: int,
    ticker:        str,
    save_path:     str = "results/forecast.png",
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(dates, actual, color=STYLE["actual"], lw=1.0, label="Actual", zorder=3)

    train_dates = dates[time_steps : time_steps + len(train_pred)]
    ax.plot(train_dates, train_pred, color=STYLE["train"], lw=1.2, label="Train fit")

    test_dates = dates[split : split + len(test_pred)]
    ax.plot(test_dates, test_pred, color=STYLE["test"], lw=1.2, label="Test fit")
    ax.fill_between(
        test_dates,
        test_pred - 2 * test_std,
        test_pred + 2 * test_std,
        color=STYLE["test"], alpha=0.15,
    )

    future_dates = pd.bdate_range(start=dates[-1], periods=forecast_days + 1)[1:]
    ax.plot(future_dates, future_pred,
            color=STYLE["forecast"], lw=1.8, ls="--", label=f"{forecast_days}-day forecast")
    ax.fill_between(
        future_dates,
        future_pred - 2 * future_std,
        future_pred + 2 * future_std,
        color=STYLE["band"], alpha=0.15, label="95% CI",
    )
    ax.axvline(dates[-1], color="gray", lw=0.8, ls=":")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price (USD)")
    ax.set_title(f"{ticker} — Stacked BiLSTM Forecast with Uncertainty Bounds", fontsize=13)
    ax.legend(framealpha=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[plot]  Saved → {save_path}")


def loss_plot(
    history,
    save_path: str = "results/loss.png",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history["loss"],     label="Train", lw=1.5)
    ax.plot(history.history["val_loss"], label="Val",   lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Training History")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[plot]  Saved → {save_path}")


def backtest_plot(
    equity:    np.ndarray,
    bh_equity: np.ndarray,
    test_dates: pd.DatetimeIndex,
    save_path: str = "results/backtest.png",
) -> None:
    dates = test_dates[1 : 1 + len(equity)]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, equity,    color=STYLE["equity"], lw=1.5, label="Model Strategy")
    ax.plot(dates, bh_equity, color=STYLE["bh"],     lw=1.5, label="Buy & Hold", ls="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (USD)")
    ax.set_title("Backtest — Model Strategy vs. Buy & Hold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[plot]  Saved → {save_path}")


def residual_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = "results/residuals.png",
) -> None:
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(residuals, lw=1.0)
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("Test observation")
    ax.set_ylabel("Actual - Predicted (USD)")
    ax.set_title("Forecast Residuals")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[plot]  Saved → {save_path}")


def error_hist_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = "results/error_hist.png",
) -> None:
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(residuals, bins=30, alpha=0.8)
    ax.axvline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("Actual - Predicted (USD)")
    ax.set_ylabel("Frequency")
    ax.set_title("Prediction Error Distribution")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[plot]  Saved → {save_path}")


def _drawdown(equity: np.ndarray) -> np.ndarray:
    equity = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(equity)
    return equity / np.clip(peak, 1e-8, None) - 1.0


def drawdown_plot(
    equity: np.ndarray,
    bh_equity: np.ndarray,
    test_dates: pd.DatetimeIndex,
    save_path: str = "results/drawdown.png",
) -> None:
    dates = test_dates[1 : 1 + len(equity)]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, _drawdown(equity) * 100, color=STYLE["equity"], lw=1.5, label="Model Strategy")
    ax.plot(dates, _drawdown(bh_equity) * 100, color=STYLE["bh"], lw=1.5, label="Buy & Hold", ls="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Drawdown Comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[plot]  Saved → {save_path}")

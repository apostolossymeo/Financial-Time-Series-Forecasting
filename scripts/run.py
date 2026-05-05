from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from src.config import load_config
from src.data import fetch, split_and_scale
from src.evaluate import backtest, metrics, moving_average_baseline, naive_last_value
from src.features import add_features
from src.model import build, forecast_future, mc_predict, train
from src.viz import backtest_plot, drawdown_plot, error_hist_plot, forecast_plot, loss_plot, residual_plot


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stock LSTM Forecaster")
    p.add_argument("--config", default="config.yaml", help="Path to YAML config")
    p.add_argument("--ticker")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--epochs", type=int)
    p.add_argument("--forecast", type=int)
    p.add_argument("--csv", help="Optional local CSV path")
    return p.parse_args()


def main() -> None:
    import numpy as np
    import tensorflow as tf

    args = parse_args()
    cfg = load_config(args.config)
    overrides = {
        "ticker": args.ticker,
        "start_date": args.start,
        "end_date": args.end,
        "epochs": args.epochs,
        "forecast_days": args.forecast,
    }
    cfg.update({k: v for k, v in overrides.items() if v is not None})

    tf.random.set_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    Path("results").mkdir(exist_ok=True)

    raw = fetch(cfg["ticker"], cfg["start_date"], cfg["end_date"], csv_path=args.csv)
    df = add_features(raw)
    feature_cols = list(df.columns)

    data = split_and_scale(df, feature_cols, cfg["time_steps"], cfg["train_ratio"])
    X_tr, y_tr = data["X_train"], data["y_train"]
    X_te, y_te = data["X_test"], data["y_test"]
    close_scaler, split = data["close_scaler"], data["split"]

    model = build(cfg["time_steps"], X_tr.shape[2], cfg["lstm_units"], cfg["dropout"])
    history = train(model, X_tr, y_tr, X_te, y_te, cfg["epochs"], cfg["batch_size"], cfg["model_path"])

    train_mean, _ = mc_predict(model, X_tr, cfg["mc_samples"])
    test_mean, test_std = mc_predict(model, X_te, cfg["mc_samples"])

    inv = lambda x: close_scaler.inverse_transform(x.reshape(-1, 1)).flatten()
    y_tr_usd, y_te_usd = inv(y_tr), inv(y_te)
    train_pred, test_pred = inv(train_mean), inv(test_mean)
    test_pred_std = test_std * (close_scaler.data_max_[0] - close_scaler.data_min_[0])

    train_metrics = metrics(y_tr_usd, train_pred)
    test_metrics = metrics(y_te_usd, test_pred)
    naive_metrics = metrics(y_te_usd, naive_last_value(y_te_usd))
    ma_metrics = metrics(y_te_usd, moving_average_baseline(y_te_usd, window=5))

    print("\n── Metrics ──")
    for name, result in [("Train", train_metrics), ("Test", test_metrics), ("Naive", naive_metrics), ("MA(5)", ma_metrics)]:
        print(name)
        for k, v in result.items():
            print(f"  {k}: {v:.4f}")

    bt = backtest(y_te_usd, test_pred, transaction_cost=cfg["transaction_cost"])
    print("\n── Backtest ──")
    for k, v in bt.items():
        if not isinstance(v, np.ndarray):
            print(f"  {k}: {v}")

    fut_mean_s, fut_std_s = forecast_future(model, X_te[-1], cfg["forecast_days"], cfg["mc_samples"])
    fut_mean = inv(fut_mean_s)
    fut_std = fut_std_s * (close_scaler.data_max_[0] - close_scaler.data_min_[0])

    dates = df.index
    test_dates = dates[split:]
    forecast_plot(dates, df["Close"].values, train_pred, test_pred, test_pred_std, fut_mean, fut_std, split, cfg["time_steps"], cfg["forecast_days"], cfg["ticker"])
    loss_plot(history)
    backtest_plot(bt["equity_curve"], bt["bh_curve"], test_dates)
    residual_plot(y_te_usd, test_pred)
    error_hist_plot(y_te_usd, test_pred)
    drawdown_plot(bt["equity_curve"], bt["bh_curve"], test_dates)

    report = {
        "config": cfg,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "naive_baseline_metrics": naive_metrics,
        "ma5_baseline_metrics": ma_metrics,
        "backtest": {k: v for k, v in bt.items() if not isinstance(v, np.ndarray)},
    }
    Path("results/metrics.json").write_text(json.dumps(report, indent=2))
    print("[report] Saved → results/metrics.json")


if __name__ == "__main__":
    main()

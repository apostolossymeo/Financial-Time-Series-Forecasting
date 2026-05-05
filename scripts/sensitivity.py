"""Hyperparameter sensitivity scaffold for quick experiments.

Runs a small grid over time steps and dropout, then saves metrics as CSV.
"""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data import fetch, split_and_scale
from src.evaluate import metrics
from src.features import add_features
from src.model import build, mc_predict, train


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="AAPL")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--out", default="results/sensitivity.csv")
    args = p.parse_args()

    tf.random.set_seed(42)
    np.random.seed(42)
    raw = fetch(args.ticker, args.start)
    df = add_features(raw)
    rows = []
    for time_steps, dropout in product([30, 60, 90], [0.1, 0.2, 0.3]):
        print(f"\n[sensitivity] time_steps={time_steps} dropout={dropout}")
        data = split_and_scale(df, list(df.columns), time_steps, 0.8)
        model = build(time_steps, data["X_train"].shape[2], units=[64, 32], dropout=dropout)
        train(model, data["X_train"], data["y_train"], data["X_test"], data["y_test"], args.epochs, 32, f"results/sens_{time_steps}_{dropout}.keras")
        pred_s, _ = mc_predict(model, data["X_test"], 50)
        inv = lambda x: data["close_scaler"].inverse_transform(x.reshape(-1, 1)).flatten()
        row = metrics(inv(data["y_test"]), inv(pred_s))
        row.update({"time_steps": time_steps, "dropout": dropout})
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[sensitivity] saved {out}")


if __name__ == "__main__":
    main()

"""Walk-forward validation scaffold.

This script is intentionally conservative: it retrains on expanding windows and records
fold-level metrics. Use small epochs first because full neural walk-forward validation is
computationally expensive.

Example:
    python scripts/walk_forward.py --epochs 20 --folds 5
"""
from __future__ import annotations

import argparse
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
    p.add_argument("--end", default=None)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--time-steps", type=int, default=60)
    p.add_argument("--out", default="results/walk_forward_metrics.csv")
    args = p.parse_args()

    tf.random.set_seed(42)
    np.random.seed(42)
    raw = fetch(args.ticker, args.start, args.end)
    df = add_features(raw)
    fold_edges = np.linspace(0.55, 0.90, args.folds)
    rows = []

    for i, ratio in enumerate(fold_edges, start=1):
        print(f"\n[wf] fold {i}/{args.folds} train_ratio={ratio:.2f}")
        data = split_and_scale(df, list(df.columns), args.time_steps, float(ratio))
        X_tr, y_tr = data["X_train"], data["y_train"]
        X_te, y_te = data["X_test"], data["y_test"]
        close_scaler = data["close_scaler"]
        model = build(args.time_steps, X_tr.shape[2], units=[64, 32], dropout=0.2)
        train(model, X_tr, y_tr, X_te, y_te, epochs=args.epochs, batch_size=32, save_path=f"results/wf_fold_{i}.keras")
        pred_s, _ = mc_predict(model, X_te, n_samples=50)
        inv = lambda x: close_scaler.inverse_transform(x.reshape(-1, 1)).flatten()
        row = metrics(inv(y_te), inv(pred_s))
        row.update({"fold": i, "train_ratio": ratio, "n_train": len(X_tr), "n_test": len(X_te)})
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[wf] saved {out}")


if __name__ == "__main__":
    main()

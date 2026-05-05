from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "ticker": "AAPL",
    "start_date": "2010-01-01",
    "end_date": None,
    "time_steps": 60,
    "train_ratio": 0.80,
    "lstm_units": [128, 64],
    "dropout": 0.20,
    "epochs": 150,
    "batch_size": 32,
    "forecast_days": 30,
    "mc_samples": 200,
    "model_path": "best_model.keras",
    "seed": 42,
    "transaction_cost": 0.0005,
}


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    cfg = DEFAULT_CONFIG.copy()
    p = Path(path)
    if p.exists():
        loaded = yaml.safe_load(p.read_text()) or {}
        cfg.update(loaded)
    return cfg

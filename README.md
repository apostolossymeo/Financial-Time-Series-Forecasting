# AAPL Stock Forecasting with Stacked BiLSTM

A reproducible deep-learning forecasting pipeline for Apple Inc. daily close prices. The project loads OHLCV data, builds technical indicators, trains a stacked Bidirectional LSTM with Monte Carlo Dropout uncertainty, evaluates against simple baselines, and stress-tests a directional strategy with transaction costs.

> This is an educational research project, not investment advice. The model's directional edge is intentionally presented conservatively.

## Highlights

- Local CSV loader for Nasdaq/Yahoo-style historical data, with yfinance fallback
- Technical features: RSI, MACD, Bollinger width, returns, and volume ratio
- Stacked BiLSTM with early stopping, model checkpointing, and ReduceLROnPlateau
- Monte Carlo Dropout uncertainty bands
- Baselines: naive last-value and moving-average forecasts
- Backtest with configurable transaction costs, Sharpe, max drawdown, and trade count
- Tests for feature engineering and evaluation utilities

## Data

| Source | Ticker | Period in included sample | Rows |
|---|---:|---:|---:|
| Nasdaq Historical Quotes CSV | AAPL | 2010-03-26 to 2020-02-28 | 2,499 sessions |

Place data at `data/AAPL.csv`. The repository ignores the `data/` folder by default, so large or proprietary datasets are not committed accidentally.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt

python run.py --csv data/AAPL.csv
```

Use a different ticker or forecast horizon:

```bash
python run.py --ticker MSFT --start 2015-01-01 --forecast 60
```

Configuration lives in `config.yaml` and can be overridden from the command line.

## Outputs

Running `python run.py` writes artifacts to `results/`:

| File | Purpose |
|---|---|
| `forecast.png` | Actual price, train fit, test fit, recursive forecast, and uncertainty bands |
| `loss.png` | Training and validation MSE by epoch |
| `backtest.png` | Strategy equity curve vs. buy-and-hold |
| `drawdown.png` | Strategy drawdown vs. buy-and-hold drawdown |
| `residuals.png` | Forecast residuals through the test set |
| `error_hist.png` | Error distribution |
| `metrics.json` | Machine-readable metrics and backtest summary |

### Forecast

![Forecast](results/forecast.png)

### Training History

![Training History](results/loss.png)

### Backtest

![Backtest](results/backtest.png)

## Model architecture

```text
Input: 60 timesteps × 8 features
  └─ Bidirectional LSTM(128), return_sequences=True
  └─ Dropout(0.20), active during MC inference
  └─ LSTM(64)
  └─ Dropout(0.20), active during MC inference
  └─ Dense(1)
```

The model is compiled with Adam and MSE loss. Uncertainty is estimated by keeping dropout active at inference and aggregating stochastic forward passes.

## Evaluation design

The project reports three categories of evidence:

1. **Forecast accuracy**: RMSE, MAE, MAPE, and directional accuracy.
2. **Baseline comparison**: naive last close and 5-day moving average.
3. **Strategy viability**: total return, Sharpe ratio, max drawdown, trade count, and transaction-cost sensitivity.

The backtest is intentionally simple: it goes long when the model predicts the next close above today's close and short otherwise. Transaction costs default to 0.05% per position change.

## Project structure

```text
stock-lstm/
├── config.yaml
├── run.py
├── src/
│   ├── config.py
│   ├── data.py
│   ├── evaluate.py
│   ├── features.py
│   ├── model.py
│   └── viz.py
├── tests/
│   ├── test_evaluate.py
│   └── test_features.py
├── docs/
│   └── methodology.md
├── data/
│   └── AAPL.csv
├── results/
│   ├── forecast.png
│   ├── loss.png
│   └── backtest.png
├── requirements.txt
├── requirements-dev.txt
└── pyproject.toml
```

## Run tests

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
```


## Extra Scripts

```bash
# Generate all non-training EDA/baseline figures
make figures

# Run tests
make test

# Walk-forward validation scaffold; expensive, start small
PYTHONPATH=. python scripts/walk_forward.py --epochs 20 --folds 5

# Hyperparameter sensitivity scaffold
PYTHONPATH=. python scripts/sensitivity.py --epochs 15

# Individual reports
PYTHONPATH=. python scripts/eda_report.py
PYTHONPATH=. python scripts/feature_report.py
PYTHONPATH=. python scripts/baseline_report.py
```

## Expanded Figure Gallery

| Figure | File | Why it matters |
|---|---|---|
| Price + volume | `results/price_volume.png` | Raw data sanity check |
| Return distribution | `results/returns_distribution.png` | Tail-risk context |
| Rolling risk | `results/rolling_risk.png` | Volatility/drawdown regimes |
| Monthly returns | `results/monthly_returns_heatmap.png` | Market regime timeline |
| Moving averages | `results/moving_averages.png` | Trend baseline context |
| RSI | `results/rsi_regime.png` | Momentum/mean-reversion context |
| MACD | `results/macd.png` | Momentum indicator behavior |
| Feature correlation | `results/feature_correlation.png` | Feature redundancy check |
| Baseline RMSE | `results/baseline_rmse.png` | Naive/SMA/EMA comparison |
| Baseline predictions | `results/baseline_predictions.png` | Visual baseline quality check |

See `docs/figures.md` for the full figure guide.

## Limitations

- A single chronological train/test split is not enough for a production-grade trading claim.
- Walk-forward validation should be added before trusting out-of-sample estimates.
- Directional accuracy near 50% is weak and may not survive costs or regime shifts.
- Recursive multi-day forecasts compound error.
- The model does not use macro, earnings, options, sentiment, or market-wide risk features.

## Suggested extensions

- Walk-forward validation on expanding windows
- Transaction-cost and slippage sweeps
- Exogenous features: VIX, rates, sector ETF returns, earnings dates
- Probability calibration for MC Dropout intervals
- A temporal fusion transformer or N-BEATS baseline

## License

MIT, unless you choose another license before publishing.

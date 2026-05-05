# AAPL Stock Forecasting with Stacked BiLSTM

A reproducible deep-learning forecasting pipeline for Apple Inc. daily close prices. The project loads OHLCV data, builds technical indicators, trains a stacked Bidirectional LSTM with Monte Carlo Dropout uncertainty, evaluates against simple baselines, and stress-tests a directional strategy with transaction costs.

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


## Run tests

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
```


## Expanded Figure Gallery

| Figure | File | Why it matters |
|---|---|---|
| Forecast | `results/forecast.png` | Model vs actual price behavior |
| Backtest | `results/backtest.png` | Strategy performance vs buy & hold |
| Loss | `results/loss.png` | Training and validation convergence |
| Residuals | `results/residuals.png` | Error structure and bias detection |
| Drawdown | `results/drawdown.png` | Risk profile and worst-case losses |
| Error histogram | `results/error_hist.png` | Distribution of prediction errors |



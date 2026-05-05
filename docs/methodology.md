# Methodology

This project is designed as a reproducible forecasting experiment, not a trading recommendation.

## Pipeline

1. Load daily OHLCV data from a local Nasdaq/Yahoo-style CSV or yfinance.
2. Build technical features from close and volume.
3. Scale features using the training partition only.
4. Convert the series into rolling 60-session windows.
5. Train a stacked BiLSTM model with early stopping and learning-rate reduction.
6. Estimate uncertainty with Monte Carlo Dropout.
7. Compare against naive and moving-average baselines.
8. Backtest a directional strategy with configurable transaction costs.

## Important caveats

- A single chronological train/test split can overstate performance on non-stationary assets.
- Directional accuracy close to 50% is weak evidence for tradability.
- Transaction costs, slippage, liquidity, tax treatment, and market impact matter.
- The 30-day recursive forecast compounds model error because predicted values are fed back into the next step.

## Next research steps

- Walk-forward validation with expanding or rolling windows.
- Exogenous features such as VIX, rates, sector ETF returns, earnings dates, and macro announcements.
- Probabilistic calibration checks for uncertainty intervals.
- Position sizing based on forecast confidence rather than binary long/short exposure.

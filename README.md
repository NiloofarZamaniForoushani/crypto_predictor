# BTC Daily Prediction + Simple Strategy (Public Data)

This project builds a BTC-USD daily direction prediction model using public Yahoo Finance data,
with time-series validation and hyperparameter optimization, and a simple probability-threshold strategy backtest.

## Project structure
- `src/` core modules (data, features, modeling, optimize, train, predict, split)
- `run_all.py` end-to-end pipeline runner
- `notebooks/` analysis + strategy notebook (EDA, rolling AUC, backtest)
- `artifacts/` (ignored) saved models and reports

## Setup
```bash
pip install -r requirements.txt

#feature engineering (indicators + lags)
import numpy as np
import pandas as pd


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()

def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """
    RSI momentum indicator.
    Why it matters:
    - Helps capture overbought/oversold style momentum signals.
    """
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features using only information available up to day t.
    
    Why it matters:
    - In quant, leakage is the #1 silent failure.
    - These features mimic what many real trading signals look like.
    """
    out = df.copy()

    # Log price and log returns are standard in finance (more stable than raw returns)
    out["log_close"] = np.log(out["close"])
    out["log_ret_1"] = out["log_close"].diff()

    # Lagged returns: momentum/reversal patterns
    for k in [1, 2, 3, 5, 10]:
        out[f"log_ret_lag_{k}"] = out["log_ret_1"].shift(k)

    # Rolling volatility: regime/risk proxy
    for w in [5, 10, 20]:
        out[f"vol_{w}"] = out["log_ret_1"].rolling(w).std()

    # Trend via moving averages
    for w in [5, 10, 20, 50]:
        out[f"sma_{w}"] = out["close"].rolling(w).mean()
        out[f"close_over_sma_{w}"] = out["close"] / (out[f"sma_{w}"] + 1e-12) - 1

    # MACD-like signals (common trend/momentum indicator)
    out["ema_12"] = _ema(out["close"], 12)
    out["ema_26"] = _ema(out["close"], 26)
    out["macd"] = out["ema_12"] - out["ema_26"]
    out["macd_signal"] = _ema(out["macd"], 9)
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    # RSI
    out["rsi_14"] = _rsi(out["close"], 14)

    # Candle/range features: intraday structure
    out["hl_range"] = (out["high"] - out["low"]) / (out["close"] + 1e-12)
    out["oc_change"] = (out["close"] - out["open"]) / (out["open"] + 1e-12)

    # Volume surprise: volume vs its own recent history
    out["vol_sma_20"] = out["volume"].rolling(20).mean()
    out["volume_over_sma20"] = out["volume"] / (out["vol_sma_20"] + 1e-12) - 1

    return out

def add_target(df: pd.DataFrame, horizon_days: int = 1) -> pd.DataFrame:
    """
    Target:
      y_t = 1 if Close_{t+h} > Close_t else 0

    Why it matters:
    - Defines the prediction problem unambiguously.
    - Horizon can be changed to 3, 7 days etc.
    """
    out = df.copy()
    out["future_log_ret"] = out["log_close"].shift(-horizon_days) - out["log_close"]
    out["y"] = (out["future_log_ret"] > 0).astype(int)
    return out

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Decide which columns are features.
    We exclude date and any target/leaky columns.
    """
    exclude = {"date", "y", "future_log_ret", "log_close", "log_ret_1"}
    return [c for c in df.columns if c not in exclude]

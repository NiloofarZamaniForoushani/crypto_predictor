#download public OHLCV

import pandas as pd
import yfinance as yf

def load_ohlcv(symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
    """
    Downloads daily OHLCV data using yfinance (public).
    
    Why it matters:
    - A consistent, reproducible data source is essential in quant workflows.
    - We standardize columns so the rest of the pipeline stays stable.
    """
    df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data returned for {symbol}. Check symbol or dates.")

    # Normalize MultiIndex columns from yfinance (multiple tickers or group_by variations)
    if isinstance(df.columns, pd.MultiIndex):
        level0 = [str(x).lower() for x in df.columns.get_level_values(0)]
        level1 = [str(x).lower() for x in df.columns.get_level_values(1)]
        ohlcv = {"open", "high", "low", "close", "adj close", "volume"}

        if set(level0) & ohlcv:
            tickers = list(dict.fromkeys(df.columns.get_level_values(1)))
            if len(tickers) != 1:
                raise ValueError(
                    "Multiple tickers returned by yfinance. "
                    "Please pass a single symbol (e.g., 'AAPL') or download one ticker at a time."
                )
            df.columns = [str(c[0]) for c in df.columns]
        elif set(level1) & ohlcv:
            tickers = list(dict.fromkeys(df.columns.get_level_values(0)))
            if len(tickers) != 1:
                raise ValueError(
                    "Multiple tickers returned by yfinance. "
                    "Please pass a single symbol (e.g., 'AAPL') or download one ticker at a time."
                )
            df.columns = [str(c[1]) for c in df.columns]
        else:
            # Fallback: flatten with a separator so columns are unique
            df.columns = ["_".join(map(str, c)) for c in df.columns]

    df = df.rename(columns=str.lower).reset_index()
    df = df.rename(columns={"Date": "date", "Datetime": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep].dropna()

    # Guard against duplicate columns after normalization
    if not df.columns.is_unique:
        df = df.loc[:, ~df.columns.duplicated()]

    return df

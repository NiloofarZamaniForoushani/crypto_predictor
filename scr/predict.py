#load model and predict latest day

import os
import joblib
import numpy as np
import pandas as pd

def load_model(artifacts_dir: str):
    path = os.path.join(artifacts_dir, "model.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}. Run training first.")
    return joblib.load(path)

def predict_latest(df: pd.DataFrame, feature_cols: list[str], artifacts_dir: str) -> dict:
    """
    Predict P(up tomorrow) for the most recent day with complete features.

    Why it matters:
    - This is the 'real usage' output: a probability forecast for next day.
    """
    model = load_model(artifacts_dir)

    latest = df.dropna(subset=feature_cols).iloc[-1]
    X = latest[feature_cols].to_numpy().reshape(1, -1)
    p_up = float(model.predict_proba(X)[0, 1])

    return {
        "date": str(latest["date"].date()),
        "p_up": p_up
    }


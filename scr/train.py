#train best model + save artifacts

import os
import json
import joblib
import numpy as np
import pandas as pd

from .optimize import tune_hyperparams
from .modeling import make_model, evaluate_classifier

def train_and_save(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    artifacts_dir: str,
    n_splits_cv: int,
    n_trials: int,
    random_seed: int
):
    """
    Train pipeline:
    1) Tune hyperparameters using train-only walk-forward CV
    2) Fit final model on full training set
    3) Save model + metadata

    Why it matters:
    - Separation of training and evaluation is the backbone of quant research.
    """
    os.makedirs(artifacts_dir, exist_ok=True)

    X_train = train_df[feature_cols].to_numpy()
    y_train = train_df["y"].to_numpy().astype(int)

    best_params, best_cv_auc = tune_hyperparams(
        X=X_train,
        y=y_train,
        n_splits=n_splits_cv,
        n_trials=n_trials,
        random_seed=random_seed
    )

    model = make_model(best_params)
    model.fit(X_train, y_train)

    train_metrics = evaluate_classifier(model, X_train, y_train)

    # Save artifacts
    joblib.dump(model, os.path.join(artifacts_dir, "model.joblib"))
    with open(os.path.join(artifacts_dir, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)
    with open(os.path.join(artifacts_dir, "train_report.json"), "w") as f:
        json.dump({
            "best_cv_auc": best_cv_auc,
            "train_accuracy": train_metrics.accuracy,
            "train_roc_auc": train_metrics.roc_auc,
            "train_logloss": train_metrics.logloss,
        }, f, indent=2)

    return model, best_params, best_cv_auc, train_metrics

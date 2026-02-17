#Optuna hyperparameter optimization (time-series CV)

import numpy as np
import optuna
from sklearn.model_selection import TimeSeriesSplit
from .modeling import make_model

def tune_hyperparams(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    n_trials: int,
    random_seed: int
) -> tuple[dict, float]:
    """
    Tune hyperparameters with walk-forward cross-validation.

    Why it matters:
    - Hyperparameters can overfit badly if you tune on the test set.
    - TimeSeriesSplit preserves ordering (no leakage).
    """
    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 255),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 200),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 10.0, log=True),
            "max_bins": trial.suggest_int("max_bins", 64, 255),
            "random_state": random_seed,
        }

        tscv = TimeSeriesSplit(n_splits=n_splits)
        aucs = []

        for tr_idx, va_idx in tscv.split(X):
            model = make_model(params)
            model.fit(X[tr_idx], y[tr_idx])
            p = model.predict_proba(X[va_idx])[:, 1]
            # We optimize AUC because it measures ranking quality
            from sklearn.metrics import roc_auc_score
            aucs.append(roc_auc_score(y[va_idx], p))

        return float(np.mean(aucs))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_seed)
    )
    study.optimize(objective, n_trials=n_trials)

    best_params = dict(study.best_params)
    best_params["random_state"] = random_seed
    best_score = float(study.best_value)

    return best_params, best_score

import numpy as np
import pandas as pd

from src.data import load_ohlcv
from src.features import add_features, add_target, get_feature_columns
from src.split import time_train_test_split
from src.modeling import evaluate_classifier
from src.train import train_and_save
from src.predict import predict_latest

def main():
    # --- Config (edit these) ---
    symbol = "BTC-USD"
    start = "2018-01-01"
    end = None

    horizon_days = 1
    test_size = 0.20

    n_splits_cv = 5
    n_trials = 50
    random_seed = 42

    artifacts_dir = "artifacts"

    # --- 1) Load data ---
    df = load_ohlcv(symbol=symbol, start=start, end=end)

    # --- 2) Features + target ---
    df_feat = add_features(df)
    df_labeled = add_target(df_feat, horizon_days=horizon_days)

    feature_cols = get_feature_columns(df_labeled)

    # Drop rows that cannot be used due to rolling windows / horizon shift
    usable = df_labeled.dropna(subset=feature_cols + ["y"]).reset_index(drop=True)

    # --- 3) Time split ---
    idx_train, idx_test = time_train_test_split(len(usable), test_size=test_size)
    train_df = usable.iloc[idx_train].copy()
    test_df  = usable.iloc[idx_test].copy()

    X_test = test_df[feature_cols].to_numpy()
    y_test = test_df["y"].to_numpy().astype(int)

    print(f"Data rows usable: {len(usable):,}")
    print(f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")
    print(f"Train period: {train_df['date'].min().date()} → {train_df['date'].max().date()}")
    print(f"Test period : {test_df['date'].min().date()} → {test_df['date'].max().date()}")

    # --- 4) Train + tune on train only ---
    model, best_params, best_cv_auc, train_metrics = train_and_save(
        train_df=train_df,
        feature_cols=feature_cols,
        artifacts_dir=artifacts_dir,
        n_splits_cv=n_splits_cv,
        n_trials=n_trials,
        random_seed=random_seed
    )

    # --- 5) Evaluate on holdout ---
    test_metrics = evaluate_classifier(model, X_test, y_test)

    # --- 6) Latest prediction ---
    latest = predict_latest(usable, feature_cols=feature_cols, artifacts_dir=artifacts_dir)

    print("\n=== Training diagnostics (NOT the real performance) ===")
    print(f"Best CV ROC-AUC: {best_cv_auc:.4f}")
    print(f"Train Accuracy: {train_metrics.accuracy:.4f} | Train AUC: {train_metrics.roc_auc:.4f} | Train LogLoss: {train_metrics.logloss:.4f}")

    print("\n=== Holdout test performance (realistic) ===")
    print(f"Test Accuracy: {test_metrics.accuracy:.4f} | Test AUC: {test_metrics.roc_auc:.4f} | Test LogLoss: {test_metrics.logloss:.4f}")

    # Baseline: constant probability = train up-rate
    base_p = float(train_df["y"].mean())
    base_pred = (np.full_like(y_test, base_p, dtype=float) >= 0.5).astype(int)
    base_acc = float((base_pred == y_test).mean())
    print("\n=== Baseline ===")
    print(f"Train up-rate p: {base_p:.4f} | Baseline accuracy: {base_acc:.4f}")

    print("\n=== Latest prediction ===")
    print(latest)

if __name__ == "__main__":
    main()

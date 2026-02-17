#pipeline + metrics
from dataclasses import dataclass
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.ensemble import HistGradientBoostingClassifier

@dataclass
class Metrics:
    accuracy: float
    roc_auc: float
    logloss: float

def make_model(params: dict) -> Pipeline:
    """
    Model pipeline:
    - Imputer: handles NaNs from rolling features
    - Scaler: standardizes features for stable training
    - Classifier: gradient boosting for nonlinear patterns

    Why it matters:
    - A pipeline guarantees identical preprocessing in train and predict.
    """
    clf = HistGradientBoostingClassifier(**params)
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", clf),
    ])

def evaluate_classifier(model: Pipeline, X: np.ndarray, y: np.ndarray) -> Metrics:
    """
    Evaluate using:
    - Accuracy: intuitive but can be misleading
    - ROC-AUC: ranking quality of probabilities
    - LogLoss: probability calibration quality (very important in quant)
    """
    p = model.predict_proba(X)[:, 1]
    pred = (p >= 0.5).astype(int)
    return Metrics(
        accuracy=float(accuracy_score(y, pred)),
        roc_auc=float(roc_auc_score(y, p)),
        logloss=float(log_loss(y, p)),
    )



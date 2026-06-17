"""Evaluation metrics for classification and regression tasks."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute accuracy, F1, precision, recall, and confusion matrix.

    Args:
        y_true:      ground-truth binary labels (0 or 1)
        y_pred_prob: predicted probabilities in [0, 1]
        threshold:   decision boundary (default 0.5)

    Returns:
        dict with keys: accuracy, f1, precision, recall, confusion_matrix
    """
    y_pred = (np.array(y_pred_prob).flatten() >= threshold).astype(int)
    y_true = np.array(y_true).flatten().astype(int)

    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy":         float(accuracy_score(y_true, y_pred)),
        "f1":               float(f1_score(y_true, y_pred, zero_division=0)),
        "precision":        float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":           float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
    }


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Compute MAE, RMSE, MAPE, and R² for regression outputs.

    Returns:
        dict with keys: mae, rmse, mape, r2
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.where(np.abs(y_true) < 1e-8, 1e-8, y_true))) * 100)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else float("nan")

    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


def sentiment_price_correlation(
    sentiment: np.ndarray,
    returns: np.ndarray,
) -> dict:
    """Pearson correlation between daily sentiment scores and price returns.

    Skips NaN values. Returns r=NaN if fewer than 5 valid pairs exist.

    Returns:
        dict with keys: pearson_r, p_value
    """
    s = np.array(sentiment).flatten()
    r = np.array(returns).flatten()
    mask = ~(np.isnan(s) | np.isnan(r))

    if mask.sum() < 5:
        return {"pearson_r": float("nan"), "p_value": float("nan")}

    rho, p = pearsonr(s[mask], r[mask])
    return {"pearson_r": float(rho), "p_value": float(p)}


def print_classification_report(metrics: dict, symbol: str = "", scenario: str = "") -> None:
    """Pretty-print classification metrics to stdout."""
    header = f"  [{symbol}] {scenario}" if symbol or scenario else ""
    if header:
        print(header)
    print(f"    Accuracy  : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.1f}%)")
    print(f"    F1-Score  : {metrics['f1']:.4f}")
    print(f"    Precision : {metrics['precision']:.4f}")
    print(f"    Recall    : {metrics['recall']:.4f}")
    cm = metrics.get("confusion_matrix")
    if cm:
        print(f"    Confusion : TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")


def print_regression_report(metrics: dict, symbol: str = "", scenario: str = "") -> None:
    """Pretty-print regression metrics to stdout."""
    header = f"  [{symbol}] {scenario}" if symbol or scenario else ""
    if header:
        print(header)
    print(f"    MAE   : {metrics['mae']:.4f}")
    print(f"    RMSE  : {metrics['rmse']:.4f}")
    print(f"    MAPE  : {metrics['mape']:.2f}%")
    print(f"    R²    : {metrics['r2']:.4f}")

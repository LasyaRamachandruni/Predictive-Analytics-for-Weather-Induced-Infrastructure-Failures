"""
Utility functions for computing regression and classification metrics.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
from sklearn import metrics


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    """
    Compute standard regression metrics.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(metrics.mean_absolute_error(y_true_arr, y_pred_arr)),
        "rmse": float(np.sqrt(metrics.mean_squared_error(y_true_arr, y_pred_arr))),
        "r2": float(metrics.r2_score(y_true_arr, y_pred_arr)),
    }


def classification_metrics(
    y_true: Iterable[int],
    y_proba: Iterable[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute classification metrics given probabilities and a threshold.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_proba_arr = np.asarray(y_proba, dtype=float)
    y_pred_arr = (y_proba_arr >= threshold).astype(int)

    results = {
        "accuracy": float(metrics.accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(metrics.precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": float(metrics.recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "f1": float(metrics.f1_score(y_true_arr, y_pred_arr, zero_division=0)),
    }

    try:
        results["roc_auc"] = float(metrics.roc_auc_score(y_true_arr, y_proba_arr))
    except ValueError:
        results["roc_auc"] = float("nan")

    try:
        results["pr_auc"] = float(metrics.average_precision_score(y_true_arr, y_proba_arr))
    except ValueError:
        results["pr_auc"] = float("nan")

    return results


def threshold_from_percentile(y: Iterable[float], percentile: float) -> Tuple[float, float]:
    """
    Compute a classification threshold based on a percentile from continuous targets.
    Returns both the percentile (0-1 scale) and the threshold value.
    """
    y_arr = np.asarray(y, dtype=float)
    percentile = np.clip(percentile, 0.0, 1.0)
    threshold_value = float(np.percentile(y_arr, percentile * 100))
    return percentile, threshold_value

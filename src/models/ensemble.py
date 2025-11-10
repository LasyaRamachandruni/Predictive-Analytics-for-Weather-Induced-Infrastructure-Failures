"""
Tabular ensemble models (Random Forest + XGBoost) and blending utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor


@dataclass
class EnsemblePrediction:
    rf: np.ndarray
    xgb: np.ndarray
    blended_tabular: np.ndarray
    hybrid: np.ndarray


class TabularEnsemble:
    """
    Wrapper around Random Forest and XGBoost supporting regression & classification.
    """

    def __init__(
        self,
        target_type: str,
        rf_params: Optional[Dict[str, Any]] = None,
        xgb_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        rf_params = rf_params or {}
        xgb_params = xgb_params or {}
        self.target_type = target_type

        if target_type == "classification":
            self.rf_model = RandomForestClassifier(**rf_params)
            self.xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", **xgb_params)
        else:
            self.rf_model = RandomForestRegressor(**rf_params)
            self.xgb_model = XGBRegressor(**xgb_params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.rf_model.fit(X, y)
        self.xgb_model.fit(X, y)

    def _predict_rf(self, X: np.ndarray) -> np.ndarray:
        if self.target_type == "classification":
            return self.rf_model.predict_proba(X)[:, 1]
        return self.rf_model.predict(X)

    def _predict_xgb(self, X: np.ndarray) -> np.ndarray:
        if self.target_type == "classification":
            return self.xgb_model.predict_proba(X)[:, 1]
        return self.xgb_model.predict(X)

    def predict(self, X: np.ndarray, alpha: float = 0.5) -> EnsemblePrediction:
        rf_pred = self._predict_rf(X)
        xgb_pred = self._predict_xgb(X)
        blended_tabular = alpha * rf_pred + (1 - alpha) * xgb_pred
        return EnsemblePrediction(
            rf=rf_pred,
            xgb=xgb_pred,
            blended_tabular=blended_tabular,
            hybrid=blended_tabular,  # placeholder, updated when combining with LSTM
        )


def blend_with_lstm(
    tabular_pred: np.ndarray,
    lstm_pred: np.ndarray,
    weight_tabular: float = 0.5,
) -> np.ndarray:
    """
    Combine tabular ensemble predictions with LSTM outputs through a weighted average.
    """
    weight_tabular = np.clip(weight_tabular, 0.0, 1.0)
    weight_lstm = 1.0 - weight_tabular
    if tabular_pred.shape != lstm_pred.shape:
        raise ValueError("Tabular and LSTM predictions must share the same shape for blending.")
    return weight_tabular * tabular_pred + weight_lstm * lstm_pred

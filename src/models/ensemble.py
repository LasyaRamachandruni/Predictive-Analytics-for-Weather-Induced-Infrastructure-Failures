"""
ensemble.py
------------
Combines LSTM, Random Forest, and XGBoost for ensemble outage prediction.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from torch import nn
from .lstm_model import LSTMOutagePredictor

class EnsembleOutageModel:
    def __init__(self, input_size):
        # Initialize submodels
        self.lstm_model = LSTMOutagePredictor(input_size=input_size)
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)

    def fit_tabular(self, X, y):
        """Train tabular ensemble models (RF + XGBoost)."""
        print("Training RandomForest & XGBoost...")
        self.rf_model.fit(X, y)
        self.xgb_model.fit(X, y)

    def predict_tabular(self, X):
        """Combine predictions from RandomForest & XGBoost."""
        rf_pred = self.rf_model.predict(X)
        xgb_pred = self.xgb_model.predict(X)
        return 0.5 * (rf_pred + xgb_pred)

    def forward_lstm(self, X_seq):
        """Run temporal prediction using the LSTM."""
        with torch.no_grad():
            return self.lstm_model(X_seq)


if __name__ == "__main__":
    print("✅ Ensemble model initialized successfully.")

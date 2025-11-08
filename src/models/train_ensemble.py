"""
train_ensemble.py
-----------------
Trains and evaluates the EnsembleOutageModel using data from the pipeline.
Also visualizes predictions.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

# Import your ensemble model and data pipeline
from .ensemble import EnsembleOutageModel
from ..data.data_pipeline import build_data_pipeline
from ..visualization.plot_results import plot_actual_vs_predicted


def train_and_evaluate():
    print("🚀 Starting ensemble model training...")

    # Step 1: Build the data pipeline
    print("🚀 Building data pipeline...")
    X_train, X_test, y_train, y_test = build_data_pipeline()

    # Step 2: Convert DataFrames to NumPy arrays
    X_train_np = np.array(X_train)
    X_test_np = np.array(X_test)
    y_train_np = np.array(y_train)
    y_test_np = np.array(y_test)

    # Step 3: Initialize the ensemble model
    model = EnsembleOutageModel(input_size=X_train.shape[1])

    # Step 4: Train Random Forest + XGBoost models
    print("🌲 Training RandomForest & XGBoost...")
    model.fit_tabular(X_train_np, y_train_np)

    # Step 5: Generate predictions
    y_pred = model.predict_tabular(X_test_np)

    # Step 6: Evaluate performance
    mae = mean_absolute_error(y_test_np, y_pred)
    r2 = r2_score(y_test_np, y_pred)

    print(f"✅ Training complete — MAE: {mae:.3f}, R²: {r2:.3f}")

    # Step 7: Visualize results
    print("📊 Generating Actual vs Predicted plot...")
    plot_actual_vs_predicted(y_test_np, y_pred)


if __name__ == "__main__":
    train_and_evaluate()

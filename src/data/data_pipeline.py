"""
data_pipeline.py
----------------
Handles the full data pipeline: ingestion → preprocessing → feature engineering → split for training.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from .download_and_preprocess import load_datasets


def build_data_pipeline(test_size=0.2, random_state=42):
    """Load, clean, and prepare the dataset for model training."""
    print("🚀 Building data pipeline...")

    # Load datasets
    outage_data, storm_data, svi_data = load_datasets()

    # Simulate merging (you’ll replace this with your actual logic later)
    data = pd.concat([outage_data, storm_data], axis=1)
    data["svi_score"] = np.random.rand(len(data))  # placeholder for SVI merge

     # Handle missing values
    data.fillna(0, inplace=True)

    # Create dummy features and labels
    X = data.drop(columns=["duration_hours"], errors="ignore")
    y = data.get("duration_hours", pd.Series(np.random.rand(len(data))))

    # 🔧 Convert only numeric columns for training
    X = X.select_dtypes(include=["number"]).copy()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


    print("✅ Data pipeline built successfully.")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = build_data_pipeline()
    print("Training data shape:", X_train.shape)
.
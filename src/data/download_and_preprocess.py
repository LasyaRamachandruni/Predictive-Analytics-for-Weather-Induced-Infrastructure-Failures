"""
download_and_preprocess.py
--------------------------
Handles data download, cleaning, and preprocessing for weather-induced infrastructure failure prediction.
"""

import pandas as pd
import numpy as np
import geopandas as gpd


def load_datasets():
    """Load datasets from local storage or APIs (demo version with fake data)."""
    print("Loading datasets...")

    # Fake sample data to allow pipeline testing
    outage_data = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=10),
        "region": ["East", "West", "North", "South", "East", "West", "North", "South", "East", "West"],
        "duration_hours": np.random.randint(1, 12, 10)
    })

    storm_data = pd.DataFrame({
        "event_id": range(10),
        "state": ["CA", "TX", "FL", "NY", "CA", "TX", "FL", "NY", "CA", "TX"],
        "damage": np.random.randint(1000, 10000, 10),
        "start_date": pd.date_range("2025-01-01", periods=10)
    })

    svi_data = pd.DataFrame({
        "county": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
        "svi_score": np.random.rand(10)
    })

    return outage_data, storm_data, svi_data


if __name__ == "__main__":
    o, s, v = load_datasets()
    print(o.head())
    print(s.head())
    print(v.head())

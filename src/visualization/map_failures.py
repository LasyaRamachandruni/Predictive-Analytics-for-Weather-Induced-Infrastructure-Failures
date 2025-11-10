"""
Geospatial visualization of infrastructure failure risk.

Usage
-----
python -m src.visualization.map_failures --artifacts models/latest/metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point


def load_artifacts(artifacts_path: Path) -> pd.DataFrame:
    """
    Load predictions from the artifact directory. Falls back to synthetic data if missing.
    """
    if artifacts_path.is_file():
        artifact_dir = artifacts_path.parent
    else:
        artifact_dir = artifacts_path

    predictions_path = artifact_dir / "predictions.csv"
    if predictions_path.exists():
        return pd.read_csv(predictions_path, parse_dates=["timestamp"])

    # Fallback synthetic data
    rng = pd.Series(range(12))
    demo = pd.DataFrame(
        {
            "region_id": [f"Region_{i:02d}" for i in range(12)],
            "timestamp": pd.Timestamp("2024-01-01"),
            "latitude": 30 + rng * 1.2,
            "longitude": -100 + rng * 1.1,
            "hybrid_pred": 1 + rng * 0.3,
            "hybrid_class": (rng % 2).astype(int),
            "split": "test",
        }
    )
    return demo


def load_threshold(artifacts_path: Path) -> Optional[float]:
    metrics_path = artifacts_path if artifacts_path.is_file() else artifacts_path / "metrics.json"
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get("classification_threshold")
    return None


def aggregate_predictions(predictions: pd.DataFrame, split: str) -> pd.DataFrame:
    df = predictions if split == "all" else predictions[predictions["split"] == split]
    if df.empty:
        raise ValueError(f"No prediction records found for split '{split}'.")
    aggregated = (
        df.groupby("region_id")
        .agg(
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            hybrid_pred=("hybrid_pred", "mean"),
            hybrid_class=("hybrid_class", "max") if "hybrid_class" in df.columns else ("hybrid_pred", "mean"),
        )
        .reset_index()
    )
    return aggregated


def plot_map(predictions: pd.DataFrame, threshold: Optional[float], title: str) -> None:
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    usa = world[world["name"] == "United States of America"]

    geometry = [Point(xy) for xy in zip(predictions["longitude"], predictions["latitude"])]
    gdf = gpd.GeoDataFrame(predictions, geometry=geometry, crs="EPSG:4326")

    fig, ax = plt.subplots(figsize=(12, 7))
    usa.plot(ax=ax, color="whitesmoke", edgecolor="gray")

    markersize = gdf["hybrid_pred"] * 250
    plot_column = "hybrid_pred"
    gdf.plot(
        ax=ax,
        column=plot_column,
        cmap="Reds",
        markersize=markersize,
        legend=True,
        alpha=0.8,
    )

    for _, row in gdf.iterrows():
        ax.text(
            row["longitude"],
            row["latitude"] + 0.3,
            row["region_id"],
            fontsize=8,
            ha="center",
        )

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", alpha=0.3)

    if threshold is not None:
        ax.annotate(
            f"Risk threshold: {threshold:.2f}",
            xy=(0.02, 0.02),
            xycoords="axes fraction",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot infrastructure failure risk on a map.")
    parser.add_argument("--artifacts", default="models/latest/metrics.json", help="Path to metrics.json or artifact directory.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"], help="Prediction split to visualise.")
    parser.add_argument("--title", default="Infrastructure Failure Risk Map", help="Custom plot title.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts_path = Path(args.artifacts)
    predictions = load_artifacts(artifacts_path)
    threshold = load_threshold(artifacts_path)
    aggregated = aggregate_predictions(predictions, args.split)
    plot_map(aggregated, threshold, args.title)


if __name__ == "__main__":
    main()

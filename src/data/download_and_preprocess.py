"""
Data adapters for weather and infrastructure datasets.

This module exposes a unified loader that can operate in two modes:

1. `demo` – generates fully synthetic data so that the repository runs end-to-end
   without any external dependencies.
2. `real` – scaffolding for plugging in real data sources (NOAA/ERA5/Kaggle/etc.).
   Implementers can point to their own loaders via the configuration file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

DEFAULT_RANDOM_SEED = 42


@dataclass
class DemoDataConfig:
    """Configuration used for generating demo data."""

    num_regions: int = 6
    start_date: str = "2024-01-01"
    periods: int = 180
    freq: str = "6H"
    base_failure_rate: float = 0.4
    temperature_mean: float = 55.0
    precipitation_scale: float = 3.0
    wind_scale: float = 15.0
    random_state: int = DEFAULT_RANDOM_SEED


def load_dataset(mode: str = "demo", config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Load the aligned dataset with columns `[region_id, timestamp, features..., failures]`.

    Parameters
    ----------
    mode
        Either `"demo"` or `"real"`.
    config
        Optional dictionary with adapter-specific configuration.
    """
    config = config or {}

    if mode == "demo":
        demo_cfg = DemoDataConfig(**config.get("demo", {}))
        return _load_demo_dataset(demo_cfg)

    if mode == "real":
        return _load_real_dataset(config.get("real", {}))

    raise ValueError(f"Unsupported data loading mode: {mode}")


def _load_demo_dataset(config: DemoDataConfig) -> pd.DataFrame:
    """
    Generate an aligned synthetic dataset covering multiple regions.

    The generated features include temperature, humidity, precipitation, wind, and
    social vulnerability indicators. The target variable `failures` is simulated to
    correlate with extreme weather conditions and vulnerability.
    """
    rng = np.random.default_rng(config.random_state)

    timestamps = pd.date_range(
        start=config.start_date,
        periods=config.periods,
        freq=config.freq,
        inclusive="left",
    )

    region_ids = [f"Region_{i:02d}" for i in range(config.num_regions)]
    latitudes = rng.uniform(30, 47, size=config.num_regions)
    longitudes = rng.uniform(-105, -70, size=config.num_regions)
    elevation = rng.normal(loc=250, scale=120, size=config.num_regions).clip(min=5)
    svi_scores = rng.uniform(0.1, 0.9, size=config.num_regions)

    records = []
    for region_idx, region_id in enumerate(region_ids):
        phase_shift = rng.uniform(0, 2 * np.pi)
        baseline_temp = config.temperature_mean + rng.normal(0, 5)
        region_variability = rng.uniform(0.8, 1.2)
        svi = svi_scores[region_idx]
        elev = elevation[region_idx]
        lon = longitudes[region_idx]
        lat = latitudes[region_idx]

        for step, ts in enumerate(timestamps):
            seasonal_component = 15 * np.sin((2 * np.pi * step / 24) + phase_shift)
            temp = baseline_temp + seasonal_component + rng.normal(0, 2)
            humidity = np.clip(55 + rng.normal(0, 10) + 0.3 * (70 - temp), 10, 100)
            precipitation = np.clip(rng.gamma(shape=2.0, scale=config.precipitation_scale) - 1.5, 0, None)
            wind_speed = np.clip(rng.normal(loc=12, scale=config.wind_scale / 3), 0, None)
            extreme_event = rng.random() < 0.05
            snow_depth = max(0.0, rng.normal(0.5, 1.0) if temp < 32 else 0)

            failure_intensity = (
                config.base_failure_rate * region_variability
                + 0.08 * precipitation
                + 0.05 * max(0, wind_speed - 20)
                + 0.03 * (humidity / 100)
                + 1.5 * extreme_event
                + 0.4 * (1 - np.tanh((temp - 40) / 15) ** 2)
                + 0.6 * svi
                + 0.001 * elev
            )

            failures = float(rng.poisson(lam=max(failure_intensity, 0.05)))

            records.append(
                {
                    "region_id": region_id,
                    "timestamp": ts,
                    "temperature": temp,
                    "humidity": humidity,
                    "precipitation": precipitation,
                    "wind_speed": wind_speed,
                    "extreme_event": int(extreme_event),
                    "snow_depth": snow_depth,
                    "svi": svi,
                    "elevation": elev,
                    "longitude": lon,
                    "latitude": lat,
                    "failures": failures,
                }
            )

    df = pd.DataFrame.from_records(records)
    df.sort_values(by=["region_id", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _load_real_dataset(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Placeholder loader for real datasets.

    Expected config keys
    --------------------
    weather_path: str
        Path to the weather dataset. Must include `region_id` and `timestamp`.
    outages_path: str
        Path to the infrastructure failure dataset with the same keys.
    context_path: Optional[str]
        Optional path to contextual data (e.g., SVI).

    Notes
    -----
    This loader intentionally contains TODO markers where users can plug in
    production data ingestion code. If paths are provided, CSV loading with
    minimal validation is performed.
    """
    weather_path = config.get("weather_path")
    outages_path = config.get("outages_path")

    if not weather_path or not outages_path:
        raise NotImplementedError(
            "Real mode requires `weather_path` and `outages_path` entries in the config."
        )

    parse_dates = config.get("parse_dates", ["timestamp"])

    weather_df = pd.read_csv(weather_path, parse_dates=parse_dates)
    outages_df = pd.read_csv(outages_path, parse_dates=parse_dates)

    context_path = config.get("context_path")
    if context_path:
        context_df = pd.read_csv(context_path)
    else:
        context_df = pd.DataFrame()

    # TODO: Replace with domain-specific joins, quality checks, and feature alignment.
    merged = pd.merge(
        weather_df,
        outages_df,
        on=["region_id", "timestamp"],
        how="inner",
        suffixes=("_weather", "_outages"),
    )

    if not context_df.empty:
        merged = merged.merge(context_df, on="region_id", how="left")

    if "failures" not in merged.columns:
        raise ValueError("Merged dataset must include a `failures` column.")

    merged.sort_values(by=["region_id", "timestamp"], inplace=True)
    merged.reset_index(drop=True, inplace=True)
    return merged
"""
Utility helpers for configuration, filesystem interactions, and artifact management.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
import yaml

PathLike = Union[str, Path]


def load_yaml_config(path: PathLike) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    path:
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_directory(path: PathLike) -> Path:
    """
    Ensure that a directory exists, creating parents if necessary.

    Parameters
    ----------
    path:
        Directory path to create.

    Returns
    -------
    Path
        The resolved directory path.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _default_serializer(obj: Any) -> Any:
    """Fallback serializer that handles numpy/pandas types gracefully."""
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_json(data: Dict[str, Any], path: PathLike) -> Path:
    """
    Persist a dictionary to disk as JSON with a deterministic format.
    """
    path = Path(path)
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_default_serializer, sort_keys=True)
    return path


def save_dataframe(df: pd.DataFrame, path: PathLike) -> Path:
    """
    Save a DataFrame to disk, inferring the storage format from the suffix.

    Supported formats: .csv, .parquet
    """
    path = Path(path)
    ensure_directory(path.parent)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported file extension for DataFrame export: {suffix}")
    return path


def create_run_directory(base_dir: PathLike, run_name: Optional[str] = None) -> Path:
    """
    Create a timestamped run directory for storing artifacts.

    Parameters
    ----------
    base_dir:
        Root artifacts directory, e.g. `models/`.
    run_name:
        Optional run name for easier identification.
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    stem = run_name or "run"
    run_dir = ensure_directory(Path(base_dir) / f"{stem}_{timestamp}")
    return run_dir


def update_latest_artifacts(source_dir: PathLike, latest_dir: PathLike) -> None:
    """
    Update the `latest` artifacts directory by copying contents from the source.
    """
    source_dir = Path(source_dir)
    latest_dir = Path(latest_dir)
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(source_dir, latest_dir)

"""
End-to-end data pipeline for hybrid weather + infrastructure modelling.

Workflow
--------
1. Load raw data via adapters defined in `download_and_preprocess.py`.
2. Align data by region and timestamp, fill gaps, add contextual metadata.
3. Engineer lagged and rolling features for both tabular and sequence models.
4. Create deterministic train/val/test splits.
5. Build sliding-window sequences for the LSTM and tabular matrices for tree models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .download_and_preprocess import load_dataset
from ..utils.metrics import threshold_from_percentile

logger = logging.getLogger(__name__)


DEFAULT_SPLITS = {"val_size": 0.15, "test_size": 0.15}


@dataclass
class TabularDataset:
    features: pd.DataFrame
    target: pd.Series
    metadata: pd.DataFrame


@dataclass
class SequenceDataset:
    features: np.ndarray  # shape: [n_samples, seq_len, n_features]
    target: np.ndarray
    metadata: pd.DataFrame


@dataclass
class PipelineArtifacts:
    tabular: Dict[str, TabularDataset]
    sequences: Dict[str, SequenceDataset]
    feature_columns: List[str]
    sequence_feature_columns: List[str]
    scaler: StandardScaler
    target_column: str
    target_type: str
    classification_threshold: Optional[float]
    metadata: Dict[str, Any]


def _resolve_sequence_length(df: pd.DataFrame, group_col: str, requested: int) -> int:
    """
    Clamp the requested sequence length so at least one training window can be created.
    """
    if requested <= 1:
        return max(1, requested)

    train_df = df[df["split"] == "train"]
    if train_df.empty:
        return max(1, min(requested, len(df)))

    train_counts = train_df.groupby(group_col).size()
    if train_counts.empty:
        return max(1, min(requested, len(df)))

    max_train_count = int(train_counts.max())
    if max_train_count <= 0:
        return 1

    return max(1, min(requested, max_train_count))


def run_data_pipeline(config: Dict[str, Any], mode: str = "real") -> PipelineArtifacts:
    """
    Execute the full data pipeline and return engineered datasets.
    """
    logger.info("Loading dataset (mode=%s)", mode)
    raw_df = load_dataset(mode=mode, config=config.get("data", {}))

    timestamp_col = config.get("data", {}).get("timestamp_column", "timestamp")
    group_col = config.get("data", {}).get("group_column", "region_id")
    target_col = config.get("target", {}).get("column", "failures")

    frequency = config.get("data", {}).get("frequency")
    aligned_df = _align_time_series(raw_df, group_col, timestamp_col, frequency)

    engineered_df = _engineer_features(
        aligned_df,
        group_col=group_col,
        timestamp_col=timestamp_col,
        target_col=target_col,
        config=config,
    )

    splits = config.get("split", DEFAULT_SPLITS)
    split_df = _assign_splits(engineered_df, timestamp_col=timestamp_col, splits=splits)

    feature_exclude = set(config.get("features", {}).get("exclude", []))
    feature_cols = _select_feature_columns(split_df, target_col, feature_exclude, config)
    sequence_feature_cols = config.get("sequence", {}).get("features") or feature_cols

    scaler = StandardScaler()
    train_mask = split_df["split"] == "train"
    scaler.fit(split_df.loc[train_mask, sequence_feature_cols])

    target_type = config.get("target", {}).get("type", "regression")
    classification_pct = config.get("target", {}).get("classification_threshold_percentile", 0.8)
    classification_threshold = None

    if target_type == "classification":
        _, classification_threshold = threshold_from_percentile(
            split_df.loc[train_mask, target_col], percentile=classification_pct
        )
        split_df["target_class"] = (split_df[target_col] >= classification_threshold).astype(int)
        target_array_name = "target_class"
    else:
        target_array_name = target_col

    seq_len_requested = config.get("sequence", {}).get("length", 24)
    seq_len = _resolve_sequence_length(split_df, group_col, seq_len_requested)
    stride = config.get("sequence", {}).get("stride", 1)

    sequences = _build_sequence_datasets(
        split_df,
        group_col=group_col,
        timestamp_col=timestamp_col,
        feature_cols=sequence_feature_cols,
        target_col=target_array_name,
        scaler=scaler,
        seq_len=seq_len,
        stride=stride,
    )

    sequence_metadata = {split: dataset.metadata for split, dataset in sequences.items()}

    tabular = _build_tabular_datasets(
        split_df,
        feature_cols=feature_cols,
        target_col=target_array_name,
        timestamp_col=timestamp_col,
        group_col=group_col,
        sequence_metadata=sequence_metadata,
    )

    metadata = {
        "group_column": group_col,
        "timestamp_column": timestamp_col,
        "sequence_length_requested": seq_len_requested,
        "sequence_length_used": seq_len,
        "tabular_sample_counts": {split: ds.features.shape[0] for split, ds in tabular.items()},
        "sequence_sample_counts": {split: ds.features.shape[0] for split, ds in sequences.items()},
    }

    return PipelineArtifacts(
        tabular=tabular,
        sequences=sequences,
        feature_columns=feature_cols,
        sequence_feature_columns=sequence_feature_cols,
        scaler=scaler,
        target_column=target_array_name,
        target_type=target_type,
        classification_threshold=classification_threshold,
        metadata=metadata,
    )


def _align_time_series(
    df: pd.DataFrame,
    group_col: str,
    timestamp_col: str,
    frequency: Optional[str],
) -> pd.DataFrame:
    """
    Ensure each region has a continuous timeline by reindexing and forward filling.
    """
    aligned_frames: List[pd.DataFrame] = []
    for region_id, region_df in df.groupby(group_col):
        region_df = region_df.sort_values(timestamp_col).copy()
        region_df.set_index(timestamp_col, inplace=True)
        freq = frequency or pd.infer_freq(region_df.index) or "1H"
        full_index = pd.date_range(region_df.index.min(), region_df.index.max(), freq=freq)
        reindexed = region_df.reindex(full_index)
        reindexed[group_col] = region_id
        reindexed.ffill(inplace=True)
        reindexed.fillna(0, inplace=True)
        aligned_frames.append(reindexed.reset_index().rename(columns={"index": timestamp_col}))

    aligned = pd.concat(aligned_frames, axis=0, ignore_index=True)
    aligned.sort_values(by=[group_col, timestamp_col], inplace=True)
    return aligned


def _engineer_features(
    df: pd.DataFrame,
    group_col: str,
    timestamp_col: str,
    target_col: str,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Add lag, rolling statistics, and weather volatility features.
    """
    feature_cfg = config.get("features", {})
    lag_steps: Sequence[int] = feature_cfg.get("lags", [1, 3, 6, 12])
    rolling_windows: Sequence[int] = feature_cfg.get("rolling_windows", [3, 6, 12])
    rolling_stats: Sequence[str] = feature_cfg.get("rolling_statistics", ["mean", "max"])

    df = df.copy()
    df.sort_values(by=[group_col, timestamp_col], inplace=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    configured_cols = feature_cfg.get("base_weather_cols")
    if configured_cols:
        base_weather_cols = [col for col in configured_cols if col in df.columns]
        if len(base_weather_cols) < len(configured_cols):
            missing = set(configured_cols) - set(base_weather_cols)
            if missing:
                logger.warning("Skipping unavailable base weather columns: %s", ", ".join(sorted(missing)))
    else:
        base_weather_cols = [col for col in numeric_cols if col not in {target_col, "longitude", "latitude"}]

    # Collect new feature columns to avoid DataFrame fragmentation
    new_features = {}
    
    # Target lag features
    for lag in lag_steps:
        new_features[f"{target_col}_lag_{lag}"] = df.groupby(group_col)[target_col].shift(lag)

    # Weather column lag and rolling features
    for column in base_weather_cols:
        for lag in lag_steps:
            new_features[f"{column}_lag_{lag}"] = df.groupby(group_col)[column].shift(lag)

        for window in rolling_windows:
            rolled = df.groupby(group_col)[column].rolling(window=window, min_periods=1)
            if "mean" in rolling_stats:
                new_features[f"{column}_rollmean_{window}"] = rolled.mean().reset_index(level=0, drop=True)
            if "max" in rolling_stats:
                new_features[f"{column}_rollmax_{window}"] = rolled.max().reset_index(level=0, drop=True)
            if "std" in rolling_stats:
                new_features[f"{column}_rollstd_{window}"] = rolled.std(ddof=0).reset_index(level=0, drop=True)

    # Target difference and percentage change
    new_features[f"{target_col}_diff_1"] = df.groupby(group_col)[target_col].diff()
    new_features[f"{target_col}_pct_change"] = (
        df.groupby(group_col)[target_col].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
    )

    # Add all new features at once using concat to avoid fragmentation
    if new_features:
        new_features_df = pd.DataFrame(new_features, index=df.index)
        df = pd.concat([df, new_features_df], axis=1)
    
    # Fill NaN values in lag features with 0 (represents "no previous value")
    lag_suffixes = tuple([f"_lag_{lag}" for lag in lag_steps])
    lag_feature_cols = [col for col in df.columns if col.endswith(lag_suffixes)]
    if lag_feature_cols:
        df[lag_feature_cols] = df[lag_feature_cols].fillna(0.0)
    
    # Fill NaN in diff_1 with 0 (no change from previous)
    if f"{target_col}_diff_1" in df.columns:
        df[f"{target_col}_diff_1"] = df[f"{target_col}_diff_1"].fillna(0.0)
    
    # Only drop rows where essential columns (target, group, timestamp) are NaN
    essential_cols = [target_col, group_col, timestamp_col]
    df = df.dropna(subset=essential_cols)
    df.reset_index(drop=True, inplace=True)
    return df


def _assign_splits(df: pd.DataFrame, timestamp_col: str, splits: Dict[str, float]) -> pd.DataFrame:
    """
    Assign deterministic train/val/test splits based on chronological order.
    """
    df = df.copy()
    unique_timestamps = np.sort(df[timestamp_col].unique())
    total = len(unique_timestamps)
    val_size = splits.get("val_size", DEFAULT_SPLITS["val_size"])
    test_size = splits.get("test_size", DEFAULT_SPLITS["test_size"])

    train_cutoff = int(total * (1 - val_size - test_size))
    val_cutoff = int(total * (1 - test_size))
    
    # Ensure at least one timestamp in train set
    if train_cutoff == 0:
        train_cutoff = max(1, total - int(total * (val_size + test_size)))
    if val_cutoff <= train_cutoff:
        val_cutoff = min(total, train_cutoff + max(1, int(total * val_size)))

    train_times = set(pd.to_datetime(unique_timestamps[:train_cutoff]))
    val_times = set(pd.to_datetime(unique_timestamps[train_cutoff:val_cutoff]))

    def assign(ts):
        ts_dt = pd.to_datetime(ts)
        if ts_dt in train_times:
            return "train"
        if ts_dt in val_times:
            return "val"
        return "test"

    df["split"] = df[timestamp_col].apply(assign)
    return df


def _select_feature_columns(
    df: pd.DataFrame,
    target_col: str,
    exclude: set,
    config: Dict[str, Any],
) -> List[str]:
    candidate = config.get("features", {}).get("include")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    disallowed = {target_col, "target_class", "split"}
    disallowed.update(exclude)
    disallowed.update(config.get("features", {}).get("exclude_auto", ["longitude", "latitude"]))
    if candidate:
        return [col for col in candidate if col in df.columns and col not in disallowed]
    return [col for col in numeric_cols if col not in disallowed]


def _build_sequence_datasets(
    df: pd.DataFrame,
    group_col: str,
    timestamp_col: str,
    feature_cols: Sequence[str],
    target_col: str,
    scaler: StandardScaler,
    seq_len: int,
    stride: int,
) -> Dict[str, SequenceDataset]:
    sequences: Dict[str, List[np.ndarray]] = {"train": [], "val": [], "test": []}
    targets: Dict[str, List[float]] = {"train": [], "val": [], "test": []}
    meta_frames: Dict[str, List[pd.DataFrame]] = {"train": [], "val": [], "test": []}

    for region_id, region_df in df.groupby(group_col):
        region_df = region_df.sort_values(timestamp_col).copy()
        split_values = region_df["split"].values
        features = scaler.transform(region_df[feature_cols])
        target_values = region_df[target_col].values

        for end_idx in range(seq_len - 1, len(region_df), stride):
            start_idx = end_idx - seq_len + 1
            window_split = split_values[start_idx : end_idx + 1]
            if np.unique(window_split).size != 1:
                continue
            split_name = window_split[-1]
            if split_name not in sequences:
                continue

            sequences[split_name].append(features[start_idx : end_idx + 1])
            targets[split_name].append(target_values[end_idx])
            meta_frames[split_name].append(
                region_df.iloc[[end_idx]][[group_col, timestamp_col, "latitude", "longitude"]].copy()
            )

    sequence_datasets: Dict[str, SequenceDataset] = {}
    for split_name in sequences.keys():
        if sequences[split_name]:
            feature_array = np.stack(sequences[split_name]).astype(np.float32)
            target_array = np.asarray(targets[split_name], dtype=np.float32)
            metadata = pd.concat(meta_frames[split_name], ignore_index=True)
        else:
            feature_array = np.empty((0, seq_len, len(feature_cols)), dtype=np.float32)
            target_array = np.empty((0,), dtype=np.float32)
            metadata = pd.DataFrame(columns=[group_col, timestamp_col, "latitude", "longitude"])

        sequence_datasets[split_name] = SequenceDataset(
            features=feature_array,
            target=target_array,
            metadata=metadata,
        )

    return sequence_datasets


def _build_tabular_datasets(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    timestamp_col: str,
    group_col: str,
    sequence_metadata: Dict[str, pd.DataFrame],
) -> Dict[str, TabularDataset]:
    datasets: Dict[str, TabularDataset] = {}

    for split_name in ("train", "val", "test"):
        split_df = df[df["split"] == split_name].copy()
        meta_df = sequence_metadata.get(split_name)

        if meta_df is not None:
            meta_df = meta_df[[group_col, timestamp_col]].drop_duplicates()
            keys = list(meta_df.itertuples(index=False, name=None))
            multi_index = pd.MultiIndex.from_tuples(keys, names=[group_col, timestamp_col])
            aligned = split_df.set_index([group_col, timestamp_col]).reindex(multi_index)
            aligned = aligned.dropna(subset=[target_col], how="any")
            split_df = aligned.reset_index()

        split_df = split_df.fillna(0)
        features = split_df[feature_cols].reset_index(drop=True)
        target = split_df[target_col].reset_index(drop=True)
        metadata = split_df[[group_col, timestamp_col, "latitude", "longitude"]].reset_index(drop=True)

        datasets[split_name] = TabularDataset(
            features=features,
            target=target,
            metadata=metadata,
        )

    return datasets

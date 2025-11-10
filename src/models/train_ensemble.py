"""
CLI entrypoint to train the hybrid LSTM + tabular ensemble.
"""

from __future__ import annotations

import argparse
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from joblib import dump
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import yaml

from src.data.data_pipeline import run_data_pipeline
from src.models.ensemble import TabularEnsemble, blend_with_lstm
from src.models.lstm_model import LSTMOutagePredictor, count_parameters
from src.utils.io import (
    create_run_directory,
    load_yaml_config,
    save_dataframe,
    save_json,
    update_latest_artifacts,
)
from src.utils.metrics import classification_metrics, regression_metrics
from src.visualization.plot_results import plot_actual_vs_predicted, plot_residuals

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hybrid ensemble for infrastructure failure prediction.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--mode", choices=["demo", "real"], help="Data loading mode override.")
    parser.add_argument("--run-name", default=None, help="Optional name prefix for the artifact run directory.")
    parser.add_argument("--quick-run", action="store_true", help="Use reduced settings for fast smoke tests.")
    parser.add_argument("--output-dir", default=None, help="Override artifact root directory.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_dataloader(dataset: Any, batch_size: int, shuffle: bool) -> Optional[DataLoader]:
    if dataset.features.size == 0:
        return None
    features = torch.from_numpy(dataset.features)
    targets = torch.from_numpy(dataset.target.astype(np.float32)).unsqueeze(-1)
    tensor_dataset = TensorDataset(features, targets)
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)


def get_device(preference: str = "auto") -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if torch.cuda.is_available() and preference == "auto":
        return torch.device("cuda")
    return torch.device("cpu")


def train_lstm(
    model: LSTMOutagePredictor,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    cfg: Dict[str, Any],
    device: torch.device,
    target_type: str,
) -> List[Dict[str, float]]:
    criterion: nn.Module
    if target_type == "classification":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.get("lr", 1e-3),
        weight_decay=cfg.get("weight_decay", 0.0),
    )
    epochs = cfg.get("epochs", 20)
    patience = cfg.get("patience", 5)
    min_delta = cfg.get("min_delta", 1e-4)
    use_tqdm = cfg.get("progress_bar", True)

    history: List[Dict[str, float]] = []
    best_val = float("inf")
    best_state = None
    patience_counter = 0

    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        batches = 0
        iterator = train_loader
        if use_tqdm:
            iterator = tqdm(train_loader, desc=f"LSTM epoch {epoch}/{epochs}", leave=False)

        for features, targets in iterator:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.get("grad_clip", 1.0))
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        train_loss = epoch_loss / max(batches, 1)

        val_loss = None
        if val_loader is not None:
            model.eval()
            loss_sum = 0.0
            val_batches = 0
            with torch.no_grad():
                for features, targets in val_loader:
                    features = features.to(device)
                    targets = targets.to(device)
                    outputs = model(features)
                    loss = criterion(outputs, targets)
                    loss_sum += loss.item()
                    val_batches += 1
            val_loss = loss_sum / max(val_batches, 1)

            if val_loss + min_delta < best_val:
                best_val = val_loss
                patience_counter = 0
                best_state = deepcopy(model.state_dict())
            else:
                patience_counter += 1
        else:
            best_state = deepcopy(model.state_dict())

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        logger.debug("Epoch %s: train_loss=%.5f val_loss=%s", epoch, train_loss, f"{val_loss:.5f}" if val_loss else "n/a")

        if patience_counter >= patience:
            logger.info("Early stopping triggered after %s epochs.", epoch)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def predict_lstm(
    model: Optional[LSTMOutagePredictor],
    dataset: Any,
    device: torch.device,
    target_type: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if model is None or dataset.features.size == 0:
        return np.array([]), np.array([])

    model.eval()
    with torch.no_grad():
        features = torch.from_numpy(dataset.features).to(device)
        outputs = model(features).squeeze(-1)
        raw = outputs.cpu().numpy()
        if target_type == "classification":
            probs = torch.sigmoid(outputs).cpu().numpy()
            return probs.astype(np.float32), raw.astype(np.float32)
        return raw.astype(np.float32), raw.astype(np.float32)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_type: str,
    threshold: Optional[float],
) -> Dict[str, float]:
    if y_pred.size == 0:
        return {}
    if target_type == "classification":
        threshold = threshold if threshold is not None else 0.5
        return classification_metrics(y_true, y_pred, threshold=threshold)
    return regression_metrics(y_true, y_pred)


def apply_quick_run_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = deepcopy(config)
    data_demo = cfg.setdefault("data", {}).setdefault("demo", {})
    data_demo["periods"] = min(data_demo.get("periods", 180), 72)
    data_demo["num_regions"] = min(data_demo.get("num_regions", 6), 3)

    training_cfg = cfg.setdefault("training", {})
    lstm_cfg = training_cfg.setdefault("lstm", {})
    lstm_cfg["epochs"] = min(lstm_cfg.get("epochs", 20), 3)
    lstm_cfg["batch_size"] = min(lstm_cfg.get("batch_size", 64), 16)
    lstm_cfg["progress_bar"] = False
    training_cfg["quick_run"] = True
    training_cfg["patience_override"] = 1

    tab_cfg = training_cfg.setdefault("tabular", {})
    rf_cfg = tab_cfg.setdefault("rf", {})
    rf_cfg["n_estimators"] = min(rf_cfg.get("n_estimators", 200), 100)
    rf_cfg["max_depth"] = min(rf_cfg.get("max_depth", 10) or 10, 8)
    rf_cfg["n_jobs"] = rf_cfg.get("n_jobs", -1)

    xgb_cfg = tab_cfg.setdefault("xgb", {})
    xgb_cfg["n_estimators"] = min(xgb_cfg.get("n_estimators", 300), 120)
    xgb_cfg["max_depth"] = min(xgb_cfg.get("max_depth", 6), 4)
    xgb_cfg["learning_rate"] = xgb_cfg.get("learning_rate", 0.05)
    xgb_cfg["subsample"] = xgb_cfg.get("subsample", 0.8)
    xgb_cfg["colsample_bytree"] = xgb_cfg.get("colsample_bytree", 0.8)
    xgb_cfg["random_state"] = xgb_cfg.get("random_state", 42)

    return cfg


def train_and_evaluate(
    config: Dict[str, Any],
    mode: str,
    output_dir: Optional[str] = None,
    run_name: Optional[str] = None,
) -> Path:
    training_cfg = config.get("training", {})
    seed = training_cfg.get("random_seed", 42)
    set_seed(seed)

    device = get_device(training_cfg.get("device", "auto"))
    logger.info("Using device: %s", device)

    logger.info("Running data pipeline...")
    pipeline = run_data_pipeline(config, mode=mode)
    sequences = pipeline.sequences
    tabular = pipeline.tabular
    target_type = pipeline.target_type
    logger.info("Sequence samples (train/val/test): %s", pipeline.metadata["sequence_sample_counts"])
    logger.info("Tabular samples (train/val/test): %s", pipeline.metadata["tabular_sample_counts"])

    lstm_cfg = training_cfg.get("lstm", {})
    if training_cfg.get("patience_override"):
        lstm_cfg["patience"] = training_cfg["patience_override"]

    batch_size = lstm_cfg.get("batch_size", 64)
    train_loader = prepare_dataloader(sequences["train"], batch_size=batch_size, shuffle=True)
    val_loader = prepare_dataloader(sequences["val"], batch_size=batch_size, shuffle=False)

    lstm_model: Optional[LSTMOutagePredictor] = None
    lstm_history: List[Dict[str, float]] = []
    if train_loader is None:
        logger.warning("Insufficient sequence data for LSTM; skipping temporal model training.")
    else:
        input_size = sequences["train"].features.shape[-1]
        lstm_model = LSTMOutagePredictor(
            input_size=input_size,
            hidden_size=lstm_cfg.get("hidden_size", 128),
            num_layers=lstm_cfg.get("num_layers", 2),
            dropout=lstm_cfg.get("dropout", 0.1),
            bidirectional=lstm_cfg.get("bidirectional", False),
        )
        logger.info("LSTM parameters: %s", count_parameters(lstm_model))
        lstm_history = train_lstm(lstm_model, train_loader, val_loader, lstm_cfg, device, target_type)

    lstm_preds: Dict[str, Dict[str, np.ndarray]] = {}
    for split_name, dataset in sequences.items():
        preds, raw = predict_lstm(lstm_model, dataset, device, target_type)
        lstm_preds[split_name] = {"pred": preds, "raw": raw}

    tabular_cfg = training_cfg.get("tabular", {})
    ensemble = TabularEnsemble(
        target_type=target_type,
        rf_params=tabular_cfg.get("rf"),
        xgb_params=tabular_cfg.get("xgb"),
    )

    X_train = tabular["train"].features.to_numpy()
    y_train = tabular["train"].target.to_numpy()
    if X_train.size == 0:
        raise RuntimeError("Tabular training set is empty; cannot fit ensemble.")
    ensemble.fit(X_train, y_train)

    blend_alpha = tabular_cfg.get("blend_alpha", 0.5)
    weight_tabular = training_cfg.get("ensemble", {}).get("weight_tabular", 0.5)
    threshold = pipeline.classification_threshold

    metrics_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    predictions_frames: List[pd.DataFrame] = []

    for split_name in ("train", "val", "test"):
        X_split = tabular[split_name].features.to_numpy()
        y_true = tabular[split_name].target.to_numpy()
        metadata = tabular[split_name].metadata.copy()
        metadata["split"] = split_name

        model_metrics: Dict[str, Dict[str, float]] = {}
        lstm_pred = lstm_preds.get(split_name, {}).get("pred", np.array([]))

        if X_split.size == 0:
            metrics_summary[split_name] = model_metrics
            continue

        tab_preds = ensemble.predict(X_split, alpha=blend_alpha)

        if lstm_pred.size and lstm_pred.shape == tab_preds.blended_tabular.shape:
            hybrid_pred = blend_with_lstm(tab_preds.blended_tabular, lstm_pred, weight_tabular=weight_tabular)
        else:
            hybrid_pred = tab_preds.blended_tabular

        metadata["target"] = y_true
        metadata["rf_pred"] = tab_preds.rf
        metadata["xgb_pred"] = tab_preds.xgb
        metadata["tabular_ensemble"] = tab_preds.blended_tabular
        metadata["hybrid_pred"] = hybrid_pred
        metadata["lstm_pred"] = lstm_pred if lstm_pred.size else np.nan
        metadata["weight_tabular"] = weight_tabular
        metadata["weight_lstm"] = 1 - weight_tabular
        if target_type == "classification":
            applied_threshold = threshold if threshold is not None else 0.5
            metadata["hybrid_class"] = (hybrid_pred >= applied_threshold).astype(int)
        else:
            metadata["residual"] = metadata["target"] - metadata["hybrid_pred"]

        predictions_frames.append(metadata)

        model_metrics["rf"] = compute_metrics(y_true, tab_preds.rf, target_type, threshold)
        model_metrics["xgb"] = compute_metrics(y_true, tab_preds.xgb, target_type, threshold)
        model_metrics["tabular_ensemble"] = compute_metrics(y_true, tab_preds.blended_tabular, target_type, threshold)
        model_metrics["hybrid_ensemble"] = compute_metrics(y_true, hybrid_pred, target_type, threshold)
        if lstm_pred.size:
            model_metrics["lstm"] = compute_metrics(y_true, lstm_pred, target_type, threshold)

        metrics_summary[split_name] = model_metrics

    predictions_df = pd.concat(predictions_frames, ignore_index=True) if predictions_frames else pd.DataFrame()

    artifacts_root = output_dir or config.get("paths", {}).get("artifacts_dir", "models")
    run_name_resolved = run_name or config.get("experiment", {}).get("name", "run")
    run_dir = create_run_directory(artifacts_root, run_name_resolved)
    logger.info("Saving artifacts to %s", run_dir)

    save_json(
        {
            "metrics": metrics_summary,
            "config": config,
            "classification_threshold": threshold,
        },
        run_dir / "metrics.json",
    )

    if not predictions_df.empty:
        save_dataframe(predictions_df, run_dir / "predictions.csv")

    history_path = run_dir / "lstm_history.json"
    save_json({"history": lstm_history}, history_path)

    config_path = run_dir / "config_used.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    if lstm_model is not None:
        torch.save(
            {
                "state_dict": lstm_model.state_dict(),
                "model_kwargs": {
                    "input_size": sequences["train"].features.shape[-1],
                    "hidden_size": lstm_cfg.get("hidden_size", 128),
                    "num_layers": lstm_cfg.get("num_layers", 2),
                    "dropout": lstm_cfg.get("dropout", 0.1),
                    "bidirectional": lstm_cfg.get("bidirectional", False),
                },
                "sequence_length": config.get("sequence", {}).get("length", 24),
                "feature_columns": pipeline.sequence_feature_columns,
            },
            run_dir / "lstm.pt",
        )

    dump(pipeline.scaler, run_dir / "sequence_scaler.joblib")
    dump(ensemble.rf_model, run_dir / "random_forest.joblib")
    dump(ensemble.xgb_model, run_dir / "xgboost.joblib")

    save_json(
        {
            "tabular_features": pipeline.feature_columns,
            "sequence_features": pipeline.sequence_feature_columns,
        },
        run_dir / "feature_columns.json",
    )

    try:
        if target_type == "regression" and not predictions_df.empty:
            test_df = predictions_df[predictions_df["split"] == "test"]
            if not test_df.empty:
                fig_path = run_dir / "actual_vs_predicted.png"
                plot_actual_vs_predicted(
                    test_df["target"].to_numpy(),
                    test_df["hybrid_pred"].to_numpy(),
                    save_path=fig_path,
                    show=False,
                    title="Hybrid Ensemble: Actual vs Predicted",
                )
                resid_path = run_dir / "residuals.png"
                plot_residuals(
                    test_df["target"].to_numpy(),
                    test_df["hybrid_pred"].to_numpy(),
                    save_path=resid_path,
                    show=False,
                )
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        logger.warning("Plot generation failed: %s", exc)

    latest_dir = Path(artifacts_root) / "latest"
    update_latest_artifacts(run_dir, latest_dir)

    logger.info("Training complete. Artifacts saved to %s", run_dir)
    return run_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()
    config = load_yaml_config(args.config)

    if args.quick_run or config.get("training", {}).get("quick_run"):
        logger.info("Applying quick-run overrides.")
        config = apply_quick_run_overrides(config)

    mode = args.mode or config.get("experiment", {}).get("mode", "demo")
    if mode == "real":
        cache_dir = config.get("data", {}).get("real", {}).get("cache_dir", "data/raw/real")
        logger.info(
            "Running in REAL mode. NOAA Storm Events and GHCN files will be cached under %s.",
            cache_dir,
        )
    train_and_evaluate(config, mode=mode, output_dir=args.output_dir, run_name=args.run_name)


if __name__ == "__main__":
    main()

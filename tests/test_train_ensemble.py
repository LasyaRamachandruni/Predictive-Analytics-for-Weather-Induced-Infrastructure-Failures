from pathlib import Path

from src.models.train_ensemble import apply_quick_run_overrides, train_and_evaluate
from src.utils.io import load_yaml_config


def test_train_and_evaluate_smoke(tmp_path):
    config = load_yaml_config("configs/default.yaml")
    config = apply_quick_run_overrides(config)

    config["data"]["demo"]["periods"] = 48
    config["training"]["lstm"]["epochs"] = 2
    config["training"]["tabular"]["rf"]["n_estimators"] = 20
    config["training"]["tabular"]["xgb"]["n_estimators"] = 20

    run_dir = train_and_evaluate(config, mode="demo", output_dir=str(tmp_path), run_name="pytest")
    metrics_path = Path(run_dir) / "metrics.json"
    assert metrics_path.exists()

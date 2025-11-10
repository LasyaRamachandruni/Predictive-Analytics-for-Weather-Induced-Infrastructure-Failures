import pytest

from src.data.data_pipeline import PipelineArtifacts, run_data_pipeline
from src.models.train_ensemble import apply_quick_run_overrides
from src.utils.io import load_yaml_config


@pytest.fixture(scope="module")
def quick_config():
    config = load_yaml_config("configs/default.yaml")
    return apply_quick_run_overrides(config)


def test_run_data_pipeline_demo(quick_config):
    artifacts = run_data_pipeline(quick_config, mode="demo")
    assert isinstance(artifacts, PipelineArtifacts)
    seq_len = quick_config["sequence"]["length"]
    assert artifacts.sequences["train"].features.shape[1] == seq_len
    assert artifacts.tabular["train"].features.shape[0] > 0


def test_sequence_and_tabular_alignment(quick_config):
    artifacts = run_data_pipeline(quick_config, mode="demo")
    for split in ("train", "val", "test"):
        seq_meta = artifacts.sequences[split].metadata
        tab_meta = artifacts.tabular[split].metadata
        assert seq_meta.shape[0] == tab_meta.shape[0]

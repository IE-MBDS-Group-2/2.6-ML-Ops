"""Tests for src/tracking.py: W&B runs, metrics, artifacts and model download, all offline with fakes."""

import os
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src import tracking


def _fake_wandb(monkeypatch) -> MagicMock:
    """Swap the wandb module inside tracking.py for a mock that records every call made to it."""
    fake = MagicMock()
    monkeypatch.setattr(tracking, "wandb", fake)
    return fake


def _cfg(enabled: bool = True) -> dict:
    """A small config with only the tracking section, shaped like config.yaml (test values, not the team's)."""
    return {
        "tracking": {
            "enabled": enabled,
            "project": "test-project",
            "name": "test-run",
            "group": "test-group",
            "tags": ["test"],
            "dataset_artifact": "test-prices",
            "model_artifact": "test-model",
            "log_predictions_table": True,
        }
    }


def _env(mode: str) -> dict[str, str]:
    """The values config.load_env returns, with the tracking mode chosen by the test and no real key."""
    return {
        "WANDB_API_KEY": "",
        "WANDB_ENTITY": "test-team",
        "WANDB_MODE": mode,
        "MODEL_SOURCE": "local",
        "WANDB_MODEL_ALIAS": "prod",
        "APP_VERSION": "test",
    }


def test_unit_every_function_does_nothing_with_run_none(monkeypatch, tmp_path):
    fake_wandb = _fake_wandb(monkeypatch)
    results = [
        tracking.log_metrics(None, {"accuracy": 0.5}, prefix="val/"),
        tracking.log_table(None, "predictions", pd.DataFrame({"action": [0.0, 1.0]})),
        tracking.log_dataset(None, tmp_path / "clean.csv", "etf-prices"),
        tracking.log_model(None, tmp_path / "model.joblib", "rebalance-model", {"gate_met": True}),
        tracking.finish_run(None),
        tracking.finish_run(None, failed=True),
    ]
    assert results == [None] * 6
    assert fake_wandb.mock_calls == []


def test_unit_start_run_returns_none_when_disabled(monkeypatch):
    fake_wandb = _fake_wandb(monkeypatch)
    assert tracking.start_run(_cfg(), _env("disabled")) is None
    assert tracking.start_run(_cfg(enabled=False), _env("offline")) is None
    fake_wandb.init.assert_not_called()


def test_unit_start_run_passes_project_entity_and_config(monkeypatch):
    fake_wandb = _fake_wandb(monkeypatch)
    cfg = _cfg()
    run = tracking.start_run(cfg, _env("offline"), job_type="train")
    fake_wandb.init.assert_called_once_with(
        project="test-project",
        entity="test-team",
        name="test-run",
        group="test-group",
        tags=["test"],
        job_type="train",
        config=cfg,
        mode="offline",
    )
    assert run is fake_wandb.init.return_value


def test_unit_start_run_logs_in_with_the_key_from_env(monkeypatch):
    fake_wandb = _fake_wandb(monkeypatch)
    placeholder = "not-a-real-value"
    env = {**_env("offline"), "WANDB_API_KEY": placeholder}
    tracking.start_run(_cfg(), env)
    fake_wandb.login.assert_called_once_with(key=placeholder)


def test_unit_log_metrics_adds_the_prefix():
    run = MagicMock()
    tracking.log_metrics(run, {"accuracy": 0.5}, prefix="val/")
    run.log.assert_called_once_with({"val/accuracy": 0.5})
    run.reset_mock()
    tracking.log_metrics(run, {"accuracy": 0.5})
    run.log.assert_called_once_with({"accuracy": 0.5})


def test_unit_finish_run_marks_failure():
    run = MagicMock()
    tracking.finish_run(run, failed=True)
    run.finish.assert_called_once_with(exit_code=1)
    run.reset_mock()
    tracking.finish_run(run, failed=False)
    run.finish.assert_called_once_with(exit_code=0)


def test_unit_log_model_builds_a_model_artifact(monkeypatch, tmp_path):
    fake_wandb = _fake_wandb(monkeypatch)
    run = MagicMock()
    model_file = tmp_path / "model.joblib"
    model_file.write_bytes(b"fake model")
    metadata = {"gate_met": True, "max_drawdown_reduction": 0.25}
    tracking.log_model(run, model_file, "test-model", metadata)
    fake_wandb.Artifact.assert_called_once_with("test-model", type="model", metadata=metadata)
    artifact = fake_wandb.Artifact.return_value
    artifact.add_file.assert_called_once_with(str(model_file))
    run.log_artifact.assert_called_once_with(artifact, aliases=["latest"])


def test_unit_download_model_returns_the_joblib_path(monkeypatch, tmp_path):
    fake_wandb = _fake_wandb(monkeypatch)
    (tmp_path / "model.joblib").write_bytes(b"fake model")
    api = fake_wandb.Api.return_value
    api.artifact.return_value.download.return_value = str(tmp_path)
    path = tracking.download_model(_env("online"), "test-project", "test-model", "prod")
    api.artifact.assert_called_once_with("test-team/test-project/test-model:prod")
    assert path.suffix == ".joblib"
    assert path.exists()


def test_unit_download_model_raises_on_missing_alias(monkeypatch):
    fake_wandb = _fake_wandb(monkeypatch)
    missing = "Artifact 'test-team/test-project/test-model:prod' not found in response"
    fake_wandb.Api.return_value.artifact.side_effect = ValueError(missing)
    with pytest.raises(FileNotFoundError, match=r"test-team/test-project/test-model.*prod"):
        tracking.download_model(_env("online"), "test-project", "test-model", "prod")


def test_unit_download_model_raises_when_no_joblib_was_downloaded(monkeypatch, tmp_path):
    fake_wandb = _fake_wandb(monkeypatch)
    fake_wandb.Api.return_value.artifact.return_value.download.return_value = str(tmp_path)
    with pytest.raises(FileNotFoundError, match=r"\.joblib"):
        tracking.download_model(_env("online"), "test-project", "test-model", "prod")


def test_unit_log_table_sends_a_wandb_table(monkeypatch):
    fake_wandb = _fake_wandb(monkeypatch)
    run = MagicMock()
    table = pd.DataFrame({"action": [0.0, 1.0]})
    tracking.log_table(run, "predictions", table)
    assert fake_wandb.Table.call_args.kwargs["dataframe"] is table
    run.log.assert_called_once_with({"predictions": fake_wandb.Table.return_value})


def test_unit_log_dataset_builds_a_dataset_artifact(monkeypatch, tmp_path):
    fake_wandb = _fake_wandb(monkeypatch)
    run = MagicMock()
    data_file = tmp_path / "clean.csv"
    data_file.write_text("date,ticker,close,volume\n", encoding="utf-8")
    tracking.log_dataset(run, data_file, "test-prices")
    fake_wandb.Artifact.assert_called_once_with("test-prices", type="dataset")
    artifact = fake_wandb.Artifact.return_value
    artifact.add_file.assert_called_once_with(str(data_file))
    run.log_artifact.assert_called_once_with(artifact)

"""Tests for scripts/fetch_prod_model.py with W&B mocked: prod found, no prod, and API errors. Never the network."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import wandb

from scripts import fetch_prod_model

NO_PROD = ValueError("artifact membership 'rebalance-model:prod' not found in 'team/etf-rebalancing'")


@pytest.fixture
def environment(monkeypatch, tmp_path):
    """W&B entity set, a $GITHUB_OUTPUT file to read back; returns that file."""
    github_output = tmp_path / "github_output.txt"
    monkeypatch.setenv("WANDB_ENTITY", "team")
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))
    return github_output


def api_with(version: str = "v7", files: tuple[str, ...] = ("model.joblib",), error: Exception | None = None):
    """A fake wandb.Api whose artifact() returns a version holding files, or raises error."""
    def download(root: str) -> str:
        for name in files:
            (Path(root) / name).write_bytes(b"model bytes")
        return root

    api = MagicMock()
    if error is not None:
        api.artifact.side_effect = error
    else:
        api.artifact.return_value = MagicMock(version=version, download=download)
    return api


def test_unit_prod_found_downloads_the_model_and_reports_the_concrete_version(environment, tmp_path, capsys):
    output = tmp_path / "models" / "model.joblib"
    api = api_with("v7")
    assert fetch_prod_model.main(["--output", str(output)], api=api) == 0
    assert output.read_bytes() == b"model bytes"
    api.artifact.assert_called_once_with("team/etf-rebalancing/rebalance-model:prod", type="model")
    assert "MODEL_VERSION=rebalance-model:v7" in capsys.readouterr().out
    assert environment.read_text(encoding="utf-8") == "MODEL_VERSION=rebalance-model:v7\n"


def test_unit_no_prod_exits_0_without_a_model_so_the_image_serves_the_rule(environment, tmp_path, caplog):
    output = tmp_path / "models" / "model.joblib"
    assert fetch_prod_model.main(["--output", str(output)], api=api_with(error=NO_PROD)) == 0
    assert not output.exists()
    assert environment.read_text(encoding="utf-8") == "MODEL_VERSION=\n"
    assert "serves today's rule" in caplog.text


@pytest.mark.parametrize("error", [
    wandb.errors.CommError("network unreachable"),
    wandb.errors.AuthenticationError("bad key"),
    ValueError("project 'etf-rebalancing' not found under entity 'tem'"),
])
def test_unit_an_api_error_fails_the_release(environment, tmp_path, error, caplog):
    output = tmp_path / "models" / "model.joblib"
    assert fetch_prod_model.main(["--output", str(output)], api=api_with(error=error)) == 1
    assert not output.exists()
    assert not environment.exists()
    assert "Could not fetch" in caplog.text


def test_unit_a_version_without_a_joblib_fails(environment, tmp_path):
    output = tmp_path / "model.joblib"
    assert fetch_prod_model.main(["--output", str(output)], api=api_with(files=("notes.txt",))) == 1
    assert not output.exists()


def test_unit_missing_entity_or_key_fails_before_calling_wandb(monkeypatch, tmp_path):
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    api = api_with()
    assert fetch_prod_model.main(["--output", str(tmp_path / "m.joblib")], api=api) == 1
    api.artifact.assert_not_called()
    monkeypatch.setenv("WANDB_ENTITY", "team")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    assert fetch_prod_model.main(["--output", str(tmp_path / "m.joblib")]) == 1


def test_unit_outside_github_actions_nothing_is_written(monkeypatch, tmp_path):
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    fetch_prod_model.write_github_output("rebalance-model:v7", None)
    assert not list(tmp_path.iterdir())


def test_unit_default_output_is_paths_model_from_config():
    _, name, output = fetch_prod_model._settings(fetch_prod_model.ROOT / "config.yaml")
    assert name == "rebalance-model"
    assert output == fetch_prod_model.ROOT / "models" / "model.joblib"

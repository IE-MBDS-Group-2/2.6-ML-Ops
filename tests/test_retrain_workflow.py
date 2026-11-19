"""Tests for .github/workflows/retrain.yml: the security shape of the retraining workflow.

The workflow runs on a self-hosted runner with the W&B key, so it must only start from a manual run or the
schedule, read-only, in the "retrain" environment, on the runner labelled "retrain".
"""
from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "retrain.yml"


def _load() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def test_triggers_are_only_manual_and_schedule() -> None:
    # PyYAML reads the bare key `on` as the boolean True.
    assert set(_load()[True]) == {"workflow_dispatch", "schedule"}


def test_job_runs_on_the_retrain_runner_in_the_retrain_environment() -> None:
    job = _load()["jobs"]["retrain"]
    assert {"self-hosted", "retrain"} <= set(job["runs-on"])
    assert job["environment"] == "retrain"
    assert job["timeout-minutes"] == 120


def test_permissions_are_read_only() -> None:
    assert _load()["permissions"] == {"contents": "read"}


def test_one_run_at_a_time_and_never_cancelled() -> None:
    assert _load()["concurrency"] == {"group": "retrain", "cancel-in-progress": False}


def test_checkout_keeps_no_credentials() -> None:
    checkout = _load()["jobs"]["retrain"]["steps"][0]
    assert checkout["uses"].startswith("actions/checkout@")
    assert checkout["with"]["persist-credentials"] is False


def test_the_key_cannot_reach_the_log() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "set -x" not in text
    assert "printenv" not in text
    assert "echo $WANDB" not in text
    assert _load()["jobs"]["retrain"]["env"]["WANDB_API_KEY"] == "${{ secrets.WANDB_API_KEY }}"


def test_checksum_matches_the_readme() -> None:
    readme = (WORKFLOW.parents[2] / "README.md").read_text(encoding="utf-8")
    assert _load()["jobs"]["retrain"]["env"]["PRICES_SHA256"] in readme

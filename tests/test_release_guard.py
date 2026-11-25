"""The release guard: a workflow_call that fails unless the tag is on main, and deploy.yml runs it first."""
from pathlib import Path

import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def _load(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def test_unit_guard_is_a_workflow_call_taking_the_tag():
    triggers = _load("release-guard.yml")[True]      # PyYAML reads the key `on` as True
    assert set(triggers) == {"workflow_call"}
    assert triggers["workflow_call"]["inputs"]["tag"]["required"] is True


def test_unit_guard_checks_history_against_main_with_the_tag_through_env():
    steps = _load("release-guard.yml")["jobs"]["on-main"]["steps"]
    assert any(step.get("with", {}).get("fetch-depth") == 0 for step in steps)
    check = next(step for step in steps if "merge-base" in step.get("run", ""))
    assert "merge-base --is-ancestor" in check["run"] and "origin/main" in check["run"]
    assert check["env"]["TAG"] == "${{ inputs.tag }}"
    assert "${{ inputs" not in check["run"]          # never interpolate the input into the script


def test_unit_deploy_runs_the_guard_before_ci_and_leaves_rollback_alone():
    jobs = _load("deploy.yml")["jobs"]
    assert jobs["guard"]["uses"] == "./.github/workflows/release-guard.yml"
    assert jobs["guard"]["with"]["tag"] == "${{ github.ref_name }}"
    assert "guard" in jobs["ci"]["needs"]
    assert "rollback" in jobs["rollback"]["if"]
    assert "guard" not in jobs["rollback"]["needs"]

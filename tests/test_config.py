"""Tests for src/config.py: loading and checking config.yaml, resolving paths, reading the environment."""

import copy
from pathlib import Path

import pytest
import yaml

from src import config


def _valid() -> dict:
    """The repo's own config.yaml: the one config that is known to be valid."""
    return yaml.safe_load(config.CONFIG_PATH.read_text(encoding="utf-8"))


def _write(path: Path, cfg: object) -> Path:
    """Write a YAML config file for a test."""
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def _load_changed(tmp_path: Path, section: str, key: str, value: object) -> dict:
    """Load a copy of the valid config with one setting changed."""
    cfg = copy.deepcopy(_valid())
    cfg[section][key] = value
    return config.load_config(_write(tmp_path / "config.yaml", cfg))


def test_unit_load_config_reads_the_repo_config():
    cfg = config.load_config()
    assert set(config.REQUIRED_SECTIONS) <= set(cfg)
    assert cfg["target_weights"]["method"] == "policy"
    assert sum(cfg["target_weights"]["policy"].values()) == pytest.approx(1.0)


def test_unit_load_config_raises_on_missing_file_and_non_mapping(tmp_path):
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        config.load_config(tmp_path / "absent.yaml")
    with pytest.raises(ValueError, match="YAML mapping"):
        config.load_config(_write(tmp_path / "list.yaml", ["not", "a", "mapping"]))


def test_unit_load_config_names_every_missing_section(tmp_path):
    cfg = {name: value for name, value in _valid().items() if name not in ("costs", "rebalancing")}
    with pytest.raises(KeyError, match="costs.*rebalancing|rebalancing.*costs"):
        config.load_config(_write(tmp_path / "partial.yaml", cfg))


@pytest.mark.parametrize(("section", "key"), sorted(config.ALLOWED_VALUES))
def test_unit_load_config_rejects_a_value_outside_the_allowed_list(tmp_path, section, key):
    # Every listed-value setting: a typo stops the run and the message lists what is allowed.
    allowed = config.ALLOWED_VALUES[(section, key)]
    with pytest.raises(ValueError, match=f"{section}.{key} is 'typo'. Allowed: {allowed[0]}"):
        _load_changed(tmp_path, section, key, "typo")


def test_unit_load_config_names_a_missing_listed_value_key(tmp_path):
    cfg = copy.deepcopy(_valid())
    del cfg["rebalancing"]["method"]
    with pytest.raises(KeyError, match="rebalancing.method"):
        config.load_config(_write(tmp_path / "config.yaml", cfg))


def test_unit_load_config_checks_the_policy_weights(tmp_path):
    with pytest.raises(KeyError, match="target_weights.policy"):
        _load_changed(tmp_path, "target_weights", "policy", {})
    with pytest.raises(ValueError, match="covers .* but data.tickers is"):
        _load_changed(tmp_path, "target_weights", "policy", {"SPY": 0.5, "EFA": 0.5})
    uneven = {"SPY": 0.50, "EFA": 0.20, "IEF": 0.30, "GLD": 0.05, "BIL": 0.05}
    with pytest.raises(ValueError, match="sums to 1.100000"):
        _load_changed(tmp_path, "target_weights", "policy", uneven)


@pytest.mark.parametrize("invalid", ["missing-ticker", "bad-sum", "nonfinite", "missing-method"])
def test_unit_load_config_rejects_broken_calculated_weights(tmp_path, invalid):
    cfg = _valid()
    policy = cfg["target_weights"]["policy"]
    calculated = {method: dict(policy) for method in config.CALCULATED_TARGETS}
    weights = calculated["min_variance"]
    if invalid == "missing-ticker":
        del weights["SPY"]
    elif invalid == "bad-sum":
        weights["SPY"] = 0.5
    elif invalid == "nonfinite":
        weights["SPY"] = float("nan")
    else:
        del calculated["max_sharpe"]
    with pytest.raises(ValueError, match=r"target_weights\.calculated"):
        _load_changed(tmp_path, "target_weights", "calculated", calculated)


@pytest.mark.parametrize("band", [0, -1, "five", None, True])
def test_unit_load_config_rejects_a_tolerance_band_that_is_not_a_positive_number(tmp_path, band):
    with pytest.raises(ValueError, match="rebalancing.tolerance_band"):
        _load_changed(tmp_path, "rebalancing", "tolerance_band", band)


@pytest.mark.parametrize("seeds", [[], None, 3, "0,1", [0, 1.5], [True], [0, None]])
def test_unit_load_config_rejects_rl_seeds_that_are_not_a_non_empty_list_of_ints(tmp_path, seeds):
    with pytest.raises(ValueError, match="rl.seeds"):
        _load_changed(tmp_path, "rl", "seeds", seeds)


def test_unit_load_config_accepts_rl_seeds_as_a_list_of_ints(tmp_path):
    assert config.load_config(_write(tmp_path / "config.yaml", _valid()))["rl"]["seeds"] == [0, 1, 2, 3, 4]


def test_unit_load_config_accepts_every_review_and_calendar_pair(tmp_path):
    # The allowed lists guarantee the calendar is never more frequent than the review.
    for review in config.ALLOWED_VALUES[("rebalancing", "review")]:
        for calendar in config.ALLOWED_VALUES[("rebalancing", "calendar")]:
            cfg = copy.deepcopy(_valid())
            cfg["rebalancing"].update({"review": review, "calendar": calendar})
            assert config.load_config(_write(tmp_path / "config.yaml", cfg))["rebalancing"]["review"] == review


def test_unit_resolve_paths_puts_every_path_under_the_root(tmp_path):
    paths = config.resolve_paths({"paths": {"raw_data": "data/raw/prices.csv", "model": "models/model.joblib"}},
                                 tmp_path)
    assert paths == {"raw_data": tmp_path / "data/raw/prices.csv", "model": tmp_path / "models/model.joblib"}


def test_unit_load_env_uses_safe_defaults_without_a_file(tmp_path, monkeypatch):
    for name in config.ENV_DEFAULTS:
        monkeypatch.delenv(name, raising=False)
    assert config.load_env(tmp_path / "absent.env") == config.ENV_DEFAULTS


def test_unit_load_env_reads_the_file_and_the_process_environment_wins(tmp_path, monkeypatch):
    for name in config.ENV_DEFAULTS:
        monkeypatch.delenv(name, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("WANDB_MODE=offline\nWANDB_ENTITY=file-team\n", encoding="utf-8")
    monkeypatch.setenv("WANDB_ENTITY", "process-team")
    env = config.load_env(env_file)
    assert env["WANDB_MODE"] == "offline"
    assert env["WANDB_ENTITY"] == "process-team"
    assert env["WANDB_MODEL_ALIAS"] == "prod"


@pytest.mark.parametrize(("name", "value"), [("WANDB_MODE", "sometimes"), ("MODEL_SOURCE", "cloud")])
def test_unit_load_env_rejects_a_value_outside_the_allowed_list(tmp_path, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=f"{name} is '{value}'. Allowed:"):
        config.load_env(tmp_path / "absent.env")


def test_unit_config_rejects_an_asset_cap_that_cannot_invest_fully(tmp_path):
    """The same feasibility bound applies at config load and in the optimizer."""
    cfg = _valid()
    cfg["benchmarks"]["max_weight"] = 0.2
    with pytest.raises(ValueError, match=r"benchmarks.max_weight.*0.25"):
        config.load_config(_write(tmp_path / "config.yaml", cfg))

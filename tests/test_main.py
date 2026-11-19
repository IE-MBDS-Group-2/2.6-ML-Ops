"""Tests for src/main.py: both splits, the pipeline and the entry point."""

import ast
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src import config, evaluate, main, rl_agent, tracking, utils

TICKERS = ["SPY", "EFA", "IEF", "GLD", "BIL"]


def _write_config(path: Path, cfg: dict) -> Path:
    """Write a YAML config file for a test."""
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def _make_xy(n: int = 100) -> tuple[pd.DataFrame, pd.Series]:
    """n rows with a balanced binary label."""
    X = pd.DataFrame({"a": range(n), "b": range(n, 2 * n)})
    y = pd.Series([0, 1] * (n // 2))
    return X, y


def _decision_table() -> pd.DataFrame:
    """Weekly rows from 2018 to 2021."""
    dates = pd.date_range("2018-01-05", "2021-12-31", freq="W-FRI")
    return pd.DataFrame({"x": range(len(dates))}, index=dates)


def test_unit_split_data_sizes_and_no_overlap():
    """100 rows at 0.15 / 0.15 of the whole dataset give 70 / 15 / 15 with no shared rows."""
    X, y = _make_xy()
    X_train, X_val, X_test, y_train, y_val, y_test = main.split_data(
        X, y, 0.15, 0.15, 42, stratify=False)
    assert (len(X_train), len(X_val), len(X_test)) == (70, 15, 15)
    assert len(y_train) + len(y_val) + len(y_test) == 100
    assert not set(X_train.index) & set(X_val.index)
    assert not set(X_train.index) & set(X_test.index)
    assert not set(X_val.index) & set(X_test.index)


def test_unit_split_data_stratify_keeps_class_share():
    """With stratify, every split keeps the 50/50 class share."""
    X, y = _make_xy()
    _, _, _, y_train, y_val, y_test = main.split_data(X, y, 0.2, 0.2, 0, stratify=True)
    for part in (y_train, y_val, y_test):
        assert part.mean() == pytest.approx(0.5)


@pytest.mark.parametrize("test_size, val_size", [(0.0, 0.2), (0.2, 1.0), (0.5, 0.5)])
def test_unit_split_data_bad_sizes_raise(test_size, val_size):
    """Sizes outside (0, 1), or summing to 1 or more, are rejected."""
    X, y = _make_xy()
    with pytest.raises(ValueError, match="sum to less than 1"):
        main.split_data(X, y, test_size, val_size, 0, stratify=False)


def test_unit_split_data_mismatched_lengths_raise():
    """X and y of different lengths are rejected."""
    X, y = _make_xy()
    with pytest.raises(ValueError, match="same number of rows"):
        main.split_data(X, y.iloc[:-1], 0.2, 0.2, 0, stratify=False)


def test_unit_split_by_date_boundaries_inclusive():
    """train_end and val_end are inclusive; every row lands in exactly one split."""
    table = _decision_table()
    train, val, test = main._split_by_date(table, "2019-12-27", "2020-12-25")
    assert train.index.max() == pd.Timestamp("2019-12-27")
    assert val.index.min() == pd.Timestamp("2020-01-03")
    assert val.index.max() == pd.Timestamp("2020-12-25")
    assert test.index.min() > pd.Timestamp("2020-12-25")
    assert len(train) + len(val) + len(test) == len(table)


def test_unit_split_by_date_requires_dates():
    """A table without a DatetimeIndex is rejected."""
    with pytest.raises(ValueError, match="DatetimeIndex"):
        main._split_by_date(pd.DataFrame({"x": [1, 2]}), "2019-12-31", "2021-12-31")


def test_unit_split_by_date_order_and_empty_raise():
    """val_end must follow train_end, and no split may be empty."""
    table = _decision_table()
    with pytest.raises(ValueError, match="must be after"):
        main._split_by_date(table, "2020-12-31", "2019-12-31")
    with pytest.raises(ValueError, match="test split is empty"):
        main._split_by_date(table, "2019-12-31", "2030-12-31")


def test_unit_main_exits_1_on_missing_config(tmp_path, monkeypatch):
    """A missing config stops the run with exit code 1."""
    monkeypatch.setattr(main, "CONFIG_PATH", tmp_path / "absent.yaml")
    with pytest.raises(SystemExit) as stopped:
        main.main()
    assert stopped.value.code == 1


def _synthetic_prices(days: pd.DatetimeIndex) -> pd.DataFrame:
    """Random-walk prices for the 5 config tickers, in the long raw format."""
    rng = np.random.default_rng(7)
    frames = []
    for ticker in TICKERS:
        close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(days)))
        frames.append(pd.DataFrame(
            {"date": days, "ticker": ticker, "close": close, "volume": 1_000_000}))
    return pd.concat(frames, ignore_index=True)


def _pipeline_setup(tmp_path: Path, monkeypatch, prices: pd.DataFrame | None = None) -> dict:
    """Synthetic data under tmp_path, and a config that points the pipeline at it.

    About 4 years of weekdays: train to 2021, validation 2022, test 2023. The inference file is
    the last 300 trading days, as scripts/fetch_prices.py would write it.
    """
    days = pd.bdate_range("2020-01-01", "2023-12-29")
    prices = _synthetic_prices(days) if prices is None else prices
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "inference").mkdir(parents=True)
    prices.to_csv(tmp_path / "data" / "raw" / "prices.csv", index=False)
    recent = prices[prices["date"].isin(days[-300:])]
    recent.to_csv(tmp_path / "data" / "inference" / "prices_latest.csv", index=False)
    cfg = config.load_config(main.CONFIG_PATH)
    cfg["target_weights"].pop("calculated", None)
    cfg["target_weights"].pop("fit", None)
    cfg["data"]["end"] = "2023-12-29"
    cfg["portfolio"]["first_decision_date"] = "2020-04-10"
    cfg["split"].update(train_end="2021-12-31", val_end="2022-12-31", test_end="2023-12-29")
    monkeypatch.setattr(main, "CONFIG_PATH", _write_config(tmp_path / "config.yaml", cfg))
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    return cfg


def _force_gate(monkeypatch, met: bool) -> None:
    """Make the quality gate pass or fail whatever the synthetic model does, to test both branches."""
    monkeypatch.setattr(evaluate, "check_gate", lambda value, comparison, threshold, tolerance=1e-9: met)


def _record(monkeypatch, name: str) -> list:
    """Replace one tracking function with a recorder and return the list of its calls."""
    calls = []
    monkeypatch.setattr(tracking, name, lambda *args, **kwargs: calls.append((args, kwargs)))
    return calls


def test_integration_pipeline_writes_every_output(tmp_path, monkeypatch):
    """main() on synthetic prices with a passing gate writes every output, and the saved model predicts."""
    cfg = _pipeline_setup(tmp_path, monkeypatch)
    _force_gate(monkeypatch, True)
    main.main()
    for name in ("processed_data", "model", "predictions", "metrics", "challengers", "robustness", "benchmarks"):
        assert (tmp_path / cfg["paths"][name]).exists()
    model = utils.load_model(tmp_path / cfg["paths"]["model"])
    metrics = json.loads((tmp_path / cfg["paths"]["metrics"]).read_text(encoding="utf-8"))
    assert {"accuracy", "f1_macro", "agent", "baseline", "gate", "guards", "challenger"} <= set(metrics)
    assert metrics["split"] == "validation"
    assert set(metrics["guards"]) == {"return_shortfall", "cost_ratio", "tracking_error"}
    assert "tracking_error" in metrics["agent"]
    predictions = pd.read_csv(tmp_path / cfg["paths"]["predictions"])
    assert predictions.columns[0] == "decision_date"
    assert predictions["decision_date"].is_unique
    assert set(predictions["recommendation"]) <= {0.0, 0.5, 1.0}
    assert len(predictions) > 20
    assert list(model.classes_) == [0.0, 1.0] or list(model.classes_) == [0.0, 0.5, 1.0]
    # 5 weights + 5 drifts + 3 state numbers + 15 market features
    assert len(model.feature_names_in_) == 28
    challengers = pd.read_csv(tmp_path / cfg["paths"]["challengers"])
    assert "current" in set(challengers["rule"])
    assert metrics["challenger"]["name"] in set(challengers["rule"])
    robustness = json.loads((tmp_path / cfg["paths"]["robustness"]).read_text(encoding="utf-8"))
    assert robustness["versions_tried"] == 1
    significance = robustness["test"]["challenger_significance"]
    assert significance["status"] == "measured"
    assert set(significance["p_values"]) == set(significance["holm_adjusted"]) == set(challengers["rule"])


def _small_rl(cfg: dict, model_type: str, enabled: bool = False) -> dict:
    """Point the config at the reinforcement learning agent with settings small enough for a test."""
    cfg["model"]["type"] = model_type
    cfg["rl"].update(enabled=enabled, n_episodes=2, n_iterations=2, seeds=[0, 1, 2])
    cfg["rl"]["estimator"]["max_iter"] = 10
    return cfg


def test_integration_pipeline_runs_the_rl_agent_when_model_type_is_fitted_q(tmp_path, monkeypatch):
    """model.type fitted_q: one agent per seed, the median one goes through the gate, the audit and inference."""
    cfg = _small_rl(_pipeline_setup(tmp_path, monkeypatch), "fitted_q")
    _force_gate(monkeypatch, True)
    registered = _record(monkeypatch, "log_model")
    logged = _record(monkeypatch, "log_metrics")
    metrics = main.run(cfg)
    report = metrics["rl"]
    assert metrics["model_type"] == "fitted_q" and "accuracy" not in metrics
    assert set(report["seed_scores"]) == {"0", "1", "2"}
    assert report["median_seed"] == rl_agent.median_seed({int(k): v for k, v in report["seed_scores"].items()})
    assert report["median_score"] == report["seed_scores"][str(report["median_seed"])]
    assert metrics["max_drawdown_reduction"] == pytest.approx(report["median_score"])    # the gate saw that agent
    assert 0.0 <= report["fallback_share"] <= 1.0
    assert "rl/fallback_share" in logged[0][0][1] and "rl/seed_scores/0" in logged[0][0][1]
    assert registered[0][0][3]["model_type"] == "fitted_q"
    assert isinstance(utils.load_model(tmp_path / cfg["paths"]["model"]), rl_agent.QAgent)
    assert metrics["test"]["split"] == "test"
    predictions = pd.read_csv(tmp_path / cfg["paths"]["predictions"])
    assert set(predictions["recommendation"]) <= {0.0, 0.5, 1.0}
    assert predictions["probability"].between(1 / 3, 1.0).all()


def test_unit_rl_enabled_reports_the_agent_but_the_classifier_stays_the_candidate(tmp_path, monkeypatch):
    cfg = _small_rl(_pipeline_setup(tmp_path, monkeypatch), "classifier", enabled=True)
    _force_gate(monkeypatch, True)
    metrics = main.run(cfg)
    assert metrics["model_type"] == "classifier"
    assert "accuracy" in metrics and set(metrics["rl"]["seed_scores"]) == {"0", "1", "2"}
    assert not isinstance(utils.load_model(tmp_path / cfg["paths"]["model"]), rl_agent.QAgent)


def test_unit_classifier_run_does_not_train_the_agent_unless_asked(tmp_path, monkeypatch):
    cfg = _pipeline_setup(tmp_path, monkeypatch)

    def never(*args, **kwargs):
        raise AssertionError("the agent must not be trained when model.type is classifier and rl.enabled is false")

    monkeypatch.setattr(rl_agent, "collect_transitions", never)
    metrics = main.run(cfg)
    assert metrics["model_type"] == "classifier" and "rl" not in metrics


def test_unit_gate_met_saves_and_registers(tmp_path, monkeypatch):
    """A model that passes the gate gets the one-time test audit, is saved, and is registered once."""
    cfg = _pipeline_setup(tmp_path, monkeypatch)
    _force_gate(monkeypatch, True)
    registered = _record(monkeypatch, "log_model")
    logged = _record(monkeypatch, "log_metrics")
    tables = _record(monkeypatch, "log_table")
    cfg["tracking"]["log_predictions_table"] = False
    metrics = main.run(cfg)
    assert [call[0][1] for call in tables] == ["robustness", "benchmarks"]
    assert (tmp_path / cfg["paths"]["model"]).exists()
    assert len(registered) == 1
    assert registered[0][0][2] == cfg["tracking"]["model_artifact"]
    assert metrics["test"]["split"] == "test"
    assert set(metrics["test"]["release"]) == {"release", "reasons"}
    assert "challenger" in metrics["test"]["guards"]
    assert [call[1]["prefix"] for call in logged] == ["val/", "test/"]


def test_unit_gate_not_met_saves_nothing(tmp_path, monkeypatch):
    """Decision D8: a failed gate saves and registers nothing, and today's rule gives every recommendation."""
    cfg = _pipeline_setup(tmp_path, monkeypatch)
    _force_gate(monkeypatch, False)
    registered = _record(monkeypatch, "log_model")
    tables = _record(monkeypatch, "log_table")
    stale = tmp_path / cfg["paths"]["model"]
    stale.parent.mkdir(parents=True)
    stale.write_text("a model left over from an earlier run", encoding="utf-8")
    main.main()                                   # exit code 0: "not met" is a result, not a crash
    assert not stale.exists()
    assert registered == []
    assert [call[0][1] for call in tables] == ["robustness", "benchmarks", "predictions"]
    robustness = json.loads((tmp_path / cfg["paths"]["robustness"]).read_text(encoding="utf-8"))
    assert robustness["test"] == {"status": "sealed", "reason": "validation_gate_not_met"}
    assert [row["cost_model"] for row in robustness["validation_cost_sensitivity"]] == ["flat", "realistic"]
    metrics = json.loads((tmp_path / cfg["paths"]["metrics"]).read_text(encoding="utf-8"))
    assert metrics["gate"]["met"] is False
    assert "test" not in metrics                  # the test years stay sealed
    predictions = pd.read_csv(tmp_path / cfg["paths"]["predictions"])
    assert predictions["fallback"].all()
    assert set(predictions["recommendation"]) <= {0.0, 1.0}      # today's rule only ever says 0 or 1
    assert (predictions["recommendation"] == predictions["prediction"]).all()


def test_integration_grid_search_reports_trials(tmp_path: Path, monkeypatch) -> None:
    """Only training rows reach the grid, and its trial count reaches the robustness report."""
    cfg = _pipeline_setup(tmp_path, monkeypatch)
    _force_gate(monkeypatch, False)
    cfg["selection"].update(enabled=True, first_train_end="2020-12-31", grid={"C": [.1, 1.]})
    logs = _record(monkeypatch, "log_metrics")
    grid_search = main.selection.grid_search

    def check_training_boundary(table: pd.DataFrame, market: tuple, settings: dict, tracker: object) -> tuple:
        """Reject validation or test dates before evaluating the real parameter grid."""
        cutoff = pd.Timestamp(settings["split"]["train_end"])
        assert table.index.max() <= cutoff
        assert all(frame.index.max() <= cutoff for frame in market)
        return grid_search(table, market, settings, tracker)

    monkeypatch.setattr(main.selection, "grid_search", check_training_boundary)

    main.run(cfg)

    trials = pd.read_csv(tmp_path / cfg["paths"]["selection"])
    assert trials["C"].tolist() == [.1, 1.]
    assert len([call for call in logs if call[1]["prefix"] == "grid/"]) == len(trials)
    report = json.loads((tmp_path / cfg["paths"]["robustness"]).read_text(encoding="utf-8"))
    assert report["versions_tried"] == 1          # grid trials are scored on the train years, not on validation
    assert report["test"]["status"] == "sealed"
    assert cfg["model"]["params"] == config.load_config()["model"]["params"]
    comparison = pd.read_csv(tmp_path / cfg["paths"]["frequency_comparison"])
    assert comparison["review"].tolist() == ["daily", "weekly", "monthly"]
    assert comparison["blocks"].gt(0).all()


def test_unit_error_marks_the_run_failed(tmp_path, monkeypatch):
    """An error inside the pipeline closes the tracked run as failed and exits with code 1."""
    _pipeline_setup(tmp_path, monkeypatch)
    tracker = object()
    monkeypatch.setattr(tracking, "start_run", lambda cfg, env, job_type="train": tracker)
    finished = _record(monkeypatch, "finish_run")

    def broken_run(cfg, tracker=None):
        raise ValueError("something in the data is wrong")

    monkeypatch.setattr(main, "run", broken_run)
    with pytest.raises(SystemExit) as stopped:
        main.main()
    assert stopped.value.code == 1
    assert finished == [((tracker,), {"failed": True})]


def test_unit_a_finished_run_is_closed_as_a_success(tmp_path, monkeypatch):
    _pipeline_setup(tmp_path, monkeypatch)
    tracker = object()
    monkeypatch.setattr(tracking, "start_run", lambda cfg, env, job_type="train": tracker)
    finished = _record(monkeypatch, "finish_run")
    monkeypatch.setattr(main, "run", lambda cfg, tracker=None: {})
    main.main()
    assert finished == [((tracker,), {})]


def test_unit_main_never_imports_wandb():
    """main.py talks to the tracker only through tracking.py, so the pipeline runs without W&B."""
    tree = ast.parse(Path(main.__file__).read_text(encoding="utf-8"))
    imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                for alias in node.names}
    imported |= {node.module.split(".")[0] for node in ast.walk(tree)
                 if isinstance(node, ast.ImportFrom) and node.module}
    assert "wandb" not in imported


def test_unit_period_gives_the_validation_and_test_dates():
    cfg = {"split": {"train_end": "2019-12-31", "val_end": "2021-12-31", "test_end": "2025-12-31"}}
    assert main._period(cfg, "validation") == ("2020-01-01", "2021-12-31")
    assert main._period(cfg, "test") == ("2022-01-01", "2025-12-31")
    with pytest.raises(ValueError, match="Unknown period 'train'"):
        main._period(cfg, "train")


def test_unit_numbers_flattens_nested_metrics_and_drops_text():
    metrics = {"split": "validation", "accuracy": 0.5, "agent": {"max_drawdown": -0.2, "n": 3},
               "gate": {"met": True, "comparison": ">="}, "reasons": ["a"]}
    assert main._numbers(metrics, "val/") == {
        "val/accuracy": 0.5, "val/agent/max_drawdown": -0.2, "val/agent/n": 3.0, "val/gate/met": 1.0}


def test_integration_pipeline_logs_the_gate(tmp_path, monkeypatch):
    """The log file says whether the quality gate was met, with the metric on validation."""
    cfg = _pipeline_setup(tmp_path, monkeypatch)
    main.main()
    for handler in logging.getLogger().handlers:
        handler.flush()
    log_lines = (tmp_path / cfg["paths"]["log_file"]).read_text(encoding="utf-8").splitlines()
    gate_lines = [line for line in log_lines if "src.main" in line and "Quality gate" in line]
    assert len(gate_lines) == 1
    assert "max_drawdown_reduction" in gate_lines[0] and "validation" in gate_lines[0]


def test_unit_main_exits_1_on_missing_raw_file(tmp_path, monkeypatch):
    """No raw price file stops the run with exit code 1."""
    cfg = _pipeline_setup(tmp_path, monkeypatch)
    (tmp_path / cfg["paths"]["raw_data"]).unlink()
    with pytest.raises(SystemExit) as stopped:
        main.main()
    assert stopped.value.code == 1


def test_unit_main_exits_1_on_bad_data(tmp_path, monkeypatch):
    """A non-positive price is caught by validate, so the pipeline stops before training."""
    prices = _synthetic_prices(pd.bdate_range("2020-01-01", "2023-12-29"))
    prices.loc[10, "close"] = -1.0
    cfg = _pipeline_setup(tmp_path, monkeypatch, prices)
    _force_gate(monkeypatch, True)
    with pytest.raises(SystemExit) as stopped:
        main.main()
    assert stopped.value.code == 1
    assert not (tmp_path / cfg["paths"]["model"]).exists()


def test_unit_recommendations_need_enough_inference_days(tmp_path, monkeypatch):
    """An inference file shorter than the feature warm-up raises a clear error."""
    cfg = _pipeline_setup(tmp_path, monkeypatch)
    short = _synthetic_prices(pd.bdate_range("2023-11-01", "2023-12-29"))
    short.to_csv(tmp_path / cfg["paths"]["inference_data"], index=False)
    with pytest.raises(ValueError, match="too short"):
        main._recommendations(None, [], cfg)


def test_unit_validation_gate_helper_reports_the_threshold(caplog):
    """The gate compares 1 - |agent| / |baseline| with the config threshold, and logs it."""
    cfg = {"quality_gate": {"metric_key": "max_drawdown_reduction", "comparison": ">=",
                            "threshold": 0.20, "split": "validation"}}
    with caplog.at_level(logging.INFO):
        value, met = main._check_quality_gate({"max_drawdown": -0.16}, {"max_drawdown": -0.20}, cfg)
    assert value == pytest.approx(0.20)
    assert met
    assert "Quality gate met" in caplog.text
    value, met = main._check_quality_gate({"max_drawdown": -0.19}, {"max_drawdown": -0.20}, cfg)
    assert not met
    assert "NOT met" in caplog.text


def test_integration_repo_config_has_every_section():
    """The committed config.yaml loads with every required section."""
    cfg = config.load_config(main.CONFIG_PATH)
    assert cfg["quality_gate"]["metric_key"] == "max_drawdown_reduction"


# ---------- benchmarks, trades files and versions tried (#71) ----------

def test_unit_benchmarks_are_reported_on_the_validation_years(tmp_path, monkeypatch):
    cfg = _pipeline_setup(tmp_path, monkeypatch)
    metrics = main.run(cfg)
    report = pd.read_csv(tmp_path / cfg["paths"]["benchmarks"])
    assert set(report["method"]) == set(cfg["benchmarks"]["methods"])
    assert len(metrics["benchmarks"]) == len(cfg["benchmarks"]["methods"])


def test_unit_trades_are_written_only_when_a_positions_file_exists(tmp_path, monkeypatch):
    cfg = _pipeline_setup(tmp_path, monkeypatch)
    main.run(cfg)
    assert not (tmp_path / cfg["paths"]["trades"]).exists()        # no positions file: nothing written
    # A drifted book: SPY 20 points above its 40% target, far outside today's 5-point band.
    values = {"SPY": 600.0, "EFA": 100.0, "IEF": 250.0, "GLD": 25.0, "BIL": 25.0}
    pd.DataFrame({"ticker": list(values), "market_value": list(values.values())}).to_csv(
        tmp_path / cfg["paths"]["positions"], index=False)
    main.run(cfg)
    proposed = pd.read_csv(tmp_path / cfg["paths"]["proposed_portfolio"])
    orders = pd.read_csv(tmp_path / cfg["paths"]["trades"])
    assert list(proposed["ticker"]) == cfg["data"]["tickers"]
    assert proposed["value"].sum() == pytest.approx(sum(values.values()))
    # No model passed the gate here, so today's rule decides for this book: the drift breach means rebalance
    # fully, back to the policy weights, and the buys and sells net to zero.
    assert proposed.set_index("ticker")["weight"].to_dict() == pytest.approx(cfg["target_weights"]["policy"])
    assert orders.set_index("ticker")["side"].to_dict() == {"SPY": "sell", "EFA": "buy", "IEF": "buy", "GLD": "buy",
                                                            "BIL": "buy"}
    signed = orders["value"].where(orders["side"] == "buy", -orders["value"])
    assert signed.sum() == pytest.approx(0.0, abs=1e-6)


def test_unit_versions_tried_counts_only_versions_scored_on_validation():
    # Grid-search trials are scored on the train years (walk-forward), so they never count.
    cfg = {"model": {"type": "classifier"}, "rl": {"seeds": [0, 1, 2]}}
    assert main._versions_tried(cfg, {}) == 1
    assert main._versions_tried(cfg, {"rl": {}}) == 4                     # classifier plus the reported agent
    assert main._versions_tried({**cfg, "model": {"type": "fitted_q"}}, {"rl": {}}) == 3


@pytest.mark.parametrize("method", config.CALCULATED_TARGETS)
def test_unit_pipeline_uses_stored_target(method):
    """Changing the configured method selects its reviewed weights throughout the pipeline."""
    cfg = config.load_config()
    cfg["target_weights"]["method"] = method
    assert main._target_weights(cfg).to_dict() == cfg["target_weights"]["calculated"][method]

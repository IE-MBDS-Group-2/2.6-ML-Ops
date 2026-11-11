"""Tests for src/main.py: config loading, logging set-up, both splits and the entry point."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src import main, utils

SECTIONS = {section: {} for section in main.REQUIRED_SECTIONS}
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


def test_unit_load_config_reads_all_sections(tmp_path):
    """A config with every section loads as a dictionary."""
    cfg = main.load_config(_write_config(tmp_path / "config.yaml", SECTIONS))
    assert set(cfg) == set(main.REQUIRED_SECTIONS)


def test_unit_load_config_errors(tmp_path):
    """Missing file, a non-mapping file and a missing section each raise a clear error."""
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        main.load_config(tmp_path / "absent.yaml")
    with pytest.raises(ValueError, match="YAML mapping"):
        main.load_config(_write_config(tmp_path / "list.yaml", ["not", "a", "mapping"]))
    partial = {k: v for k, v in SECTIONS.items() if k != "costs"}
    with pytest.raises(KeyError, match="costs"):
        main.load_config(_write_config(tmp_path / "partial.yaml", partial))


def test_unit_configure_logging_writes_file(tmp_path):
    """Log lines reach the configured file, creating its folder."""
    log_file = tmp_path / "logs" / "pipeline.log"
    main.configure_logging("info", log_file)
    logging.getLogger("src.test").info("hello %s", "log")
    for handler in logging.getLogger().handlers:
        handler.flush()
    assert "hello log" in log_file.read_text(encoding="utf-8")


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
    cfg = main.load_config(main.CONFIG_PATH)
    cfg["data"]["end"] = "2023-12-29"
    cfg["portfolio"]["first_decision_date"] = "2020-04-10"
    cfg["split"].update(train_end="2021-12-31", val_end="2022-12-31", test_end="2023-12-29")
    monkeypatch.setattr(main, "CONFIG_PATH", _write_config(tmp_path / "config.yaml", cfg))
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    return cfg


def test_integration_pipeline_writes_every_output(tmp_path, monkeypatch):
    """main() on synthetic prices writes the 4 outputs, and the saved model predicts."""
    cfg = _pipeline_setup(tmp_path, monkeypatch)
    main.main()
    for name in ("processed_data", "model", "predictions", "metrics"):
        assert (tmp_path / cfg["paths"][name]).exists()
    model = utils.load_model(tmp_path / cfg["paths"]["model"])
    metrics = json.loads((tmp_path / cfg["paths"]["metrics"]).read_text(encoding="utf-8"))
    assert {"accuracy", "f1_macro", "agent", "baseline", "gate"} <= set(metrics)
    assert metrics["split"] == "validation"
    assert metrics["gate"]["met"] == (metrics["max_drawdown_reduction"] >= 0.20)
    predictions = pd.read_csv(tmp_path / cfg["paths"]["predictions"])
    assert predictions.columns[0] == "decision_date"
    assert predictions["decision_date"].is_unique
    assert set(predictions["recommendation"]) <= {0.0, 0.5, 1.0}
    assert len(predictions) > 20
    assert list(model.classes_) == [0.0, 1.0] or list(model.classes_) == [0.0, 0.5, 1.0]
    # 5 weights + 5 drifts + 3 state numbers + 15 market features
    assert len(model.feature_names_in_) == 28


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
    cfg = main.load_config(main.CONFIG_PATH)
    assert cfg["quality_gate"]["metric_key"] == "max_drawdown_reduction"

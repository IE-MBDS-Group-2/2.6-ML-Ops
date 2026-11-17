"""Tests for scripts/make_sample_prices.py: deterministic synthetic prices for CI and tests."""

import logging
import runpy
import sys

import pandas as pd
import pytest

from scripts import make_sample_prices as sample
from src import config, validate

TICKERS = ["SPY", "EFA", "IEF", "GLD", "BIL"]
COLUMNS = {
    "date_column": "date",
    "ticker_column": "ticker",
    "close_column": "close",
    "volume_column": "volume",
}


def test_unit_same_seed_gives_identical_prices():
    """The same tickers, dates and seed give the same frame; a different seed gives different closes."""
    first = sample.make_sample_prices(TICKERS, "2024-01-01", "2024-03-29", 42, COLUMNS)
    second = sample.make_sample_prices(TICKERS, "2024-01-01", "2024-03-29", 42, COLUMNS)
    other = sample.make_sample_prices(TICKERS, "2024-01-01", "2024-03-29", 43, COLUMNS)

    pd.testing.assert_frame_equal(first, second)
    assert not first[COLUMNS["close_column"]].equals(other[COLUMNS["close_column"]])


def test_unit_every_ticker_has_every_business_day():
    """Five tickers over a quarter give business days x 5 rows, with no gaps and no repeated (date, ticker)."""
    frame = sample.make_sample_prices(TICKERS, "2024-01-01", "2024-03-29", 42, COLUMNS)
    business_days = pd.bdate_range("2024-01-01", "2024-03-29")

    assert len(frame) == len(business_days) * len(TICKERS)
    assert not frame.isna().any().any()
    assert not frame.duplicated([COLUMNS["date_column"], COLUMNS["ticker_column"]]).any()
    dates_per_ticker = frame.groupby(COLUMNS["ticker_column"])[COLUMNS["date_column"]].nunique()
    assert (dates_per_ticker == len(business_days)).all()


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 2024])
def test_unit_prices_and_volumes_are_positive(seed):
    """Whatever the seed, over many years every close and every volume stays above zero."""
    frame = sample.make_sample_prices(TICKERS, "2007-05-30", "2025-12-31", seed, COLUMNS)

    assert (frame[COLUMNS["close_column"]] > 0).all()
    assert (frame[COLUMNS["volume_column"]] > 0).all()


def test_unit_columns_come_from_config():
    """Column names passed in (day, symbol, px, qty) are the only columns in the frame."""
    other_names = {
        "date_column": "day",
        "ticker_column": "symbol",
        "close_column": "px",
        "volume_column": "qty",
    }
    frame = sample.make_sample_prices(TICKERS, "2024-01-01", "2024-01-31", 42, other_names)

    assert list(frame.columns) == ["day", "symbol", "px", "qty"]


def test_integration_output_passes_validate():
    """The synthetic table passes validate_dataframe with the rules from config.yaml."""
    data = config.load_config()["data"]
    frame = sample.make_sample_prices(data["tickers"], data["start"], data["end"], 42, data["columns"])

    assert validate.validate_dataframe(
        frame,
        required_columns=data["required_columns"],
        tickers=data["tickers"],
        max_abs_daily_return=data["max_abs_daily_return"],
        price_columns=data["columns"],
    ) is True


def _use_config_in(tmp_path, monkeypatch) -> dict:
    """Load config.yaml, point the two output paths into tmp_path and make config.load_config return that copy."""
    cfg = config.load_config()
    cfg["paths"]["raw_data"] = str(tmp_path / "raw" / "prices.csv")
    cfg["paths"]["inference_data"] = str(tmp_path / "inference" / "prices_latest.csv")
    monkeypatch.setattr(config, "load_config", lambda *args, **kwargs: cfg)
    return cfg


def test_integration_main_writes_both_files(tmp_path, monkeypatch):
    """main() writes the raw file and the inference file (inference_days dates) to the configured paths."""
    cfg = _use_config_in(tmp_path, monkeypatch)

    sample.main()

    date_column = cfg["data"]["columns"]["date_column"]
    raw = pd.read_csv(cfg["paths"]["raw_data"])
    latest = pd.read_csv(cfg["paths"]["inference_data"])
    assert not raw.empty
    assert latest[date_column].nunique() == cfg["data"]["inference_days"]
    assert set(latest[date_column]) <= set(raw[date_column])


def test_integration_running_as_a_script_writes_both_files(tmp_path, monkeypatch):
    """python -m scripts.make_sample_prices (the CI call) runs main() and writes both files."""
    _use_config_in(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["make_sample_prices"])  # pytest's own arguments must not reach the script
    monkeypatch.delitem(sys.modules, "scripts.make_sample_prices")  # run it fresh, as python -m would

    runpy.run_module("scripts.make_sample_prices", run_name="__main__")

    assert (tmp_path / "raw" / "prices.csv").exists()
    assert (tmp_path / "inference" / "prices_latest.csv").exists()


def test_unit_refuses_to_overwrite_existing_raw_file(tmp_path, monkeypatch):
    """With the raw file already there and no --force, main() raises FileExistsError and writes nothing."""
    _use_config_in(tmp_path, monkeypatch)
    raw = tmp_path / "raw" / "prices.csv"
    raw.parent.mkdir()
    raw.write_text("keep me")

    with pytest.raises(FileExistsError, match=r"prices\.csv.*--force"):
        sample.main()

    assert raw.read_text() == "keep me"
    assert not (tmp_path / "inference").exists()


def test_unit_refuses_when_only_inference_file_exists(tmp_path, monkeypatch):
    """An existing inference file alone also stops main(): the raw file is not written either."""
    _use_config_in(tmp_path, monkeypatch)
    inference = tmp_path / "inference" / "prices_latest.csv"
    inference.parent.mkdir()
    inference.write_text("keep me")

    with pytest.raises(FileExistsError, match=r"prices_latest\.csv.*--force"):
        sample.main()

    assert inference.read_text() == "keep me"
    assert not (tmp_path / "raw").exists()


def test_integration_force_overwrites_both_files(tmp_path, monkeypatch):
    """With --force, main() replaces both existing files with fresh synthetic tables."""
    cfg = _use_config_in(tmp_path, monkeypatch)
    for name in ("raw", "inference"):
        (tmp_path / name).mkdir()
        (tmp_path / name / ("prices.csv" if name == "raw" else "prices_latest.csv")).write_text("old")

    sample.main(["--force"])

    raw = pd.read_csv(cfg["paths"]["raw_data"])
    latest = pd.read_csv(cfg["paths"]["inference_data"])
    assert list(raw.columns) == list(cfg["data"]["columns"].values())
    assert len(latest) == cfg["data"]["inference_days"] * len(cfg["data"]["tickers"])


def test_unit_logs_synthetic_warning(caplog):
    """Generating prices logs a WARNING that says the data is SYNTHETIC, so nobody mistakes it for the snapshot."""
    with caplog.at_level(logging.WARNING):
        sample.make_sample_prices(TICKERS, "2024-01-01", "2024-01-31", 42, COLUMNS)

    assert any(record.levelno == logging.WARNING and "SYNTHETIC" in record.getMessage()
               for record in caplog.records)


def test_unit_raises_on_end_before_start():
    """An end date before the start date stops with a ValueError that names both dates."""
    with pytest.raises(ValueError, match=r"2024-03-29.*2024-01-01"):
        sample.make_sample_prices(TICKERS, "2024-03-29", "2024-01-01", 42, COLUMNS)

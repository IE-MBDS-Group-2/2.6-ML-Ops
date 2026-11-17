"""Tests for scripts/make_sample_prices.py: deterministic synthetic prices for CI and tests."""

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


def test_integration_main_writes_both_files(tmp_path, monkeypatch):
    """main() writes the raw file and the inference file (inference_days dates) to the configured paths."""
    cfg = config.load_config()
    cfg["paths"]["raw_data"] = str(tmp_path / "raw" / "prices.csv")
    cfg["paths"]["inference_data"] = str(tmp_path / "inference" / "prices_latest.csv")
    monkeypatch.setattr(config, "load_config", lambda *args, **kwargs: cfg)

    sample.main()

    date_column = cfg["data"]["columns"]["date_column"]
    raw = pd.read_csv(cfg["paths"]["raw_data"])
    latest = pd.read_csv(cfg["paths"]["inference_data"])
    assert not raw.empty
    assert latest[date_column].nunique() == cfg["data"]["inference_days"]
    assert set(latest[date_column]) <= set(raw[date_column])


def test_unit_raises_on_end_before_start():
    """An end date before the start date stops with a ValueError that names both dates."""
    with pytest.raises(ValueError, match=r"2024-03-29.*2024-01-01"):
        sample.make_sample_prices(TICKERS, "2024-03-29", "2024-01-01", 42, COLUMNS)

"""Tests for src/validate.py: validate_dataframe, the data quality gate."""

import pandas as pd
import pytest

from src import clean_data, validate

REQUIRED = ["date", "ticker", "close", "volume"]
TICKERS = ["SPY", "EFA"]


def make_valid_long() -> pd.DataFrame:
    """Clean long table that passes every check: 3 dates x 2 tickers, small daily moves."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"] * 2),
            "ticker": ["SPY"] * 3 + ["EFA"] * 3,
            "close": [100.0, 101.0, 102.0, 70.0, 71.0, 72.0],
            "volume": [1000, 1100, 1200, 500, 550, 600],
        }
    )


def run_gate(df: pd.DataFrame, tickers: list[str] | None = None, max_move: float | None = 0.25) -> bool:
    """Call the gate with the settings main.py will pass (config: max_abs_daily_return 0.25)."""
    return validate.validate_dataframe(
        df, required_columns=REQUIRED, tickers=TICKERS if tickers is None else tickers,
        max_abs_daily_return=max_move,
    )


def test_unit_validate_returns_true_for_valid_data():
    """Good data passes every check and returns exactly True."""
    assert run_gate(make_valid_long()) is True


def test_unit_validate_skips_optional_checks_when_none():
    """With tickers and max_abs_daily_return left as None, those two checks are skipped: a 30% jump passes."""
    df = make_valid_long()
    df.loc[1, "close"] = 130.0
    assert validate.validate_dataframe(df, required_columns=REQUIRED) is True


def test_unit_validate_raises_on_missing_required_column():
    """No volume column: the message names it."""
    with pytest.raises(ValueError, match="Missing required columns.*volume"):
        run_gate(make_valid_long().drop(columns=["volume"]))


def test_unit_validate_raises_on_empty_frame():
    """Right columns but zero rows is not usable data."""
    with pytest.raises(ValueError, match="empty"):
        run_gate(make_valid_long().iloc[0:0])


def test_unit_validate_raises_on_absent_ticker():
    """Config asks for GLD but the data has none."""
    with pytest.raises(ValueError, match="GLD"):
        run_gate(make_valid_long(), tickers=["SPY", "EFA", "GLD"])


def test_unit_validate_raises_on_ticker_with_fewer_dates():
    """EFA is missing 2024-01-04 while SPY has it: the calendars do not match."""
    df = make_valid_long()
    df = df[~((df["ticker"] == "EFA") & (df["date"] == "2024-01-04"))]
    with pytest.raises(ValueError, match="fewer dates.*EFA"):
        run_gate(df)


def test_unit_validate_raises_on_non_positive_close():
    """A price of 0 is impossible for an ETF."""
    df = make_valid_long()
    df.loc[0, "close"] = 0.0
    with pytest.raises(ValueError, match="close <= 0"):
        run_gate(df, max_move=None)


def test_unit_validate_raises_on_negative_volume():
    """Traded volume cannot be negative."""
    df = make_valid_long()
    df.loc[2, "volume"] = -5
    with pytest.raises(ValueError, match="volume < 0"):
        run_gate(df)


def test_unit_validate_raises_on_extreme_daily_return():
    """SPY 100 -> 130 is a 30% one-day move, above the 25% limit: likely a bad price."""
    df = make_valid_long()
    df.loc[1, "close"] = 130.0
    with pytest.raises(ValueError, match="daily return.*SPY"):
        run_gate(df)


@pytest.mark.skip(reason="Awaiting clean_data in dev")
def test_integration_clean_output_passes_validate():
    """clean_dataframe output on messy raw data is accepted by the gate."""
    raw = make_valid_long().rename(columns=str.upper)
    raw["DATE"] = raw["DATE"].dt.strftime("%Y-%m-%d")
    assert run_gate(clean_data.clean_dataframe(raw)) is True

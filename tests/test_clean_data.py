"""Tests for src/clean_data.py: clean_dataframe and to_wide."""

import pandas as pd
import pytest

from src import clean_data, validate


def make_raw_prices() -> pd.DataFrame:
    """Tiny long-format price table: 2 dates x 2 tickers, plus 1 duplicate row, in shuffled order."""
    return pd.DataFrame(
        {
            "Date": ["2024-01-03", "2024-01-02", "2024-01-02", "2024-01-03", "2024-01-02"],
            "Ticker": ["SPY", "SPY", "EFA", "EFA", "SPY"],
            "Close": [101.0, 100.0, 70.0, 71.0, 100.0],
            "Volume": [1000, 900, 500, 550, 900],
        }
    )


def test_unit_clean_dataframe_drops_duplicate_date_ticker_rows():
    """5 rows with 1 duplicate (2024-01-02, SPY) must give 4 rows."""
    result = clean_data.clean_dataframe(make_raw_prices())
    assert len(result) == 4


def test_unit_clean_dataframe_does_not_modify_input():
    """Cleaning returns a new frame; the caller's raw frame stays exactly as it was."""
    raw = make_raw_prices()
    original = raw.copy()
    clean_data.clean_dataframe(raw)
    pd.testing.assert_frame_equal(raw, original)


def make_clean_long() -> pd.DataFrame:
    """Clean long table: 3 dates, 2 tickers, but EFA has no row on 2024-01-03."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-04"]),
            "ticker": ["EFA", "SPY", "SPY", "EFA", "SPY"],
            "close": [70.0, 100.0, 101.0, 72.0, 102.0],
        }
    )


def test_unit_to_wide_drops_dates_missing_a_ticker():
    """2024-01-03 has SPY but no EFA, so only the 2 complete dates survive."""
    result = clean_data.to_wide(make_clean_long(), "close", ["SPY", "EFA"])
    assert list(result.index) == list(pd.to_datetime(["2024-01-02", "2024-01-04"]))
    assert result.loc[pd.Timestamp("2024-01-04"), "EFA"] == 72.0


def test_unit_to_wide_columns_follow_tickers_order():
    """Columns come out in the order given (SPY before EFA), not alphabetical."""
    result = clean_data.to_wide(make_clean_long(), "close", ["SPY", "EFA"])
    assert list(result.columns) == ["SPY", "EFA"]


def test_unit_to_wide_raises_on_missing_ticker():
    """Asking for a ticker that is not in the data (GLD) must stop with a clear error."""
    with pytest.raises(ValueError, match="GLD"):
        clean_data.to_wide(make_clean_long(), "close", ["SPY", "EFA", "GLD"])


@pytest.mark.skip(reason="Awaiting validate in dev")
def test_integration_clean_output_passes_validate():
    """Whatever clean_dataframe returns must be accepted by the quality gate."""
    cleaned = clean_data.clean_dataframe(make_raw_prices())
    assert validate.validate_dataframe(
        cleaned,
        required_columns=["date", "ticker", "close", "volume"],
        tickers=["SPY", "EFA"],
        max_abs_daily_return=0.25,
    )

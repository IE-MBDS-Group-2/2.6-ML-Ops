"""Tests for src/clean_data.py: clean_dataframe and to_wide."""

import pandas as pd
import pytest

from src import clean_data, validate


def make_raw_prices() -> pd.DataFrame:
    """Tiny long price table: 2 dates x 2 tickers, plus 1 exact duplicate row, shuffled."""
    return pd.DataFrame(
        {
            "Date": ["2024-01-03", "2024-01-02", "2024-01-02", "2024-01-03", "2024-01-02"],
            "Ticker": ["SPY", "SPY", "EFA", "EFA", "SPY"],
            "Close": [101.0, 100.0, 70.0, 71.0, 100.0],
            "Volume": [1000, 900, 500, 550, 900],
        }
    )


def test_unit_clean_dataframe_drops_exact_duplicate_rows():
    """5 rows with 1 exact duplicate (2024-01-02, SPY, 100.0, 900) must give 4 rows."""
    result = clean_data.clean_dataframe(make_raw_prices())
    assert len(result) == 4


def test_unit_clean_dataframe_keeps_conflicting_rows_for_validate():
    """Same (date, ticker) but another price (100 vs 999) is no exact duplicate: both stay."""
    raw = make_raw_prices()
    raw.loc[4, "Close"] = 999.0
    result = clean_data.clean_dataframe(raw)
    spy_day = result[(result["ticker"] == "SPY") & (result["date"] == "2024-01-02")]
    assert sorted(spy_day["close"]) == [100.0, 999.0]


def test_unit_clean_dataframe_standardises_names_types_and_order():
    """Columns lowercase, date parsed to datetime, rows sorted by date then ticker, index 0..n-1."""
    result = clean_data.clean_dataframe(make_raw_prices())
    assert list(result.columns) == ["date", "ticker", "close", "volume"]
    assert pd.api.types.is_datetime64_any_dtype(result["date"])
    assert list(zip(result["date"].dt.strftime("%Y-%m-%d"), result["ticker"])) == [
        ("2024-01-02", "EFA"), ("2024-01-02", "SPY"), ("2024-01-03", "EFA"), ("2024-01-03", "SPY"),
    ]
    assert list(result.index) == [0, 1, 2, 3]


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
            "date": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-04"]
            ),
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


def test_integration_clean_output_passes_validate():
    """Whatever clean_dataframe returns must be accepted by the quality gate."""
    cleaned = clean_data.clean_dataframe(make_raw_prices())
    assert validate.validate_dataframe(
        cleaned,
        required_columns=["date", "ticker", "close", "volume"],
        tickers=["SPY", "EFA"],
        max_abs_daily_return=0.25,
    )


def test_unit_clean_dataframe_training_mode_drops_rows_without_target():
    """Training mode (target_column given): 3 rows, 1 with no label, gives 2 rows."""
    raw = pd.DataFrame({"x": [1.0, 2.0, 3.0], "label": [0.0, None, 1.0]})
    clean = clean_data.clean_dataframe(raw, target_column="label")
    assert len(clean) == 2
    assert clean.index.tolist() == [0, 1]


def test_unit_clean_dataframe_inference_mode_keeps_every_row():
    """Inference mode (target_column None): no row is dropped for a missing label."""
    raw = pd.DataFrame({"x": [1.0, 2.0, 3.0], "label": [0.0, None, 1.0]})
    assert len(clean_data.clean_dataframe(raw)) == 3


def test_unit_clean_dataframe_missing_target_column_raises():
    """Training mode needs the target column: name it in the error."""
    with pytest.raises(ValueError, match="label"):
        clean_data.clean_dataframe(pd.DataFrame({"x": [1.0]}), target_column="label")


def test_unit_clean_dataframe_works_without_date_and_ticker_columns():
    """Generic columns: names become snake case, the input is untouched, nothing else is assumed."""
    raw = pd.DataFrame({" Feature One ": [2.0, 1.0], "Label": [1, 0]})
    clean = clean_data.clean_dataframe(raw)
    assert list(clean.columns) == ["feature_one", "label"]
    assert list(raw.columns) == [" Feature One ", "Label"]

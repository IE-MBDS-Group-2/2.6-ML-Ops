"""Tests for src/validate.py: validate_dataframe, the data quality gate."""

import pandas as pd
import pytest

from src import clean_data, validate

REQUIRED = ["date", "ticker", "close", "volume"]
COLUMNS = {"date": "date", "ticker": "ticker", "close": "close", "volume": "volume"}  # config data.columns
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


def run_gate(
    df: pd.DataFrame, tickers: list[str] | None = None, max_move: float | None = 0.25
) -> bool:
    """Call the gate with the settings main.py will pass (config: max_abs_daily_return 0.25)."""
    return validate.validate_dataframe(
        df, required_columns=REQUIRED, tickers=TICKERS if tickers is None else tickers,
        max_abs_daily_return=max_move, price_columns=COLUMNS,
    )


def test_unit_validate_returns_true_for_valid_data():
    """Good data passes every check and returns exactly True."""
    assert run_gate(make_valid_long()) is True


def test_unit_validate_skips_optional_checks_when_none():
    """tickers and max_abs_daily_return left as None skip those checks: a 30% jump passes."""
    df = make_valid_long()
    df.loc[1, "close"] = 130.0
    assert validate.validate_dataframe(df, required_columns=REQUIRED) is True
    assert validate.validate_dataframe(df, required_columns=REQUIRED, price_columns=COLUMNS) is True


def test_unit_validate_raises_on_missing_required_column():
    """No volume column: the message names it."""
    with pytest.raises(ValueError, match="Missing required columns.*volume"):
        run_gate(make_valid_long().drop(columns=["volume"]))


def test_unit_validate_raises_on_empty_frame():
    """Right columns but zero rows is not usable data."""
    with pytest.raises(ValueError, match="empty"):
        run_gate(make_valid_long().iloc[0:0])


@pytest.mark.parametrize("column", ["close", "volume"])
def test_unit_validate_raises_on_missing_value(column):
    """An empty (NaN) close or volume must stop the gate, not be dropped quietly by to_wide."""
    df = make_valid_long()
    df[column] = df[column].astype(float)
    df.loc[3, column] = float("nan")
    with pytest.raises(ValueError, match=f"missing values in \\['{column}'\\].*EFA on 2024-01-02"):
        run_gate(df)


def test_unit_validate_raises_on_conflicting_date_ticker_rows():
    """Two different SPY prices for 2024-01-02 (cleaning keeps both): the gate must reject them."""
    df = make_valid_long()
    conflict = df.iloc[[0]].assign(close=999.0)
    with pytest.raises(ValueError, match="share a \\(date, ticker\\).*SPY on 2024-01-02"):
        run_gate(pd.concat([df, conflict], ignore_index=True))


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


def test_integration_clean_output_passes_validate():
    """clean_dataframe output on messy raw data is accepted by the gate."""
    raw = make_valid_long().rename(columns=str.upper)
    raw["DATE"] = raw["DATE"].dt.strftime("%Y-%m-%d")
    assert run_gate(clean_data.clean_dataframe(raw, date_column="date", sort_by=["date", "ticker"])) is True


def test_unit_validate_raises_on_missing_target_column():
    """In training data the target must be present: name it in the error."""
    with pytest.raises(ValueError, match="best_action"):
        validate.validate_dataframe(make_valid_long(), REQUIRED, target_column="best_action")


def test_unit_validate_raises_on_empty_target_values():
    """A label with empty values cannot be trained on: stop and say how many rows."""
    df = make_valid_long().assign(best_action=1.0)
    df.loc[0, "best_action"] = None
    with pytest.raises(ValueError, match="1 rows have no value"):
        validate.validate_dataframe(df, REQUIRED, target_column="best_action")


def test_unit_validate_accepts_complete_target():
    """A present, complete target passes."""
    df = make_valid_long().assign(best_action=1.0)
    assert validate.validate_dataframe(df, REQUIRED, target_column="best_action") is True


def make_generic_table() -> pd.DataFrame:
    """A non-price table (no date or ticker columns), as the contract probes use."""
    return pd.DataFrame({"tenure": [1.0, 5.0, 9.0], "charges": [20.0, 35.5, 50.0], "churn": [0, 1, 0]})


def test_unit_validate_accepts_a_generic_table_without_price_columns():
    table = make_generic_table()
    assert validate.validate_dataframe(table, required_columns=["tenure", "charges"], target_column="churn")


def test_unit_validate_raises_on_missing_value_in_a_generic_table():
    table = make_generic_table()
    table.loc[1, "charges"] = None
    with pytest.raises(ValueError, match=r"missing values in \['charges'\], first: row 1"):
        validate.validate_dataframe(table, required_columns=["tenure", "charges"])


def test_unit_validate_skips_price_rules_without_price_columns():
    # Without data.columns the gate checks only the generic rules, so a repeated (date, ticker) row passes.
    table = make_valid_long()
    assert validate.validate_dataframe(pd.concat([table, table.iloc[[0]]]), required_columns=REQUIRED)


def test_unit_validate_raises_when_a_configured_price_column_is_missing():
    with pytest.raises(ValueError, match=r"Price columns \['volume'\] are missing"):
        validate.validate_dataframe(make_valid_long().drop(columns="volume"), required_columns=["date"],
                                    price_columns=COLUMNS)


def test_unit_validate_uses_the_configured_column_names():
    renamed = {"date": "day", "ticker": "symbol", "close": "px", "volume": "qty"}
    table = make_valid_long().rename(columns=renamed)
    assert validate.validate_dataframe(table, list(renamed.values()), tickers=TICKERS, max_abs_daily_return=0.25,
                                       price_columns=renamed)
    table.loc[0, "px"] = -1.0
    with pytest.raises(ValueError, match="close <= 0, first: SPY on 2024-01-02"):
        validate.validate_dataframe(table, list(renamed.values()), price_columns=renamed)


def test_unit_validate_raises_type_error_on_non_dataframe():
    with pytest.raises(TypeError, match="expects a pandas DataFrame"):
        validate.validate_dataframe([1, 2, 3], required_columns=[])

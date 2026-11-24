"""
validate.py  [Step 3: VALIDATE, the data quality gate]
======================================================
Purpose   Check the clean long price table BEFORE any expensive step runs
Input     Clean long DataFrame (date, ticker, close, volume) + the rules from config.yaml
Output    True if valid, otherwise a ValueError saying what is wrong and how to fix it
Called by main.py (training) and api.py (stage 3, every live request)
Next      to_wide in clean_data.py, then the split in main.py

Why it matters
- Garbage in, garbage out: a bad price would silently train a bad model
- Validate, never fix. Fixing belongs in clean_data.py or in the raw data source
- Rules (tickers, max daily move) come in as arguments from config, so they are versioned

Tests     tests/test_validate.py
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: list[str],
    target_column: str | None = None,
    tickers: list[str] | None = None,
    max_abs_daily_return: float | None = None,
    price_columns: dict[str, str] | None = None,
) -> bool:
    """Stop the pipeline if the clean price table is not trustworthy.

    Inputs:  df, clean long table from clean_dataframe; required_columns, columns that must
             exist; target_column, the label that must be present and complete in training
             data (None for prices and inference); tickers, ETFs that must be present (None skips the check);
             max_abs_daily_return, largest believable one-day move, e.g. 0.25 (None skips it);
             price_columns, config data.columns (date_column, ticker_column, close_column, volume_column). The
             price-table rules run only when it is given; tickers and max_abs_daily_return need it.
    Outputs: True when every check passes.
    Why:     catching bad data here is cheap; finding it after training is not.
    Raises:  TypeError if df is not a DataFrame. ValueError naming the problem and how to fix it, for:
             tickers or max_abs_daily_return given without price_columns, missing columns, empty table,
             empty values in a required column, and, for a price table (price_columns given), a missing
             price column, conflicting (date, ticker) rows, absent ticker, a ticker with fewer dates,
             close <= 0, volume < 0, an extreme daily return.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"validate_dataframe expects a pandas DataFrame, got {type(df).__name__}.")
    if price_columns is None and (tickers is not None or max_abs_daily_return is not None):
        raise ValueError("tickers and max_abs_daily_return are price-table rules and need price_columns. "
                         "Pass data.columns from config.yaml, or leave both as None.")
    _check_columns_and_rows(df, required_columns, price_columns)
    if target_column is not None:
        _check_target(df, target_column)
    _check_no_missing_values(df, required_columns, price_columns)
    if price_columns is not None:
        _check_price_table(df, price_columns, tickers, max_abs_daily_return)
    logger.info("Validated %d rows and %d columns", len(df), df.shape[1])
    return True


def _check_columns_and_rows(df: pd.DataFrame, required_columns: list[str], cols: dict[str, str] | None) -> None:
    """Raise if a required column or a configured price column is absent, or if the table has no rows.

    Inputs: df, the clean table; required_columns, names that must exist; cols, data.columns from config or None.
    Outputs: None.
    Why: Every later check indexes these columns, so a missing one should fail here with a clear message.
    Raises: ValueError for a missing required column, a missing price column, or an empty table.
    """
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns {missing}. Check the raw file header and clean_dataframe."
        )
    absent = [name for name in (cols or {}).values() if name not in df.columns]
    if absent:
        raise ValueError(f"Price columns {absent} are missing. Check data.columns in config.yaml.")
    if df.empty:
        raise ValueError(
            "The table is empty. Check that the raw file has rows and was loaded correctly."
        )


def _check_price_table(df: pd.DataFrame, cols: dict[str, str], tickers: list[str] | None,
                       max_abs_daily_return: float | None) -> None:
    """Price-table rules (one row per date and ticker); cols maps date, ticker, close, volume to column names.

    Inputs: df, the clean long table; cols, data.columns from config; tickers, ETFs that must be present or None;
        max_abs_daily_return, the largest believable one-day move or None.
    Outputs: None.
    Why: Groups the rules that only make sense for a (date, ticker) price table, in the order they should run.
    Raises: ValueError from the checks it calls.
    """
    _check_column_types(df, cols)
    _check_one_row_per_date_ticker(df, cols)
    if tickers is not None:
        _check_tickers_present(df, cols, tickers)
    _check_same_number_of_dates(df, cols)
    _check_value_ranges(df, cols)
    if max_abs_daily_return is not None:
        _check_daily_returns(df, cols, max_abs_daily_return)


def _check_target(df: pd.DataFrame, target_column: str) -> None:
    """Raise if the target is absent or has empty values (training data only; prices have no target).

    Inputs: df, the clean table; target_column, the label column (problem.target_column).
    Outputs: None.
    Why: A label with gaps cannot train a model, and the fix belongs in clean_dataframe or the source data.
    Raises: ValueError if the column is absent or has empty values.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is missing. Check problem.target_column in config.yaml.")
    empty = int(df[target_column].isna().sum())
    if empty:
        raise ValueError(f"{empty} rows have no value in target column '{target_column}'. "
                         "Drop them in clean_dataframe (training mode) or fix the source data.")


def _check_column_types(df: pd.DataFrame, cols: dict[str, str]) -> None:
    """Raise, naming the column and the type found, if a price column has the wrong type.

    The rules: date is a date, ticker is text, close and volume are numbers.

    Inputs:  df, the clean long table; cols, the configured price column names.
    Outputs: None when the four types are right.
    Why:     a price read as text (one bad cell turns the whole column into text) or a date left as text would
             otherwise fail later with a pandas error far from the cause, or sort and compare wrongly.
    """
    wrong = [(cols["date_column"], "dates (datetime)", pd.api.types.is_datetime64_any_dtype),
             (cols["ticker_column"], "text", _is_text),
             (cols["close_column"], "numbers", pd.api.types.is_numeric_dtype),
             (cols["volume_column"], "numbers", pd.api.types.is_numeric_dtype)]
    for column, expected, is_right in wrong:
        if pd.api.types.is_bool_dtype(df[column]) or not is_right(df[column]):
            raise ValueError(f"Column '{column}' must hold {expected}, but its type is {df[column].dtype}. "
                             "Check the raw file for stray text in the column and clean_dataframe's date step.")


def _is_text(values: pd.Series) -> bool:
    """True when every non-missing value in the column is a string.

    Inputs: values, a column.
    Outputs: True if its dtype is text-like and every non-missing value is a str.
    Why: pandas treats a mixed object column (strings and numbers) as text, which would hide a bad ticker.
    """
    return pd.api.types.is_string_dtype(values) and values.dropna().map(lambda item: isinstance(item, str)).all()


def _first_row(rows: pd.DataFrame, cols: dict[str, str] | None) -> str:
    """Describe the first offending row, e.g. 'first: SPY on 2024-01-03' (or 'first: row 7'), so it can be found.

    Inputs: rows, the offending rows (not empty); cols, data.columns from config, or None for a plain table.
    Outputs: Text naming the ticker and date of the first row, or its row label when cols is None.
    Why: An error that points at one row is much faster to fix than a bare count.
    """
    first = rows.iloc[0]
    if cols is not None:
        return f"first: {first[cols['ticker_column']]} on {pd.Timestamp(first[cols['date_column']]):%Y-%m-%d}"
    return f"first: row {rows.index[0]}"


def _check_no_missing_values(df: pd.DataFrame, required_columns: list[str], cols: dict[str, str] | None) -> None:
    """Raise if a required column has an empty (NaN) value; for prices, to_wide would otherwise drop the date.

    Inputs: df, the clean table; required_columns, names to check; cols, data.columns from config or None.
    Outputs: None.
    Why: A gap in one ticker's price would make to_wide drop the whole date for every ticker, silently.
    Raises: ValueError with the number of affected rows and the first one.
    """
    is_missing = df[list(required_columns)].isna()
    columns = [column for column in required_columns if is_missing[column].any()]
    if columns:
        bad = df[is_missing.any(axis=1)]
        raise ValueError(
            f"{len(bad)} rows have missing values in {columns}, {_first_row(bad, cols)}. "
            "Fill them in the raw data source or re-download the snapshot."
        )


def _check_one_row_per_date_ticker(df: pd.DataFrame, cols: dict[str, str]) -> None:
    """Raise if a (date, ticker) appears more than once, e.g. two different prices for one day.

    Inputs: df, the clean table; cols, data.columns from config.
    Outputs: None.
    Why: Exact duplicates are removed in cleaning, so a repeat here has conflicting values and picking one would
        be a silent fix.
    Raises: ValueError with the number of rows involved and the first one.
    """
    repeated = df[df.duplicated([cols["date_column"], cols["ticker_column"]], keep=False)]
    if not repeated.empty:
        raise ValueError(
            f"{len(repeated)} rows share a (date, ticker) with conflicting values, "
            f"{_first_row(repeated, cols)}. Keep one correct row per day in the raw data source."
        )


def _check_tickers_present(df: pd.DataFrame, cols: dict[str, str], tickers: list[str]) -> None:
    """Raise if any configured ticker has no rows.

    Inputs: df, the clean table; cols, data.columns from config; tickers, data.tickers from config.
    Outputs: None.
    Why: A typo in the ticker list or a short download would otherwise surface later as a missing column.
    Raises: ValueError naming the absent tickers.
    """
    present = set(df[cols["ticker_column"]])
    absent = [ticker for ticker in tickers if ticker not in present]
    if absent:
        raise ValueError(
            f"Tickers {absent} not found in the data. Check config.yaml tickers or the raw file."
        )


def _check_same_number_of_dates(df: pd.DataFrame, cols: dict[str, str]) -> None:
    """Raise if some ticker has fewer dates than the others (gaps in its history).

    Inputs: df, the clean table; cols, data.columns from config.
    Outputs: None.
    Why: to_wide keeps only dates where every ticker has a price, so one short history would shrink them all.
    Raises: ValueError naming the tickers with fewer dates and the expected number.
    """
    dates_per_ticker = df.groupby(cols["ticker_column"])[cols["date_column"]].nunique()
    expected = dates_per_ticker.max()
    short = dates_per_ticker[dates_per_ticker < expected].to_dict()
    if short:
        raise ValueError(
            f"Tickers with fewer dates than the others (expected {expected}): {short}. "
            "Re-download the snapshot or check the raw file for missing days."
        )


def _check_value_ranges(df: pd.DataFrame, cols: dict[str, str]) -> None:
    """Raise on impossible values: a price at or below zero, or negative volume.

    Inputs: df, the clean table; cols, data.columns from config.
    Outputs: None.
    Why: Returns and costs are built from these numbers, and such values mean the source data is wrong.
    Raises: ValueError for close <= 0 or volume < 0, with the first offending row.
    """
    bad_close = df[df[cols["close_column"]] <= 0]
    if not bad_close.empty:
        raise ValueError(
            f"{len(bad_close)} rows have close <= 0, {_first_row(bad_close, cols)}. "
            "Prices must be positive; fix the raw data source."
        )
    bad_volume = df[df[cols["volume_column"]] < 0]
    if not bad_volume.empty:
        raise ValueError(
            f"{len(bad_volume)} rows have volume < 0, {_first_row(bad_volume, cols)}. "
            "Volume cannot be negative; fix the raw data source."
        )


def _check_daily_returns(df: pd.DataFrame, cols: dict[str, str], max_abs_daily_return: float) -> None:
    """Raise if any ticker moves more than max_abs_daily_return in one day (likely a bad price).

    Inputs: df, the clean table; cols, data.columns from config; max_abs_daily_return, e.g. 0.25.
    Outputs: None.
    Why: A move that large in a diversified ETF is far more likely a data error or a missed split than a real day.
    Raises: ValueError with the number of such moves, the first one and its size.
    """
    ordered = df.sort_values([cols["ticker_column"], cols["date_column"]])
    moves = ordered.groupby(cols["ticker_column"])[cols["close_column"]].pct_change().abs()
    too_big = ordered[moves > max_abs_daily_return]
    if not too_big.empty:
        biggest = moves[too_big.index[0]]
        raise ValueError(
            f"{len(too_big)} daily returns above {max_abs_daily_return:.0%}, "
            f"{_first_row(too_big, cols)} ({biggest:.1%}). "
            "Check for a bad price or a missing split adjustment."
        )

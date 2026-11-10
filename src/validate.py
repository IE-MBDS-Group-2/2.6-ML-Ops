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

DATE_COLUMN = "date"
TICKER_COLUMN = "ticker"
CLOSE_COLUMN = "close"
VOLUME_COLUMN = "volume"


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: list[str],
    target_column: str | None = None,
    tickers: list[str] | None = None,
    max_abs_daily_return: float | None = None,
) -> bool:
    """Stop the pipeline if the clean price table is not trustworthy.

    Inputs:  df, clean long table from clean_dataframe; required_columns, columns that must exist;
             target_column, unused (price data has no target), kept for the course contract;
             tickers, ETFs that must be present (None skips the check);
             max_abs_daily_return, largest believable one-day move, e.g. 0.25 (None skips the check).
    Outputs: True when every check passes.
    Why:     catching bad data here is cheap; finding it after training is not.
    Raises:  ValueError naming the problem and how to fix it, for: missing columns, empty table,
             absent ticker, a ticker with fewer dates, close <= 0, volume < 0, an extreme daily return.
    """
    del target_column  # contract parameter, not used for price data

    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing}. Check the raw file header and clean_dataframe.")
    if df.empty:
        raise ValueError("The price table is empty. Check that the raw file has rows and was loaded correctly.")

    if tickers is not None:
        _check_tickers_present(df, tickers)
    _check_same_number_of_dates(df)
    _check_value_ranges(df)
    if max_abs_daily_return is not None:
        _check_daily_returns(df, max_abs_daily_return)

    logger.info("Validated %d rows for %d tickers", len(df), df[TICKER_COLUMN].nunique())
    return True


def _check_tickers_present(df: pd.DataFrame, tickers: list[str]) -> None:
    """Raise if any configured ticker has no rows."""
    present = set(df[TICKER_COLUMN])
    absent = [ticker for ticker in tickers if ticker not in present]
    if absent:
        raise ValueError(f"Tickers {absent} not found in the data. Check config.yaml tickers or the raw price file.")


def _check_same_number_of_dates(df: pd.DataFrame) -> None:
    """Raise if some ticker has fewer dates than the others (gaps in its history)."""
    dates_per_ticker = df.groupby(TICKER_COLUMN)[DATE_COLUMN].nunique()
    expected = dates_per_ticker.max()
    short = dates_per_ticker[dates_per_ticker < expected].to_dict()
    if short:
        raise ValueError(
            f"Tickers with fewer dates than the others (expected {expected}): {short}. "
            "Re-download the snapshot or check the raw file for missing days."
        )


def _check_value_ranges(df: pd.DataFrame) -> None:
    """Raise on impossible values: a price at or below zero, or negative volume."""
    bad_close = df[df[CLOSE_COLUMN] <= 0]
    if not bad_close.empty:
        first = bad_close.iloc[0]
        raise ValueError(
            f"{len(bad_close)} rows have close <= 0, first: {first[TICKER_COLUMN]} on "
            f"{first[DATE_COLUMN]:%Y-%m-%d}. Prices must be positive; fix the raw data source."
        )
    bad_volume = df[df[VOLUME_COLUMN] < 0]
    if not bad_volume.empty:
        first = bad_volume.iloc[0]
        raise ValueError(
            f"{len(bad_volume)} rows have volume < 0, first: {first[TICKER_COLUMN]} on "
            f"{first[DATE_COLUMN]:%Y-%m-%d}. Volume cannot be negative; fix the raw data source."
        )


def _check_daily_returns(df: pd.DataFrame, max_abs_daily_return: float) -> None:
    """Raise if any ticker moves more than max_abs_daily_return in one day (likely a bad price)."""
    ordered = df.sort_values([TICKER_COLUMN, DATE_COLUMN])
    moves = ordered.groupby(TICKER_COLUMN)[CLOSE_COLUMN].pct_change().abs()
    too_big = ordered[moves > max_abs_daily_return]
    if not too_big.empty:
        first = too_big.iloc[0]
        raise ValueError(
            f"{len(too_big)} daily returns above {max_abs_daily_return:.0%}, first: {first[TICKER_COLUMN]} on "
            f"{first[DATE_COLUMN]:%Y-%m-%d} ({moves[first.name]:.1%}). "
            "Check for a bad price or a missing split adjustment."
        )

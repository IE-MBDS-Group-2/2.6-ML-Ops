"""
clean_data.py  [Step 2: CLEAN]
==============================
Purpose   Make the raw long price table consistent so every later step behaves predictably,
          then reshape it to one row per date and one column per ticker
Input     Raw long DataFrame (date, ticker, close, volume) from load_data.py
Output    clean_dataframe: clean long DataFrame; to_wide: date x ticker table on a common calendar
Called by main.py (training) and api.py (stage 3, same cleaning for live requests)
Next      validate.py, then to_wide for features.py and the portfolio code

Why it matters
- Stateless fixes only: column names, types, order, exact duplicates. Nothing that learns from data
- Never modifies the caller's DataFrame, and logs every row it drops
- Bad values (prices <= 0, extreme moves) are NOT fixed here: validate.py stops the pipeline on them

Tests     tests/test_clean_data.py
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def clean_dataframe(df_raw: pd.DataFrame, target_column: str | None = None, date_column: str | None = None,
                    sort_by: list[str] | None = None) -> pd.DataFrame:
    """Return a cleaned copy of the raw long price table.

    Inputs:  df_raw, one row per (date, ticker); target_column, given in training mode (rows with no target are
             dropped), None in inference mode (no rows dropped for a target). Prices have no target.
             date_column, parsed to datetime if given; sort_by, columns to sort on if given (both from
             config data.columns; None leaves the table as it is).
    Outputs: new DataFrame with lowercase column names, a datetime date column, sorted by date then
             ticker, exact duplicate rows removed, and a fresh 0..n-1 index. Rows that share a
             (date, ticker) but differ in a value are KEPT: choosing one would be a silent fix, so
             validate.py rejects them instead.
    Raises:  TypeError if df_raw is not a DataFrame; ValueError if target_column is given but absent.
    Why:     later steps rely on consistent names, real dates and a stable order; exact duplicates
             would double-count a day.
    """
    if not isinstance(df_raw, pd.DataFrame):
        raise TypeError(f"clean_dataframe expects a pandas DataFrame, got {type(df_raw).__name__}. "
                        "Pass the output of load_raw_data.")
    df = df_raw.copy()
    df.columns = [str(column).strip().lower().replace(" ", "_") for column in df.columns]
    if date_column is not None and date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])

    n_before = len(df)
    df = df.drop_duplicates()
    _log_dropped("exact duplicate", n_before, len(df))

    if target_column is not None:
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' is missing. "
                             "Check problem.target_column in config.yaml.")
        n_before = len(df)
        df = df.dropna(subset=[target_column])
        _log_dropped(f"missing-{target_column}", n_before, len(df))

    order = [column for column in (sort_by or []) if column in df.columns]
    if order:
        df = df.sort_values(order)
    return df.reset_index(drop=True)


def _log_dropped(kind: str, before: int, after: int) -> None:
    """Warn when rows were dropped, with the counts before and after, so nothing vanishes silently."""
    if after < before:
        logger.warning("Dropped %d %s rows: %d before, %d after", before - after, kind, before, after)
    else:
        logger.info("No %s rows dropped (%d rows)", kind, before)


def to_wide(df: pd.DataFrame, value_column: str, tickers: list[str], date_column: str,
            ticker_column: str) -> pd.DataFrame:
    """Pivot a clean long table to one row per date and one column per ticker.

    Inputs:  df, clean long table from clean_dataframe; value_column, the column to spread
             (e.g. "close"); tickers, the ETFs to keep, in the order the columns should appear;
             date_column and ticker_column, the names from config data.columns.
    Outputs: DataFrame indexed by date (sorted), columns exactly `tickers`, keeping only dates
             where every ticker has a value, so all ETFs share one trading calendar.
    Why:     features and the portfolio code compare ETFs date by date and match columns to target
             weights by position.
    Raises:  ValueError if a requested ticker does not appear in df.
    """
    present = set(df[ticker_column])
    missing = [ticker for ticker in tickers if ticker not in present]
    if missing:
        raise ValueError(
            f"Tickers {missing} not found in the data. Check the tickers list in config.yaml "
            "or that the raw price file contains them."
        )

    requested = df[df[ticker_column].isin(tickers)]
    wide = requested.pivot(index=date_column, columns=ticker_column, values=value_column)
    wide = wide[tickers]
    wide.columns.name = None

    n_before = len(wide)
    wide = wide.dropna(how="any")
    logger.info(
        "Dropped %d of %d dates where not every ticker has a %s value",
        n_before - len(wide), n_before, value_column,
    )
    return wide

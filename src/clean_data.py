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

DATE_COLUMN = "date"
TICKER_COLUMN = "ticker"


def clean_dataframe(df_raw: pd.DataFrame, target_column: str | None = None) -> pd.DataFrame:
    """Return a cleaned copy of the raw long price table.

    Inputs:  df_raw, one row per (date, ticker); target_column, unused because the price file has no
             target, kept so the signature matches the course contract.
    Outputs: new DataFrame with lowercase column names, a datetime date column, sorted by date then
             ticker, one row per (date, ticker), and a fresh 0..n-1 index.
    Why:     later steps rely on consistent names, real dates and a stable order; duplicates would
             double-count a day.
    """
    del target_column  # contract parameter, not used for price data

    df = df_raw.copy()
    df.columns = [str(column).strip().lower() for column in df.columns]
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])

    n_before = len(df)
    df = df.drop_duplicates(subset=[DATE_COLUMN, TICKER_COLUMN], keep="first")
    logger.info("Dropped %d duplicate (date, ticker) rows out of %d", n_before - len(df), n_before)

    return df.sort_values([DATE_COLUMN, TICKER_COLUMN]).reset_index(drop=True)


def to_wide(df: pd.DataFrame, value_column: str, tickers: list[str]) -> pd.DataFrame:
    """Pivot a clean long table to one row per date and one column per ticker.

    Inputs:  df, clean long table from clean_dataframe; value_column, the column to spread (e.g. "close");
             tickers, the ETFs to keep, in the order the columns should appear.
    Outputs: DataFrame indexed by date (sorted), columns exactly `tickers`, keeping only dates where every
             ticker has a value, so all ETFs share one trading calendar.
    Why:     features and the portfolio code compare ETFs date by date and match columns to target
             weights by position.
    Raises:  ValueError if a requested ticker does not appear in df.
    """
    present = set(df[TICKER_COLUMN])
    missing = [ticker for ticker in tickers if ticker not in present]
    if missing:
        raise ValueError(
            f"Tickers {missing} not found in the data. Check the tickers list in config.yaml "
            "or that the raw price file contains them."
        )

    wide = df[df[TICKER_COLUMN].isin(tickers)].pivot(index=DATE_COLUMN, columns=TICKER_COLUMN, values=value_column)
    wide = wide[tickers]
    wide.columns.name = None

    n_before = len(wide)
    wide = wide.dropna(how="any")
    logger.info("Dropped %d of %d dates where not every ticker has a %s value", n_before - len(wide), n_before,
                value_column)
    return wide

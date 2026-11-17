"""
make_sample_prices.py  [Synthetic data: deterministic sample prices for CI and tests]
=====================================================================================
Purpose   Build fake but realistic ETF prices so CI and the tests can run the whole pipeline without the
          real Yahoo snapshot (which is licensed for personal use and never goes in Git)
Input     config.yaml: data.tickers, data.start, data.end, data.columns, data.inference_days,
          paths.raw_data, paths.inference_data
Output    the same two files fetch_prices.py writes: paths.raw_data and paths.inference_data (long format)
Run       python -m scripts.make_sample_prices          (from the repository root)
Not for   real results: the prices are random. Every run logs a WARNING that the data is SYNTHETIC

Why it matters
- Same seed, same file: a CI failure can be reproduced on any machine
- Each ticker has its own random stream, so adding or removing a ticker never changes the others' prices

Tests     tests/test_make_sample_prices.py
"""

import logging

import numpy as np
import pandas as pd

from src import config, utils

logger = logging.getLogger(__name__)

# Daily volatility per ticker (spec). A ticker not listed here gets DEFAULT_VOLATILITY.
DAILY_VOLATILITY = {"SPY": 0.012, "EFA": 0.013, "IEF": 0.004, "GLD": 0.010, "BIL": 0.0002}
DEFAULT_VOLATILITY = 0.010
DAILY_DRIFT = 0.0003          # a small positive drift, about 8% a year
START_PRICE = 100.0
MEDIAN_VOLUME = 1_000_000
VOLUME_SPREAD = 0.3           # log-normal spread of the daily volume


def make_sample_prices(tickers: list[str], start: str, end: str, seed: int,
                       columns: dict[str, str]) -> pd.DataFrame:
    """Build a long table of synthetic daily prices, one row per business day and ticker.

    Inputs:  tickers, the ETFs to simulate; start and end, YYYY-MM-DD, inclusive; seed, the random seed;
             columns, config data.columns (date_column, ticker_column, close_column, volume_column).
    Outputs: a DataFrame with exactly those four columns, in that order, sorted by date and then ticker.
             Closes are above 0, volumes above 0, dates are text (YYYY-MM-DD) like the real snapshot.
    Why:     CI has no access to the real price file. The same seed always gives the same table.
    Raises:  ValueError if end is before start (it names both dates).
    """
    if pd.Timestamp(end) < pd.Timestamp(start):
        raise ValueError(f"Start date {start} is after end date {end}. Check data.start and data.end in config.yaml.")
    days = pd.bdate_range(start, end)
    frames = [_one_ticker(ticker, days, seed, columns) for ticker in tickers]
    prices = pd.concat(frames, ignore_index=True)
    prices = prices.sort_values([columns["date_column"], columns["ticker_column"]]).reset_index(drop=True)
    logger.warning("SYNTHETIC prices generated: %d rows for %d tickers from %s to %s. "
                   "This is random data, not the real snapshot.", len(prices), len(tickers), start, end)
    return prices


def _one_ticker(ticker: str, days: pd.DatetimeIndex, seed: int, columns: dict[str, str]) -> pd.DataFrame:
    """Simulate one ticker: a random walk for the close and a log-normal volume.

    Inputs:  ticker; days, the business days; seed; columns, config data.columns.
    Outputs: a table with the four configured columns for this ticker.
    Why:     the random stream depends on the seed and the ticker's own name, not on its position in the list,
             so one ticker's prices never change when another ticker is added or removed.
             The walk works on log returns, so the close can never reach zero or below.
    """
    rng = np.random.default_rng([seed, *ticker.encode("utf-8")])
    volatility = DAILY_VOLATILITY.get(ticker, DEFAULT_VOLATILITY)
    log_returns = DAILY_DRIFT + volatility * rng.standard_normal(len(days))
    close = START_PRICE * np.exp(np.cumsum(log_returns))
    volume = MEDIAN_VOLUME * rng.lognormal(mean=0.0, sigma=VOLUME_SPREAD, size=len(days))
    return pd.DataFrame({
        columns["date_column"]: days.strftime("%Y-%m-%d"),
        columns["ticker_column"]: ticker,
        columns["close_column"]: np.round(close, 4),
        columns["volume_column"]: np.maximum(np.round(volume), 1).astype("int64"),
    })


def _latest_days(prices: pd.DataFrame, date_column: str, days: int) -> pd.DataFrame:
    """Keep the rows of the last `days` trading days.

    Inputs:  prices, the long table; date_column, its date column; days, how many trading days to keep.
    Outputs: the rows of the latest `days` distinct dates (all rows if there are fewer dates).
    Why:     the inference file is the recent slice of the same table, like the real snapshot's.
    """
    latest = sorted(prices[date_column].unique())[-days:]
    return prices[prices[date_column].isin(latest)]


def main() -> None:
    """Write the synthetic raw file and the inference file, using only the settings in config.yaml.

    Inputs:  none; reads config.yaml through config.load_config (data.* and paths.*; the seed is model.random_state).
    Outputs: None. Writes paths.raw_data (the full table) and paths.inference_data (the last
             data.inference_days trading days). Missing folders are created.
    Why:     CI runs this before python -m src.main, so the pipeline has data without the real snapshot.
    """
    cfg = config.load_config()
    paths = config.resolve_paths(cfg)
    data = cfg["data"]
    columns = data["columns"]
    prices = make_sample_prices(data["tickers"], data["start"], data["end"], cfg["model"]["random_state"], columns)
    utils.save_csv(prices, paths["raw_data"])
    utils.save_csv(_latest_days(prices, columns["date_column"], data["inference_days"]), paths["inference_data"])
    logger.warning("SYNTHETIC data written to %s and %s. Never use it for real results.",
                   paths["raw_data"], paths["inference_data"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()

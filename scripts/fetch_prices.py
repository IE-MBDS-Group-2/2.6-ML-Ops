"""
fetch_prices.py  [Data snapshot: download the frozen price file]
================================================================
Purpose   Download daily prices for the portfolio ETFs from Yahoo Finance and freeze them to disk
Input     config.yaml: data.tickers, data.start, data.end, data.inference_days,
          paths.raw_data, paths.inference_data
Output    data/raw/prices.csv (long format: date, ticker, close, volume)
          data/inference/prices_latest.csv (the last N trading days of the same table)
Run       python scripts/fetch_prices.py            (from the repository root)
          python scripts/fetch_prices.py --config path/to/other_config.yaml
Not in    src/: the pipeline never downloads. It reads the frozen file through load_data.py

Before you run it
- yfinance comes with environment.yml (conda env create -f environment.yml)
- Yahoo's data is for personal use only: share the file directly, never commit it to Git
- The script logs the SHA-256 checksum of prices.csv: send it to the team so everyone can check
  they hold the same file
"""

import argparse
import hashlib
import logging
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

logger = logging.getLogger(__name__)

DATA_KEYS = ("tickers", "start", "end", "inference_days")
PATH_KEYS = ("raw_data", "inference_data")


def read_settings(config_path: Path) -> dict:
    """Read the settings this script needs from the team config and check that every key is there.

    Inputs:
        config_path: path of the YAML config file (for example config.yaml).
    Outputs:
        A flat dictionary with tickers, start, end, inference_days (from the "data" section)
        and raw_data, inference_data (from the "paths" section).
    Why:
        Tickers, dates and paths live in the team config, so each path is defined once and a
        change never needs an edit here. A missing key stops the run with a message that says
        what to add.

    Raises:
        FileNotFoundError: if the config file does not exist.
        ValueError: if a section or any required key is missing.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}. Run from the repository root.")
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        config = {}

    settings = {}
    problems = []
    for section, keys in (("data", DATA_KEYS), ("paths", PATH_KEYS)):
        values = config.get(section)
        if not isinstance(values, dict):
            problems.append(f"no '{section}' section (needs {', '.join(keys)})")
            continue
        missing = [key for key in keys if key not in values]
        if missing:
            problems.append(f"'{section}' is missing {', '.join(missing)}")
        settings.update({key: values[key] for key in keys if key in values})
    if problems:
        raise ValueError(f"Config {config_path} is incomplete: " + "; ".join(problems))
    logger.info("Read settings from %s", config_path)
    return settings


def download_prices(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Download daily adjusted prices from Yahoo Finance and return them in long format.

    Inputs:
        tickers: ticker symbols to download (from the config).
        start_date: first day to include, as YYYY-MM-DD.
        end_date: last day to include, as YYYY-MM-DD (Yahoo's own end date is exclusive,
            so one day is added here to keep end_date in the result).
    Outputs:
        A DataFrame with the columns date (text), ticker, close and volume, one row per
        ticker and trading day, sorted by date and then ticker.
    Why:
        auto_adjust=True gives prices adjusted for splits and dividends, so returns are right.
        One ticker at a time, so an empty answer names the ticker that failed.

    Raises:
        ValueError: if Yahoo returns no data for a ticker.
    """
    end_exclusive = (pd.Timestamp(str(end_date)) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    frames = []
    for ticker in tickers:
        history = yf.Ticker(ticker).history(start=str(start_date), end=end_exclusive, auto_adjust=True)
        if history.empty:
            raise ValueError(f"Yahoo returned no data for {ticker}. Check the ticker, the dates and your network.")
        frames.append(
            pd.DataFrame(
                {
                    "date": history.index.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "close": history["Close"].to_numpy(),
                    "volume": history["Volume"].to_numpy(),
                }
            )
        )
        logger.info("Downloaded %d rows for %s", len(history), ticker)
    prices = pd.concat(frames, ignore_index=True)
    return prices.sort_values(["date", "ticker"]).reset_index(drop=True)


def check_prices(prices: pd.DataFrame, tickers: list, max_gap_days: int = 7) -> None:
    """Check that the downloaded table is complete: every ticker present, same days, no gaps.

    Inputs:
        prices: the long table from download_prices (date, ticker, close, volume).
        tickers: the tickers that must be present (from the config).
        max_gap_days: longest stretch of calendar days allowed between two trading days
            (7 allows weekends, holidays and a short market closure).
    Outputs:
        None. It only logs a summary when everything is fine.
    Why:
        A ticker that starts late or skips days would silently bend the backtest. The data is
        frozen once, so it is checked once, here, and the run stops if anything is off.

    Raises:
        ValueError: listing every problem found (missing ticker, missing days, long gap).
    """
    problems = []
    missing_tickers = [ticker for ticker in tickers if ticker not in set(prices["ticker"])]
    if missing_tickers:
        problems.append("no data for " + ", ".join(missing_tickers))

    days_by_ticker = {ticker: set(days) for ticker, days in prices.groupby("ticker")["date"]}
    all_days = set().union(*days_by_ticker.values())
    for ticker, days in days_by_ticker.items():
        absent = sorted(all_days - days)
        if absent:
            problems.append(f"{ticker} is missing {len(absent)} days that other tickers have (first: {absent[0]})")

    calendar = pd.Series(pd.to_datetime(sorted(all_days)))
    biggest_gap = calendar.diff().dt.days.max()
    if biggest_gap > max_gap_days:
        problems.append(f"the longest stretch without data is {biggest_gap:.0f} days")

    if problems:
        raise ValueError("Price check failed: " + "; ".join(problems))
    logger.info(
        "Price check passed: %d tickers share %d trading days, %s to %s",
        len(days_by_ticker), len(all_days), min(all_days), max(all_days),
    )


def sha256_of_file(path: Path) -> str:
    """Return the SHA-256 checksum of a file as a hex string.

    Inputs:
        path: the file to fingerprint.
    Outputs:
        64 hex characters. Two people with the same checksum hold byte-for-byte the same file.
    Why:
        The snapshot is shared outside Git, so the checksum is how the team knows it is the same one.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_outputs(prices: pd.DataFrame, raw_path: Path, inference_path: Path, inference_days: int) -> str:
    """Write the full snapshot and the recent slice, and return the checksum of the full file.

    Inputs:
        prices: the checked long table (date, ticker, close, volume).
        raw_path: where the full snapshot goes (data/raw/prices.csv).
        inference_path: where the recent slice goes (data/inference/prices_latest.csv).
        inference_days: how many of the latest trading days go in the recent slice.
    Outputs:
        The SHA-256 checksum of the full snapshot file.
    Why:
        Training reads the full frozen file; the weekly inference run needs only the latest days.
        Both folders are created if missing. Neither file is ever committed to Git.
    """
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(raw_path, index=False)
    logger.info("Wrote %d rows to %s", len(prices), raw_path)

    latest_days = sorted(prices["date"].unique())[-inference_days:]
    latest = prices[prices["date"].isin(latest_days)]
    inference_path.parent.mkdir(parents=True, exist_ok=True)
    latest.to_csv(inference_path, index=False)
    logger.info("Wrote %d rows (%d trading days) to %s", len(latest), len(latest_days), inference_path)
    return sha256_of_file(raw_path)


def main(config_path: Path) -> None:
    """Run the whole download: read settings, download, check, write, log the checksum.

    Inputs:
        config_path: path of the YAML config file with the "data" and "paths" sections.
    Outputs:
        None. Writes the two CSV files and logs the SHA-256 of the full snapshot.
    Why:
        One command freezes the data. The checks run before anything is written, so a bad
        download never replaces a good file.
    """
    settings = read_settings(config_path)
    prices = download_prices(settings["tickers"], settings["start"], settings["end"])
    check_prices(prices, settings["tickers"])
    checksum = write_outputs(
        prices, Path(settings["raw_data"]), Path(settings["inference_data"]), int(settings["inference_days"])
    )
    logger.info("SHA-256 of %s: %s", settings["raw_data"], checksum)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Download and freeze the ETF price snapshot.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="YAML file with a data section")
    main(parser.parse_args().config)

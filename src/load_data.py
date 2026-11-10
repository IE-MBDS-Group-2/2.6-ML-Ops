"""
load_data.py  [Step 1: LOAD]
============================
Purpose   Read the raw dataset into a DataFrame, exactly as it is on disk
Input     Path to the raw file (config: paths.raw_data)
Output    Raw pandas DataFrame, untouched
Called by main.py          Uses   utils.py (file reading)
Next      clean_data.py

Why it matters
- One entry point for data: swap CSV for a database or API later without touching other files
- Read only: the raw file is never modified, so every run is reproducible
- Fail fast with a clear message if the file is missing, a folder, or empty

Tests     tests/test_load_data.py (stage 2)
"""

import logging
from pathlib import Path

import pandas as pd

from src import utils

logger = logging.getLogger(__name__)


def load_raw_data(raw_data_path: Path) -> pd.DataFrame:
    """Read the frozen raw price file and return it untouched.

    Inputs:
        raw_data_path: path of the raw CSV file (for example data/raw/prices.csv).
    Outputs:
        The raw table as a pandas DataFrame, with the columns and text values exactly as on disk.
    Why:
        One entry point for the data that stops early, with a clear message, when the file is
        missing, is a folder, is empty or has no rows. It never cleans, converts or downloads.

    Raises:
        FileNotFoundError: if the path does not exist.
        IsADirectoryError: if the path is a folder, not a file.
        ValueError: if the file has zero bytes, or has a header but no rows.
    """
    if not raw_data_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_data_path}. Share or fetch the snapshot first.")
    if raw_data_path.is_dir():
        raise IsADirectoryError(f"Raw data path is a folder, not a file: {raw_data_path}")
    if raw_data_path.stat().st_size == 0:
        raise ValueError(f"Raw data file is empty: {raw_data_path}")

    df = utils.load_csv(raw_data_path)
    if len(df) == 0:
        raise ValueError(f"Raw data file has no rows: {raw_data_path}")

    logger.info("Raw data ready: %d rows, %d columns", len(df), len(df.columns))
    return df

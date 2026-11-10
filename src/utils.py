"""
utils.py  [Shared plumbing]
===========================
Purpose   Small, reusable read and write helpers
Examples  read and write CSV files, save and load the model file, save a JSON report
Used by   load_data.py, main.py, infer.py, api.py (stage 3)

Why it matters
- One place for file I/O (Input/Output): fix a bug once, every module benefits
- No business logic: if a helper knows your target column, it belongs elsewhere

Tests     tests/test_utils.py (stage 2)
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_csv(filepath: Path) -> pd.DataFrame:
    """Read a CSV file into a DataFrame, exactly as it is on disk.

    Inputs:
        filepath: path of the CSV file to read.
    Outputs:
        The table as a pandas DataFrame (no cleaning, no type changes).
    Why:
        One reader for the whole pipeline, with a clear error when the file is missing.

    Raises:
        FileNotFoundError: if the file does not exist.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}. Check the path or create the file first.")
    df = pd.read_csv(filepath)
    logger.info("Loaded %d rows from %s", len(df), filepath)
    return df


def save_csv(df: pd.DataFrame, filepath: Path) -> None:
    """Write a DataFrame to a CSV file, creating missing parent folders.

    Inputs:
        df: the table to save.
        filepath: where to write it.
    Outputs:
        None. The file is written to disk, without the index column.
    Why:
        A first run on a clean checkout has no data/ or reports/ folder; the writer creates it.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    logger.info("Saved %d rows to %s", len(df), filepath)

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

import json
import logging
from pathlib import Path
from typing import Any

import joblib
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
        raise FileNotFoundError(f"File not found: {filepath}. Check the path or create the file.")
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


def save_json(data: dict, filepath: Path) -> None:
    """Write a dictionary to a readable JSON file, creating missing parent folders.

    Inputs:
        data: the dictionary to save (numbers, text, lists and nested dictionaries).
        filepath: where to write it, for example reports/metrics.json.
    Outputs:
        None. The file is written to disk with a 2-space indent.
    Why:
        The evaluation step saves its metrics here, so a person and main.py can read them.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Saved JSON to %s", filepath)


def save_model(model: Any, filepath: Path) -> None:
    """Save a trained model to disk with joblib, creating missing parent folders.

    Inputs:
        model: any trained object joblib can store, for example a scikit-learn pipeline.
        filepath: where to write it, for example models/model.joblib.
    Outputs:
        None. The model file is written to disk.
    Why:
        Training and inference run at different times; the saved file is the hand-off between them.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    logger.info("Saved model to %s", filepath)


def load_model(filepath: Path) -> Any:
    """Load a model saved by save_model.

    Inputs:
        filepath: path of the model file.
    Outputs:
        The model object, ready to call predict on.
    Why:
        Inference must use the exact model that training saved, and fail clearly if it is missing.

    Raises:
        FileNotFoundError: if the file does not exist (train a model first).
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}. Train and save a model first.")
    model = joblib.load(filepath)
    logger.info("Loaded model from %s", filepath)
    return model

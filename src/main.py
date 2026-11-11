"""
main.py  [Step 0: ORCHESTRATOR]
================================
Purpose   Run the whole pipeline, in order, with one command: python -m src.main
Owns      Loading config.yaml once, logging set-up, the date-based split, passing data between steps
Never     Cleaning, feature or model logic. main.py only CALLS other modules

Flow (filled in by issue "Module: main.py runs the pipeline end to end")
    config.yaml -> load_data -> clean_data -> validate -> to_wide -> features
      -> portfolio (decision table) -> _split_by_date (train / validation / test sealed)
      -> train -> evaluate + quality gate (validation) -> save model -> infer -> reports/

Why it matters
- Every setting comes from config.yaml, read once here and passed to modules as arguments
- Split by DATE before anything is fitted, so no future prices leak into training
- Handled errors stop the run with exit code 1 and a message saying what to fix

Tests     tests/test_main.py
"""

import logging
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
REQUIRED_SECTIONS = (
    "paths", "data", "portfolio", "problem", "split", "features",
    "model", "quality_gate", "fallback", "costs", "logging",
)
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def load_config(path: Path) -> dict:
    """Read config.yaml once and fail fast if a section is missing.

    Inputs:  path, the YAML file (normally CONFIG_PATH).
    Outputs: the settings as a dictionary.
    Why:     one source of truth for every setting; a missing section stops the run before any work.
    Raises:  FileNotFoundError if the file is missing; ValueError if it is not a mapping;
             KeyError naming every missing section.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}. Run from the repository root.")
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} must contain a YAML mapping of sections.")
    missing = [section for section in REQUIRED_SECTIONS if section not in cfg]
    if missing:
        raise KeyError(f"config.yaml is missing sections: {missing}")
    return cfg


def configure_logging(level: str, log_file: Path) -> None:
    """Send every module's log lines to the terminal and to one log file.

    Inputs:  level, e.g. "INFO" (from config.yaml); log_file, e.g. logs/pipeline.log.
    Outputs: None. The root logger gets a console and a file handler with one format.
    Why:     modules only call logging.getLogger(__name__); where lines go is decided once, here.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level.upper(),
        format=LOG_FORMAT,
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")],
        force=True,
    )


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float, val_size: float,
               random_state: int, stratify: bool) -> tuple:
    """Course contract: random split into train, validation and test.

    Inputs:  X, y; test_size and val_size as shares of the WHOLE dataset; random_state; stratify
             (keep class shares in every split, classification only).
    Outputs: (X_train, X_val, X_test, y_train, y_val, y_test).
    Why:     kept and tested for the course contract. Our pipeline uses _split_by_date instead,
             because a random split of weekly decisions would leak future prices (TEAM_BRIEF Decisions).
    Raises:  ValueError for sizes outside (0, 1), sizes that sum to 1 or more, or X and y of different lengths.
    """
    if not (0 < test_size < 1 and 0 < val_size < 1) or test_size + val_size >= 1:
        raise ValueError(
            f"test_size ({test_size}) and val_size ({val_size}) must each be between 0 and 1 "
            "and sum to less than 1."
        )
    if len(X) != len(y):
        raise ValueError(f"X and y must have the same number of rows, got {len(X)} and {len(y)}.")
    # Exact row counts, not fractions: 0.15 / 0.85 * 85 is 15.000000000000002, which sklearn rounds up to 16.
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=round(test_size * len(X)), random_state=random_state,
        stratify=y if stratify else None,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=round(val_size * len(X)), random_state=random_state,
        stratify=y_rest if stratify else None,
    )
    logger.info("Split %d rows into train %d, validation %d, test %d",
                len(X), len(X_train), len(X_val), len(X_test))
    return X_train, X_val, X_test, y_train, y_val, y_test


def _split_by_date(table: pd.DataFrame, train_end: str, val_end: str) -> tuple:
    """Split the decision table by date: train up to train_end, validation up to val_end, test after.

    Inputs:  table indexed by decision date (DatetimeIndex); train_end and val_end as YYYY-MM-DD,
             both inclusive.
    Outputs: (train, validation, test) DataFrames; every row is in exactly one of them.
    Why:     the model is judged on later years than it learned from, as it would be in operation.
    Raises:  ValueError if the index is not dates, val_end is not after train_end, or a split is empty.
    """
    if not isinstance(table.index, pd.DatetimeIndex):
        raise ValueError("_split_by_date needs a table indexed by date (DatetimeIndex).")
    train_cut, val_cut = pd.Timestamp(train_end), pd.Timestamp(val_end)
    if val_cut <= train_cut:
        raise ValueError(f"val_end ({val_end}) must be after train_end ({train_end}) in config.yaml.")
    train = table[table.index <= train_cut]
    val = table[(table.index > train_cut) & (table.index <= val_cut)]
    test = table[table.index > val_cut]
    for name, part in (("train", train), ("validation", val), ("test", test)):
        if part.empty:
            raise ValueError(f"The {name} split is empty. Check the split dates in config.yaml.")
    logger.info("Date split: train %d, validation %d, test %d rows", len(train), len(val), len(test))
    return train, val, test


def main() -> None:
    """Course contract entry point: python -m src.main.

    Inputs:  none (reads CONFIG_PATH).
    Outputs: none. Stage 2 steps are added by the main.py module issue; for now it loads the
             config and sets up logging, proving the entry point and error handling work.
    Why:     one command that CI, teammates and the grader all run the same way.
    """
    try:
        cfg = load_config(CONFIG_PATH)
        configure_logging(cfg["logging"]["level"], PROJECT_ROOT / cfg["paths"]["log_file"])
        logger.info("Config loaded from %s: %d sections", CONFIG_PATH, len(cfg))
    except (ValueError, OSError, KeyError) as exc:
        logger.error("Pipeline stopped: %s. Next: check config.yaml and the input files.", exc)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()

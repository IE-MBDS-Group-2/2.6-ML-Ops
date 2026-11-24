"""
split.py  [Step 4: SPLIT]
=========================
Purpose   Cut the data into train, validation and test
Input     A table (or X and y) and the split settings
Output    The parts; every row is in exactly one of them
Called by main.py          Next   train.py

Why it matters
- The pipeline splits by DATE: a random split of weekly decisions would leak future prices into training
- The test years stay sealed: they are split off here and only read once, at the release decision
- split_data (random) is kept for the course contract; the pipeline does not use it

Tests     tests/test_split.py
"""
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float, val_size: float,
               random_state: int, stratify: bool) -> tuple:
    """Course contract: random split into train, validation and test.

    Inputs:  X, y; test_size and val_size as shares of the WHOLE dataset; random_state;
             stratify (keep class shares in every split, classification only).
    Outputs: (X_train, X_val, X_test, y_train, y_val, y_test).
    Why:     kept and tested for the course contract. Our pipeline uses split_by_date instead,
             because a random split of weekly decisions would leak future prices
             (TEAM_BRIEF Decisions).
    Raises:  ValueError for sizes outside (0, 1), sizes that sum to 1 or more, or X and y of
             different lengths.
    """
    if not (0 < test_size < 1 and 0 < val_size < 1) or test_size + val_size >= 1:
        raise ValueError(
            f"test_size ({test_size}) and val_size ({val_size}) must each be between 0 and 1 "
            "and sum to less than 1."
        )
    if len(X) != len(y):
        raise ValueError(f"X and y must have the same number of rows, got {len(X)} and {len(y)}.")
    # Exact row counts, not fractions: 0.15 / 0.85 * 85 is 15.000000000000002, which sklearn
    # rounds up to 16.
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


def split_by_date(table: pd.DataFrame, train_end: str, val_end: str) -> tuple:
    """Split the decision table by date: train to train_end, validation to val_end, test after.

    Inputs:  table indexed by decision date (DatetimeIndex); train_end and val_end as YYYY-MM-DD,
             both inclusive.
    Outputs: (train, validation, test) DataFrames; every row is in exactly one of them.
    Why:     the model is judged on later years than it learned from, as it would be in operation.
    Raises:  ValueError if the index is not dates, val_end is not after train_end, or a split
             is empty.
    """
    if not isinstance(table.index, pd.DatetimeIndex):
        raise ValueError("split_by_date needs a table indexed by date (DatetimeIndex).")
    train_cut, val_cut = pd.Timestamp(train_end), pd.Timestamp(val_end)
    if val_cut <= train_cut:
        raise ValueError(
            f"val_end ({val_end}) must be after train_end ({train_end}) in config.yaml.")
    train_rows = table[table.index <= train_cut]
    val_rows = table[(table.index > train_cut) & (table.index <= val_cut)]
    test_rows = table[table.index > val_cut]
    for name, part in (("train", train_rows), ("validation", val_rows), ("test", test_rows)):
        if part.empty:
            raise ValueError(f"The {name} split is empty. Check the split dates in config.yaml.")
    logger.info("Date split: train %d, validation %d, test %d rows",
                len(train_rows), len(val_rows), len(test_rows))
    return train_rows, val_rows, test_rows


def up_to_date(data: pd.DataFrame | tuple, last_date: str) -> pd.DataFrame | tuple:
    """Keep the rows up to and including a date, in one table or in every table of a tuple.

    Inputs:  data, a table indexed by date or a tuple of such tables (the market frames); last_date, YYYY-MM-DD.
    Outputs: the same kind of object with no row after last_date.
    Why:     a step that must not see later years (selection on the train years, robustness on the history)
             gets its data cut in one named place, so the cut is never re-written inline.
    """
    if isinstance(data, tuple):
        return tuple(frame.loc[:last_date] for frame in data)
    return data.loc[:last_date]

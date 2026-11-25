"""Tests for src/split.py: the random split (course contract), the split by date and the date cut."""

import pandas as pd
import pytest

from src import split


def _make_xy(n: int = 100) -> tuple[pd.DataFrame, pd.Series]:
    """n rows with a balanced binary label."""
    X = pd.DataFrame({"a": range(n), "b": range(n, 2 * n)})
    y = pd.Series([0, 1] * (n // 2))
    return X, y


def _decision_table() -> pd.DataFrame:
    """Weekly rows from 2018 to 2021."""
    dates = pd.date_range("2018-01-05", "2021-12-31", freq="W-FRI")
    return pd.DataFrame({"x": range(len(dates))}, index=dates)


def test_unit_split_data_sizes_and_no_overlap():
    """100 rows at 0.15 / 0.15 of the whole dataset give 70 / 15 / 15 with no shared rows."""
    X, y = _make_xy()
    X_train, X_val, X_test, y_train, y_val, y_test = split.split_data(
        X, y, 0.15, 0.15, 42, stratify=False)
    assert (len(X_train), len(X_val), len(X_test)) == (70, 15, 15)
    assert len(y_train) + len(y_val) + len(y_test) == 100
    assert not set(X_train.index) & set(X_val.index)
    assert not set(X_train.index) & set(X_test.index)
    assert not set(X_val.index) & set(X_test.index)


def test_unit_split_data_stratify_keeps_class_share():
    """With stratify, every split keeps the 50/50 class share."""
    X, y = _make_xy()
    _, _, _, y_train, y_val, y_test = split.split_data(X, y, 0.2, 0.2, 0, stratify=True)
    for part in (y_train, y_val, y_test):
        assert part.mean() == pytest.approx(0.5)


@pytest.mark.parametrize("test_size, val_size", [(0.0, 0.2), (0.2, 1.0), (0.5, 0.5)])
def test_unit_split_data_bad_sizes_raise(test_size, val_size):
    """Sizes outside (0, 1), or summing to 1 or more, are rejected."""
    X, y = _make_xy()
    with pytest.raises(ValueError, match="sum to less than 1"):
        split.split_data(X, y, test_size, val_size, 0, stratify=False)


def test_unit_split_data_mismatched_lengths_raise():
    """X and y of different lengths are rejected."""
    X, y = _make_xy()
    with pytest.raises(ValueError, match="same number of rows"):
        split.split_data(X, y.iloc[:-1], 0.2, 0.2, 0, stratify=False)


def test_unitsplit_by_date_boundaries_inclusive():
    """train_end and val_end are inclusive; every row lands in exactly one split."""
    table = _decision_table()
    train, val, test = split.split_by_date(table, "2019-12-27", "2020-12-25")
    assert train.index.max() == pd.Timestamp("2019-12-27")
    assert val.index.min() == pd.Timestamp("2020-01-03")
    assert val.index.max() == pd.Timestamp("2020-12-25")
    assert test.index.min() > pd.Timestamp("2020-12-25")
    assert len(train) + len(val) + len(test) == len(table)


def test_unitsplit_by_date_requires_dates():
    """A table without a DatetimeIndex is rejected."""
    with pytest.raises(ValueError, match="DatetimeIndex"):
        split.split_by_date(pd.DataFrame({"x": [1, 2]}), "2019-12-31", "2021-12-31")


def test_unitsplit_by_date_order_and_empty_raise():
    """val_end must follow train_end, and no split may be empty."""
    table = _decision_table()
    with pytest.raises(ValueError, match="must be after"):
        split.split_by_date(table, "2020-12-31", "2019-12-31")
    with pytest.raises(ValueError, match="test split is empty"):
        split.split_by_date(table, "2019-12-31", "2030-12-31")


def test_unit_up_to_date_cuts_a_table_and_every_table_of_a_tuple():
    """The cut date is inclusive, for one table and for a tuple of tables."""
    table = _decision_table()
    cut = split.up_to_date(table, "2019-12-27")
    assert cut.index.max() == pd.Timestamp("2019-12-27")
    first, second = split.up_to_date((table, table * 2), "2018-03-02")
    assert first.index.max() == second.index.max() == pd.Timestamp("2018-03-02")
    assert second["x"].iloc[0] == 0 and second["x"].iloc[1] == 2

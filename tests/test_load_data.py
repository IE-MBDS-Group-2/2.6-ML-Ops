"""Tests for src/load_data.py: load_raw_data reads the raw CSV and fails fast on a bad path."""

import pandas as pd
import pytest

from src import load_data


def _make_df() -> pd.DataFrame:
    """Tiny raw table, 3 rows and 3 columns, independent of the chosen dataset."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "feature": [0.5, 1.5, 2.5],
            "target": [0, 1, 0],
        }
    )


def test_unit_load_raw_data_returns_expected_shape(tmp_path):
    """A valid CSV loads with the same rows and columns that were written."""
    path = tmp_path / "raw.csv"
    _make_df().to_csv(path, index=False)

    df = load_data.load_raw_data(path)

    assert df.shape == (3, 3)
    assert list(df.columns) == ["id", "feature", "target"]


def test_unit_load_raw_data_missing_file_raises(tmp_path):
    """A path that does not exist stops the run and names the file."""
    with pytest.raises(FileNotFoundError, match="raw.csv"):
        load_data.load_raw_data(tmp_path / "raw.csv")


def test_unit_load_raw_data_folder_raises(tmp_path):
    """A folder instead of a file stops the run before pandas tries to read it."""
    with pytest.raises(IsADirectoryError):
        load_data.load_raw_data(tmp_path)


def test_unit_load_raw_data_empty_file_raises(tmp_path):
    """A zero-byte file is rejected with a clear message, not a pandas error."""
    path = tmp_path / "raw.csv"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="is empty"):
        load_data.load_raw_data(path)


def test_unit_load_raw_data_header_only_raises(tmp_path):
    """A file with column names but no rows is rejected: there is nothing to train on."""
    path = tmp_path / "raw.csv"
    path.write_text("id,feature,target\n", encoding="utf-8")

    with pytest.raises(ValueError, match="has no rows"):
        load_data.load_raw_data(path)


def test_unit_load_raw_data_leaves_file_unchanged(tmp_path):
    """Loading is read only: the raw file on disk is byte-for-byte the same afterwards."""
    path = tmp_path / "raw.csv"
    _make_df().to_csv(path, index=False)
    before = path.read_bytes()

    load_data.load_raw_data(path)

    assert path.read_bytes() == before


def test_integration_load_raw_data_reads_file_from_utils_save_csv(tmp_path):
    """The hand-off with utils works: a file written by save_csv loads back unchanged."""
    from src import utils

    path = tmp_path / "raw.csv"
    utils.save_csv(_make_df(), path)

    df = load_data.load_raw_data(path)

    pd.testing.assert_frame_equal(df, _make_df())

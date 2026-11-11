"""Tests for src/utils.py: generic file helpers to read and write CSV, model and JSON files."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src import utils


def _make_df() -> pd.DataFrame:
    """Tiny table, 3 rows and 3 columns, independent of the chosen dataset."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "feature": [0.5, 1.5, 2.5],
            "target": [0, 1, 0],
        }
    )


def test_unit_save_csv_then_load_csv_round_trip(tmp_path):
    """A table written with save_csv comes back identical from load_csv."""
    path = tmp_path / "table.csv"

    utils.save_csv(_make_df(), path)
    loaded = utils.load_csv(path)

    pd.testing.assert_frame_equal(loaded, _make_df())


def test_unit_load_missing_file_raises(tmp_path):
    """Reading a CSV or a model that does not exist stops the run and names the file."""
    with pytest.raises(FileNotFoundError, match="table.csv"):
        utils.load_csv(tmp_path / "table.csv")

    with pytest.raises(FileNotFoundError, match="model.joblib"):
        utils.load_model(tmp_path / "model.joblib")


def test_unit_save_csv_and_save_json_create_missing_folders(tmp_path):
    """Writers create the parent folders themselves, so a first run on a clean checkout works."""
    csv_path = tmp_path / "data" / "raw" / "table.csv"
    json_path = tmp_path / "reports" / "metrics.json"

    utils.save_csv(_make_df(), csv_path)
    utils.save_json({"metric": 0.25}, json_path)

    assert csv_path.is_file()
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"metric": 0.25}


def test_unit_save_csv_writes_no_index_column(tmp_path):
    """The row numbers are not saved: 3 columns written means 3 columns read back."""
    path = tmp_path / "table.csv"

    utils.save_csv(_make_df(), path)

    assert path.read_text(encoding="utf-8").splitlines()[0] == "id,feature,target"
    assert utils.load_csv(path).shape[1] == 3


def test_integration_save_model_then_load_model_gives_same_predictions(tmp_path):
    """A real scikit-learn model saved to disk and reloaded predicts exactly as before."""
    features = np.array([[0.0], [1.0], [2.0], [3.0]])
    labels = np.array([0, 0, 1, 1])
    model = LogisticRegression().fit(features, labels)
    path = tmp_path / "models" / "model.joblib"

    utils.save_model(model, path)
    reloaded = utils.load_model(path)

    np.testing.assert_array_equal(reloaded.predict(features), model.predict(features))

"""Tests for src/train.py: one fitted pipeline (preprocessing + LogisticRegression), fail-fast inputs.

Tiny synthetic data only; the preprocessor is built inside the tests so they do not wait for features.py.
"""
import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from src import train

COLUMNS = ["drift_SPY", "drawdown"]
MODEL_PARAMS = {"max_iter": 1000, "class_weight": "balanced"}


def make_preprocessor() -> ColumnTransformer:
    """Same recipe as features.get_feature_preprocessor: median impute, then scale."""
    steps = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    return ColumnTransformer([("numeric", steps, COLUMNS)])


def make_data(rows_per_class: int = 10) -> tuple[pd.DataFrame, pd.Series]:
    """Three well separated clusters, one per action (0.0, 0.5, 1.0)."""
    rng = np.random.default_rng(0)
    centres = {0.0: (-3.0, 0.0), 0.5: (0.0, 3.0), 1.0: (3.0, 0.0)}
    frames, labels = [], []
    for action, (x, z) in centres.items():
        frames.append(pd.DataFrame({"drift_SPY": x + rng.normal(0, 0.3, rows_per_class),
                                    "drawdown": z + rng.normal(0, 0.3, rows_per_class)}))
        labels += [action] * rows_per_class
    return pd.concat(frames, ignore_index=True), pd.Series(labels, name="best_action")


def test_unit_train_model_returns_fitted_pipeline_with_one_prediction_per_row():
    X, y = make_data()
    model = train.train_model(X, y, make_preprocessor(), "classification", MODEL_PARAMS)
    assert isinstance(model, Pipeline)
    assert list(model.named_steps) == ["preprocess", "model"]
    check_is_fitted(model)
    assert len(model.predict(X)) == len(X)


def test_unit_train_model_known_value_separable_clusters_are_learned():
    # Known value: each cluster centre is predicted as its own action.
    X, y = make_data()
    model = train.train_model(X, y, make_preprocessor(), "classification", MODEL_PARAMS)
    centres = pd.DataFrame({"drift_SPY": [-3.0, 0.0, 3.0], "drawdown": [0.0, 3.0, 0.0]})
    assert list(model.predict(centres)) == [0.0, 0.5, 1.0]
    # The real action labels come back (sklearn alone rejects 0.5 as a "continuous" label).
    assert isinstance(model.named_steps["model"], LogisticRegression)
    assert list(model.named_steps["model"].classes_) == [0.0, 0.5, 1.0]
    assert model.predict_proba(centres).shape == (3, 3)


def test_unit_train_model_passes_config_params_and_seed_to_the_model():
    X, y = make_data()
    model = train.train_model(X, y, make_preprocessor(), "classification", MODEL_PARAMS, random_state=7)
    estimator = model.named_steps["model"]
    assert (estimator.max_iter, estimator.class_weight, estimator.random_state) == (1000, "balanced", 7)
    default = train.train_model(X, y, make_preprocessor(), "classification")  # model_params=None works
    assert default.named_steps["model"].random_state == 42


def test_unit_train_model_logs_class_counts(caplog):
    X, y = make_data(rows_per_class=4)
    with caplog.at_level(logging.INFO, logger="src.train"):
        train.train_model(X, y, make_preprocessor(), "classification", MODEL_PARAMS)
    assert "class counts" in caplog.text and "0.5: 4" in caplog.text


def test_unit_train_model_raises_on_bad_problem_type():
    X, y = make_data()
    with pytest.raises(ValueError, match="problem_type"):
        train.train_model(X, y, make_preprocessor(), "clustering", MODEL_PARAMS)


@pytest.mark.parametrize("cut_x, cut_y, message", [
    (slice(0, 0), slice(0, 0), "empty"),        # no rows at all
    (slice(None), slice(0, 5), "same number"),  # X and y of different lengths
])
def test_unit_train_model_raises_on_empty_or_mismatched_data(cut_x, cut_y, message):
    X, y = make_data()
    with pytest.raises(ValueError, match=message):
        train.train_model(X.iloc[cut_x], y.iloc[cut_y], make_preprocessor(), "classification", MODEL_PARAMS)


def test_unit_train_model_raises_when_random_state_is_given_twice():
    X, y = make_data()
    with pytest.raises(ValueError, match="random_state"):
        train.train_model(X, y, make_preprocessor(), "classification", {"random_state": 1})


def test_integration_preprocessor_then_train_then_saved_model_predicts_the_same(tmp_path):
    from src import features, utils  # pylint: disable=import-outside-toplevel

    X, y = make_data()
    model = train.train_model(X, y, features.get_feature_preprocessor(COLUMNS), "classification", MODEL_PARAMS)
    path = tmp_path / "models" / "model.joblib"
    utils.save_model(model, path)
    reloaded = utils.load_model(path)
    assert np.array_equal(model.predict(X), reloaded.predict(X))


def test_unit_train_model_regression_fits_ridge_and_predicts_floats():
    """problem_type "regression" picks Ridge from the settings and predicts one float per row."""
    X, _ = make_data()
    y = pd.Series(np.linspace(0.0, 1.0, len(X)), name="reward")
    model = train.train_model(X, y, make_preprocessor(), "regression")
    assert type(model.named_steps["model"]).__name__ == "Ridge"
    predictions = model.predict(X)
    assert len(predictions) == len(X)
    assert predictions.dtype.kind == "f"


def test_unit_train_model_leaves_the_callers_preprocessor_unfitted():
    X, y = make_data()
    preprocessor = make_preprocessor()
    model = train.train_model(X, y, preprocessor, "classification", MODEL_PARAMS)
    check_is_fitted(model)
    assert not hasattr(preprocessor, "transformers_")

"""Tests for src/infer.py: batch predictions and the fall back to today's rule."""

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src import infer

FRIDAYS = pd.to_datetime(["2021-01-08", "2021-01-15", "2021-01-22"])


def _prior_model() -> Pipeline:
    """Tiny fitted pipeline that always predicts action "1.0" with probability 0.75 (3 of its 4 training labels)."""
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    y = pd.Series(["0.0", "1.0", "1.0", "1.0"])
    return Pipeline([("model", DummyClassifier(strategy="prior"))]).fit(X, y)


def _X_infer() -> pd.DataFrame:
    """Three Friday rows indexed by date, as main.py will pass them."""
    return pd.DataFrame({"x": [0.5, 1.5, 2.5]}, index=FRIDAYS)


# ---------------------------------------------------------------- run_inference


def test_unit_run_inference_keeps_index_and_prediction_column():
    """Output rows line up with the input dates and carry only the agreed 'prediction' column."""
    preds = infer.run_inference(_prior_model(), _X_infer())
    assert preds.index.equals(FRIDAYS)
    assert list(preds.columns) == ["prediction"]
    assert preds["prediction"].tolist() == ["1.0", "1.0", "1.0"]


def test_unit_run_inference_with_proba_known_value():
    """With include_proba, 'probability' is the top class's probability: 3 of 4 training labels -> 0.75."""
    preds = infer.run_inference(_prior_model(), _X_infer(), include_proba=True)
    assert list(preds.columns) == ["prediction", "probability"]
    assert preds["probability"].tolist() == pytest.approx([0.75, 0.75, 0.75])


def test_unit_run_inference_does_not_modify_input():
    """Inference must never change the caller's data (no added columns, no changed values)."""
    X = _X_infer()
    before = X.copy()
    infer.run_inference(_prior_model(), X, include_proba=True)
    pd.testing.assert_frame_equal(X, before)


def test_unit_run_inference_empty_input_raises():
    """Nothing to predict means an upstream step failed: fail loudly."""
    with pytest.raises(ValueError, match="empty"):
        infer.run_inference(_prior_model(), pd.DataFrame({"x": []}))


def test_unit_run_inference_proba_on_model_without_proba_raises():
    """A model with no predict_proba cannot supply confidence, so the fallback could not work: say so clearly."""
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    model = Pipeline([("model", LinearSVC())]).fit(X, ["0.0", "0.0", "1.0", "1.0"])
    with pytest.raises(ValueError, match="predict_proba"):
        infer.run_inference(model, _X_infer(), include_proba=True)


# ---------------------------------------------------------------- apply_fallback


def _preds(probabilities: list[float]) -> pd.DataFrame:
    """Model output for the three Fridays: the model always says 'rebalance fully' (1.0)."""
    return pd.DataFrame({"prediction": [1.0, 1.0, 1.0], "probability": probabilities}, index=FRIDAYS)


RULE = pd.Series([0.0, 0.0, 0.5], index=FRIDAYS)  # today's rule: nothing, nothing, halfway
NOTHING_MISSING = pd.Series([False, False, False], index=FRIDAYS)


def test_unit_apply_fallback_low_probability_uses_rule():
    """Only the middle row is unsure (0.40 < 0.60): it gets today's rule action and fallback=True."""
    out = infer.apply_fallback(_preds([0.90, 0.40, 0.80]), RULE, NOTHING_MISSING, min_probability=0.60)
    assert out["recommendation"].tolist() == [1.0, 0.0, 1.0]
    assert out["fallback"].tolist() == [False, True, False]


def test_unit_apply_fallback_missing_input_uses_rule():
    """A confident model still falls back when one of its inputs was missing (last row)."""
    missing = pd.Series([False, False, True], index=FRIDAYS)
    out = infer.apply_fallback(_preds([0.90, 0.90, 0.90]), RULE, missing, min_probability=0.60)
    assert out["recommendation"].tolist() == [1.0, 1.0, 0.5]
    assert out["fallback"].tolist() == [False, False, True]


def test_unit_apply_fallback_confident_keeps_prediction():
    """Probabilities at or above the threshold (0.60 included) keep the model's prediction; input is not modified."""
    preds = _preds([0.60, 0.70, 0.99])
    before = preds.copy()
    out = infer.apply_fallback(preds, RULE, NOTHING_MISSING, min_probability=0.60)
    assert out["recommendation"].tolist() == [1.0, 1.0, 1.0]
    assert not out["fallback"].any()
    assert list(out.columns) == ["prediction", "probability", "recommendation", "fallback"]
    pd.testing.assert_frame_equal(preds, before)


def test_unit_apply_fallback_without_probability_column_raises():
    """No confidence column means the fallback cannot be judged: ask for include_proba=True."""
    preds = pd.DataFrame({"prediction": [1.0, 1.0, 1.0]}, index=FRIDAYS)
    with pytest.raises(ValueError, match="include_proba"):
        infer.apply_fallback(preds, RULE, NOTHING_MISSING, min_probability=0.60)


def test_unit_apply_fallback_misaligned_index_raises():
    """Rule actions for different dates would silently mix up weeks: fail loudly instead."""
    other_dates = pd.Series([0.0, 0.0, 0.5], index=pd.to_datetime(["2021-02-05", "2021-02-12", "2021-02-19"]))
    with pytest.raises(ValueError, match="same index"):
        infer.apply_fallback(_preds([0.9, 0.9, 0.9]), other_dates, NOTHING_MISSING, min_probability=0.60)


# ---------------------------------------------------------------- integration


@pytest.mark.skip(reason="Awaiting utils.py in dev (issue #14)")
def test_integration_saved_model_round_trip_then_run_inference(tmp_path):
    """A model saved and re-loaded with utils gives the same predictions as the original."""
    from src import utils  # pylint: disable=import-outside-toplevel

    model = _prior_model()
    path = tmp_path / "model.joblib"
    utils.save_model(model, path)
    loaded = utils.load_model(path)
    pd.testing.assert_frame_equal(infer.run_inference(loaded, _X_infer(), include_proba=True),
                                  infer.run_inference(model, _X_infer(), include_proba=True))

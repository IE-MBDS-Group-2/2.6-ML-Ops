"""Synthetic checks for Farah's market features and training-only recipe."""
import numpy as np
import pandas as pd
import pytest
from sklearn import exceptions
from sklearn.utils import validation

from src import features


def prices():
    return pd.DataFrame({"A": np.linspace(100, 200, 61), "B": np.linspace(50, 100, 61)},
                        index=pd.date_range("2020-01-01", periods=61, freq="B"))


def test_unit_return_known_value_and_columns():
    frame = prices()
    result = features.compute_market_features(frame, [4, 12], 20)
    assert result.loc[frame.index[-1], "ret_12w_A"] == pytest.approx(1.0)
    assert list(result.columns) == ["ret_4w_A", "ret_12w_A", "vol_20d_A",
                                    "ret_4w_B", "ret_12w_B", "vol_20d_B"]
    pd.testing.assert_index_equal(result.index, frame.index)
    assert result["ret_12w_A"].iloc[:60].isna().all()
    assert result["vol_20d_A"].iloc[:20].isna().all()


def test_unit_volatility_known_value():
    frame = prices().iloc[:3].copy()
    frame["A"] = [100, 110, 99]
    result = features.compute_market_features(frame, [1], 2)
    assert result["vol_2d_A"].iloc[-1] == pytest.approx(np.sqrt(0.02 * 252))


def test_unit_no_lookahead_or_input_mutation():
    frame = prices()
    before = frame.copy(deep=True)
    expected = features.compute_market_features(frame, [1, 4], 5)
    changed = frame.copy()
    changed.iloc[31:] *= 10
    actual = features.compute_market_features(changed, [1, 4], 5)
    pd.testing.assert_frame_equal(actual.iloc[:31], expected.iloc[:31])
    pd.testing.assert_frame_equal(frame, before)


def test_unit_missing_prices_are_not_filled():
    frame = prices()
    frame.iloc[10, 0] = np.nan
    result = features.compute_market_features(frame, [1], 2)
    assert np.isnan(result["ret_1w_A"].iloc[10])
    assert np.isnan(result["ret_1w_A"].iloc[15])
    assert result["vol_2d_A"].iloc[10:13].isna().all()


@pytest.mark.parametrize("weeks,days", [([], 2), ([0], 2), ([-1], 2), ([1.5], 2),
                                        ([True], 2), ([1, 1], 2), ([1], 0), ([1], 1),
                                        ([1], 2.5), ([1], True)])
def test_unit_invalid_windows(weeks, days):
    with pytest.raises(ValueError, match="window|vol_days|return_weeks"):
        features.compute_market_features(prices(), weeks, days)


@pytest.mark.parametrize("kind", ["empty", "no_columns", "dates", "duplicate_dates", "unsorted",
                                  "duplicate_columns", "column_names", "text", "zero", "negative", "infinite"])
def test_unit_invalid_prices(kind):
    frame = prices()
    if kind == "empty":
        frame = frame.iloc[:0]
    elif kind == "no_columns":
        frame = frame.iloc[:, :0]
    elif kind == "dates":
        frame = frame.reset_index(drop=True)
    elif kind == "duplicate_dates":
        frame.index = [frame.index[0]] * len(frame)
    elif kind == "unsorted":
        frame = frame.iloc[::-1]
    elif kind == "duplicate_columns":
        frame.columns = ["A", "A"]
    elif kind == "column_names":
        frame.columns = ["", "B"]
    elif kind == "text":
        frame["A"] = "text"
    else:
        frame.iloc[0, 0] = {"zero": 0, "negative": -1, "infinite": np.inf}[kind]
    with pytest.raises(ValueError):
        features.compute_market_features(frame, [1], 2)


@pytest.mark.parametrize("columns", [[], [""], ["A", "A"], [1], ["A", None]])
def test_unit_invalid_preprocessor_columns(columns):
    with pytest.raises(ValueError, match="numeric_columns"):
        features.get_feature_preprocessor(columns)


def test_unit_preprocessor_is_unfitted():
    recipe = features.get_feature_preprocessor(["A"])
    with pytest.raises(exceptions.NotFittedError):
        validation.check_is_fitted(recipe)


def test_integration_preprocessor_training_only_and_column_width():
    recipe = features.get_feature_preprocessor(["A", "B", "empty"])
    train = pd.DataFrame({"A": [1., np.nan, 3.], "B": [2., 4., 6.], "empty": [np.nan] * 3,
                          "ignored": [10, 20, 30]})
    output = recipe.fit_transform(train)
    assert output.shape == (3, 3)
    assert np.isfinite(output).all()
    np.testing.assert_allclose(output.mean(axis=0), 0, atol=1e-12)
    steps = recipe.named_transformers_["numeric"].named_steps
    medians = steps["imputer"].statistics_.copy()
    means = steps["scaler"].mean_.copy()
    transformed = recipe.transform(pd.DataFrame({"A": [1000.], "B": [np.nan], "empty": [np.nan]}))
    assert transformed.shape == (1, 3)
    assert np.isfinite(transformed).all()
    np.testing.assert_array_equal(steps["imputer"].statistics_, medians)
    np.testing.assert_array_equal(steps["scaler"].mean_, means)

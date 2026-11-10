"""Tests for src/evaluate.py: model metrics, portfolio metrics and the quality gate."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from src import evaluate

# ---------------------------------------------------------------- portfolio_metrics


def test_unit_portfolio_metrics_known_max_drawdown():
    """Peak 120 falls to 90: the worst peak-to-trough loss is 90 / 120 - 1 = -0.25."""
    values = pd.Series([100.0, 120.0, 90.0, 110.0])
    costs = pd.Series([0.0, 0.0, 0.0, 0.0])
    result = evaluate.portfolio_metrics(values, costs)
    assert result["max_drawdown"] == pytest.approx(-0.25)


def test_unit_portfolio_metrics_one_year_known_values():
    """253 daily values = 252 returns = exactly 1 year, so CAGR equals the total return and cost per year the sum."""
    values = pd.Series(np.linspace(1.0, 1.10, 253))
    costs = pd.Series([0.0] * 252 + [0.002])
    result = evaluate.portfolio_metrics(values, costs)
    assert result["annual_return"] == pytest.approx(0.10)
    assert result["annual_trading_cost"] == pytest.approx(0.002)
    assert result["sharpe"] == pytest.approx(result["annual_return"] / result["annual_volatility"])


def test_unit_portfolio_metrics_returns_plain_floats_with_expected_keys():
    """main.py and the logs need plain Python floats under fixed names."""
    values = pd.Series([1.0, 1.01, 0.99, 1.02])
    costs = pd.Series([0.0, 0.001, 0.0, 0.0])
    result = evaluate.portfolio_metrics(values, costs)
    expected = {"max_drawdown", "annual_return", "annual_volatility", "sharpe", "annual_trading_cost"}
    assert set(result) == expected
    assert all(isinstance(v, float) for v in result.values())


def test_unit_portfolio_metrics_flat_values_give_zero_sharpe():
    """A flat portfolio has zero volatility: Sharpe is reported as 0.0 instead of dividing by zero."""
    values = pd.Series([1.0, 1.0, 1.0])
    costs = pd.Series([0.0, 0.0, 0.0])
    result = evaluate.portfolio_metrics(values, costs)
    assert result["annual_volatility"] == 0.0
    assert result["sharpe"] == 0.0


def test_unit_portfolio_metrics_too_few_values_raises():
    """One value has no return to measure: fail loudly."""
    with pytest.raises(ValueError, match="at least 2"):
        evaluate.portfolio_metrics(pd.Series([1.0]), pd.Series([0.0]))


# ---------------------------------------------------------------- max_drawdown_reduction


def test_unit_max_drawdown_reduction_known_value():
    """Agent loses 16% at worst, today's rule 20%: the agent cut the drawdown by 1 - 0.16 / 0.20 = 0.20."""
    assert evaluate.max_drawdown_reduction(-0.16, -0.20) == pytest.approx(0.20)


def test_unit_max_drawdown_reduction_worse_agent_is_negative():
    """An agent with a deeper drawdown than the baseline gets a negative reduction (it made things worse)."""
    assert evaluate.max_drawdown_reduction(-0.30, -0.20) == pytest.approx(-0.50)


def test_unit_max_drawdown_reduction_zero_baseline_raises():
    """A baseline that never lost money cannot be divided by: fail loudly instead of returning inf."""
    with pytest.raises(ValueError, match="baseline"):
        evaluate.max_drawdown_reduction(-0.10, 0.0)


# ---------------------------------------------------------------- check_gate


def test_unit_check_gate_greater_equal_at_threshold_passes():
    """Exactly on the threshold counts as a pass for '>='."""
    assert evaluate.check_gate(0.20, ">=", 0.20) is True


def test_unit_check_gate_greater_equal_below_threshold_fails():
    """Just under the threshold is a fail for '>='."""
    assert evaluate.check_gate(0.19, ">=", 0.20) is False


def test_unit_check_gate_less_equal():
    """'<=' works the other way round (useful for cost-type metrics)."""
    assert evaluate.check_gate(0.01, "<=", 0.02) is True
    assert evaluate.check_gate(0.03, "<=", 0.02) is False


def test_unit_check_gate_passes_reduction_hit_exactly_despite_float_rounding():
    """1 - 0.16 / 0.20 is 0.19999999999999996 in floating point; reaching the target exactly must still pass."""
    assert evaluate.check_gate(evaluate.max_drawdown_reduction(-0.16, -0.20), ">=", 0.20) is True


def test_unit_check_gate_unknown_comparison_raises():
    """A typo in config.yaml such as '>' or '=>' must stop the pipeline, not silently pass or fail."""
    with pytest.raises(ValueError, match="comparison"):
        evaluate.check_gate(0.20, ">", 0.20)


# ---------------------------------------------------------------- evaluate_model


def _always_one_model() -> Pipeline:
    """Tiny fitted pipeline that always predicts action 1.0, so its scores are known in advance."""
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    y = pd.Series([0.0, 0.5, 1.0, 1.0])
    return Pipeline([("model", DummyClassifier(strategy="constant", constant=np.array([1.0])))]).fit(X, y)


def test_unit_evaluate_model_known_values():
    """Model says 1.0 every time; 2 of 4 labels are 1.0, so accuracy = 0.5.
    F1 per class: 1.0 -> 2/3, 0.0 and 0.5 -> 0, so f1_macro = (2/3) / 3 = 0.2222."""
    X_eval = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    y_eval = pd.Series([0.0, 0.5, 1.0, 1.0])
    result = evaluate.evaluate_model(_always_one_model(), X_eval, y_eval, "classification")
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["f1_macro"] == pytest.approx(2 / 9)


def test_unit_evaluate_model_returns_plain_floats_with_expected_keys():
    """Exactly the agreed keys, as plain Python floats (not numpy types), so they log and compare cleanly."""
    X_eval = pd.DataFrame({"x": [0.0, 1.0]})
    y_eval = pd.Series([1.0, 1.0])
    result = evaluate.evaluate_model(_always_one_model(), X_eval, y_eval, "classification")
    assert set(result) == {"accuracy", "f1_macro"}
    assert all(type(v) is float for v in result.values())


def test_unit_evaluate_model_empty_input_raises():
    """Scoring zero rows would give a meaningless number: fail loudly."""
    with pytest.raises(ValueError, match="empty"):
        evaluate.evaluate_model(_always_one_model(), pd.DataFrame({"x": []}), pd.Series([], dtype=float),
                                "classification")


def test_unit_evaluate_model_mismatched_lengths_raises():
    """3 feature rows but 2 labels means the split went wrong upstream."""
    with pytest.raises(ValueError, match="same number of rows"):
        evaluate.evaluate_model(_always_one_model(), pd.DataFrame({"x": [0.0, 1.0, 2.0]}), pd.Series([1.0, 1.0]),
                                "classification")


def test_unit_evaluate_model_unknown_problem_type_raises():
    """Stage 2 is a classifier; any other problem_type is a config error."""
    with pytest.raises(ValueError, match="problem_type"):
        evaluate.evaluate_model(_always_one_model(), pd.DataFrame({"x": [0.0]}), pd.Series([1.0]), "regression")


@pytest.mark.skip(reason="Awaiting train.py in dev (issue #22)")
def test_integration_train_model_scored_by_evaluate_model():
    """A pipeline fitted by train.train_model can be scored by evaluate_model end to end."""
    from src import train  # pylint: disable=import-outside-toplevel

    X = pd.DataFrame({"x": np.arange(30, dtype=float)})
    y = pd.Series([0.0, 0.5, 1.0] * 10)
    model = train.train_model(X, y)
    result = evaluate.evaluate_model(model, X, y, "classification")
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["f1_macro"] <= 1.0

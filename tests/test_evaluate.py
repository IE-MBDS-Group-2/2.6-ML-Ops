"""Tests for src/evaluate.py: model metrics, portfolio metrics and the quality gate."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.pipeline import Pipeline

from src import evaluate, features, train

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
    """Stage 2 is a classifier; any other problem_type (e.g. clustering) is a config error."""
    with pytest.raises(ValueError, match="problem_type"):
        evaluate.evaluate_model(_always_one_model(), pd.DataFrame({"x": [0.0]}), pd.Series([1.0]), "clustering")


def test_integration_train_model_scored_by_evaluate_model():
    """A pipeline fitted by train.train_model can be scored by evaluate_model end to end."""
    X = pd.DataFrame({"x": np.arange(30, dtype=float)})
    y = pd.Series([0.0, 0.5, 1.0] * 10)
    model = train.train_model(X, y, features.get_feature_preprocessor(["x"]), "classification")
    result = evaluate.evaluate_model(model, X, y, "classification")
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["f1_macro"] <= 1.0


def test_unit_evaluate_model_regression_known_values():
    """A model predicting the mean of [0, 2] scores rmse 1.0, mae 1.0 and r2 0.0."""
    X = pd.DataFrame({"x": [0.0, 1.0]})
    y = pd.Series([0.0, 2.0])
    model = Pipeline([("model", DummyRegressor(strategy="mean"))]).fit(X, y)
    result = evaluate.evaluate_model(model, X, y, "regression")
    assert result == pytest.approx({"rmse": 1.0, "mae": 1.0, "r2": 0.0})
    assert all(isinstance(value, float) for value in result.values())


def test_unit_evaluate_model_without_predict_raises_type_error():
    """An object with no predict method is not a model: fail with a TypeError that says so."""
    with pytest.raises(TypeError, match="predict"):
        evaluate.evaluate_model(object(), pd.DataFrame({"x": [0.0]}), pd.Series([1.0]), "classification")


# ---------------------------------------------------------------- guards and the release decision (Stage 3, #64)

LIMITS = {"max_return_shortfall": 0.0025, "max_cost_ratio": 2.0, "max_tracking_error": 0.02}


def metrics(annual_return: float = 0.05, annual_trading_cost: float = 0.001, tracking_error: float = 0.01) -> dict:
    """The metric keys guard_report reads (portfolio_metrics plus tracking_error), with easy defaults."""
    return {"annual_return": annual_return, "annual_trading_cost": annual_trading_cost,
            "tracking_error": tracking_error}


def test_unit_tracking_error_is_zero_for_identical_paths():
    values = pd.Series([1.0, 1.02, 0.99, 1.05, 1.04])
    assert evaluate.tracking_error(values, values.copy(), 52) == 0.0


def test_unit_tracking_error_known_value():
    """Target flat, portfolio +1% then -1% each week: differences alternate +0.01 / -0.01."""
    differences = np.array([0.01, -0.01] * 4)
    portfolio = pd.Series(np.cumprod(np.r_[1.0, 1 + differences]))
    target = pd.Series(np.ones(len(portfolio)))
    expected = np.std(differences, ddof=1) * np.sqrt(52)
    assert evaluate.tracking_error(portfolio, target, 52) == pytest.approx(expected, abs=1e-6)


def test_unit_each_guard_passes_and_fails_at_its_limit():
    baseline = metrics(annual_return=0.05, annual_trading_cost=0.001)
    at_limit = evaluate.guard_report(metrics(annual_return=0.0475, annual_trading_cost=0.002, tracking_error=0.02),
                                     baseline, LIMITS)
    beyond = evaluate.guard_report(metrics(annual_return=0.0474, annual_trading_cost=0.00201,
                                           tracking_error=0.0201), baseline, LIMITS)
    for name in ("return_shortfall", "cost_ratio", "tracking_error"):
        assert at_limit[name]["passed"] is True, name
        assert beyond[name]["passed"] is False, name


def test_unit_guard_report_shape():
    report = evaluate.guard_report(metrics(annual_return=0.045, annual_trading_cost=0.003), metrics(), LIMITS)
    assert set(report) == {"return_shortfall", "cost_ratio", "tracking_error"}
    for entry in report.values():
        assert set(entry) == {"value", "limit", "passed"}
    assert report["return_shortfall"] == {"value": pytest.approx(0.005), "limit": 0.0025, "passed": False}
    assert report["cost_ratio"]["value"] == pytest.approx(3.0)


def test_unit_cost_ratio_when_the_baseline_trades_nothing():
    """No baseline cost: an agent that also trades nothing passes; any cost at all fails."""
    baseline = metrics(annual_trading_cost=0.0)
    assert evaluate.guard_report(metrics(annual_trading_cost=0.0), baseline, LIMITS)["cost_ratio"]["passed"] is True
    assert evaluate.guard_report(metrics(annual_trading_cost=1e-6), baseline, LIMITS)["cost_ratio"]["passed"] is False


def test_unit_release_decision_lists_every_failed_reason():
    report = evaluate.guard_report(metrics(annual_return=0.04, annual_trading_cost=0.005), metrics(), LIMITS)
    decision = evaluate.release_decision(True, report)
    assert decision["release"] is False
    assert len(decision["reasons"]) == 2
    assert any("return_shortfall" in reason for reason in decision["reasons"])
    assert any("cost_ratio" in reason for reason in decision["reasons"])
    all_passed = evaluate.release_decision(True, evaluate.guard_report(metrics(), metrics(), LIMITS))
    assert all_passed == {"release": True, "reasons": []}


def test_unit_release_decision_never_releases_a_failed_gate():
    decision = evaluate.release_decision(False, evaluate.guard_report(metrics(), metrics(), LIMITS))
    assert decision["release"] is False
    assert len(decision["reasons"]) == 1 and "gate" in decision["reasons"][0]


def test_unit_release_decision_accepts_an_extra_guard_entry():
    """#66 adds the selected simple rule as one more entry in the same shape; it must count like the others."""
    guards = evaluate.guard_report(metrics(), metrics(), LIMITS)
    guards["challenger"] = {"value": -0.15, "limit": -0.18, "passed": False}
    decision = evaluate.release_decision(True, guards)
    assert decision["release"] is False and "challenger" in decision["reasons"][0]


@pytest.mark.parametrize("call, message", [
    (lambda: evaluate.tracking_error(pd.Series([1.0, 1.1, 1.2]), pd.Series([1.0, 1.1]), 52), "same length"),
    (lambda: evaluate.tracking_error(pd.Series([], dtype=float), pd.Series([], dtype=float), 52), "at least 2"),
    (lambda: evaluate.guard_report(metrics(), metrics(), {"max_return_shortfall": 0.0025, "max_cost_ratio": 2.0}),
     "max_tracking_error"),
    (lambda: evaluate.guard_report(metrics(), metrics(), {**LIMITS, "max_tracking_error": None}),
     "max_tracking_error"),
    (lambda: evaluate.guard_report({"annual_return": 0.05}, metrics(), LIMITS), "annual_trading_cost"),
    (lambda: evaluate.release_decision(True, {"cost_ratio": {"value": 1.0, "limit": 2.0}}), "passed"),
], ids=["different lengths", "empty series", "missing limit", "limit not set (null)", "missing metric",
        "malformed guard entry"])
def test_unit_guards_raise(call, message):
    with pytest.raises(ValueError, match=message):
        call()

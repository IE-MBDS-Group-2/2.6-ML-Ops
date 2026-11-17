"""Tests for src/allocation.py: the target-weight and rebalancing choices, and the plain-words reason.

Tiny synthetic prices on business days and a tiny fitted model. Never the real data/ folder.
"""
import numpy as np
import pandas as pd
import pytest

from src import allocation, config, features, portfolio, train

TICKERS = ["SPY", "EFA", "IEF", "GLD", "BIL"]
POLICY = {"SPY": 0.40, "EFA": 0.20, "IEF": 0.30, "GLD": 0.05, "BIL": 0.05}
TARGET = pd.Series(POLICY, dtype=float)[TICKERS]
STATE_COLUMNS = [f"w_{t}" for t in TICKERS] + [f"drift_{t}" for t in TICKERS] + ["drawdown", "weeks_since_rebalance"]


def make_cfg(calendar: str = "quarterly", band: float = 5) -> dict:
    """The config sections allocation.py, rules.py and portfolio.py read, with the team's values."""
    return {
        "data": {"tickers": TICKERS},
        "target_weights": {"method": "policy", "policy": dict(POLICY)},
        "rebalancing": {"method": "smart", "review": "weekly", "calendar": calendar, "tolerance_band": band},
        "portfolio": {
            "actions": [0.0, 0.5, 1.0],
            "current_rule": {"drift_band": 0.05, "quarter_months": [1, 4, 7, 10]},
            "reward": {"penalty_lambda": 1.0},
        },
        "fallback": {"min_probability": 0.60},
        "costs": {"aum": 100_000_000, "model": "flat", "flat": {"rate": 0.001}, "slippage": {"adv_window": 20}},
        "problem": {"target_column": "best_action", "id_column": "decision_date"},
        "benchmarks": {"methods": ["min_variance", "max_sharpe", "mean_variance", "risk_parity"],
                       "risk_aversion": 3.0},
    }


def make_prices(periods: int = 120, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random-walk closes and constant volumes for the 5 ETFs on business days from 1 Jan 2024."""
    days = pd.bdate_range("2024-01-01", periods=periods, name="date")
    returns = np.random.default_rng(seed).normal(0.0003, 0.01, size=(periods, len(TICKERS)))
    close = pd.DataFrame(100 * np.cumprod(1 + returns, axis=0), index=days, columns=TICKERS)
    return close, pd.DataFrame(1_000_000.0, index=days, columns=TICKERS)


def state_row(date: str, shifts: dict | None = None) -> pd.Series:
    """A Friday state row as simulate builds it: weights = target + shifts, named by its decision date."""
    weights = TARGET + pd.Series(shifts or {}, dtype=float).reindex(TICKERS, fill_value=0.0)
    row = {f"w_{t}": weights[t] for t in TICKERS}
    row.update({f"drift_{t}": weights[t] - TARGET[t] for t in TICKERS})
    row.update({"drawdown": 0.0, "weeks_since_rebalance": 3.0})
    return pd.Series(row, name=pd.Timestamp(date))


def fitted_model():
    """A tiny Stage 2 model (train.train_model) fitted on random state rows with all three actions."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(0.0, 0.05, size=(60, len(STATE_COLUMNS))), columns=STATE_COLUMNS)
    y = pd.Series([0.0, 0.5, 1.0] * 20)
    params = {"max_iter": 1000, "solver": "lbfgs", "class_weight": "balanced"}
    return train.train_model(X, y, features.get_feature_preprocessor(STATE_COLUMNS), "classification", params, 42)


def resolve(method: str, close: pd.DataFrame, cfg: dict, model=None):
    """allocation.resolve_policy on the decision dates of `close`, towards the policy target."""
    decisions = portfolio.decision_dates(close.index, str(close.index[0].date()), str(close.index[-1].date()), "weekly")
    return allocation.resolve_policy(method, model, TARGET, decisions, close, cfg)


# ---------- acceptance tests ----------

def test_unit_defaults_reproduce_stage_2():
    cfg, (close, _) = make_cfg(), make_prices()
    target = allocation.resolve_target("policy", cfg, None)
    pd.testing.assert_series_equal(target, TARGET, check_names=False)
    model = fitted_model()
    policy = resolve("smart", close, cfg, model)
    for shifts in ({}, {"SPY": 0.06, "IEF": -0.06}, {"GLD": -0.02, "BIL": 0.02}):
        row = state_row("2024-02-09", shifts)
        expected = float(model.predict(row[STATE_COLUMNS].to_frame().T.astype(float))[0])
        assert policy(row) == expected


@pytest.mark.skip(reason="Awaiting #63 (benchmarks.fit_weights) in dev")
def test_unit_each_calculated_target_is_valid():
    returns = make_prices(periods=250)[0].pct_change().dropna()
    for method in ("min_variance", "max_sharpe", "mean_variance", "risk_parity"):
        weights = allocation.resolve_target(method, make_cfg(), returns)
        assert (weights >= 0).all()
        assert weights.sum() == pytest.approx(1.0)
        assert list(weights.index) == TICKERS


def test_unit_calendar_and_band_equals_todays_rule():
    close, volume = make_prices()
    cfg = make_cfg("quarterly", 5)
    decisions = portfolio.decision_dates(close.index, "2024-01-01", "2024-06-14", "weekly")
    market = pd.DataFrame(index=close.index)
    policies = (allocation.resolve_policy("calendar_and_band", None, TARGET, decisions, close, cfg),
                portfolio.make_current_rule_policy(TARGET, decisions, cfg))
    runs = [portfolio.simulate(close, volume, market, TARGET, policy, "2024-01-01", "2024-06-14", cfg)[1]
            for policy in policies]
    pd.testing.assert_series_equal(runs[0]["action"], runs[1]["action"])


def test_unit_each_rebalancing_method_returns_a_policy():
    close, _ = make_prices()
    model = fitted_model()
    for method in config.ALLOWED_VALUES[("rebalancing", "method")]:
        policy = resolve(method, close, make_cfg(), model)
        action = policy(state_row("2024-02-09", {"SPY": 0.03, "IEF": -0.03}))
        assert isinstance(action, float)
        assert 0.0 <= action <= 1.0


def test_unit_unknown_names_raise_with_the_allowed_list():
    close, _ = make_prices()
    with pytest.raises(ValueError) as target_error:
        allocation.resolve_target("min_risk", make_cfg(), None)
    for allowed in config.ALLOWED_VALUES[("target_weights", "method")]:
        assert allowed in str(target_error.value)
    with pytest.raises(ValueError) as policy_error:
        resolve("smrt", close, make_cfg())
    for allowed in config.ALLOWED_VALUES[("rebalancing", "method")]:
        assert allowed in str(policy_error.value)


def test_unit_explain_three_known_states():
    cfg = make_cfg(band=5)
    breach = allocation.explain(state_row("2024-02-09", {"SPY": 0.061, "IEF": -0.061}), 1.0, False, cfg)
    assert breach == "Rebalance fully: SPY is 6.1 points above target, outside the 5-point band."
    calm = allocation.explain(state_row("2024-02-09", {"SPY": 0.02, "IEF": -0.02}), 0.0, False, cfg)
    assert calm == "No trade: every holding is within 2.0 points of target."
    fallback = allocation.explain(state_row("2024-02-09", {"SPY": 0.02, "IEF": -0.02}), 0.0, True, cfg)
    assert fallback.startswith("The model was less than 60% sure, so today's rule decides. ")
    assert fallback.endswith("No trade: every holding is within 2.0 points of target.")


@pytest.mark.skip(reason="Awaiting #63 (benchmarks.fit_weights) in dev")
def test_unit_calculated_target_falls_back_to_the_rule():
    returns = make_prices(periods=250)[0].pct_change().dropna()
    target = allocation.resolve_target("min_variance", make_cfg(), returns)
    fit = allocation.target_fit(target, make_cfg(), returns)
    assert fit["fallback"] is True
    assert fit["reason"]
    assert fit["expected_tracking_error"] >= 0.0


# ---------- extra cases the table does not cover ----------

def test_unit_calendar_fires_only_on_its_dates_and_band_only_on_drift():
    close, _ = make_prices()
    calendar, band = resolve("calendar", close, make_cfg()), resolve("tolerance_band", close, make_cfg())
    quarter_start, mid_quarter = "2024-04-05", "2024-05-10"
    small, large = {"SPY": 0.01, "IEF": -0.01}, {"SPY": 0.06, "IEF": -0.06}
    assert calendar(state_row(quarter_start, small)) == 1.0
    assert calendar(state_row(mid_quarter, large)) == 0.0
    assert band(state_row(quarter_start, small)) == 0.0
    assert band(state_row(mid_quarter, large)) == 1.0


def test_unit_calendar_months_follow_the_setting():
    close, _ = make_prices(periods=260)
    monthly = resolve("calendar", close, make_cfg("monthly"))
    annually = resolve("calendar", close, make_cfg("annually"))
    assert monthly(state_row("2024-02-02")) == 1.0
    assert annually(state_row("2024-04-05")) == 0.0
    assert annually(state_row("2024-01-05")) == 1.0


def test_unit_smart_without_a_model_uses_the_rule():
    close, _ = make_prices()
    smart = resolve("smart", close, make_cfg(), model=None)
    assert smart(state_row("2024-05-10", {"SPY": 0.06, "IEF": -0.06})) == 1.0
    assert smart(state_row("2024-05-10", {"SPY": 0.01, "IEF": -0.01})) == 0.0


def test_unit_calculated_target_waits_for_benchmarks():
    with pytest.raises(ValueError, match="train_returns"):
        allocation.resolve_target("min_variance", make_cfg(), None)
    with pytest.raises(NotImplementedError, match="#63"):
        allocation.resolve_target("min_variance", make_cfg(), alternating_returns())


def test_unit_explain_halfway_and_full_without_a_breach():
    cfg = make_cfg(band=5)
    row = state_row("2024-02-09", {"EFA": -0.032, "GLD": 0.032})
    assert allocation.explain(row, 0.5, False, cfg) == "Rebalance halfway: EFA is 3.2 points below target."
    assert allocation.explain(row, 1.0, False, cfg) == "Rebalance fully: EFA is 3.2 points below target."


def alternating_returns(days: int = 100) -> pd.DataFrame:
    """Daily returns where only SPY moves, +1% then -1%, so its variance is known exactly."""
    returns = pd.DataFrame(0.0, index=pd.bdate_range("2015-01-01", periods=days), columns=TICKERS)
    returns["SPY"] = [0.01, -0.01] * (days // 2)
    return returns


def test_unit_target_fit_policy_target_has_no_fallback():
    fit = allocation.target_fit(TARGET, make_cfg(), None)
    assert fit == {"fallback": False, "reason": "", "expected_tracking_error": 0.0}


def test_unit_target_fit_other_target_falls_back_with_known_tracking_error():
    other = TARGET + pd.Series({"SPY": 0.10, "IEF": -0.10}).reindex(TICKERS, fill_value=0.0)
    fit = allocation.target_fit(other, make_cfg(), alternating_returns(100))
    # Only SPY varies: variance 0.0001 x 100 / 99; the weight difference on SPY is 0.10; 252 trading days.
    expected = 0.10 * np.sqrt(0.0001 * 100 / 99 * 252)
    assert fit["fallback"] is True
    assert "calendar-and-band" in fit["reason"]
    assert fit["expected_tracking_error"] == pytest.approx(expected, abs=1e-12)


def test_unit_target_fit_raises_on_bad_inputs():
    other = TARGET + pd.Series({"SPY": 0.10, "IEF": -0.10}).reindex(TICKERS, fill_value=0.0)
    with pytest.raises(ValueError, match="train_returns"):
        allocation.target_fit(other, make_cfg(), None)
    with pytest.raises(ValueError, match="GLD"):
        allocation.target_fit(other, make_cfg(), alternating_returns().drop(columns="GLD"))
    with pytest.raises(ValueError, match="QQQ"):
        allocation.target_fit(other.rename({"BIL": "QQQ"}), make_cfg(), alternating_returns())


def test_unit_explain_raises_on_an_unknown_action():
    with pytest.raises(ValueError, match="0.3"):
        allocation.explain(state_row("2024-02-09"), 0.3, False, make_cfg())


class ColumnsAgent:
    """A model shaped like rl_agent.QAgent: feature_columns and predict, but no sklearn feature_names_in_."""

    def __init__(self, feature_columns: list[str]) -> None:
        """Keep the columns the agent reads."""
        self.feature_columns = feature_columns

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Rebalance fully when SPY drifts above 5 points, else do nothing; fails on any extra column."""
        assert list(X.columns) == self.feature_columns
        return np.where(X["drift_SPY"] > 0.05, 1.0, 0.0)


def test_unit_smart_works_with_a_model_that_lists_feature_columns():
    close, _ = make_prices()
    smart = resolve("smart", close, make_cfg(), model=ColumnsAgent(["drift_SPY", "drawdown"]))
    assert smart(state_row("2024-02-09", {"SPY": 0.06, "IEF": -0.06})) == 1.0
    assert smart(state_row("2024-02-09", {"SPY": 0.01, "IEF": -0.01})) == 0.0


def test_unit_smart_raises_on_a_model_without_feature_columns():
    close, _ = make_prices()
    with pytest.raises(ValueError, match="feature_names_in_ or feature_columns"):
        resolve("smart", close, make_cfg(), model=object())

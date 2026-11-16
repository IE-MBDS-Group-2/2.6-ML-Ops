"""Tests for src/rules.py: the business case's simple-rule challengers and the selection among them.

Tiny synthetic prices on business days (every full week ends on a Friday). Never the real data/ folder.
"""
import numpy as np
import pandas as pd
import pytest

from src import portfolio, rules

TICKERS = ["SPY", "EFA", "IEF", "GLD", "BIL"]
TARGET = pd.Series([0.40, 0.20, 0.30, 0.05, 0.05], index=TICKERS)


def make_cfg() -> dict:
    """The config sections rules.py and portfolio.py read, with the team's values from config.yaml."""
    return {
        "portfolio": {
            "actions": [0.0, 0.5, 1.0],
            "current_rule": {"drift_band": 0.05, "quarter_months": [1, 4, 7, 10]},
            "reward": {"penalty_lambda": 1.0},
        },
        "costs": {"aum": 100_000_000, "model": "flat", "flat": {"rate": 0.001}, "slippage": {"adv_window": 20}},
        "problem": {"target_column": "best_action", "id_column": "decision_date"},
        "challengers": {
            "bands": [0.025, 0.015],
            "persistent_drift": {"min_assets": 2, "band": 0.025, "days": 5},
            "asymmetric": {"drift_threshold": 0.05, "drawdown_threshold": -0.1, "step": 0.5, "min_wait": 4},
            "equity_tickers": ["SPY", "EFA"],
        },
        "guards": {"max_return_shortfall": 0.0025, "max_cost_ratio": 2.0, "max_tracking_error": 0.012},
    }


def make_prices(periods: int = 60, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random-walk closes and constant volumes for the 5 ETFs on business days from 1 Jan 2024."""
    days = pd.bdate_range("2024-01-01", periods=periods, name="date")
    returns = np.random.default_rng(seed).normal(0.0003, 0.01, size=(periods, len(TICKERS)))
    close = pd.DataFrame(100 * np.cumprod(1 + returns, axis=0), index=days, columns=TICKERS)
    return close, pd.DataFrame(1_000_000.0, index=days, columns=TICKERS)


def flat_prices(start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Constant closes and volumes on business days from start to end."""
    days = pd.bdate_range(start, end, name="date")
    return (pd.DataFrame(100.0, index=days, columns=TICKERS),
            pd.DataFrame(1_000_000.0, index=days, columns=TICKERS))


def state_row(date: str, shifts: dict | None = None, weeks: int = 0, drawdown: float = 0.0,
              target: pd.Series = TARGET) -> pd.Series:
    """A Friday state row as simulate builds it: weights = target + shifts, named by its decision date."""
    weights = target + pd.Series(shifts or {}, dtype=float).reindex(target.index, fill_value=0.0)
    row = {f"w_{ticker}": weights[ticker] for ticker in target.index}
    row.update({f"drift_{ticker}": weights[ticker] - target[ticker] for ticker in target.index})
    row.update({"drawdown": drawdown, "weeks_since_rebalance": weeks})
    return pd.Series(row, name=pd.Timestamp(date))


def policy_for(name: str, close: pd.DataFrame, target: pd.Series = TARGET, cfg: dict | None = None):
    """make_rule_policy on the decision dates of `close`."""
    decisions = portfolio.decision_dates(close.index, str(close.index[0].date()), str(close.index[-1].date()))
    return rules.make_rule_policy(name, target, decisions, close, cfg or make_cfg())


def report(passed: bool) -> dict:
    """A guard_report-shaped dict whose single guard passes or fails."""
    return {"cost_ratio": {"value": 1.0 if passed else 3.0, "limit": 2.0, "passed": passed}}


# ---------- the rules ----------

def test_unit_tolerance_band_fires_only_outside_the_band():
    close, _ = flat_prices("2024-02-01", "2024-02-29")
    policy = policy_for("band_2_5", close)
    assert policy(state_row("2024-02-09", {"SPY": 0.03, "IEF": -0.03})) == 1.0
    assert policy(state_row("2024-02-09", {"SPY": 0.02, "IEF": -0.02})) == 0.0


def test_unit_calendar_fires_only_on_the_calendar_date():
    # The matched-exposure mix is rebalanced on the calendar only: first decision of a quarter month, never on drift.
    close, _ = flat_prices("2024-03-01", "2024-05-31")
    policy = policy_for("matched_exposure", close)
    assert policy(state_row("2024-04-05", {"SPY": 0.10, "IEF": -0.10})) == 1.0  # first Friday of April
    assert policy(state_row("2024-04-12", {"SPY": 0.10, "IEF": -0.10})) == 0.0  # mid-quarter, big drift ignored


def test_unit_persistent_drift_needs_every_day():
    # SPY +3 and IEF -3 points on Fri 12 Jan. Flat prices: the same drift on each of the last 5 days (8 to 12 Jan).
    close, _ = flat_prices("2024-01-01", "2024-01-12")
    drifted = state_row("2024-01-12", {"SPY": 0.03, "IEF": -0.03})
    assert policy_for("persistent_drift", close)(drifted) == 1.0
    # On Mon 8 Jan SPY was cheaper and IEF dearer, so that day both were inside the 2.5-point band: 4 of 5 days only.
    close.loc["2024-01-08", ["SPY", "IEF"]] = [95.0, 105.0]
    assert policy_for("persistent_drift", close)(drifted) == 0.0


def test_unit_persistent_drift_needs_at_least_two_assets():
    close, _ = flat_prices("2024-01-01", "2024-01-12")
    one_asset = state_row("2024-01-12", {"SPY": 0.03, "EFA": -0.01, "IEF": -0.02})
    assert policy_for("persistent_drift", close)(one_asset) == 0.0


def test_unit_asymmetric_steps_back_gradually():
    # SPY jumps 30% (weight 0.40 -> 0.46, a 6-point breach), then drifts up slowly: never another breach.
    close, volume = flat_prices("2024-01-01", "2024-03-22")
    close.loc["2024-01-09":, "SPY"] = 130.0
    calm = close.loc["2024-01-16":].index
    close.loc[calm, "SPY"] = 130.0 * 1.003 ** np.arange(1, len(calm) + 1)
    policy = policy_for("asymmetric", close)
    _, decisions = portfolio.simulate(close, volume, pd.DataFrame(index=close.index), TARGET, policy,
                                      "2024-01-01", "2024-03-22", make_cfg())
    traded = decisions["action"][decisions["action"] > 0]
    assert traded.index[0] == pd.Timestamp("2024-01-12") and traded.iloc[0] == 1.0  # full rebalance at the breach
    assert len(traded) > 1 and (traded.iloc[1:] == 0.5).all()                     # then only half steps
    positions = [decisions.index.get_loc(day) for day in traded.index]
    assert min(np.diff(positions)) >= make_cfg()["challengers"]["asymmetric"]["min_wait"]


def test_unit_asymmetric_fires_fully_on_a_drawdown_breach():
    close, _ = flat_prices("2024-02-01", "2024-02-29")
    policy = policy_for("asymmetric", close)
    assert policy(state_row("2024-02-09", drawdown=-0.12, weeks=1)) == 1.0
    assert policy(state_row("2024-02-09", drawdown=-0.05, weeks=1)) == 0.0


def test_unit_today_rule_equals_stage_2():
    close, volume = make_prices(periods=120)
    decisions = portfolio.decision_dates(close.index, "2024-01-01", "2024-06-14")
    market, cfg = pd.DataFrame(index=close.index), make_cfg()
    runs = [portfolio.simulate(close, volume, market, TARGET, policy, "2024-01-01", "2024-06-14", cfg)[1]
            for policy in (rules.make_rule_policy("current", TARGET, decisions, close, cfg),
                           portfolio.make_current_rule_policy(TARGET, decisions, cfg))]
    pd.testing.assert_series_equal(runs[0]["action"], runs[1]["action"])


def test_unit_matched_exposure_weights_known_value():
    # The agent held 50% in equities on average; the target holds 60% (SPY 40 + EFA 20). Scale each side.
    agent_daily = pd.DataFrame({"w_SPY": [0.30, 0.36], "w_EFA": [0.20, 0.14], "w_IEF": [0.40, 0.40],
                                "w_GLD": [0.05, 0.05], "w_BIL": [0.05, 0.05]})
    mix = rules.matched_exposure_weights(TARGET, agent_daily, ["SPY", "EFA"])
    expected = pd.Series([0.40 * 0.5 / 0.6, 0.20 * 0.5 / 0.6, 0.30 * 0.5 / 0.4, 0.05 * 0.5 / 0.4,
                          0.05 * 0.5 / 0.4], index=TICKERS)
    pd.testing.assert_series_equal(mix, expected, check_names=False)
    assert mix.sum() == pytest.approx(1.0)


# ---------- selection ----------

def test_unit_select_challenger_ignores_rules_that_fail_a_guard():
    results = pd.DataFrame({"max_drawdown": [-0.20, -0.15, -0.18]}, index=["current", "band_2_5", "band_1_5"])
    guards = {"current": report(True), "band_2_5": report(False), "band_1_5": report(True)}
    assert rules.select_challenger(results, guards) == "band_1_5"  # band_2_5 loses least but fails a guard


# ---------- errors ----------

def test_unit_unknown_rule_raises():
    close, _ = flat_prices("2024-02-01", "2024-02-29")
    with pytest.raises(ValueError, match="current, band_2_5, band_1_5, persistent_drift, asymmetric, matched_exposure"):
        policy_for("momentum", close)


@pytest.mark.parametrize("call, message", [
    (lambda: policy_for("band_2_5", flat_prices("2024-02-01", "2024-02-29")[0],
                        cfg={**make_cfg(), "challengers": {**make_cfg()["challengers"], "bands": [0.03, 0.015]}}),
     "challengers.bands"),
    (lambda: rules.matched_exposure_weights(TARGET, pd.DataFrame({"w_SPY": [0.4]}), ["SPY", "EFA"]), "w_EFA"),
    (lambda: rules.select_challenger(pd.DataFrame({"max_drawdown": [-0.2]}, index=["current"]),
                                     {"current": report(False)}), "No rule passes"),
    (lambda: rules.select_challenger(pd.DataFrame({"max_drawdown": [-0.2]}, index=["current"]), {}), "current"),
], ids=["band not in config", "agent weights missing", "no rule passes the guards", "guard report missing"])
def test_unit_rules_raise(call, message):
    with pytest.raises(ValueError, match=message):
        call()


# ---------- integration ----------

def test_integration_run_challengers_one_row_per_rule():
    close, volume = make_prices(periods=120)
    market, cfg = pd.DataFrame(index=close.index), make_cfg()
    agent_daily, _ = portfolio.simulate(close, volume, market, TARGET, lambda row: 0.5, "2024-01-01", "2024-06-14", cfg)
    results = rules.run_challengers((close, volume, market), TARGET, cfg, "2024-01-01", "2024-06-14", agent_daily)
    assert list(results.index) == list(rules.RULES)
    for column in ("max_drawdown", "annual_return", "annual_trading_cost", "tracking_error", "guards_passed",
                   "guard_return_shortfall", "guard_cost_ratio", "guard_tracking_error"):
        assert column in results.columns, column
    assert results.loc["current", "guards_passed"]  # today's rule against itself passes every guard
    assert set(results.attrs["guards"]) == set(rules.RULES)


def test_integration_run_challengers_without_the_agent_skips_matched_exposure():
    close, volume = make_prices(periods=120)
    results = rules.run_challengers((close, volume, pd.DataFrame(index=close.index)), TARGET, make_cfg(),
                                    "2024-01-01", "2024-06-14")
    assert list(results.index) == [name for name in rules.RULES if name != "matched_exposure"]
    assert rules.select_challenger(results, results.attrs["guards"]) in results.index

"""Tests for src/portfolio.py: timing rule, today's rule, trading costs and labels.

Tiny synthetic prices only (business days, so every full week ends on a Friday). Never the real data/ folder.
"""
import numpy as np
import pandas as pd
import pytest

from src import portfolio

TICKERS = ["SPY", "EFA", "IEF", "GLD", "BIL"]
TARGET = pd.Series([0.40, 0.20, 0.30, 0.05, 0.05], index=TICKERS)
COST_CFG = {"model": "flat", "flat": {"rate": 0.001}}
AUM = 100_000_000


def make_cfg() -> dict:
    """The config.yaml sections portfolio.py reads, with the team's Stage 2 values."""
    return {
        "portfolio": {
            "actions": [0.0, 0.5, 1.0],
            "current_rule": {"drift_band": 0.05, "quarter_months": [1, 4, 7, 10]},
            "reward": {"penalty_lambda": 1.0},
        },
        "costs": {"aum": AUM, "model": "flat", "flat": {"rate": 0.001}, "slippage": {"adv_window": 20}},
    }


def make_prices(start: str = "2024-01-01", periods: int = 60, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random-walk closes and constant volumes for the 5 ETFs on business days."""
    days = pd.bdate_range(start, periods=periods, name="date")
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0003, 0.01, size=(periods, len(TICKERS)))
    close = pd.DataFrame(100 * np.cumprod(1 + returns, axis=0), index=days, columns=TICKERS)
    volume = pd.DataFrame(1_000_000.0, index=days, columns=TICKERS)
    return close, volume


# ---------- decision_dates ----------

def test_unit_decision_dates_picks_last_trading_day_of_each_week():
    # Known value: Friday 12 Jan 2024 is a holiday, so that week's decision moves to Thursday 11 Jan.
    days = pd.bdate_range("2024-01-01", "2024-01-19").drop(pd.Timestamp("2024-01-12"))
    result = portfolio.decision_dates(days, "2024-01-01", "2024-01-19")
    expected = pd.DatetimeIndex(["2024-01-05", "2024-01-11", "2024-01-19"])
    assert list(result) == list(expected)


def test_unit_decision_dates_raises_when_no_trading_days_in_range():
    days = pd.bdate_range("2024-01-01", "2024-01-19")
    with pytest.raises(ValueError, match="No trading days"):
        portfolio.decision_dates(days, "2025-01-01", "2025-12-31")


# ---------- apply_action ----------

def test_unit_apply_action_half_moves_exactly_halfway_and_sums_to_one():
    weights = TARGET + pd.Series([0.06, -0.06, 0.0, 0.0, 0.0], index=TICKERS)
    result = portfolio.apply_action(weights, TARGET, 0.5)
    assert result["SPY"] == pytest.approx(0.43)
    assert result["EFA"] == pytest.approx(0.17)
    assert result.sum() == pytest.approx(1.0)


def test_unit_apply_action_raises_on_action_outside_zero_one():
    with pytest.raises(ValueError, match="between 0 and 1"):
        portfolio.apply_action(TARGET, TARGET, 1.5)


# ---------- current_rule_action ----------

def test_unit_current_rule_fires_on_quarter_start_and_6_point_drift_not_4_point():
    friday = pd.Timestamp("2024-02-09")
    drift_6 = TARGET + pd.Series([0.06, -0.06, 0.0, 0.0, 0.0], index=TICKERS)
    drift_4 = TARGET + pd.Series([0.04, -0.04, 0.0, 0.0, 0.0], index=TICKERS)
    assert portfolio.current_rule_action(TARGET, TARGET, friday, True, 0.05) == 1.0
    assert portfolio.current_rule_action(drift_6, TARGET, friday, False, 0.05) == 1.0
    assert portfolio.current_rule_action(drift_4, TARGET, friday, False, 0.05) == 0.0


# ---------- rebalance_cost (calls costs.trade_costs) ----------

def test_unit_full_rebalance_cost_known_value():
    # README known value: flat 0.1%, weights off by +0.05 and -0.05 -> 0.05 x 0.001 x 2 = 0.0001.
    weights = TARGET + pd.Series([0.05, -0.05, 0.0, 0.0, 0.0], index=TICKERS)
    assert portfolio.rebalance_cost(weights, TARGET, 1.0, AUM, None, COST_CFG) == pytest.approx(0.0001)
    assert portfolio.rebalance_cost(weights, TARGET, 0.5, AUM, None, COST_CFG) == pytest.approx(0.00005)


# ---------- execution_schedule (README timing tests) ----------

def test_unit_first_reward_date_is_after_execution_date():
    days = pd.bdate_range("2024-01-01", periods=30)
    decisions = portfolio.decision_dates(days, "2024-01-01", "2024-02-28")
    schedule = portfolio.execution_schedule(days, decisions)
    full = schedule.dropna()
    assert (full["execution_date"] > full.index).all()
    assert (full["reward_start"] > full["execution_date"]).all()


def test_unit_reward_windows_join_with_no_gaps_or_overlaps():
    days = pd.bdate_range("2024-01-01", periods=30)
    decisions = portfolio.decision_dates(days, "2024-01-01", "2024-02-28")
    full = portfolio.execution_schedule(days, decisions).dropna()
    covered = []
    for _, row in full.iterrows():
        covered.extend(days[(days >= row["reward_start"]) & (days <= row["reward_end"])])
    first, last = full["reward_start"].iloc[0], full["reward_end"].iloc[-1]
    expected = list(days[(days >= first) & (days <= last)])
    assert covered == expected  # every trading day in exactly one window, in order


def test_unit_execution_schedule_raises_on_decision_date_that_is_not_a_trading_day():
    days = pd.bdate_range("2024-01-01", periods=10)
    with pytest.raises(ValueError, match="not trading days"):
        portfolio.execution_schedule(days, pd.DatetimeIndex(["2024-01-06"]))  # a Saturday


# ---------- make_current_rule_policy ----------

def test_unit_current_rule_policy_fires_only_on_first_decision_of_a_quarter_month():
    days = pd.bdate_range("2024-03-01", "2024-05-31")
    decisions = portfolio.decision_dates(days, "2024-03-01", "2024-05-31")
    policy = portfolio.make_current_rule_policy(TARGET, decisions, make_cfg())

    def at_target(date: str) -> pd.Series:
        return pd.Series({f"w_{t}": TARGET[t] for t in TICKERS}, name=pd.Timestamp(date))

    assert policy(at_target("2024-04-05")) == 1.0  # first Friday of April
    assert policy(at_target("2024-04-12")) == 0.0  # second Friday of April
    assert policy(at_target("2024-05-03")) == 0.0  # May is not a quarter month


# ---------- simulate ----------

def test_unit_simulate_known_value_buy_and_hold():
    # SPY (40%) rises 10% once; with a do-nothing policy the portfolio ends at 1 + 0.4 x 0.10 = 1.04, no costs.
    days = pd.bdate_range("2024-01-01", "2024-01-19", name="date")
    close = pd.DataFrame(100.0, index=days, columns=TICKERS)
    close.loc["2024-01-08":, "SPY"] = 110.0
    volume = pd.DataFrame(1_000_000.0, index=days, columns=TICKERS)
    daily, _ = portfolio.simulate(close, volume, pd.DataFrame(index=days), TARGET, lambda row: 0.0,
                                  "2024-01-01", "2024-01-19", make_cfg())
    assert daily.index[0] == pd.Timestamp("2024-01-05")  # starts on the first decision date
    assert daily["value"].iloc[0] == pytest.approx(1.0)
    assert daily["value"].iloc[-1] == pytest.approx(1.04)
    assert daily["cost_paid"].sum() == 0.0


def test_unit_simulate_calls_any_policy_once_per_decision_and_charges_only_on_execution_days():
    close, volume = make_prices(periods=60)
    market = pd.DataFrame({"ret_4w_SPY": 0.0}, index=close.index)
    seen = []

    def always_rebalance(state_row: pd.Series) -> float:
        seen.append(state_row)
        return 1.0

    daily, decisions = portfolio.simulate(close, volume, market, TARGET, always_rebalance,
                                          "2024-01-01", "2024-03-22", make_cfg())
    assert len(seen) == len(decisions)
    assert (decisions["action"] == 1.0).all()
    assert decisions.index.name == "decision_date"
    for column in ["w_SPY", "drift_SPY", "drawdown", "weeks_since_rebalance", "full_rebalance_cost", "ret_4w_SPY"]:
        assert column in seen[0].index
    assert decisions["weeks_since_rebalance"].tolist() == [0] + [1] * (len(decisions) - 1)
    schedule = portfolio.execution_schedule(daily.index, decisions.index)
    charged = set(daily.index[daily["cost_paid"] > 0])
    assert charged and charged <= set(schedule["execution_date"].dropna())


@pytest.mark.parametrize("target", [
    TARGET.drop("BIL") / TARGET.drop("BIL").sum(),   # a ticker missing
    TARGET * 0.9,                                    # weights do not sum to 1
])
def test_unit_simulate_raises_when_target_does_not_fit_prices(target):
    close, volume = make_prices(periods=20)
    with pytest.raises(ValueError, match="target"):
        portfolio.simulate(close, volume, pd.DataFrame(index=close.index), target, lambda row: 0.0,
                           "2024-01-01", "2024-01-26", make_cfg())


# ---------- build_decision_table ----------

def test_unit_build_decision_table_runs_todays_rule_through_simulate(monkeypatch):
    # README test: the agent and the baseline use the SAME simulate; only the policy differs.
    calls = []
    real_simulate = portfolio.simulate

    def spy(*args, **kwargs):
        calls.append(args)
        return real_simulate(*args, **kwargs)

    monkeypatch.setattr(portfolio, "simulate", spy)
    close, volume = make_prices(periods=40)
    table = portfolio.build_decision_table(close, volume, pd.DataFrame(index=close.index), TARGET,
                                           "2024-01-01", "2024-02-23", make_cfg())
    assert len(calls) == 1
    assert (table["rule_action"].isin([0.0, 1.0])).all()


def test_unit_build_decision_table_one_row_per_week_labels_and_last_row_nan():
    close, volume = make_prices(periods=60)
    market = pd.DataFrame({"ret_4w_SPY": 0.0}, index=close.index)
    table = portfolio.build_decision_table(close, volume, market, TARGET, "2024-01-01", "2024-03-22", make_cfg())
    weeks = portfolio.decision_dates(close.index, "2024-01-01", "2024-03-22")
    assert table.index.name == "decision_date"
    assert list(table.index) == list(weeks)
    assert set(table["best_action"].iloc[:-2]) <= {0.0, 0.5, 1.0}
    assert "ret_4w_SPY" in table.columns and "w_SPY" in table.columns


def test_unit_last_two_decisions_have_no_full_window_so_their_label_is_nan():
    # The last decision is the last trading day: it never executes, so the one before has no window end either.
    close, volume = make_prices(periods=60)
    table = portfolio.build_decision_table(close, volume, pd.DataFrame(index=close.index), TARGET,
                                           "2024-01-01", "2024-03-22", make_cfg())
    assert table["best_action"].isna().tolist() == [False] * (len(table) - 2) + [True, True]


def test_unit_best_action_tie_goes_to_the_smaller_action():
    # Flat prices: weights never drift, every action earns the same reward -> label 0.0 (trade less).
    days = pd.bdate_range("2024-01-01", "2024-02-29", name="date")
    close = pd.DataFrame(100.0, index=days, columns=TICKERS)
    volume = pd.DataFrame(1_000_000.0, index=days, columns=TICKERS)
    table = portfolio.build_decision_table(close, volume, pd.DataFrame(index=days), TARGET,
                                           "2024-01-01", "2024-02-29", make_cfg())
    assert (table["best_action"].iloc[:-2] == 0.0).all()


def test_unit_best_action_is_full_rebalance_before_a_crash_in_the_overweight_etf():
    # SPY jumps 50% (weight 0.40 -> 0.50), then falls 40% in the next window: going back to target loses less.
    days = pd.bdate_range("2024-01-01", "2024-01-31", name="date")
    close = pd.DataFrame(100.0, index=days, columns=TICKERS)
    close.loc["2024-01-09":, "SPY"] = 150.0
    close.loc["2024-01-16":, "SPY"] = 90.0
    volume = pd.DataFrame(1_000_000.0, index=days, columns=TICKERS)
    table = portfolio.build_decision_table(close, volume, pd.DataFrame(index=days), TARGET,
                                           "2024-01-01", "2024-01-31", make_cfg())
    assert table.loc["2024-01-12", "w_SPY"] == pytest.approx(0.5)
    assert table.loc["2024-01-12", "best_action"] == 1.0


def test_integration_market_features_then_decision_table():
    from src import features  # pylint: disable=import-outside-toplevel

    close, volume = make_prices(periods=260)
    market = features.compute_market_features(close, return_weeks=[4, 12], vol_days=20)
    table = portfolio.build_decision_table(close, volume, market, TARGET, "2024-01-01", "2024-12-31", make_cfg())
    weeks = portfolio.decision_dates(close.index, "2024-01-01", "2024-12-31")
    assert len(table) == len(weeks)
    assert set(table["best_action"].iloc[:-2]) <= {0.0, 0.5, 1.0}
    assert table["best_action"].iloc[-2:].isna().all()
    assert set(market.columns) <= set(table.columns)

"""Tests for src/rl_agent.py: transitions from the simulator, fitted Q iteration and the QAgent."""

import numpy as np
import pandas as pd
import pytest

from src import infer, portfolio, rl_agent

TICKERS = ["SPY", "EFA", "IEF", "GLD", "BIL"]
TARGET = pd.Series([0.40, 0.20, 0.30, 0.05, 0.05], index=TICKERS)
ACTIONS = [0.0, 0.5, 1.0]
PARAMS = {"gamma": 0.9, "n_iterations": 5, "temperature": 0.5,
          "estimator": {"max_depth": 3, "max_iter": 50, "learning_rate": 0.2, "min_samples_leaf": 1,
                        "early_stopping": False}}


def make_cfg() -> dict:
    """The config sections the simulator and the agent read."""
    return {
        "portfolio": {"actions": ACTIONS, "reward": {"penalty_lambda": 1.0},
                      "current_rule": {"drift_band": 0.05, "quarter_months": [1, 4, 7, 10]}},
        "costs": {"aum": 100_000_000, "model": "flat", "flat": {"rate": 0.001}, "slippage": {"adv_window": 20}},
        "problem": {"target_column": "best_action", "id_column": "decision_date"},
    }


def make_market(periods: int = 120, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Random-walk closes, constant volumes and one market feature on business days from 2024-01-01."""
    days = pd.bdate_range("2024-01-01", periods=periods, name="date")
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0003, 0.01, size=(periods, len(TICKERS)))
    close = pd.DataFrame(100 * np.cumprod(1 + returns, axis=0), index=days, columns=TICKERS)
    volume = pd.DataFrame(1_000_000.0, index=days, columns=TICKERS)
    market = pd.DataFrame({"ret_4w_SPY": close["SPY"].pct_change(20).fillna(0.0)}, index=days)
    return close, volume, market


def toy_transitions(rows: list[tuple[float, float, float, float, bool]], copies: int = 40) -> pd.DataFrame:
    """Hand-made transitions with one state column x: (x, action, reward, next_x, done), repeated."""
    frame = pd.DataFrame(rows, columns=["x", "action", "reward", "next_x", "done"])
    return pd.concat([frame] * copies, ignore_index=True)


def end_date(market: tuple) -> str:
    """The last trading day of the synthetic market, as text."""
    return str(market[0].index[-1].date())


# ---------- fitted Q iteration on toy problems ----------

def test_unit_agent_learns_the_better_action_on_a_toy_problem():
    # Known value: action 1.0 always earns 1, the others 0, so the agent must pick 1.0 in every state.
    rows = [(x, action, 1.0 if action == 1.0 else 0.0, x, True) for x in (0.0, 1.0) for action in ACTIONS]
    agent = rl_agent.fit_q_agent(toy_transitions(rows), ["x"], ACTIONS, PARAMS, random_state=0)
    assert agent.predict(pd.DataFrame({"x": [0.0, 1.0]})).tolist() == [1.0, 1.0]


@pytest.mark.parametrize(("gamma", "expected"), [(0.9, 0.0), (0.0, 1.0)])
def test_unit_agent_prefers_delayed_reward_with_gamma(gamma, expected):
    # State 0: action 0.0 pays 0 now and leads to state 1 (worth 10); action 1.0 pays 1 now and ends.
    # Looking ahead (gamma 0.9) the agent takes 0.0; with gamma 0 it only sees the 1 and takes 1.0.
    rows = [(0.0, 0.0, 0.0, 1.0, False), (0.0, 1.0, 1.0, 0.0, True),
            (1.0, 0.0, 10.0, 1.0, True), (1.0, 1.0, 10.0, 1.0, True)]
    agent = rl_agent.fit_q_agent(toy_transitions(rows), ["x"], [0.0, 1.0], {**PARAMS, "gamma": gamma},
                                 random_state=0)
    assert agent.predict(pd.DataFrame({"x": [0.0]})).tolist() == [expected]


def test_unit_same_seed_same_agent():
    rows = [(x, action, x * action, x, True) for x in (0.0, 0.5, 1.0) for action in ACTIONS]
    states = pd.DataFrame({"x": [0.0, 0.25, 0.5, 1.0]})
    first = rl_agent.fit_q_agent(toy_transitions(rows), ["x"], ACTIONS, PARAMS, random_state=3)
    second = rl_agent.fit_q_agent(toy_transitions(rows), ["x"], ACTIONS, PARAMS, random_state=3)
    assert np.allclose(first.q_values(states), second.q_values(states))


def test_unit_different_seeds_learn_from_different_resamples():
    # Noisy rewards: two seeds see different bootstrap samples, so their values differ a little.
    rng = np.random.default_rng(0)
    rows = [(x, action, x * action + rng.normal(0, 0.5), x, True)
            for x in np.linspace(0, 1, 20) for action in ACTIONS]
    transitions = toy_transitions(rows, copies=1)
    states = pd.DataFrame({"x": [0.0, 0.5, 1.0]})
    first = rl_agent.fit_q_agent(transitions, ["x"], ACTIONS, PARAMS, random_state=1)
    second = rl_agent.fit_q_agent(transitions, ["x"], ACTIONS, PARAMS, random_state=2)
    assert not np.allclose(first.q_values(states), second.q_values(states))


def test_unit_predict_proba_rows_sum_to_one():
    rows = [(x, action, action, x, True) for x in (0.0, 1.0) for action in ACTIONS]
    agent = rl_agent.fit_q_agent(toy_transitions(rows), ["x"], ACTIONS, PARAMS, random_state=0)
    proba = agent.predict_proba(pd.DataFrame({"x": [0.0, 1.0, 0.5]}))
    assert proba.shape == (3, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert agent.classes_.tolist() == ACTIONS
    assert proba.argmax(axis=1).tolist() == [2, 2, 2]      # the highest score is on the best action, 1.0


def test_unit_tie_goes_to_the_smaller_action():
    rows = [(0.0, action, 1.0, 0.0, True) for action in ACTIONS]      # every action is worth the same
    agent = rl_agent.fit_q_agent(toy_transitions(rows), ["x"], ACTIONS, PARAMS, random_state=0)
    assert agent.predict(pd.DataFrame({"x": [0.0]})).tolist() == [0.0]


def test_unit_q_values_raises_on_missing_feature_column():
    rows = [(0.0, action, action, 0.0, True) for action in ACTIONS]
    agent = rl_agent.fit_q_agent(toy_transitions(rows), ["x"], ACTIONS, PARAMS, random_state=0)
    with pytest.raises(ValueError, match=r"missing feature columns \['x'\]"):
        agent.predict(pd.DataFrame({"y": [0.0]}))


@pytest.mark.parametrize(("change", "message"), [
    ({"gamma": 1.5}, "rl.gamma is 1.5"),
    ({"gamma": -0.1}, "rl.gamma is -0.1"),
    ({"temperature": 0}, "rl.temperature is 0"),
    ({"n_iterations": 0}, "rl.n_iterations is 0"),
])
def test_unit_fit_raises_on_settings_that_cannot_work(change, message):
    rows = [(0.0, action, action, 0.0, True) for action in ACTIONS]
    with pytest.raises(ValueError, match=message):
        rl_agent.fit_q_agent(toy_transitions(rows), ["x"], ACTIONS, {**PARAMS, **change}, random_state=0)


def test_unit_fit_raises_on_empty_incomplete_or_unknown_action_transitions():
    rows = [(0.0, action, action, 0.0, True) for action in ACTIONS]
    transitions = toy_transitions(rows)
    with pytest.raises(ValueError, match="No transitions to learn from"):
        rl_agent.fit_q_agent(transitions.iloc[0:0], ["x"], ACTIONS, PARAMS, random_state=0)
    with pytest.raises(ValueError, match=r"missing columns \['next_x'\]"):
        rl_agent.fit_q_agent(transitions.drop(columns="next_x"), ["x"], ACTIONS, PARAMS, random_state=0)
    with pytest.raises(ValueError, match=r"actions \[0.5\] that are not in portfolio.actions"):
        rl_agent.fit_q_agent(transitions, ["x"], [0.0, 1.0], PARAMS, random_state=0)


# ---------- transitions from the simulator ----------

def test_unit_transitions_have_state_action_reward_next_state_and_done():
    market = make_market()
    transitions = rl_agent.collect_transitions(market, TARGET, make_cfg(), "2024-01-01", end_date(market), 2, seed=1)
    assert {"action", "reward", "done", "drawdown", "next_drawdown", "w_SPY", "next_w_SPY"} <= set(transitions.columns)
    assert set(transitions["action"]) <= set(ACTIONS)
    assert transitions["reward"].notna().all()
    assert not transitions["done"].any()                # the data ends, the portfolio does not
    assert transitions.filter(like="next_").notna().all().all()      # so every row has a real next state
    # The next state of a row is the state of the following decision in the same run.
    one_run = transitions.iloc[: len(transitions) // 2]
    assert np.allclose(one_run["next_drawdown"].iloc[:-1], one_run["drawdown"].iloc[1:])


def test_unit_transitions_stay_inside_the_train_years():
    market = make_market(periods=200)
    cut = "2024-06-28"
    transitions = rl_agent.collect_transitions(market, TARGET, make_cfg(), "2024-01-01", cut, 2, seed=0)
    assert transitions.index.max() <= pd.Timestamp(cut)


def test_unit_collect_is_reproducible_per_seed():
    market = make_market()
    args = (market, TARGET, make_cfg(), "2024-01-01", end_date(market), 2)
    first = rl_agent.collect_transitions(*args, seed=5)
    second = rl_agent.collect_transitions(*args, seed=5)
    other = rl_agent.collect_transitions(*args, seed=6)
    assert first["action"].tolist() == second["action"].tolist()
    assert first["action"].tolist() != other["action"].tolist()


def test_unit_collect_raises_on_fewer_than_two_episodes():
    market = make_market()
    with pytest.raises(ValueError, match="n_episodes is 1. Give at least 2"):
        rl_agent.collect_transitions(market, TARGET, make_cfg(), "2024-01-01", end_date(market), 1, seed=0)


def test_unit_rewards_are_measured_against_the_average_run_on_each_date():
    # Every run lives through the same market: after removing each date's average, what is left sums to 0 per date.
    market = make_market()
    transitions = rl_agent.collect_transitions(market, TARGET, make_cfg(), "2024-01-01", end_date(market), 4, seed=2)
    per_date = transitions.groupby(level=0)["reward"]
    assert np.allclose(per_date.mean(), 0.0, atol=1e-12)
    assert (per_date.size() == 4).all()
    assert transitions["reward"].abs().max() < 0.01      # the market's weekly move is gone; only small gaps remain


def test_integration_transition_rewards_are_portfolios_realised_rewards():
    # One reward function for labels and agent: replay the first run and compare, before centring by date.
    market = make_market()
    cfg, end = make_cfg(), end_date(market)
    transitions = rl_agent.collect_transitions(market, TARGET, cfg, "2024-01-01", end, 2, seed=3)
    rng = np.random.default_rng(3)
    runs = []
    for _ in range(2):
        daily, decisions = portfolio.simulate(*market, TARGET, lambda row: float(rng.choice(ACTIONS)),
                                              "2024-01-01", end, cfg)
        runs.append(portfolio.realised_rewards(daily, decisions, market[0], market[1], TARGET, cfg).dropna())
    expected = pd.concat(runs)
    expected = expected - expected.groupby(level=0).transform("mean")
    assert np.allclose(transitions["reward"].to_numpy(), expected.to_numpy())


# ---------- the confidence score and the fallback ----------

def test_unit_confidence_score_does_not_depend_on_the_size_of_the_rewards():
    # Weekly rewards are tiny. Shrinking every reward 1000 times must not change how sure the agent looks.
    rows = [(x, action, action * (1 + x), x, True) for x in (0.0, 1.0) for action in ACTIONS]
    small = [(x, action, reward / 1000, next_x, done) for x, action, reward, next_x, done in rows]
    states = pd.DataFrame({"x": [0.0, 1.0]})
    big_agent = rl_agent.fit_q_agent(toy_transitions(rows), ["x"], ACTIONS, PARAMS, random_state=0)
    small_agent = rl_agent.fit_q_agent(toy_transitions(small), ["x"], ACTIONS, PARAMS, random_state=0)
    assert small_agent.typical_gap == pytest.approx(big_agent.typical_gap / 1000, rel=0.05)
    assert np.allclose(small_agent.predict_proba(states), big_agent.predict_proba(states), atol=0.02)
    assert small_agent.predict_proba(states).max(axis=1).min() > 0.5       # not stuck at one third


def test_unit_typical_gap_known_values():
    q_values = pd.DataFrame({0.0: [1.0, 5.0, 2.0], 0.5: [3.0, 5.0, 2.5], 1.0: [2.0, 5.0, 4.5]})
    assert rl_agent._typical_gap(q_values) == pytest.approx(1.5)       # leads of 1 and 2; the tied row is left out
    assert rl_agent._typical_gap(pd.DataFrame({0.0: [1.0], 1.0: [1.0]})) == 1.0      # nobody leads
    assert rl_agent._typical_gap(pd.DataFrame({0.0: [1.0, 2.0]})) == 1.0             # one action


def test_unit_fallback_share_known_values():
    rows = [(x, action, action, x, True) for x in (0.0, 1.0) for action in ACTIONS]
    agent = rl_agent.fit_q_agent(toy_transitions(rows), ["x"], ACTIONS, PARAMS, random_state=0)
    states = pd.DataFrame({"x": [0.0, 1.0]})
    top = agent.predict_proba(states).max(axis=1)
    assert rl_agent.fallback_share(agent, states, min_probability=float(top.min()) - 0.01) == 0.0
    assert rl_agent.fallback_share(agent, states, min_probability=1.01) == 1.0
    with pytest.raises(ValueError, match="No states given"):
        rl_agent.fallback_share(agent, states.iloc[0:0], min_probability=0.6)


def test_unit_value_carries_past_a_transition_that_is_not_done():
    # Same data, one flag: where the data just stops (done False) the next state's value still counts.
    def best_value(done: bool) -> float:
        rows = [(0.0, 0.0, 0.0, 1.0, done), (1.0, 0.0, 1.0, 1.0, True)]
        agent = rl_agent.fit_q_agent(toy_transitions(rows), ["x"], [0.0], {**PARAMS, "gamma": 0.9}, random_state=0)
        return float(agent.q_values(pd.DataFrame({"x": [0.0]})).iloc[0, 0])
    assert best_value(done=True) == pytest.approx(0.0, abs=0.05)
    assert best_value(done=False) == pytest.approx(0.9, abs=0.05)


def test_unit_estimator_settings_reach_the_regressor():
    rows = [(x, action, action, x, True) for x in (0.0, 1.0) for action in ACTIONS]
    agent = rl_agent.fit_q_agent(toy_transitions(rows), ["x"], ACTIONS, PARAMS, random_state=0)
    assert agent.regressor.early_stopping is False
    assert agent.regressor.n_iter_ == PARAMS["estimator"]["max_iter"]      # every round runs in full


# ---------- the agent in the pipeline ----------

def test_integration_agent_runs_through_infer_and_simulate():
    market = make_market()
    cfg, end = make_cfg(), end_date(market)
    transitions = rl_agent.collect_transitions(market, TARGET, cfg, "2024-01-01", end, 3, seed=0)
    columns = [name for name in transitions.columns
               if not name.startswith("next_") and name not in ("action", "reward", "done")]
    agent = rl_agent.fit_q_agent(transitions, columns, ACTIONS, PARAMS, random_state=0)
    predictions = infer.run_inference(agent, transitions[columns].head(5), include_proba=True)
    assert list(predictions.columns) == ["prediction", "probability"]
    assert set(predictions["prediction"]) <= set(ACTIONS)
    assert predictions["probability"].between(1 / 3, 1.0).all()

    def policy(state_row: pd.Series) -> float:
        return float(agent.predict(state_row[columns].to_frame().T.astype(float))[0])

    daily, decisions = portfolio.simulate(*market, TARGET, policy, "2024-01-01", end, cfg)
    assert set(decisions["action"]) <= set(ACTIONS)
    assert daily["value"].gt(0).all()


# ---------- the median seed ----------

def test_unit_median_seed_known_values():
    assert rl_agent.median_seed({0: 0.10, 1: -0.20, 2: 0.05, 3: 0.30, 4: 0.00}) == 2
    assert rl_agent.median_seed({7: 0.4, 8: 0.1}) == 8          # even count: the lower middle
    assert rl_agent.median_seed({5: 0.2}) == 5
    with pytest.raises(ValueError, match="No seed scores given"):
        rl_agent.median_seed({})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_unit_median_seed_raises_on_a_score_that_is_not_finite(bad):
    with pytest.raises(ValueError, match=r"Seeds \[1\] have a score that is not a finite number"):
        rl_agent.median_seed({0: 0.1, 1: bad, 2: 0.3})

"""
rl_agent.py  [The reinforcement learning challenger]
====================================================
Purpose   Learn, from simulated experience, how valuable each rebalancing action is over MANY weeks,
          and act on it: the agent the business case promises as a Stage 3 challenger
Method    Fitted Q iteration (batch reinforcement learning, scikit-learn only)
Input     Transitions (state, action, reward, next state) collected by running random policies through
          portfolio.simulate on the TRAIN years only
Output    A QAgent with predict and predict_proba, so infer.py, the fallback and the simulator use it unchanged
Reward    portfolio.realised_rewards: the one reward function the labels use too
Called by main.py when model.type is "fitted_q"

Why it matters
- The Stage 2 classifier looks one week ahead. The agent values an action by what follows it too:
  Q(state, action) = reward now + gamma x the best Q in the next state
- Same simulator, same costs, same reward and same quality gate as the classifier: a fair challenger
- Little data (about 640 weekly decisions): small model, several seeds, the median reported

Tests     tests/test_rl_agent.py
"""

import logging
import math
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src import portfolio

logger = logging.getLogger(__name__)

ACTION_COLUMN = "action"     # the action taken, as recorded by portfolio.simulate
REWARD_COLUMN = "reward"
DONE_COLUMN = "done"         # True only where the problem itself ends; never because the data ends
NEXT_PREFIX = "next_"        # the next state's columns carry this prefix


class QAgent:
    """A fitted action-value model that picks the action with the highest value.

    Why a class: infer.run_inference, the 60% fallback and the simulator all expect predict and
    predict_proba, the way a scikit-learn classifier offers them. classes_ lists the actions in column order.
    """

    def __init__(self, regressor: HistGradientBoostingRegressor, feature_columns: list[str],
                 actions: list[float], temperature: float, typical_gap: float = 1.0) -> None:
        """Keep the fitted regressor and what it needs to score a state for every action.

        Inputs:  regressor, the fitted model of Q(state, action); feature_columns, the state columns it reads;
                 actions, the allowed actions; temperature, how sharp predict_proba is (rl.temperature);
                 typical_gap, the usual distance between the best and the second-best Q value on the states
                 the agent learned from (fit_q_agent sets it). predict_proba measures every gap against it.
        Outputs: None; the values are stored on the agent (classes_ holds the sorted actions).
        Why:     the agent must carry everything needed to score a new state, so serving needs nothing else.
        """
        self.regressor = regressor
        self.feature_columns = list(feature_columns)
        self.classes_ = np.array(sorted(actions), dtype=float)
        self.temperature = float(temperature)
        self.typical_gap = float(typical_gap)

    def q_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """The value of every action in every state.

        Inputs:  X, one row per state, holding the feature columns.
        Outputs: frame with X's index and one column per action.
        Why:     every decision (predict, predict_proba, the fitted-Q update) is made from these values.
        Raises:  ValueError naming any feature column that is missing.
        """
        missing = [column for column in self.feature_columns if column not in X.columns]
        if missing:
            raise ValueError(f"State is missing feature columns {missing}. Pass the columns the agent was fitted on.")
        states = X[self.feature_columns].astype(float)
        values = {action: self.regressor.predict(states.assign(**{ACTION_COLUMN: action}))
                  for action in self.classes_}
        return pd.DataFrame(values, index=X.index)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """The best action per row; a tie goes to the smaller action (it trades less).

        Inputs:  X, one row per state, holding the feature columns.
        Outputs: array with one action per row.
        Why:     gives the agent the predict method of a scikit-learn classifier, so callers need no change.
        """
        return self.classes_[self.q_values(X).to_numpy().argmax(axis=1)]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """A score per action that sums to 1 per row: a softmax of the Q values.

        Inputs:  X, one row per state, holding the feature columns.
        Outputs: array with one row per state and one column per action (the order of classes_).
        Why:     the fallback asks "is the preferred action above 60%?". Q values differ by tiny amounts
                 (weekly returns), so they are first divided by typical_gap: a state where the best action
                 leads by the usual margin scores the same whatever the size of the rewards. This is a ranking
                 score, not a calibrated probability; rl.temperature sets how sharp it is (1.0: a lead of one
                 usual margin gives 0.58 to 0.73; smaller is sharper).
        """
        scaled = self.q_values(X).to_numpy() / (self.temperature * self.typical_gap)
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        weights = np.exp(scaled)
        return weights / weights.sum(axis=1, keepdims=True)


def _random_policy(actions: list[float], rng: np.random.Generator) -> Callable[[pd.Series], float]:
    """A policy that explores.

    Inputs:  actions, the allowed actions; rng, the random generator of this collection.
    Outputs: a policy for portfolio.simulate that picks one of the actions at random at every decision.
    Why:     the agent can only learn the value of an action it has seen tried in many states.
    """
    def policy(_state_row: pd.Series) -> float:
        """One random action, whatever the state.

        Inputs:  _state_row, the state (ignored).
        Outputs: one of the actions, chosen uniformly at random.
        Why:     exploration must not depend on what the agent has learned so far.
        """
        return float(rng.choice(actions))
    return policy


def _episode_transitions(decisions: pd.DataFrame, rewards: pd.Series) -> pd.DataFrame:
    """Turn one simulated run into rows of (state, action, reward, next state, done).

    Inputs:  decisions, the decision frame from portfolio.simulate (state columns and the action taken);
             rewards, the reward of each decision (portfolio.realised_rewards), NaN without a full window.
    Outputs: one row per decision that has a reward. done is False on every row.
    Why:     the run stops because the data stops, not because the portfolio does: the last row still has a
             real next state, so its value must keep counting what follows. Marking it as an ending would
             teach the agent that nothing comes after the last week of the train years.
    """
    features = decisions.drop(columns=ACTION_COLUMN)
    following = features.shift(-1).add_prefix(NEXT_PREFIX)
    rows = pd.concat([features, decisions[ACTION_COLUMN], following], axis=1)
    rows[REWARD_COLUMN] = rewards
    rows = rows.dropna(subset=[REWARD_COLUMN])
    rows[DONE_COLUMN] = False
    return rows


def _center_rewards_by_date(transitions: pd.DataFrame) -> pd.DataFrame:
    """Subtract, from every reward, the average reward of all runs on the same decision date.

    Inputs:  transitions of several runs, indexed by decision date.
    Outputs: a copy with the reward column centred per date; every other column unchanged.
    Why:     every run lives through the same market, and the market's move that week dwarfs the effect of
             the action. Removing the date's average leaves what the agent can actually influence: how much
             better or worse this state and action did than the average random run that week. The amount
             removed depends only on the date, never on the action, so the best action in every state is
             unchanged.
    """
    centered = transitions.copy()
    by_date = centered.groupby(level=0)[REWARD_COLUMN]
    centered[REWARD_COLUMN] = centered[REWARD_COLUMN] - by_date.transform("mean")
    return centered


def collect_transitions(market_data: tuple, target: pd.Series, cfg: dict, start: str, end: str,
                        n_episodes: int, seed: int) -> pd.DataFrame:
    """Gather experience by running random policies through the simulator.

    Inputs:  market_data (close_wide, volume_wide, market_features); target weights; cfg, the full config
             (reads portfolio.actions, plus what simulate and portfolio.realised_rewards read);
             start and end (YYYY-MM-DD): pass the TRAIN years only; n_episodes, how many runs; seed.
    Outputs: one row per decision per run: the state columns, action, reward, the next state's columns
             (prefixed next_) and done (always False here: the data ends, the portfolio does not). No row is
             dated after `end`. The reward is portfolio.realised_rewards, measured against the average of all
             runs on the same date (see _center_rewards_by_date).
    Why:     the agent can only learn the value of an action it has seen tried; random runs try them all.
    Raises:  ValueError if n_episodes is below 2 (and the errors of portfolio.simulate).
    """
    if n_episodes < 2:
        raise ValueError(f"n_episodes is {n_episodes}. Give at least 2 (rl.n_episodes in config.yaml): rewards "
                         "are measured against the average of the runs on each date.")
    actions = [float(action) for action in cfg["portfolio"]["actions"]]
    rng = np.random.default_rng(seed)
    episodes = []
    for _ in range(n_episodes):
        daily, decisions = portfolio.simulate(*market_data, target, _random_policy(actions, rng), start, end, cfg)
        rewards = portfolio.realised_rewards(daily, decisions, market_data[0], market_data[1], target, cfg)
        episodes.append(_episode_transitions(decisions, rewards))
    transitions = _center_rewards_by_date(pd.concat(episodes))
    logger.info("Collected %d transitions from %d random runs, %s to %s", len(transitions), n_episodes, start, end)
    return transitions


def _check_transitions(transitions: pd.DataFrame, feature_columns: list[str], actions: list[float]) -> None:
    """Check the experience before learning from it.

    Inputs:  transitions; feature_columns, the state columns the agent will use; actions, the allowed actions.
    Outputs: nothing.
    Why:     a missing column or a stray action would otherwise surface as an unreadable scikit-learn error.
    Raises:  ValueError if the transitions are empty, lack a needed column or hold an action outside the list.
    """
    if transitions.empty:
        raise ValueError("No transitions to learn from. Collect them with collect_transitions on the train years.")
    needed = [*feature_columns, ACTION_COLUMN, REWARD_COLUMN, DONE_COLUMN,
              *(NEXT_PREFIX + column for column in feature_columns)]
    missing = [column for column in needed if column not in transitions.columns]
    if missing:
        raise ValueError(f"Transitions are missing columns {missing}. Build them with collect_transitions.")
    unknown = sorted(set(transitions[ACTION_COLUMN]) - set(actions))
    if unknown:
        raise ValueError(f"Transitions hold actions {unknown} that are not in portfolio.actions {sorted(actions)}.")


def _check_settings(params: dict) -> None:
    """Check the rl settings before fitting.

    Inputs:  params, the rl config section.
    Outputs: nothing.
    Why:     a wrong setting should name its config key, not fail deep inside the fitting loop.
    Raises:  ValueError on fewer than 1 round, a gamma outside 0 to 1 or a temperature that is not above 0.
    """
    if params["n_iterations"] < 1:
        raise ValueError(f"rl.n_iterations is {params['n_iterations']}. Give at least 1.")
    if not 0.0 <= params["gamma"] <= 1.0:
        raise ValueError(f"rl.gamma is {params['gamma']}. Give a number from 0 to 1.")
    if params["temperature"] <= 0:
        raise ValueError(f"rl.temperature is {params['temperature']}. Give a number above 0.")


def _typical_gap(q_values: pd.DataFrame) -> float:
    """The usual lead of the best action over the second best.

    Inputs:  q_values, one row per state and one column per action.
    Outputs: the median, over the states where the best action leads at all, of best minus second-best Q.
             1.0 when no state has a leader (or there is one action): the scores are then equal anyway.
    Why:     Q values live on the scale of weekly returns, so a fixed temperature would be right for one
             dataset and wrong for the next. Dividing by this gap makes the confidence score scale-free.
    """
    ranked = np.sort(q_values.to_numpy(), axis=1)
    if ranked.shape[1] < 2:
        return 1.0
    gaps = ranked[:, -1] - ranked[:, -2]
    leading = gaps[gaps > 0]
    return float(np.median(leading)) if leading.size else 1.0


def fallback_share(agent: QAgent, X: pd.DataFrame, min_probability: float) -> float:
    """How often serving would set the agent aside and use today's rule instead.

    Inputs:  agent; X, one row per state; min_probability, the confidence needed (fallback.min_probability).
    Outputs: the share of rows, 0 to 1, whose best score is below min_probability.
    Why:     the quality gate scores the agent's own choices, but the service applies the fallback on top.
             If this share is close to 1 the service is really running today's rule, whatever the gate says:
             log it next to the gate result.
    Raises:  ValueError if X has no rows.
    """
    if X.empty:
        raise ValueError("No states given. Pass at least one row to measure the fallback share.")
    return float((agent.predict_proba(X).max(axis=1) < min_probability).mean())


def fit_q_agent(transitions: pd.DataFrame, feature_columns: list[str], actions: list[float], params: dict,
                random_state: int) -> QAgent:
    """Fitted Q iteration: learn the long-run value of every action from recorded transitions.

    Inputs:  transitions from collect_transitions; feature_columns, the state columns the agent may use;
             actions, the allowed actions (portfolio.actions); params, the rl config section (gamma,
             n_iterations, temperature, estimator settings: set early_stopping there, or scikit-learn
             switches it on by itself above 10,000 rows and the rounds stop being comparable);
             random_state, the seed of this agent: it picks the bootstrap resample of the transitions the
             agent learns from, and seeds the regressor.
    Outputs: a QAgent, with typical_gap measured on the states it learned from.
    Why:     each round fits one regressor on (state, action) against "reward now + gamma x the best value
             the previous round saw in the next state". After enough rounds the values look many weeks ahead.
    Raises:  ValueError on empty or incomplete transitions, an action outside the list, or a gamma,
             temperature or number of rounds that cannot work.
    """
    actions = sorted(float(action) for action in actions)
    _check_transitions(transitions, feature_columns, actions)
    _check_settings(params)
    # Each seed learns from its own resample of the experience (a bootstrap), so seeds really differ and the
    # median over seeds says how much the result depends on which transitions the agent happened to see.
    transitions = transitions.sample(frac=1.0, replace=True, random_state=random_state)
    inputs = transitions[[*feature_columns, ACTION_COLUMN]].astype(float)
    next_states = transitions[[NEXT_PREFIX + column for column in feature_columns]].astype(float)
    next_states.columns = feature_columns
    rewards = transitions[REWARD_COLUMN].to_numpy(dtype=float)
    continues = ~transitions[DONE_COLUMN].to_numpy(dtype=bool)

    targets, agent = rewards, None
    for _ in range(params["n_iterations"]):
        regressor = HistGradientBoostingRegressor(random_state=random_state, **params["estimator"])
        regressor.fit(inputs, targets)
        agent = QAgent(regressor, feature_columns, actions, params["temperature"])
        best_next = agent.q_values(next_states).to_numpy().max(axis=1)
        targets = rewards + params["gamma"] * continues * best_next
    agent.typical_gap = _typical_gap(agent.q_values(inputs[feature_columns]))
    logger.info("Fitted Q agent: %d transitions, %d features, %d rounds, gamma %.2f, seed %d, typical gap %.6f",
                len(transitions), len(feature_columns), params["n_iterations"], params["gamma"], random_state,
                agent.typical_gap)
    return agent


def median_seed(scores: dict[int, float]) -> int:
    """The seed whose score is the median: the agent to report and register.

    Inputs:  scores, seed -> validation score (for example max_drawdown_reduction).
    Outputs: the seed in the middle when the scores are sorted (the lower middle for an even count).
    Why:     with little data one lucky seed proves nothing; the README asks for the median of 5.
    Raises:  ValueError if there are no scores, or one of them is NaN or infinite (a run that failed must
             not be ranked as if it were a result).
    """
    if not scores:
        raise ValueError("No seed scores given. Fit and score at least one agent (rl.seeds in config.yaml).")
    broken = sorted(seed for seed, score in scores.items() if not math.isfinite(score))
    if broken:
        raise ValueError(f"Seeds {broken} have a score that is not a finite number. Check those runs before "
                         "taking the median.")
    ranked = sorted(scores, key=lambda seed: (scores[seed], seed))
    return ranked[(len(ranked) - 1) // 2]

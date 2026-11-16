"""
rl_agent.py  [The reinforcement learning challenger]
====================================================
Purpose   Learn, from simulated experience, how valuable each rebalancing action is over MANY weeks,
          and act on it: the agent the business case promises as a Stage 3 challenger
Method    Fitted Q iteration (batch reinforcement learning, scikit-learn only)
Input     Transitions (state, action, reward, next state) collected by running random policies through
          portfolio.simulate on the TRAIN years only
Output    A QAgent with predict and predict_proba, so infer.py, the fallback and the simulator use it unchanged
Called by main.py when model.type is "fitted_q"

Why it matters
- The Stage 2 classifier looks one week ahead. The agent values an action by what follows it too:
  Q(state, action) = reward now + gamma x the best Q in the next state
- Same simulator, same costs, same reward and same quality gate as the classifier: a fair challenger
- Little data (about 640 weekly decisions): small model, several seeds, the median reported

Tests     tests/test_rl_agent.py
"""

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src import portfolio

logger = logging.getLogger(__name__)

ACTION_COLUMN = "action"     # the action taken, as recorded by portfolio.simulate
REWARD_COLUMN = "reward"
DONE_COLUMN = "done"         # True on the last transition of an episode
NEXT_PREFIX = "next_"        # the next state's columns carry this prefix


class QAgent:
    """A fitted action-value model that picks the action with the highest value.

    Why a class: infer.run_inference, the 60% fallback and the simulator all expect predict and
    predict_proba, the way a scikit-learn classifier offers them. classes_ lists the actions in column order.
    """

    def __init__(self, regressor: HistGradientBoostingRegressor, feature_columns: list[str],
                 actions: list[float], temperature: float) -> None:
        """Keep the fitted regressor and what it needs to score a state for every action."""
        self.regressor = regressor
        self.feature_columns = list(feature_columns)
        self.classes_ = np.array(sorted(actions), dtype=float)
        self.temperature = float(temperature)

    def q_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """The value of every action in every state.

        Inputs:  X, one row per state, holding the feature columns.
        Outputs: frame with X's index and one column per action.
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
        """The best action per row; a tie goes to the smaller action (it trades less)."""
        return self.classes_[self.q_values(X).to_numpy().argmax(axis=1)]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """A score per action that sums to 1 per row: a softmax of the Q values.

        Why: the fallback asks "is the preferred action above 60%?". This is a ranking score, not a
        calibrated probability; temperature (config rl.temperature) sets how sharp it is.
        """
        scaled = self.q_values(X).to_numpy() / self.temperature
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        weights = np.exp(scaled)
        return weights / weights.sum(axis=1, keepdims=True)


def window_rewards(daily: pd.DataFrame, decisions: pd.DatetimeIndex, penalty_lambda: float) -> pd.Series:
    """The reward each decision actually earned in a simulated run.

    Inputs:  daily, the daily frame from portfolio.simulate (value after costs, cost_paid); decisions, its
             decision dates; penalty_lambda, the weight of new drops below the previous high.
    Outputs: reward per decision date: window return - trading cost - penalty_lambda x increase in drawdown,
             over the window from the day after the trade to the next trade day. NaN where the window is not
             complete (the last decisions of the run).
    Why:     the same reward the labels use (README), measured on the path the policy really took.
    """
    gross = daily["value"] / (1.0 - daily["cost_paid"])       # value before that day's trading cost
    rewards = pd.Series(np.nan, index=decisions, name=REWARD_COLUMN)
    for decided, when in portfolio.execution_schedule(daily.index, decisions).dropna().iterrows():
        rewards.loc[decided] = _one_window_reward(daily, gross, when, penalty_lambda)
    return rewards


def _one_window_reward(daily: pd.DataFrame, gross: pd.Series, when: pd.Series, penalty_lambda: float) -> float:
    """The reward of one decision: return over its window, minus its trading cost and the drawdown penalty.

    gross is the value before each day's trading cost; when holds the execution day and the window's first
    and last day (one row of portfolio.execution_schedule).
    """
    traded, last = when["execution_date"], when["reward_end"]
    window = daily["value"].loc[when["reward_start"]:last].copy()
    window.loc[last] = gross.loc[last]                        # the next trade's cost belongs to the next window
    start_peak = gross.loc[:traded].max()
    drawdown = window / np.maximum(start_peak, window.cummax()) - 1.0
    deeper = max(0.0, gross.loc[traded] / start_peak - 1.0 - float(drawdown.min()))
    window_return = float(window.iloc[-1] / daily["value"].loc[traded]) - 1.0
    return window_return - float(daily["cost_paid"].loc[traded]) - penalty_lambda * deeper


def _random_policy(actions: list[float], rng: np.random.Generator) -> Callable[[pd.Series], float]:
    """A policy that explores: every decision picks one of the actions at random."""
    def policy(_state_row: pd.Series) -> float:
        """One random action, whatever the state."""
        return float(rng.choice(actions))
    return policy


def _episode_transitions(daily: pd.DataFrame, decisions: pd.DataFrame, penalty_lambda: float) -> pd.DataFrame:
    """Turn one simulated run into rows of (state, action, reward, next state, done)."""
    features = decisions.drop(columns=ACTION_COLUMN)
    following = features.shift(-1).add_prefix(NEXT_PREFIX)
    rows = pd.concat([features, decisions[ACTION_COLUMN], following], axis=1)
    rows[REWARD_COLUMN] = window_rewards(daily, decisions.index, penalty_lambda)
    rows = rows.dropna(subset=[REWARD_COLUMN])
    rows[DONE_COLUMN] = rows.index == rows.index.max()        # the last transition of the run ends the episode
    return rows


def _center_rewards_by_date(transitions: pd.DataFrame) -> pd.DataFrame:
    """Subtract, from every reward, the average reward of all runs on the same decision date.

    Why: every run lives through the same market, and the market's move that week dwarfs the effect of the
    action. Removing the date's average leaves what the agent can actually influence: how much better or worse
    this state and action did than the average random run that week. The amount removed depends only on the
    date, never on the action, so the best action in every state is unchanged.
    """
    centered = transitions.copy()
    by_date = centered.groupby(level=0)[REWARD_COLUMN]
    centered[REWARD_COLUMN] = centered[REWARD_COLUMN] - by_date.transform("mean")
    return centered


def collect_transitions(market_data: tuple, target: pd.Series, cfg: dict, start: str, end: str,
                        n_episodes: int, seed: int) -> pd.DataFrame:
    """Gather experience by running random policies through the simulator.

    Inputs:  market_data (close_wide, volume_wide, market_features); target weights; cfg, the full config
             (reads portfolio.actions and portfolio.reward.penalty_lambda, plus what simulate reads);
             start and end (YYYY-MM-DD): pass the TRAIN years only; n_episodes, how many runs; seed.
    Outputs: one row per decision per run: the state columns, action, reward, the next state's columns
             (prefixed next_) and done. No row is dated after `end`. The reward is measured against the
             average of all runs on the same date (see _center_rewards_by_date).
    Why:     the agent can only learn the value of an action it has seen tried; random runs try them all.
    Raises:  ValueError if n_episodes is below 2 (and the errors of portfolio.simulate).
    """
    if n_episodes < 2:
        raise ValueError(f"n_episodes is {n_episodes}. Give at least 2 (rl.n_episodes in config.yaml): rewards "
                         "are measured against the average of the runs on each date.")
    actions = [float(action) for action in cfg["portfolio"]["actions"]]
    penalty_lambda = cfg["portfolio"]["reward"]["penalty_lambda"]
    rng = np.random.default_rng(seed)
    episodes = []
    for _ in range(n_episodes):
        daily, decisions = portfolio.simulate(*market_data, target, _random_policy(actions, rng), start, end, cfg)
        episodes.append(_episode_transitions(daily, decisions, penalty_lambda))
    transitions = _center_rewards_by_date(pd.concat(episodes))
    logger.info("Collected %d transitions from %d random runs, %s to %s", len(transitions), n_episodes, start, end)
    return transitions


def _check_transitions(transitions: pd.DataFrame, feature_columns: list[str], actions: list[float]) -> None:
    """Raise a clear error if the transitions are empty, incomplete or hold an action outside the list."""
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
    """Raise a clear error if the rl settings cannot work."""
    if params["n_iterations"] < 1:
        raise ValueError(f"rl.n_iterations is {params['n_iterations']}. Give at least 1.")
    if not 0.0 <= params["gamma"] <= 1.0:
        raise ValueError(f"rl.gamma is {params['gamma']}. Give a number from 0 to 1.")
    if params["temperature"] <= 0:
        raise ValueError(f"rl.temperature is {params['temperature']}. Give a number above 0.")


def fit_q_agent(transitions: pd.DataFrame, feature_columns: list[str], actions: list[float], params: dict,
                random_state: int) -> QAgent:
    """Fitted Q iteration: learn the long-run value of every action from recorded transitions.

    Inputs:  transitions from collect_transitions; feature_columns, the state columns the agent may use;
             actions, the allowed actions (portfolio.actions); params, the rl config section (gamma,
             n_iterations, temperature, estimator settings); random_state, the seed of this agent: it picks
             the bootstrap resample of the transitions the agent learns from, and seeds the regressor.
    Outputs: a QAgent.
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
    logger.info("Fitted Q agent: %d transitions, %d features, %d rounds, gamma %.2f, seed %d",
                len(transitions), len(feature_columns), params["n_iterations"], params["gamma"], random_state)
    return agent


def median_seed(scores: dict[int, float]) -> int:
    """The seed whose score is the median: the agent to report and register.

    Inputs:  scores, seed -> validation score (for example max_drawdown_reduction).
    Outputs: the seed in the middle when the scores are sorted (the lower middle for an even count).
    Why:     with little data one lucky seed proves nothing; the README asks for the median of 5.
    Raises:  ValueError if there are no scores.
    """
    if not scores:
        raise ValueError("No seed scores given. Fit and score at least one agent (rl.seeds in config.yaml).")
    ranked = sorted(scores, key=lambda seed: (scores[seed], seed))
    return ranked[(len(ranked) - 1) // 2]

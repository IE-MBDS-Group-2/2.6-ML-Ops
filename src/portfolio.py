"""
portfolio.py  [Step 4b: PORTFOLIO SIMULATOR, TODAY'S RULE AND LABELS]
=====================================================================
Purpose   Apply a weekly rebalancing decision to the 5-ETF portfolio with the README timing rule and trading
          costs, run today's rule (the baseline), and build the decision table the model learns from
Input     close_wide, volume_wide (date x ticker) from clean_data.to_wide, market features from features.py,
          target weights and the config sections portfolio and costs (passed in by main.py)
Output    simulate: a daily frame (value, cost_paid, weights) and a decision frame (one row per Friday)
          build_decision_table: the decision frame plus rule_action and the label best_action
Called by main.py (decision table, and model vs today's rule on validation)     Calls   costs.trade_costs

Timing rule (README, do not change)
- Decide at the close of decision day t (last trading day of the week, usually Friday)
- Trade at the close of the next trading day e
- The new weights earn returns from the day after e up to and including the next execution day
- Between trades the weights drift with prices: w_new = w x (1 + r) / sum(w x (1 + r))

Why it matters
- One simulate for every policy: today's rule, the trained model and test policies are charged by the same code
- The state row only uses data known at the close of t, so the label never leaks into the features

Tests     tests/test_portfolio.py
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterator

import numpy as np
import pandas as pd

from src import config, costs

logger = logging.getLogger(__name__)
RULE_ACTION_COLUMN = "rule_action"  # today's rule in the decision table; never a model input

_SUM_TOLERANCE = 1e-9    # target weights must sum to 1 within this
_TIE_TOLERANCE = 1e-12   # rewards closer than this count as a tie (the smaller action wins)
_REVIEWS = config.ALLOWED_VALUES[("rebalancing", "review")]          # daily, weekly, monthly
_REVIEW_PERIODS = {"daily": "D", "weekly": "W-SUN", "monthly": "M"}  # pandas period: one review per period
_PERIODS_PER_YEAR = {"daily": 252, "weekly": 52, "monthly": 12}      # reviews in a year, to annualise


def _check_review(review: str) -> None:
    """Raise if review is not one of config.ALLOWED_VALUES for rebalancing.review.

    Inputs:  review, a review frequency string (daily, weekly or monthly).
    Outputs: None; raises ValueError when review is not allowed.
    Why:     a typo in config.yaml should name the key and the allowed values, not fail on a later lookup.
    Raises:  ValueError if review is not daily, weekly or monthly.
    """
    if review not in _REVIEWS:
        raise ValueError(f"rebalancing.review is {review!r}. Use one of {', '.join(_REVIEWS)} in config.yaml.")


def review_frequency(cfg: dict) -> str:
    """The review frequency of a run: rebalancing.review from config (daily, weekly or monthly).

    Inputs:  cfg, the full config (reads rebalancing.review).
    Outputs: the review frequency string.
    Why:     one reader for the setting, so every module takes the frequency from the same place.
    """
    return cfg["rebalancing"]["review"]


def periods_per_year(review: str) -> int:
    """Reviews in a year for a review frequency (daily 252, weekly 52, monthly 12), to annualise per-review numbers.

    Inputs:  review, daily, weekly or monthly.
    Outputs: the number of reviews in a year, as an int.
    Why:     per-review returns and tracking errors must be annualised with the right factor.
    Raises:  ValueError if review is not daily, weekly or monthly.
    """
    _check_review(review)
    return _PERIODS_PER_YEAR[review]


def decision_dates(trading_days: pd.DatetimeIndex, start: str, end: str, review: str) -> pd.DatetimeIndex:
    """Pick the last trading day of each review period (day, week or month) inside [start, end].

    Inputs:  trading_days, the common calendar (index of close_wide); start and end as YYYY-MM-DD, inclusive;
             review, rebalancing.review from config: daily, weekly (today's process) or monthly.
    Outputs: a DatetimeIndex, one date per period (weekly: a Thursday when Friday is a holiday).
    Why:     the portfolio manager decides after the last close of each review period.
    Raises:  ValueError if review is not daily, weekly or monthly, or no trading day falls inside [start, end].
    """
    _check_review(review)
    days = pd.DatetimeIndex(trading_days)
    in_range = days[(days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))]
    if in_range.empty:
        raise ValueError(f"No trading days between {start} and {end}. Check the dates in config.yaml.")
    last_per_period = pd.Series(in_range, index=in_range).groupby(in_range.to_period(_REVIEW_PERIODS[review])).max()
    result = pd.DatetimeIndex(last_per_period.to_numpy())
    logger.info("Found %d %s decision dates from %s to %s", len(result), review, result[0].date(), result[-1].date())
    return result


def apply_action(weights: pd.Series, target: pd.Series, action: float) -> pd.Series:
    """Move the weights `action` of the way back to target.

    Inputs:  weights and target, fractions per ticker; action, 0.0 (do nothing) to 1.0 (fully back to target).
    Outputs: the new weights, same index as weights. They still sum to 1 when weights and target do.
    Why:     one formula, weights + action x (target - weights), for every policy.
    Raises:  ValueError if action is outside [0, 1] (or NaN).
    """
    if not 0.0 <= action <= 1.0:
        raise ValueError(f"action must be between 0 and 1 (share of the way back to target), got {action}.")
    return weights + action * (target.reindex(weights.index) - weights)


def current_rule_action(weights: pd.Series, target: pd.Series, date: pd.Timestamp, is_quarter_start: bool,
                        band: float) -> float:
    """Today's rule: rebalance fully on a quarter start, or when any weight drifts more than `band`.

    Inputs:  weights and target, fractions per ticker; date, the decision date (for the log);
             is_quarter_start, True on the first decision date of January, April, July and October;
             band, the drift band in absolute points (config: 0.05).
    Outputs: 1.0 (rebalance fully) or 0.0 (do nothing).
    Why:     this is the baseline the model must beat on max drawdown.
    """
    max_drift = float((weights - target.reindex(weights.index)).abs().max())
    fires = is_quarter_start or max_drift > band
    logger.debug("Current rule on %s: quarter start %s, max drift %.4f, fires %s",
                 date, is_quarter_start, max_drift, fires)
    return 1.0 if fires else 0.0


def rebalance_cost(weights: pd.Series, target: pd.Series, action: float, aum: float,
                   adv: pd.Series | None, cost_cfg: dict) -> float:
    """Total cost of applying `action`, as a fraction of portfolio value.

    Inputs:  weights and target per ticker; action in [0, 1]; aum, capital in the model portfolio;
             adv, trailing average daily dollar volume per ticker (or None); cost_cfg, the costs section.
    Outputs: one float, the sum over ETFs of costs.trade_costs for the weight change.
    Why:     every trade, and the full_rebalance_cost state feature, goes through Miguel's cost model.
    """
    delta = apply_action(weights, target, action) - weights
    adv_values = None if adv is None else np.asarray(adv, dtype=float)
    return float(np.sum(costs.trade_costs(delta.to_numpy(), aum, adv_values, cost_cfg)))


def execution_schedule(trading_days: pd.DatetimeIndex, decisions: pd.DatetimeIndex) -> pd.DataFrame:
    """For each decision date: when it trades and which days its reward window covers.

    Inputs:  trading_days, the calendar; decisions, from decision_dates (each must be a trading day).
    Outputs: a frame indexed by decision_date with execution_date (next trading day), reward_start (the day
             after execution) and reward_end (the next decision's execution day). NaT where the data ends.
    Why:     the README timing rule in one place, so simulate and the labels cannot disagree, and the
             "no gaps, no overlaps" rule can be tested directly.
    Raises:  ValueError if a decision date is not in trading_days.
    """
    days = pd.DatetimeIndex(trading_days)
    decisions = pd.DatetimeIndex(decisions)
    positions = days.get_indexer(decisions)
    if (positions < 0).any():
        missing = [str(day.date()) for day in decisions[positions < 0]]
        raise ValueError(f"Decision dates are not trading days: {missing}. Build them with decision_dates().")
    padded = days.append(pd.DatetimeIndex([pd.NaT, pd.NaT]))
    execution = padded[positions + 1]
    return pd.DataFrame(
        {
            "execution_date": execution,
            "reward_start": padded[positions + 2],
            "reward_end": execution[1:].append(pd.DatetimeIndex([pd.NaT])),
        },
        index=decisions,
    )


def make_current_rule_policy(target: pd.Series, decisions: pd.DatetimeIndex,
                             cfg: dict) -> Callable[[pd.Series], float]:
    """Wrap today's rule as a policy: a function from a state row to an action.

    Inputs:  target weights; decisions, the decision dates of the run (to find each quarter's first one);
             cfg, the full config (reads portfolio.current_rule: drift_band and quarter_months).
    Outputs: policy(state_row) -> 1.0 or 0.0. The state row's name is its decision date.
    Why:     the baseline then runs through simulate exactly like the trained model does.
    """
    rule = cfg["portfolio"]["current_rule"]
    decisions = pd.DatetimeIndex(decisions)
    months = decisions.to_period("M")
    first_in_month = np.r_[True, months[1:] != months[:-1]]
    in_quarter_month = decisions.to_series().dt.month.isin(rule["quarter_months"]).to_numpy()
    quarter_starts = set(decisions[first_in_month & in_quarter_month])
    tickers = list(target.index)

    def policy(state_row: pd.Series) -> float:
        """Today's rule for one decision date: 1.0 on a quarter start or a breached band, else 0.0.

        Inputs:  state_row, one decision's state (w_<T> columns), named by its decision date.
        Outputs: 1.0 (rebalance fully) or 0.0 (do nothing).
        Why:     lets today's rule run through simulate like any other policy.
        """
        weights = pd.Series([state_row[f"w_{ticker}"] for ticker in tickers], index=tickers)
        date = pd.Timestamp(state_row.name)
        return current_rule_action(weights, target, date, date in quarter_starts, rule["drift_band"])

    return policy


def _check_target(close_wide: pd.DataFrame, target: pd.Series) -> list[str]:
    """Fail fast if the target weights do not fit the price table; return the tickers in price order.

    Inputs:  close_wide (date x ticker); target, weights per ticker.
    Outputs: the tickers, in the column order of close_wide.
    Why:     a mismatch or weights that do not sum to 1 would silently corrupt every simulated value.
    Raises:  ValueError if the tickers differ from the price columns or the weights do not sum to 1.
    """
    if set(target.index) != set(close_wide.columns):
        raise ValueError(f"target tickers {sorted(target.index)} do not match the price columns "
                         f"{sorted(close_wide.columns)}. Check target_weights.policy in config.yaml.")
    if abs(float(target.sum()) - 1.0) > _SUM_TOLERANCE:
        raise ValueError(f"target weights must sum to 1, got {float(target.sum()):.6f}. "
                         "Check target_weights.policy in config.yaml.")
    return list(close_wide.columns)


def _problem_column(cfg: dict, key: str) -> str:
    """Read problem.<key> (a column name) from the config; a clear error beats a bare KeyError.

    Inputs:  cfg, the full config; key, the entry of the problem section (for example id_column).
    Outputs: the column name.
    Why:     the table's index and label names come from config, and a missing entry should say where to add it.
    Raises:  ValueError if problem.<key> is missing or empty.
    """
    name = cfg.get("problem", {}).get(key)
    if not name:
        raise ValueError(f"Config has no problem.{key}. Add it to the problem section of config.yaml.")
    return name


def _prepare_run(prices: tuple[pd.DataFrame, pd.DataFrame], market_features: pd.DataFrame, target: pd.Series,
                 period: tuple[str, str], cfg: dict) -> dict:
    """Everything simulate needs before the daily loop: calendar, schedule, returns, ADV, market rows.

    Inputs:  prices, (close_wide, volume_wide); market_features (date x feature); target weights;
             period, (start, end) as YYYY-MM-DD; cfg, the full config (reads rebalancing, costs, problem).
    Outputs: a dict with target, days, decisions, id_column, to_execute (execution day -> decision day),
             returns, adv and market (market features on the decision dates).
    Why:     keeps the daily loop in simulate short and puts the one-off set-up in a single place.
    """
    close_wide, volume_wide = prices
    tickers = _check_target(close_wide, target)
    days = close_wide.index[close_wide.index <= pd.Timestamp(period[1])]
    decisions = decision_dates(days, *period, review_frequency(cfg))
    days = days[days >= decisions[0]]
    schedule = execution_schedule(days, decisions)
    window = cfg["costs"]["slippage"]["adv_window"]
    return {
        "target": target.reindex(tickers),
        "days": days,
        "decisions": decisions,
        "id_column": _problem_column(cfg, "id_column"),
        "to_execute": {executed: decided for decided, executed in schedule["execution_date"].dropna().items()},
        "returns": close_wide[tickers].pct_change(fill_method=None).reindex(days),
        "adv": costs.average_daily_dollar_volume(close_wide[tickers], volume_wide[tickers], window),
        "market": market_features.reindex(decisions),
    }


def _grow(book: dict, day_returns: pd.Series) -> None:
    """One day of price moves: the value grows and the weights drift, w x (1 + r) / sum(w x (1 + r)).

    Inputs:  book, the running state (weights and value); day_returns, that day's return per ticker.
    Outputs: None; book["weights"] and book["value"] are updated in place.
    Why:     between trades the weights drift with prices, as the README timing rule says.
    """
    grown = book["weights"] * (1.0 + day_returns)
    growth = float(grown.sum())
    book["weights"] = grown / growth
    book["value"] *= growth


def _trade(book: dict, target: pd.Series, action: float, adv_row: pd.Series, cost_cfg: dict) -> float:
    """Execute an action at today's close: pay the cost out of the value, then move the weights.

    Inputs:  book, the running state; target weights; action in [0, 1]; adv_row, average daily dollar volume
             per ticker on the decision day; cost_cfg, the costs section (reads aum and the cost model).
    Outputs: the cost paid, as a fraction of portfolio value; book is updated in place.
    Why:     every trade is charged by the same cost model, whichever policy asked for it.
    """
    cost = rebalance_cost(book["weights"], target, action, cost_cfg["aum"], adv_row, cost_cfg)
    book["weights"] = apply_action(book["weights"], target, action)
    book["value"] *= 1.0 - cost
    return cost


def _state_row(day: pd.Timestamp, book: dict, run: dict, cfg: dict) -> pd.Series:
    """The state at the close of a decision day, before any trade: only data known at that close.

    Inputs:  day, the decision date; book, the running state (weights, value, peak, weeks); run, from
             _prepare_run (target, adv, market); cfg, the full config (reads costs).
    Outputs: a float Series named day: w_<T>, drift_<T>, drawdown, weeks_since_rebalance,
             full_rebalance_cost and the market features.
    Why:     the same row feeds the policy, the decision table and live requests, and holds nothing from
             after the close, so a label cannot leak into the features.
    """
    weights, target = book["weights"], run["target"]
    row = {f"w_{ticker}": weights[ticker] for ticker in target.index}
    row.update({f"drift_{ticker}": weights[ticker] - target[ticker] for ticker in target.index})
    row["drawdown"] = book["value"] / book["peak"] - 1.0
    row["weeks_since_rebalance"] = book["weeks"]
    row["full_rebalance_cost"] = rebalance_cost(weights, target, 1.0, cfg["costs"]["aum"], run["adv"].loc[day],
                                                cfg["costs"])
    row.update(run["market"].loc[day].to_dict())
    return pd.Series(row, dtype=float, name=day)


def _decide(day: pd.Timestamp, book: dict, run: dict, policy: Callable[[pd.Series], float],
            cfg: dict) -> pd.Series:
    """Build the state row, ask the policy, remember the action for the execution day.

    Inputs:  day, the decision date; book, the running state; run, from _prepare_run; policy, a function
             from a state row to an action; cfg, the full config.
    Outputs: the state row with the action appended; book["actions"] and book["weeks"] are updated.
    Why:     the action is decided at the close of day t but traded at the next close, so it is stored until then.
    """
    book["weeks"] += 1
    state = _state_row(day, book, run, cfg)
    action = float(policy(state))
    if action > 0:
        book["weeks"] = 0
    book["actions"][day] = action
    state["action"] = action
    return state


def simulate(close_wide: pd.DataFrame, volume_wide: pd.DataFrame, market_features: pd.DataFrame,
             target: pd.Series, policy: Callable[[pd.Series], float], start: str, end: str,
             cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run any policy through the README timing rule and trading costs.

    Inputs:  close_wide, volume_wide (date x ticker); market_features (date x feature, may have no columns);
             target weights; policy, a function state_row -> action in [0, 1]; start and end (YYYY-MM-DD);
             cfg, the full config (reads costs: aum, model, rates and slippage.adv_window).
    Outputs: daily frame indexed by date: value (starts at 1.0), cost_paid, w_<T> after any trade;
             decision frame indexed by decision_date: the state features, market features and the action.
    Why:     today's rule, the trained model and every test policy go through this same code.
    Raises:  ValueError if the target does not match the price columns or does not sum to 1 (and the
             errors of decision_dates, execution_schedule and apply_action).
    """
    run = _prepare_run((close_wide, volume_wide), market_features, target, (start, end), cfg)
    book = {"weights": run["target"].copy(), "value": 1.0, "peak": 1.0, "weeks": -1, "actions": {}}
    daily_rows, decision_rows = [], []
    for day in run["days"]:
        cost = 0.0
        if day > run["days"][0]:
            _grow(book, run["returns"].loc[day])
        decided_on = run["to_execute"].get(day)
        if decided_on is not None:
            cost = _trade(book, run["target"], book["actions"][decided_on], run["adv"].loc[decided_on],
                          cfg["costs"])
        book["peak"] = max(book["peak"], book["value"])
        daily_rows.append({"value": book["value"], "cost_paid": cost,
                           **{f"w_{ticker}": book["weights"][ticker] for ticker in run["target"].index}})
        if day in run["decisions"]:
            decision_rows.append(_decide(day, book, run, policy, cfg))
    return _finish(run, daily_rows, decision_rows)


def _finish(run: dict, daily_rows: list[dict], decision_rows: list[pd.Series]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn the loop's rows into the daily and decision frames, and log a one-line summary.

    Inputs:  run, from _prepare_run (days, id_column); daily_rows, one dict per day; decision_rows, one
             state Series per decision date.
    Outputs: the daily frame and the decision frame, as simulate returns them.
    Why:     keeps the frame building and the log line out of the daily loop.
    """
    daily = pd.DataFrame(daily_rows, index=pd.DatetimeIndex(run["days"]))
    decisions = pd.DataFrame(decision_rows).rename_axis(run["id_column"])
    logger.info("Simulated %d days and %d decisions: final value %.4f, total cost %.6f",
                len(daily), len(decisions), daily["value"].iloc[-1], daily["cost_paid"].sum())
    return daily, decisions


def _hold(weights: pd.Series, returns_block: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Buy and hold over a block of days: cumulative growth on each day, and the drifted weights at the end.

    Inputs:  weights at the start of the block, per ticker; returns_block, daily returns (day x ticker).
    Outputs: the growth of one unit on each day of the block, and the weights after the last day.
    Why:     the reward and the label measure a window with no trading inside it, so they share one formula.
    """
    paths = (1.0 + returns_block).cumprod()
    growth = (paths * weights).sum(axis=1)
    return growth, weights * paths.iloc[-1] / growth.iloc[-1]


def _window_start(state: dict, pre_block: pd.DataFrame) -> dict:
    """Drift the decision-day state over (t, e] to the execution close e, the same for every action.

    Inputs:  state, the decision-day weights, value, peak and adv; pre_block, daily returns on (t, e].
    Outputs: a dict with the weights, value, peak, drawdown and adv at the close of the execution day.
    Why:     the trade happens at e, so every candidate action starts from the same pre-trade position.
    """
    pre_growth, weights_e = _hold(state["weights"], pre_block)
    value = state["value"] * pre_growth.iloc[-1]
    peak = max(state["peak"], state["value"] * pre_growth.max())
    return {"weights": weights_e, "value": value, "peak": peak, "drawdown": value / peak - 1.0, "adv": state["adv"]}


def _reward(start: dict, action: float, window: pd.DataFrame, target: pd.Series, cfg: dict) -> float:
    """reward = window return - cost - penalty_lambda x increase in drawdown during the window.

    Inputs:  start, the state at the trade (from _window_start); action in [0, 1]; window, daily returns on the
             reward window; target weights; cfg, the full config (reads costs and portfolio.reward).
    Outputs: the reward, as a float.
    Why:     one reward function scores the labels and the reinforcement learning agent alike.
    """
    cost = rebalance_cost(start["weights"], target, action, cfg["costs"]["aum"], start["adv"], cfg["costs"])
    growth, _ = _hold(apply_action(start["weights"], target, action), window)
    values = start["value"] * (1.0 - cost) * growth
    drawdown = values / np.maximum(start["peak"], values.cummax()) - 1.0
    deeper = max(0.0, start["drawdown"] - float(drawdown.min()))
    return float(growth.iloc[-1]) - 1.0 - cost - cfg["portfolio"]["reward"]["penalty_lambda"] * deeper


def _label(start: dict, window: pd.DataFrame, target: pd.Series, cfg: dict) -> float:
    """Try every action from one decision's state at the trade and return the one with the highest reward.

    Inputs:  start, weights, value, peak, drawdown and adv at the close of the execution day, before the trade;
             window, daily returns on the reward window (execution day, next execution day]; target; cfg.
    Outputs: the best action (one of portfolio.actions). Ties go to the smaller action (it trades less).
    Why:     the label the classifier learns is the action that would have done best one window ahead.
    """
    best_action, best_reward = np.nan, -np.inf
    for action in sorted(cfg["portfolio"]["actions"]):
        reward = _reward(start, action, window, target, cfg)
        if reward > best_reward + _TIE_TOLERANCE:
            best_action, best_reward = float(action), reward
    return best_action


def _decision_windows(daily: pd.DataFrame, decisions: pd.DataFrame, close_wide: pd.DataFrame,
                      volume_wide: pd.DataFrame, target: pd.Series,
                      cfg: dict) -> Iterator[tuple[pd.Timestamp, dict, pd.DataFrame]]:
    """Walk through the decisions that have a full reward window.

    Inputs:  daily and decisions, the two frames from simulate; close_wide, volume_wide; target; cfg.
    Outputs: yields (decision date, state at the trade, reward-window returns) for each such decision.
    Why:     the labels and the rewards an agent learns from are measured on exactly the same windows.
    """
    returns = close_wide[target.index].pct_change(fill_method=None)
    adv = costs.average_daily_dollar_volume(close_wide[target.index], volume_wide[target.index],
                                            cfg["costs"]["slippage"]["adv_window"])
    peaks = daily["value"].cummax()
    weight_columns = [f"w_{ticker}" for ticker in target.index]
    for decided, when in execution_schedule(daily.index, decisions.index).dropna().iterrows():
        state = {"weights": decisions.loc[decided, weight_columns].set_axis(target.index).astype(float),
                 "value": daily.at[decided, "value"], "peak": peaks.at[decided], "adv": adv.loc[decided]}
        start = _window_start(state, returns.loc[decided:when["execution_date"]].iloc[1:])
        yield decided, start, returns.loc[when["reward_start"]:when["reward_end"]]


def _labels(daily: pd.DataFrame, decisions: pd.DataFrame, close_wide: pd.DataFrame, volume_wide: pd.DataFrame,
            target: pd.Series, cfg: dict) -> pd.Series:
    """best_action for every decision with a full reward window; NaN for the rest (the last one).

    Inputs:  daily and decisions, the two frames from simulate; close_wide, volume_wide; target; cfg.
    Outputs: a Series indexed by decision date, named problem.target_column.
    Why:     the label column of the decision table, built on the windows the rewards also use.
    """
    labels = pd.Series(np.nan, index=decisions.index, name=_problem_column(cfg, "target_column"))
    for decided, start, window in _decision_windows(daily, decisions, close_wide, volume_wide, target, cfg):
        labels.loc[decided] = _label(start, window, target, cfg)
    return labels


def realised_rewards(daily: pd.DataFrame, decisions: pd.DataFrame, close_wide: pd.DataFrame,
                     volume_wide: pd.DataFrame, target: pd.Series, cfg: dict) -> pd.Series:
    """The reward each decision of a simulated run earned for the action it actually took.

    Inputs:  daily and decisions, the two frames from simulate (decisions holds the action taken);
             close_wide, volume_wide; target weights; cfg, the full config (reads costs and portfolio.reward).
    Outputs: reward per decision date: window return - trading cost - penalty x increase in drawdown. NaN where
             the reward window is not complete (the last decisions of the run).
    Why:     the same reward function and the same windows as the labels, so what the classifier is taught
             and what the reinforcement learning agent is rewarded for cannot drift apart.
    """
    rewards = pd.Series(np.nan, index=decisions.index, name="reward")
    for decided, start, window in _decision_windows(daily, decisions, close_wide, volume_wide, target, cfg):
        rewards.loc[decided] = _reward(start, float(decisions.at[decided, "action"]), window, target, cfg)
    return rewards


def build_decision_table(close_wide: pd.DataFrame, volume_wide: pd.DataFrame, market_features: pd.DataFrame,
                         target: pd.Series, start: str, end: str, cfg: dict) -> pd.DataFrame:
    """The table the model learns from: one row per decision date, state + market features + label.

    Inputs:  as simulate, without the policy: today's rule is the policy, so the states follow the path the
             portfolio manager actually takes today.
    Outputs: frame indexed by decision_date: w_<T>, drift_<T>, drawdown, weeks_since_rebalance,
             full_rebalance_cost, the market features, rule_action (today's rule; NOT a model input, but
             infer's fallback and the evaluation use it) and best_action (0.0, 0.5 or 1.0; NaN on the last
             row, which has no full window: main drops it before training).
    Why:     labels say which action would have done best one window ahead, net of costs and drawdown.
    """
    days = close_wide.index[close_wide.index <= pd.Timestamp(end)]
    policy = make_current_rule_policy(target, decision_dates(days, start, end, review_frequency(cfg)), cfg)
    daily, decisions = simulate(close_wide, volume_wide, market_features, target, policy, start, end, cfg)
    target = target.reindex(close_wide.columns)
    target_column = _problem_column(cfg, "target_column")
    table = decisions.rename(columns={"action": RULE_ACTION_COLUMN})
    table[target_column] = _labels(daily, decisions, close_wide, volume_wide, target, cfg)
    logger.info("Decision table: %d rows, label counts %s", len(table),
                table[target_column].value_counts().sort_index().to_dict())
    return table


def _check_live_inputs(close_wide: pd.DataFrame, weights: pd.Series, drawdown: float,
                       weeks_since_rebalance: int) -> None:
    """Fail fast on the client's position inputs before any maths.

    Inputs:  close_wide (its columns are the portfolio's tickers); the client's weights, drawdown and weeks.
    Outputs: None; raises on the first problem found.
    Why:     a live request comes from outside the pipeline, so nothing about it can be assumed.
    Raises:  ValueError naming what is wrong and how to fix it.
    """
    if set(weights.index) != set(close_wide.columns):
        raise ValueError(f"Position weights are for tickers {sorted(weights.index)}, but the prices have "
                         f"{sorted(close_wide.columns)}. Send exactly one weight per portfolio ETF.")
    if abs(float(weights.sum()) - 1.0) > _SUM_TOLERANCE:
        raise ValueError(f"Position weights must sum to 1, got {float(weights.sum()):.6f}. "
                         "Send fractions of portfolio value (0.40, not 40).")
    if drawdown > 0:
        raise ValueError(f"drawdown must be 0 or negative (value / running peak - 1), got {drawdown}. "
                         "A portfolio at its peak has drawdown 0.")
    if weeks_since_rebalance < 0:
        raise ValueError(f"weeks_since_rebalance must be 0 or more, got {weeks_since_rebalance}. "
                         "Send the number of reviews since the last rebalance.")


def _last_market_row(market_features: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """The market features on the request date, as a one-row frame; every value must be known.

    Inputs:  market_features from features.compute_market_features; day, the last price date.
    Outputs: market_features restricted to [day] (what _state_row reads).
    Why:     the model was trained without gaps on its inputs, so a missing value means too little history.
    Raises:  ValueError if the date is absent or any feature is NaN on it.
    """
    market = market_features.reindex([day])
    missing = [str(column) for column in market.columns[market.iloc[0].isna()]]
    if missing:
        raise ValueError(f"Market features {missing} are missing on {day.date()}: too little price history. "
                         "Send more trading days of prices (the longest return window needs them).")
    return market


def live_state(close_wide: pd.DataFrame, volume_wide: pd.DataFrame, market_features: pd.DataFrame,
               weights: pd.Series, drawdown: float, weeks_since_rebalance: int, target: pd.Series,
               cfg: dict) -> pd.DataFrame:
    """The model's state row for one live request, built by the same code as the decision table.

    Inputs:  close_wide, volume_wide (date x ticker) of recent prices, ending on the request date;
             market_features from features.compute_market_features on those prices; the client's current
             weights (fractions per ticker), drawdown (value / peak - 1, at most 0) and weeks_since_rebalance;
             target weights; cfg, the full config (reads costs, portfolio.current_rule, problem.id_column).
    Outputs: one row indexed by the last date of close_wide (index name problem.id_column) with the decision
             table's columns in its order (w_<T>, drift_<T>, drawdown, weeks_since_rebalance,
             full_rebalance_cost, market features) plus rule_action, today's rule on that date.
    Why:     the API must feed the model exactly what it learned from; reusing _state_row and today's rule
             policy means a live request and the training table cannot drift apart.
    Raises:  ValueError if the weights do not fit the tickers or do not sum to 1, drawdown > 0,
             weeks_since_rebalance < 0, the market features are missing on the last date, or the target
             does not fit the prices.
    """
    tickers = _check_target(close_wide, target)
    _check_live_inputs(close_wide, weights, drawdown, weeks_since_rebalance)
    day = close_wide.index[-1]
    run = {
        "target": target.reindex(tickers),
        "adv": costs.average_daily_dollar_volume(close_wide[tickers], volume_wide[tickers],
                                                 cfg["costs"]["slippage"]["adv_window"]),
        "market": _last_market_row(market_features, day),
    }
    # value 1 + drawdown against a peak of 1 gives back exactly the client's drawdown in _state_row.
    book = {"weights": weights.reindex(tickers).astype(float), "value": 1.0 + drawdown, "peak": 1.0,
            "weeks": weeks_since_rebalance}
    state = _state_row(day, book, run, cfg)
    decisions = decision_dates(close_wide.index, str(close_wide.index[0].date()), str(day.date()),
                               review_frequency(cfg))
    state[RULE_ACTION_COLUMN] = make_current_rule_policy(run["target"], decisions, cfg)(state)
    logger.info("Live state on %s: max drift %.4f, today's rule %.1f", day.date(),
                float((book["weights"] - run["target"]).abs().max()), state[RULE_ACTION_COLUMN])
    return pd.DataFrame([state.to_numpy()], columns=state.index,
                        index=pd.DatetimeIndex([day], name=_problem_column(cfg, "id_column")))

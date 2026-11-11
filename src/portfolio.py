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

import numpy as np
import pandas as pd

from src import costs

logger = logging.getLogger(__name__)

_SUM_TOLERANCE = 1e-9    # target weights must sum to 1 within this
_TIE_TOLERANCE = 1e-12   # rewards closer than this count as a tie (the smaller action wins)


def decision_dates(trading_days: pd.DatetimeIndex, start: str, end: str) -> pd.DatetimeIndex:
    """Pick the last trading day of each calendar week inside [start, end].

    Inputs:  trading_days, the common calendar (index of close_wide); start and end as YYYY-MM-DD, inclusive.
    Outputs: a DatetimeIndex named decision_date, one date per week (a Thursday when Friday is a holiday).
    Why:     the portfolio manager decides once a week, after the last close of the week.
    Raises:  ValueError if no trading day falls inside [start, end].
    """
    days = pd.DatetimeIndex(trading_days)
    in_range = days[(days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))]
    if in_range.empty:
        raise ValueError(f"No trading days between {start} and {end}. Check the dates in config.yaml.")
    last_per_week = pd.Series(in_range, index=in_range).groupby(in_range.to_period("W-SUN")).max()
    result = pd.DatetimeIndex(last_per_week.to_numpy(), name="decision_date")
    logger.info("Found %d decision dates from %s to %s", len(result), result[0].date(), result[-1].date())
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


def rebalance_cost(weights: pd.Series, target: pd.Series, action: float, aum: float, adv,
                   cost_cfg: dict) -> float:
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
    decisions = pd.DatetimeIndex(decisions, name="decision_date")
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

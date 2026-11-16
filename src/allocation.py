"""
allocation.py  [Stage 3: THE TWO CHOICES A USER MAKES]
======================================================
Purpose   Turn the portfolio manager's two choices into what the pipeline understands: the target weights, and
          the rebalancing method as a policy (state row -> action); and say in one plain sentence why
Input     The choice names (target_weights.method, rebalancing.method), the config, the fitted model (or None),
          the target weights, the decision dates and close prices; for explain, one state row and its action
Output    resolve_target: weights per ticker; resolve_policy: policy(state_row) -> action in [0, 1];
          explain: one sentence
Called by main.py and api.py          Calls   config.ALLOWED_VALUES, rules.make_rule_policy

Why it matters
- The tool only decides how to return to the targets; it never moves away from them on a market view, so this is
  not "tactical" allocation
- No new rule logic: calendar, tolerance band and both are today's rule (rules.make_rule_policy "current") with
  the calendar and band from the rebalancing settings; smart is the trained model
- A wrong name stops with the allowed list, from the same config.ALLOWED_VALUES the config check and the API use

Tests     tests/test_allocation.py
"""

import logging

import numpy as np
import pandas as pd

from src import config, features

logger = logging.getLogger(__name__)

TARGET_KEY = ("target_weights", "method")
REBALANCING_KEY = ("rebalancing", "method")
SAME_WEIGHTS = 1e-9   # weights this close to the policy mix count as the policy mix


def _check_choice(key: tuple[str, str], value: str) -> None:
    """Raise if a choice is not one of its allowed values.

    Inputs:  key, (section, setting) in config.ALLOWED_VALUES; value, the name the user chose.
    Outputs: None.
    Why:     the modules check against the same list as load_config and the API, so the lists cannot drift.
    Raises:  ValueError naming the setting, the value and every allowed value.
    """
    allowed = config.ALLOWED_VALUES[key]
    if value not in allowed:
        raise ValueError(f"{key[0]}.{key[1]} is {value!r}. Allowed: {', '.join(allowed)}.")


def resolve_target(method: str, cfg: dict, train_returns: pd.DataFrame | None) -> pd.Series:
    """The target weights the portfolio is rebalanced towards, in config ticker order.

    Inputs:  method, target_weights.method ("policy" or a calculated method); cfg, the full config (reads
             data.tickers and target_weights.policy); train_returns, daily returns of the train years (only for
             the calculated methods; None is fine for "policy").
    Outputs: weights per ticker, long only, summing to 1 (config.load_config checks the policy mix).
    Why:     "policy" is the mix the Investment Committee approved, and the default.
    Raises:  ValueError for an unknown method; NotImplementedError for a calculated method until
             benchmarks.fit_weights is on dev (issue #63).
    """
    _check_choice(TARGET_KEY, method)
    if method != "policy":
        raise NotImplementedError(f"target_weights.method {method!r} calculates the weights with "
                                  "benchmarks.fit_weights (issue #63), which is not in this version yet. "
                                  "Use 'policy' for now.")
    tickers = cfg["data"]["tickers"]
    target = pd.Series(cfg["target_weights"]["policy"], dtype=float).reindex(tickers)
    logger.info("Target weights: %s (%d tickers)", method, len(target))
    return target


def _check_fit_inputs(target: pd.Series, tickers: list[str], train_returns: pd.DataFrame | None) -> None:
    """Raise if the target or the train returns do not cover exactly the configured tickers.

    Inputs:  target weights; tickers, data.tickers from config; train_returns, daily returns or None.
    Outputs: None.
    Why:     a missing ticker would silently drop out of the covariance and understate the tracking error.
    Raises:  ValueError naming what is missing or extra.
    """
    if sorted(target.index) != sorted(tickers):
        raise ValueError(f"The target covers {sorted(target.index)} but data.tickers is {sorted(tickers)}. "
                         "Give one weight for every configured ticker, and no others.")
    if train_returns is None:
        raise ValueError("target_fit needs train_returns (daily returns of the train years) to estimate the "
                         "tracking error of a target that is not the policy mix.")
    missing = [ticker for ticker in tickers if ticker not in train_returns.columns]
    if missing:
        raise ValueError(f"train_returns has no column for {missing}. Pass the daily returns of every ticker.")


def target_fit(target: pd.Series, cfg: dict, train_returns: pd.DataFrame | None) -> dict:
    """Say whether the model can serve this target, and how far the target is from the approved policy mix.

    Inputs:  target, the weights in use (from resolve_target); cfg, the full config (reads data.tickers and
             target_weights.policy); train_returns, daily returns of the train years (not needed for the
             policy mix itself).
    Outputs: {"fallback": bool, "reason": str, "expected_tracking_error": float}. The policy mix gives
             False, "" and 0.0. Any other target gives True, a reason, and the ex-ante annual tracking error
             sqrt((w - w_policy)' S (w - w_policy) x trading days), S the covariance of the daily returns.
    Why:     the model is trained for the policy mix, so another target falls back to the calendar-and-band
             rule; and a different target changes the mandate the Investment Committee approved, so the size of
             that change is always reported. Ex-ante, because one live request has no path to measure.
    Raises:  ValueError if the target or the returns do not cover exactly the configured tickers, or the
             returns are missing for a target that is not the policy mix.
    """
    tickers = cfg["data"]["tickers"]
    policy = pd.Series(cfg["target_weights"]["policy"], dtype=float).reindex(tickers)
    if sorted(target.index) == sorted(tickers) and np.allclose(target.reindex(tickers), policy, atol=SAME_WEIGHTS):
        return {"fallback": False, "reason": "", "expected_tracking_error": 0.0}
    _check_fit_inputs(target, tickers, train_returns)
    difference = (target.reindex(tickers) - policy).to_numpy()
    covariance = train_returns[tickers].cov().to_numpy()
    tracking = float(np.sqrt(difference @ covariance @ difference * features.TRADING_DAYS_PER_YEAR))
    reason = ("The model was trained for the policy weights, so with this target the calendar-and-band rule "
              f"decides. Expected tracking error to the policy weights: {tracking:.1%} a year.")
    logger.info("Target is not the policy mix: fallback to the rule, expected tracking error %.4f", tracking)
    return {"fallback": True, "reason": reason, "expected_tracking_error": tracking}

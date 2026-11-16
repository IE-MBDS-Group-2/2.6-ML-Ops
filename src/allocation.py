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
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src import config, features, rules

logger = logging.getLogger(__name__)

TARGET_KEY = ("target_weights", "method")
REBALANCING_KEY = ("rebalancing", "method")
SAME_WEIGHTS = 1e-9   # weights this close to the policy mix count as the policy mix
POINTS = 100          # rebalancing.tolerance_band is in percentage points; today's rule works in fractions
MONTHS_IN_YEAR = 12
NEVER = float("inf")  # a band no drift can exceed: the calendar-only method never trades on drift

Policy = Callable[[pd.Series], float]


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


def _calendar_months(calendar: str, cfg: dict) -> list[int]:
    """The months whose first review rebalances, for a calendar setting.

    Inputs:  calendar, rebalancing.calendar (monthly, quarterly or annually); cfg (reads
             portfolio.current_rule.quarter_months).
    Outputs: month numbers, for example [1, 4, 7, 10] for quarterly.
    Why:     quarterly uses today's quarter months from config, so calendar_and_band with quarterly and 5 is
             exactly today's rule; annually is the first of those months.
    """
    quarter_months = list(cfg["portfolio"]["current_rule"]["quarter_months"])
    by_calendar = {
        "monthly": list(range(1, MONTHS_IN_YEAR + 1)),
        "quarterly": quarter_months,
        "annually": quarter_months[:1],
    }
    return by_calendar[calendar]


def _rule_settings(method: str, cfg: dict) -> dict:
    """Today's rule's two settings (drift band, calendar months) for one of the rule-based methods.

    Inputs:  method, calendar, tolerance_band or calendar_and_band; cfg (reads rebalancing).
    Outputs: {"drift_band": fraction, "quarter_months": [...]}, the shape of portfolio.current_rule.
    Why:     all three methods are today's rule with a part switched off: calendar has no band (NEVER),
             tolerance_band has no calendar months.
    """
    settings = cfg["rebalancing"]
    months = [] if method == "tolerance_band" else _calendar_months(settings["calendar"], cfg)
    band = NEVER if method == "calendar" else float(settings["tolerance_band"]) / POINTS
    return {"drift_band": band, "quarter_months": months}


def _model_policy(model: Pipeline) -> Policy:
    """Wrap the fitted model as a policy, exactly as main.py does for the Stage 2 run.

    Inputs:  model, the fitted pipeline (it remembers its feature columns in feature_names_in_).
    Outputs: policy(state_row) -> the model's action.
    Why:     smart is our model deciding at each review; reading the columns from the model means a caller
             cannot pass the wrong list.
    """
    columns = list(model.feature_names_in_)

    def policy(state_row: pd.Series) -> float:
        """Inputs: a state row. Outputs: the model's action. Why: smart rebalancing."""
        return float(model.predict(state_row[columns].to_frame().T.astype(float))[0])

    return policy


def resolve_policy(method: str, model: Pipeline | None, target: pd.Series, decisions: pd.DatetimeIndex,
                   close_wide: pd.DataFrame, cfg: dict) -> Policy:
    """The rebalancing method as a policy: a function from a state row to an action in [0, 1].

    Inputs:  method, rebalancing.method; model, the fitted pipeline or None; target, the weights to return to;
             decisions, the run's decision dates; close_wide (date x ticker); cfg, the full config (reads
             rebalancing and portfolio.current_rule).
    Outputs: policy(state_row) -> action, ready for portfolio.simulate or one live state row.
    Why:     smart is the model (today's rule when there is no model, D8); calendar, tolerance_band and
             calendar_and_band are today's rule from rules.make_rule_policy with the user's calendar and band.
    Raises:  ValueError for an unknown method, listing the allowed ones.
    """
    _check_choice(REBALANCING_KEY, method)
    if method == "smart" and model is not None:
        logger.info("Rebalancing: smart (the trained model decides)")
        return _model_policy(model)
    rule_method = "calendar_and_band" if method == "smart" else method
    if method == "smart":
        logger.warning("Rebalancing: smart, but no model is loaded, so the calendar-and-band rule decides (D8)")
    rule_cfg = {**cfg, "portfolio": {**cfg["portfolio"], "current_rule": _rule_settings(rule_method, cfg)}}
    logger.info("Rebalancing: %s (%s, band %s)", rule_method, cfg["rebalancing"]["calendar"],
                rule_cfg["portfolio"]["current_rule"]["drift_band"])
    return rules.make_rule_policy("current", target, decisions, close_wide, rule_cfg)

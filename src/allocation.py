"""
allocation.py  [Stage 3: THE TWO CHOICES A USER MAKES]
======================================================
Purpose   Turn the portfolio manager's two choices into what the pipeline understands: the target weights, and
          the rebalancing method as a policy (state row -> action); and say in one plain sentence why
Input     The choice names (target_weights.method, rebalancing.method), the config, the fitted model (or None),
          the target weights, the decision dates and close prices; for explain, one state row and its action
Output    resolve_target: weights per ticker; resolve_policy: policy(state_row) -> action in [0, 1];
          explain: one sentence
Called by main.py and api.py          Calls   config.ALLOWED_VALUES, rules.make_rule_policy, benchmarks.fit_weights

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

from src import benchmarks, config, features, rules

logger = logging.getLogger(__name__)

TARGET_KEY = ("target_weights", "method")
REBALANCING_KEY = ("rebalancing", "method")
SAME_WEIGHTS = 1e-9   # weights this close to the policy mix count as the policy mix
POINTS = 100          # rebalancing.tolerance_band is in percentage points; today's rule works in fractions
MONTHS_IN_YEAR = 12
NEVER = float("inf")  # a band no drift can exceed: the calendar-only method never trades on drift
DRIFT_DECIMALS = 9    # drifts are compared rounded, so float noise never decides which holding is named
ACTION_WORDS = {0.0: "No trade", 0.5: "Rebalance halfway", 1.0: "Rebalance fully"}

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
             data.tickers and target_weights.policy; the calculated methods also read benchmarks);
             train_returns, daily returns of the train years (only for the calculated methods; None is fine
             for "policy").
    Outputs: weights per ticker, long only, summing to 1 (config.load_config checks the policy mix).
    Why:     "policy" is the mix the Investment Committee approved, and the default. The calculated methods
             call benchmarks.fit_weights with the benchmarks settings, so they match the benchmark report.
    Raises:  ValueError for an unknown method, a calculated method without train_returns or with a ticker
             missing from them, plus the errors of benchmarks.fit_weights.
    """
    _check_choice(TARGET_KEY, method)
    tickers = cfg["data"]["tickers"]
    if method != "policy":
        if train_returns is None:
            raise ValueError(f"target_weights.method {method!r} fits the weights on the train years: pass "
                             "train_returns (daily returns of the train years).")
        missing = [ticker for ticker in tickers if ticker not in train_returns.columns]
        if missing:
            raise ValueError(f"train_returns has no column for {missing}. Pass the daily returns of every ticker.")
        target = benchmarks.fit_weights(train_returns[tickers], method, cfg["benchmarks"])
        logger.info("Target weights: %s fitted on %d train days (%d tickers)", method, len(train_returns),
                    len(target))
        return target
    target = pd.Series(cfg["target_weights"]["policy"], dtype=float).reindex(tickers)
    logger.info("Target weights: %s (%d tickers)", method, len(target))
    return target


def stored_target(method: str, cfg: dict) -> pd.Series:
    """Read the configured policy or a previously fitted allocation in ticker order."""
    _check_choice(TARGET_KEY, method)
    settings = cfg["target_weights"]
    weights = settings["policy"] if method == "policy" else settings.get("calculated", {}).get(method)
    if weights is None:
        raise ValueError(f"No stored weights for {method!r}. Run python -m scripts.fit_targets at retrain time.")
    return pd.Series(weights, dtype=float).reindex(cfg.get("data", {}).get("tickers", list(weights)))


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


def _target_tracking_error(target: pd.Series, cfg: dict, train_returns: pd.DataFrame | None) -> float:
    """Use training returns during fitting, or the reviewed training-risk estimate during serving."""
    tickers = cfg["data"]["tickers"]
    policy = pd.Series(cfg["target_weights"]["policy"], dtype=float).reindex(tickers)
    if train_returns is None:
        settings = cfg["target_weights"]
        for method, weights in settings.get("calculated", {}).items():
            if np.allclose(target.reindex(tickers), pd.Series(weights).reindex(tickers), atol=SAME_WEIGHTS, rtol=0):
                return float(settings["fit"]["tracking_error"][method])
    _check_fit_inputs(target, tickers, train_returns)
    difference = (target.reindex(tickers) - policy).to_numpy()
    covariance = train_returns[tickers].cov().to_numpy()
    return float(np.sqrt(difference @ covariance @ difference * features.TRADING_DAYS_PER_YEAR))


def model_target(cfg: dict) -> dict:
    """Describe the target used to train a model, stored inside its joblib artifact."""
    method = cfg["target_weights"]["method"]
    return {"method": method, "weights": stored_target(method, cfg).to_dict()}


def model_target_matches(model: object, cfg: dict) -> bool:
    """Only serve artifacts with recorded training weights that match the current target."""
    return getattr(model, "target_metadata_", None) == model_target(cfg)


def target_fit(target: pd.Series, cfg: dict, train_returns: pd.DataFrame | None) -> dict:
    """Say whether the model can serve this target, and how far the target is from the approved policy mix.

    Inputs:  target, the weights in use (from resolve_target); cfg, the full config (reads data.tickers and
             target_weights.policy); train_returns, daily training returns at fit time, or None to read
             the stored training-period estimate for a calculated target during serving.
    Outputs: {"fallback": bool, "reason": str, "expected_tracking_error": float}. The policy mix gives
             False, "" and 0.0. Any other target gives True, a reason, and the ex-ante annual tracking error
             sqrt((w - w_policy)' S (w - w_policy) x trading days), S the covariance of the daily returns.
    Why:     smart serving is restricted to the policy mix, so another target uses calendar-and-band.
             Training-risk estimates are computed at refit time and stored with the reviewed weights.
             Serving reads that estimate when train_returns is None.
    Raises:  ValueError if the target or the returns do not cover exactly the configured tickers, or the
             returns are missing for a target that is not the policy mix.
    """
    tickers = cfg["data"]["tickers"]
    policy = pd.Series(cfg["target_weights"]["policy"], dtype=float).reindex(tickers)
    if sorted(target.index) == sorted(tickers) and np.allclose(target.reindex(tickers), policy, atol=SAME_WEIGHTS):
        return {"fallback": False, "reason": "", "expected_tracking_error": 0.0}
    tracking = _target_tracking_error(target, cfg, train_returns)
    reason = ("Smart serving is restricted to the policy weights, so with this target the calendar-and-band "
              f"rule decides. Expected tracking error to the policy weights: {tracking:.1%} a year.")
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


def _feature_columns(model: Pipeline) -> list[str]:
    """The state columns a fitted model reads.

    Inputs:  model, a scikit-learn pipeline (feature_names_in_) or the RL agent, rl_agent.QAgent (feature_columns).
    Outputs: the column names, in the order the model was fitted on.
    Why:     smart works with either model type (model.type classifier or fitted_q).
    Raises:  ValueError if the model has neither attribute.
    """
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "feature_columns"):
        return list(model.feature_columns)
    raise ValueError(f"The model ({type(model).__name__}) has no feature_names_in_ or feature_columns, so smart "
                     "cannot tell which state columns to pass. Fit it with train.train_model or rl_agent.fit_q_agent.")


def _model_policy(model: Pipeline) -> Policy:
    """Wrap the fitted model as a policy, exactly as main.py does for the Stage 2 run.

    Inputs:  model, the fitted pipeline or the RL agent (both remember their feature columns).
    Outputs: policy(state_row) -> the model's action.
    Why:     smart is our model deciding at each review; reading the columns from the model means a caller
             cannot pass the wrong list.
    Raises:  ValueError if the model does not say which columns it reads.
    """
    columns = _feature_columns(model)

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


def _furthest(state_row: pd.Series, tickers: list[str]) -> tuple[str, float]:
    """The holding furthest from target, and its drift as a fraction (positive = above target).

    Inputs:  state_row, with drift_<T> columns; tickers, in config order.
    Outputs: (ticker, drift). On a tie the first ticker in config order is named.
    Why:     one holding is enough to explain a rebalance; the worst one is the one the bands react to.
    """
    drifts = pd.Series({ticker: float(state_row[f"drift_{ticker}"]) for ticker in tickers})
    ticker = str(drifts.abs().round(DRIFT_DECIMALS).idxmax())
    return ticker, float(drifts[ticker])


def _detail(action: float, ticker: str, drift: float, band: float) -> str:
    """The sentence for one action, without the fallback part.

    Inputs:  action, 0.0, 0.5 or 1.0; ticker and drift, from _furthest; band, rebalancing.tolerance_band (points).
    Outputs: for example "Rebalance fully: SPY is 6.1 points above target, outside the 5-point band."
    Why:     a portfolio manager reads the action and the one number behind it.
    """
    points = abs(drift) * POINTS
    if action == 0.0:
        return f"{ACTION_WORDS[action]}: every holding is within {points:.1f} points of target."
    side = "above" if drift > 0 else "below"
    outside = f", outside the {band:g}-point band" if points > band else ""
    return f"{ACTION_WORDS[action]}: {ticker} is {points:.1f} points {side} target{outside}."


def explain(state_row: pd.Series, action: float, fallback: bool, cfg: dict) -> str:
    """One plain sentence that says what to do and why, for the API's reason field.

    Inputs:  state_row, the live state (drift_<T> columns); action, the recommendation (0.0, 0.5 or 1.0);
             fallback, True when today's rule decided because the model was not confident enough; cfg, the full
             config (reads data.tickers, rebalancing.tolerance_band and fallback.min_probability).
    Outputs: the sentence, for example "No trade: every holding is within 2.0 points of target."
    Why:     every answer comes with a reason a portfolio manager can check against the positions.
    Raises:  ValueError if the action is not 0.0, 0.5 or 1.0.
    """
    if action not in ACTION_WORDS:
        raise ValueError(f"explain got action {action}; expected one of {sorted(ACTION_WORDS)}.")
    ticker, drift = _furthest(state_row, cfg["data"]["tickers"])
    sentence = _detail(action, ticker, drift, float(cfg["rebalancing"]["tolerance_band"]))
    if not fallback:
        return sentence
    confidence = cfg["fallback"]["min_probability"]
    return f"The model was less than {confidence:.0%} sure, so today's rule decides. {sentence}"

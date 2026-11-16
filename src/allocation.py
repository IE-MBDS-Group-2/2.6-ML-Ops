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

import pandas as pd

from src import config

logger = logging.getLogger(__name__)

TARGET_KEY = ("target_weights", "method")
REBALANCING_KEY = ("rebalancing", "method")


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

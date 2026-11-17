"""
evaluate.py  [Step 6: EVALUATE]
===============================
Purpose   Score the fitted pipeline on data it has never seen, and decide go / no-go with the quality gate
Input     Fitted pipeline + X_eval, y_eval (validation split; test split once, at the end)
          Daily portfolio values and costs for the agent and for today's rule
Output    Dictionaries of plain floats, e.g. {"accuracy": 0.61, "f1_macro": 0.48}, and a True / False gate result
Called by main.py          Feeds  the quality gate (stage 2) and experiment tracking (stage 3)

Why it matters
- Never score on training data: it hides overfitting
- The business goal is a smaller worst loss (max drawdown) than today's rule, so the gate is measured in money terms
- Plain numbers are easy to log, compare between runs and turn into a go / no-go decision

Tests     tests/test_evaluate.py
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def evaluate_model(model: Pipeline, X_eval: pd.DataFrame, y_eval: pd.Series, problem_type: str) -> dict[str, float]:
    """Score a fitted pipeline on a held-out split.

    Inputs:  model: fitted pipeline; X_eval, y_eval: validation (or, once, test) features and labels;
             problem_type: "classification" (the stage 2 model predicts the best action 0.0 / 0.5 / 1.0)
             or "regression".
    Outputs: {"accuracy", "f1_macro"} for classification, {"rmse", "mae", "r2"} for regression, as
             plain Python floats.
    Why:     f1_macro weighs the three actions equally, so a model that always says "do nothing" scores badly
             even if that action is the most common one.
    """
    if not hasattr(model, "predict"):
        raise TypeError(f"evaluate_model: model must have a predict method, got {type(model).__name__}. "
                        "Pass the fitted pipeline from train_model.")
    if problem_type not in ("classification", "regression"):
        raise ValueError(f"evaluate_model: problem_type must be 'classification' or 'regression', "
                         f"got '{problem_type}'.")
    if len(X_eval) == 0:
        raise ValueError("evaluate_model: X_eval is empty. Check the validation split dates in config.yaml.")
    if len(X_eval) != len(y_eval):
        raise ValueError(
            f"evaluate_model: X_eval and y_eval must have the same number of rows, got {len(X_eval)} and "
            f"{len(y_eval)}. Check the split keeps features and labels aligned."
        )
    if problem_type == "regression":
        return _regression_metrics(model, X_eval, y_eval)
    # Actions 0.0 / 0.5 / 1.0 are categories, but sklearn reads 0.5 as a continuous number and refuses to score it.
    # Comparing them as text keeps the metrics correct whatever numeric or text encoding the labels use.
    y_true = pd.Series(y_eval).astype(str).to_numpy()
    y_pred = pd.Series(model.predict(X_eval)).astype(str).to_numpy()
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    logger.info("Scored %d rows: accuracy %.4f, f1_macro %.4f", len(X_eval), result["accuracy"], result["f1_macro"])
    return result


def _regression_metrics(model: Pipeline, X_eval: pd.DataFrame, y_eval: pd.Series) -> dict[str, float]:
    """Fixed regression metrics: rmse, mae and r2, as plain floats."""
    y_pred = model.predict(X_eval)
    result = {
        "rmse": float(np.sqrt(mean_squared_error(y_eval, y_pred))),
        "mae": float(mean_absolute_error(y_eval, y_pred)),
        "r2": float(r2_score(y_eval, y_pred)),
    }
    logger.info("Scored %d rows: rmse %.4f, mae %.4f, r2 %.4f",
                len(X_eval), result["rmse"], result["mae"], result["r2"])
    return result


def portfolio_metrics(values: pd.Series, costs_paid: pd.Series, periods_per_year: int = 252) -> dict[str, float]:
    """Summarise a simulated portfolio path in the risk and return numbers the business case uses.

    Inputs:  values: daily portfolio value (e.g. starting at 1.0); costs_paid: trading cost paid each day;
             periods_per_year: trading days in a year (252 by convention).
    Outputs: {"max_drawdown", "annual_return", "annual_volatility", "sharpe", "annual_trading_cost"} as floats.
    Why:     The quality gate compares the agent's max drawdown with today's rule; the other numbers show
             what that protection cost in return, risk and fees.
    """
    if len(values) < 2:
        raise ValueError(
            f"portfolio_metrics needs at least 2 values to measure a return, got {len(values)}. "
            "Check the simulation produced a full daily path."
        )
    drawdown = values / values.cummax() - 1
    years = (len(values) - 1) / periods_per_year
    annual_return = (values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1
    annual_volatility = values.pct_change().dropna().std() * np.sqrt(periods_per_year)
    sharpe = annual_return / annual_volatility if annual_volatility > 0 else 0.0
    result = {
        "max_drawdown": float(drawdown.min()),
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_volatility),
        "sharpe": float(sharpe),
        "annual_trading_cost": float(costs_paid.sum() / years),
    }
    logger.info("Portfolio max drawdown %.4f, annual return %.4f over %.2f years",
                result["max_drawdown"], result["annual_return"], years)
    return result


def max_drawdown_reduction(agent_mdd: float, baseline_mdd: float) -> float:
    """Share of today's rule's worst loss that the agent avoided.

    Inputs:  agent_mdd, baseline_mdd: max drawdowns (negative numbers, e.g. -0.16 and -0.20).
    Outputs: 1 - |agent| / |baseline|; 0.20 means a 20% smaller worst loss, below 0 means the agent did worse.
    Why:     This is the metric the quality gate checks (>= 0.20 on validation).
    """
    if baseline_mdd == 0:
        raise ValueError(
            "max_drawdown_reduction: baseline max drawdown is 0, so a reduction cannot be computed. "
            "Check the baseline simulation covers a period with price moves."
        )
    reduction = 1 - abs(agent_mdd) / abs(baseline_mdd)
    logger.info("Max drawdown reduction %.4f (agent %.4f vs baseline %.4f)", reduction, agent_mdd, baseline_mdd)
    return float(reduction)


def check_gate(value: float, comparison: str, threshold: float, tolerance: float = 1e-9) -> bool:
    """Go / no-go decision: does the metric clear the agreed threshold?

    Inputs:  value: the measured metric; comparison: ">=" or "<="; threshold: the target from config;
             tolerance: values this close to the threshold count as equal (absorbs floating-point rounding).
    Outputs: True if the gate is met, otherwise False.
    Why:     One explicit pass / fail that main.py reports honestly, whichever way it goes.
    """
    if comparison == ">=":
        passed = value >= threshold - tolerance
    elif comparison == "<=":
        passed = value <= threshold + tolerance
    else:
        raise ValueError(f"check_gate: comparison must be '>=' or '<=', got '{comparison}'. Fix it in config.yaml.")
    logger.info("Quality gate %s: %.4f %s %.4f", "met" if passed else "NOT met", value, comparison, threshold)
    return bool(passed)


# ---------------------------------------------------------------- guards and the release decision (Stage 3)

GUARD_LIMIT_KEYS = ("max_return_shortfall", "max_cost_ratio", "max_tracking_error")  # guards section of config.yaml
GUARD_METRIC_KEYS = ("annual_return", "annual_trading_cost", "tracking_error")      # per portfolio, agent side
_GUARD_TOLERANCE = 1e-9  # a value this close to its limit counts as at the limit (floating-point rounding)
_GUARD_ENTRY_KEYS = {"value", "limit", "passed"}


def tracking_error(portfolio_values: pd.Series, target_values: pd.Series, periods_per_year: int) -> float:
    """How far the portfolio strays from the target-weight portfolio, as an annualised spread of returns.

    Inputs:  portfolio_values, target_values: value paths on the same dates (weekly in the business case);
             periods_per_year: 52 for weekly values.
    Outputs: std of (portfolio return - target return) x sqrt(periods_per_year), a plain float; 0.0 when the
             two paths move identically.
    Why:     the README guard: an agent must not cut drawdown by drifting away from the approved allocation.
    Raises:  ValueError if the two series differ in length or have fewer than 3 values (2 returns).
    """
    if len(portfolio_values) != len(target_values):
        raise ValueError(
            f"tracking_error: portfolio_values and target_values must have the same length, got "
            f"{len(portfolio_values)} and {len(target_values)}. Resample both to the same dates first."
        )
    if len(portfolio_values) < 3:
        raise ValueError(
            f"tracking_error needs at least 2 returns (3 values) per series, got {len(portfolio_values)} values. "
            "Check the simulation covers more than two periods."
        )
    portfolio_returns = pd.Series(np.asarray(portfolio_values, dtype=float)).pct_change().dropna()
    target_returns = pd.Series(np.asarray(target_values, dtype=float)).pct_change().dropna()
    result = float((portfolio_returns - target_returns).std() * np.sqrt(periods_per_year))
    logger.info("Tracking error %.6f over %d returns", result, len(portfolio_returns))
    return result


def _require(source: dict, keys: tuple[str, ...], label: str) -> None:
    """Fail fast when a limit or a metric guard_report needs is absent or not set (None).

    Inputs:  source, the dict to check; keys, the names it must hold; label, what it is (for the message).
    Outputs: None.
    Why:     a guard compared with a missing number would silently pass or crash later; say which one is missing.
    Raises:  ValueError naming every missing key.
    """
    missing = [key for key in keys if source.get(key) is None]
    if missing:
        raise ValueError(f"guard_report: {label} has no value for {missing}. Fill them in before reporting guards.")


def _guard(value: float, limit: float) -> dict:
    """One guard entry: the measured value, its limit, and whether it stays within the limit.

    A value that is not finite (a cost ratio against a baseline that trades nothing) fails and is reported as
    None, so metrics.json stays strict JSON (json.dumps would write Infinity, which strict readers reject).
    """
    if not np.isfinite(value):
        return {"value": None, "limit": float(limit), "passed": False}
    return {"value": float(value), "limit": float(limit), "passed": bool(value <= limit + _GUARD_TOLERANCE)}


def _cost_ratio(agent_cost: float, baseline_cost: float) -> float:
    """Agent's annual trading cost as a multiple of the baseline's; a baseline that trades nothing allows only 0."""
    if baseline_cost == 0:
        return 0.0 if agent_cost == 0 else float("inf")
    return agent_cost / baseline_cost


def guard_report(agent: dict, baseline: dict, limits: dict) -> dict:
    """Check the README's three guards for the agent against today's rule.

    Inputs:  agent: portfolio_metrics(...) plus "tracking_error"; baseline: portfolio_metrics(...) of today's rule;
             limits: the guards section of config.yaml (max_return_shortfall, max_cost_ratio, max_tracking_error).
    Outputs: {"return_shortfall", "cost_ratio", "tracking_error"}, each {"value", "limit", "passed"}, for
             metrics.json under "guards".
    Why:     a smaller drawdown is only worth having if return, cost and mandate stay acceptable (README).
             Reported with every run; they are not part of the quality gate.
    Raises:  ValueError if a limit is missing or null, or a metric is missing.
    """
    _require(limits, GUARD_LIMIT_KEYS, "limits (the guards section of config.yaml)")
    _require(agent, GUARD_METRIC_KEYS, "agent metrics")
    _require(baseline, GUARD_METRIC_KEYS[:2], "baseline metrics")
    report = {
        "return_shortfall": _guard(baseline["annual_return"] - agent["annual_return"],
                                   limits["max_return_shortfall"]),
        "cost_ratio": _guard(_cost_ratio(agent["annual_trading_cost"], baseline["annual_trading_cost"]),
                             limits["max_cost_ratio"]),
        "tracking_error": _guard(agent["tracking_error"], limits["max_tracking_error"]),
    }
    summary = ", ".join(name + (" passed" if entry["passed"] else " FAILED") for name, entry in report.items())
    logger.info("Guards: %s", summary)
    return report


def release_decision(gate_met: bool, guards: dict) -> dict:
    """The one-time go / no-go on test 2022-2025: the gate and every guard must pass (README "Do not release if").

    Inputs:  gate_met: the quality gate on test; guards: guard_report(...) output, plus any extra entry in the
             same shape (e.g. "challenger": the selected simple rule's max drawdown, from rules.py).
    Outputs: {"release": bool, "reasons": [one line per failed condition]}; reasons is empty on a release.
    Why:     one explicit decision that names everything that blocked it, so nobody has to dig through metrics.
    Raises:  ValueError if a guard entry lacks value, limit or passed.
    """
    reasons = [] if gate_met else ["quality gate not met on test: max drawdown misses the KPI target"]
    for name, entry in guards.items():
        if not _GUARD_ENTRY_KEYS <= set(entry):
            raise ValueError(f"release_decision: guard {name!r} must have value, limit and passed (from "
                             f"guard_report), got {sorted(entry)}.")
        if not entry["passed"]:
            value = "infinite" if entry["value"] is None else f"{entry['value']:.6g}"
            reasons.append(f"guard {name} failed: {value} is beyond the limit {entry['limit']:.6g}")
    decision = {"release": not reasons, "reasons": reasons}
    logger.info("Release decision: %s (%d blocking reasons)", "release" if decision["release"] else "do not release",
                len(reasons))
    return decision

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

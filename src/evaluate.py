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

logger = logging.getLogger(__name__)


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

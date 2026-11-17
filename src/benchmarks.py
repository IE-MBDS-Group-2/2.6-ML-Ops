"""Fit and simulate traditional long-only portfolio benchmarks."""

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from scipy import optimize

from src import evaluate, portfolio, tracking, utils

logger = logging.getLogger(__name__)
METHODS = ("min_variance", "max_sharpe", "mean_variance", "risk_parity")
_NUMERIC_FLOOR = 1e-10


def _risk_budget(weights: np.ndarray, covariance: np.ndarray) -> float:
    """Inputs: Positive weights and covariance. Output: Objective for equal risk budgets."""
    return float(0.5 * weights @ covariance @ weights - np.log(weights).mean())


def _objective(method: str, mean: np.ndarray, covariance: np.ndarray, params: dict) -> Callable:
    """Inputs: Method, moments and params. Output: Loss. Why: SLSQP minimizes each loss."""
    if method == "min_variance":
        return lambda weights: float(weights @ covariance @ weights)
    if method == "max_sharpe":
        return lambda weights: -float(mean @ weights) / np.sqrt(max(weights @ covariance @ weights, _NUMERIC_FLOOR))
    aversion = float(params["risk_aversion"])
    if not np.isfinite(aversion) or aversion < 0:
        raise ValueError("benchmarks.risk_aversion must be finite and nonnegative.")
    return lambda weights: float(0.5 * aversion * (weights @ covariance @ weights) - mean @ weights)


def fit_weights(train_returns: pd.DataFrame, method: str, params: dict) -> pd.Series:
    """Fit a long-only allocation using pre-cutoff daily returns.

    Inputs: Finite training returns in ticker order, benchmark name and method settings.
    Outputs: Weights in input column order, summing to one.
    Why: Estimate targets strictly within the training period.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown benchmark {method!r}. Choose from {', '.join(METHODS)}.")
    if train_returns.shape[0] < 2 or train_returns.shape[1] == 0:
        raise ValueError("Benchmarks need at least two training return rows and one asset.")
    if not np.isfinite(train_returns.to_numpy(dtype=float)).all():
        raise ValueError("Benchmark training returns must be finite. Check the source prices.")
    size = train_returns.shape[1]
    initial = np.full(size, 1.0 / size)
    covariance = train_returns.cov().to_numpy()
    mean = train_returns.mean().to_numpy()
    scale = max(float(covariance.diagonal().max()), _NUMERIC_FLOOR)
    covariance /= scale
    mean /= np.sqrt(scale) if method == "max_sharpe" else scale
    if method == "risk_parity":
        # The log barrier imposes equal risk budgets. Normalize to a fully invested portfolio.
        stable_covariance = covariance + _NUMERIC_FLOOR * np.eye(size)
        result = optimize.minimize(
            _risk_budget, initial, args=(stable_covariance,), method="SLSQP",
            bounds=[(_NUMERIC_FLOOR, None)] * size,
        )
    else:
        result = optimize.minimize(
            _objective(method, mean, covariance, params), initial, method="SLSQP",
            bounds=[(0.0, 1.0)] * size,
            constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        )
    if not result.success:
        raise ValueError(f"Benchmark {method} could not fit weights: {result.message}. Check training returns.")
    weights = np.maximum(result.x, 0.0)
    return pd.Series(weights / weights.sum(), index=train_returns.columns, dtype=float)


def run_benchmarks(market_data: tuple, cfg: dict, start: str, end: str) -> pd.DataFrame:
    """Fit configured benchmarks and compare them through the project's simulator.

    Inputs: Simulator market frames, config and inclusive evaluation dates.
    Outputs: One row per method with weights, drawdown, return, volatility, Sharpe and trading cost.
    Why: Assess the traditional allocations under the same rebalancing rule and trading costs.
    """
    close_wide = market_data[0]
    training_prices = close_wide.loc[cfg["portfolio"]["first_decision_date"]:cfg["split"]["train_end"],
                                     cfg["data"]["tickers"]]
    train_returns = training_prices.pct_change(fill_method=None).iloc[1:]
    settings = cfg["benchmarks"]
    bounded_market = tuple(frame.loc[:end] for frame in market_data)
    decisions = portfolio.decision_dates(bounded_market[0].index, start, end)
    results = []
    for method in settings["methods"]:
        target = fit_weights(train_returns, method, settings)
        policy = portfolio.make_current_rule_policy(target, decisions, cfg)
        daily, _ = portfolio.simulate(*bounded_market, target, policy, start, end, cfg)
        results.append({"method": method, "weights": target.to_dict(),
                        **evaluate.portfolio_metrics(daily["value"], daily["cost_paid"])})
    logger.info("Simulated %d benchmark portfolios from %s to %s", len(results), start, end)
    return pd.DataFrame(results)


def report_benchmarks(market_data: tuple, cfg: dict, start: str, end: str, path: Path,
                      metrics: dict, run: wandb.Run | None) -> None:
    """Inputs: Benchmark inputs, CSV path, metrics and run. Output: CSV, metrics entry and W&B table."""
    if not cfg["benchmarks"]["enabled"]:
        return
    table = run_benchmarks(market_data, cfg, start, end)
    utils.save_csv(table, path)
    tracking.log_table(run, "benchmarks", table)
    metrics["benchmarks"] = table.to_dict(orient="records")

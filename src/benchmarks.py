"""Fit and simulate traditional long-only portfolio benchmarks."""

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

from src import evaluate, portfolio, tracking, utils

logger = logging.getLogger(__name__)
METHODS = ("min_variance", "max_sharpe", "mean_variance", "risk_parity")
_NUMERIC_FLOOR = 1e-10


def _risk_budget(weights: np.ndarray, covariance: np.ndarray) -> float:
    """Evaluate the risk-parity objective.

    Inputs: Positive weights and a covariance matrix.
    Outputs: Objective value for the optimizer to minimize.
    Why: The log term sets equal risk budgets for all assets.
    """
    return float(0.5 * weights @ covariance @ weights - np.log(weights).mean())


def _objective(method: str, mean: np.ndarray, covariance: np.ndarray, params: dict) -> Callable:
    """Build the optimization objective for the selected allocation.

    Inputs: Method, scaled return moments and benchmark parameters.
    Outputs: Callable that computes the loss for a weight vector.
    Why: SLSQP uses a common minimization interface for all objectives.
    """
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
    covariance = covariance / scale
    mean = mean / (np.sqrt(scale) if method == "max_sharpe" else scale)
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


def _benchmark_weights(train_returns: pd.DataFrame, method: str, params: dict) -> pd.Series:
    """Fit allocations using BIL as cash when it is available for maximum Sharpe.

    Inputs: Training returns, allocation method and benchmark settings.
    Outputs: Target weights in the original ticker order.
    Why: The maximum Sharpe portfolio earns excess returns over cash and only holds risky assets.
    """
    if method != "max_sharpe" or "BIL" not in train_returns.columns:
        return fit_weights(train_returns, method, params)
    excess_returns = train_returns.drop(columns="BIL").sub(train_returns["BIL"], axis="index")
    weights = fit_weights(excess_returns, method, params)
    return weights.reindex(train_returns.columns, fill_value=0.0)


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
        target = _benchmark_weights(train_returns, method, settings)
        policy = portfolio.make_current_rule_policy(target, decisions, cfg)
        daily, _ = portfolio.simulate(*bounded_market, target, policy, start, end, cfg)
        results.append({"method": method, "weights": target.to_dict(),
                        **evaluate.portfolio_metrics(daily["value"], daily["cost_paid"])})
    logger.info("Simulated %d benchmark portfolios from %s to %s", len(results), start, end)
    return pd.DataFrame(results)


def report_benchmarks(market_data: tuple, cfg: dict, start: str, end: str, path: Path,
                      metrics: dict, run: object | None) -> None:
    """Save the configured benchmark report and log its metrics.

    Inputs: Market frames, configuration, evaluation dates, CSV path, metrics and run.
    Outputs: CSV, metrics entry and W&B table when benchmarks are enabled.
    Why: Keep the local reports and tracking system on the same results.
    """
    if not cfg["benchmarks"]["enabled"]:
        return
    table = run_benchmarks(market_data, cfg, start, end)
    utils.save_csv(table, path)
    tracking.log_table(run, "benchmarks", table)
    metrics["benchmarks"] = table.to_dict(orient="records")

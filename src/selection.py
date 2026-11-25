"""Date windows for validating candidate portfolio models within the training period."""

import logging
from collections.abc import Callable
from itertools import product
from math import isfinite

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src import (allocation, benchmarks, config, evaluate, features, portfolio,
                 tracking, train)

logger = logging.getLogger(__name__)


def walk_forward_windows(dates: pd.DatetimeIndex, first_train_end: str, last_end: str, block: str,
                         gap_days: int) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Construct expanding training windows and disjoint scoring blocks.

    Inputs: Sorted, unique trading dates; the first inclusive training cutoff;
            the final inclusive scoring cutoff; a pandas offset such as YE;
            and the number of trading dates to exclude before each scoring block.
    Outputs: Pairs of training dates and scoring dates, in chronological order.
    Why: A gap before scoring keeps forward-looking training labels out of the scoring period.
    """
    if dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("dates must contain sorted, unique trading dates with no missing values.")
    if gap_days < 0:
        raise ValueError("gap_days must be zero or a positive number of trading dates.")

    first_end = pd.Timestamp(first_train_end, tz=dates.tz)
    final_end = pd.Timestamp(last_end, tz=dates.tz)
    if first_end >= final_end:
        raise ValueError("first_train_end must be earlier than last_end.")

    offset = pd.tseries.frequencies.to_offset(block)
    if offset.n <= 0:
        raise ValueError("block must be a positive pandas offset, such as YE.")
    boundaries = pd.date_range(first_end, final_end, freq=offset, inclusive="neither")
    start = dates.searchsorted(first_end, side="right")
    windows = []
    for end in boundaries.append(pd.DatetimeIndex([final_end])):
        stop = dates.searchsorted(end, side="right")
        train_stop = start - gap_days
        if train_stop <= 0 or stop == start:
            raise ValueError(
                f"Window ending {end.date()} has no training or scoring dates. "
                "Check the trading calendar, split dates and gap_days."
            )
        windows.append((dates[:train_stop], dates[start:stop]))
        start = stop

    logger.info("Built %d walk-forward windows through %s", len(windows), final_end.date())
    return windows


def _model_policy(model: Pipeline, columns: list[str]) -> Callable[[pd.Series], float]:
    """Wrap a fitted model for the simulator.

    Inputs: Fitted model and feature order.
    Outputs: Policy that predicts the action for a simulated state.
    Why: Apply the same feature order used during training.
    """
    def policy(state: pd.Series) -> float:
        """Predict the action for one decision.

        Inputs: Current simulated state.
        Outputs: Predicted rebalancing action.
        Why: The simulator scores one state at a time.
        """
        return float(model.predict(state[columns].to_frame().T.astype(float))[0])

    return policy


def _policy_metrics(policy: Callable[[pd.Series], float], market_data: tuple, target_weights: pd.Series,
                    period: tuple[str, str], cfg: dict) -> dict:
    """Simulate one policy over one scoring window and measure it.

    Inputs: Policy, market frames, target weights, (start, end) as YYYY-MM-DD and configuration.
    Outputs: evaluate.portfolio_metrics of the simulated values (net of trading costs).
    Why: The model and the current rule are measured by exactly the same steps, with no prices after the window.
    """
    bounded_market = tuple(frame.loc[:period[1]] for frame in market_data)
    daily, _ = portfolio.simulate(*bounded_market, target_weights, policy, *period, cfg)
    return evaluate.portfolio_metrics(daily["value"], daily["cost_paid"])


def _score_window(
    model: Pipeline, columns: list[str], market_data: tuple, score_dates: pd.DatetimeIndex, cfg: dict,
) -> dict:
    """Compare the model and current rule on a single scoring window.

    Inputs: Fitted model, feature order, market frames, scoring dates and configuration.
    Outputs: Block dates, both drawdowns, the relative drawdown reduction, and both annual returns and trading
        costs (the simulated values are net of trading costs).
    Why: Score both policies with the same simulator used for the quality gate.
    """
    start, end = score_dates[0], score_dates[-1]
    period = start.date().isoformat(), end.date().isoformat()
    target_weights = allocation.stored_target(cfg["target_weights"].get("method", "policy"), cfg)
    rule = portfolio.make_current_rule_policy(
        target_weights, portfolio.decision_dates(score_dates, *period, portfolio.review_frequency(cfg)), cfg)
    model_metrics = _policy_metrics(_model_policy(model, columns), market_data, target_weights, period, cfg)
    rule_metrics = _policy_metrics(rule, market_data, target_weights, period, cfg)
    return {
        "block_start": start,
        "block_end": end,
        "max_drawdown_reduction": evaluate.max_drawdown_reduction(model_metrics["max_drawdown"],
                                                                  rule_metrics["max_drawdown"]),
        "agent_mdd": model_metrics["max_drawdown"],
        "baseline_mdd": rule_metrics["max_drawdown"],
        "agent_return": model_metrics["annual_return"],
        "baseline_return": rule_metrics["annual_return"],
        "agent_cost": model_metrics["annual_trading_cost"],
        "baseline_cost": rule_metrics["annual_trading_cost"],
    }


def _fit_candidate(rows: pd.DataFrame, columns: list[str], cfg: dict, model_params: dict) -> Pipeline:
    """Fit a candidate model using the eligible decisions for a window.

    Inputs: Eligible rows, feature columns, configuration and model parameters.
    Outputs: Fitted sklearn pipeline.
    Why: Training only uses decisions whose labels are available before scoring.
    """
    problem = cfg["problem"]
    target = problem["target_column"]
    labelled = rows.dropna(subset=[target])
    if labelled.empty:
        raise ValueError("No labelled training rows in a walk-forward window. Check decision dates and split settings.")
    return train.train_model(
        labelled[columns], labelled[target], features.get_feature_preprocessor(columns),
        problem["problem_type"], model_params, cfg["model"]["random_state"],
    )


def _window_target(table: pd.DataFrame, market_data: tuple, cfg: dict,
                   train_dates: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    """Refit a calculated target before each scoring window to prevent future allocation leakage.

    Inputs: Decision table, market frames, configuration and the window's training dates.
    Outputs: The decision table and configuration for this window: unchanged for the policy mix, else rebuilt
        with weights fitted on returns up to the last training date.
    Why: Weights fitted on later years would leak the future into an earlier scoring block.
    """
    method = cfg["target_weights"].get("method", "policy")
    if method == "policy":
        return table, cfg
    cutoff = str(train_dates[-1].date())
    settings = {**cfg, "split": {**cfg["split"], "train_end": cutoff}}
    returns = benchmarks.training_returns(market_data[0], settings)
    weights = benchmarks.fit_weights(returns, method, cfg["benchmarks"])
    settings["target_weights"] = {**cfg["target_weights"], "calculated": {
        **cfg["target_weights"].get("calculated", {}), method: weights.to_dict()}}
    bounded = tuple(frame.loc[:cutoff] for frame in market_data)
    rebuilt = portfolio.build_decision_table(
        *bounded, weights, cfg["portfolio"]["first_decision_date"], cutoff, settings)
    return rebuilt, settings


def _prepare_window(table: pd.DataFrame, market_data: tuple, cfg: dict, dates: tuple) -> tuple:
    """Build target and labels once per cutoff, ready to reuse across parameter trials.

    Inputs: Decision table, market frames, configuration and a (training dates, scoring dates) pair.
    Outputs: Eligible rows, the window's configuration, feature columns and scoring dates.
    Why: Rows whose label window reaches the scoring block are dropped, and the work is not repeated per trial.
    """
    train_dates, score_dates = dates
    table, settings = _window_target(table, market_data, cfg, train_dates)
    calendar = market_data[0].loc[:cfg["split"]["train_end"]].index
    columns = table.columns.drop([cfg["problem"]["target_column"], portfolio.RULE_ACTION_COLUMN],
                                 errors="ignore").tolist()
    reward_end = portfolio.execution_schedule(calendar, table.index)["reward_end"]
    eligible = table.index.isin(train_dates) & reward_end.lt(score_dates[0])
    return table.loc[eligible], settings, columns, score_dates


def _prepared_windows(table: pd.DataFrame, market_data: tuple, cfg: dict) -> list:
    """Prepare chronological training windows independently of candidate parameters.

    Inputs: Decision table, market frames and configuration (reads selection and split.train_end).
    Outputs: One prepared window per walk-forward block, in chronological order.
    Why: A calculated target does not depend on the model parameters, so the grid search prepares it once.
    """
    settings = cfg["selection"]
    last_end = cfg["split"]["train_end"]
    calendar = market_data[0].loc[:last_end].index
    windows = walk_forward_windows(
        calendar, settings["first_train_end"], last_end, settings["block"], settings["gap_days"])
    return [_prepare_window(table.loc[:last_end], market_data, cfg, window) for window in windows]


def _score_prepared(windows: list, market_data: tuple, model_params: dict) -> pd.DataFrame:
    """Fit each candidate on the prepared labels and score on the following block.

    Inputs: Prepared windows, market frames and the candidate model parameters.
    Outputs: One row per window, as _score_window returns.
    Why: Each parameter trial reuses the prepared windows and only refits the model.
    """
    results = []
    for rows, cfg, columns, score_dates in windows:
        model = _fit_candidate(rows, columns, cfg, model_params)
        results.append(_score_window(model, columns, market_data, score_dates, cfg))
    return pd.DataFrame(results)


def walk_forward_scores(table: pd.DataFrame, market_data: tuple, cfg: dict, model_params: dict) -> pd.DataFrame:
    """Fit and score one candidate per chronological block inside the training period.

    Inputs: Decision table, simulator market frames, config and candidate model parameters.
    Outputs: One row per block with its dates, drawdown reduction, both drawdowns, annual returns and costs.
    Why: Use the gate's trading simulation while sealing the validation and test years.
    Raises: ValueError for invalid date windows or a window with no labelled training rows.
    """
    return _score_prepared(_prepared_windows(table, market_data, cfg), market_data, model_params)


def _parameter_grid(cfg: dict) -> dict:
    """Validate the grid against the estimator before fitting.

    Inputs: Project configuration.
    Outputs: Nonempty parameter grid in configuration order.
    Why: Reject unknown search parameters before starting the first trial.
    """
    grid = cfg["selection"]["grid"]
    if not grid or any(not isinstance(values, (tuple, list)) or not values for values in grid.values()):
        raise ValueError("selection.grid must specify a nonempty list for every parameter.")
    estimator = Ridge() if cfg["problem"]["problem_type"] == "regression" else train.ActionClassifier()
    unknown = set(grid).difference(estimator.get_params()).union({"random_state"}.intersection(grid))
    if unknown:
        raise ValueError(f"Unsupported selection.grid parameter(s): {', '.join(sorted(unknown))}.")
    return grid


def grid_search(table: pd.DataFrame, market_data: tuple, cfg: dict, run: object = None) -> tuple[dict, pd.DataFrame]:
    """Rank the configured parameter combinations on the training years.

    Inputs: Decision table, market data, project configuration and optional W&B run.
    Outputs: Best full parameter set and one row per attempted combination.
    Why: Pick the highest median walk-forward score before looking at validation.
    Raises: ValueError for an empty grid, an unsupported parameter or a nonfinite score.
    """
    grid = _parameter_grid(cfg)
    prepared = None
    if cfg.get("target_weights", {}).get("method", "policy") != "policy":
        prepared = _prepared_windows(table, market_data, cfg)
    trials = []
    best_params = cfg["model"]["params"].copy()
    best_score = -float("inf")
    for index, combination in enumerate(product(*grid.values()), 1):
        overrides = dict(zip(grid, combination))
        params = {**cfg["model"]["params"], **overrides}
        scores = (walk_forward_scores(table, market_data, cfg, params) if prepared is None
                  else _score_prepared(prepared, market_data, params))
        score = float(scores["max_drawdown_reduction"].median(skipna=False))
        if not isfinite(score):
            raise ValueError(f"Grid trial {index} produced no finite median walk-forward score.")
        tracking.log_metrics(run, {"trial": index, "median_max_drawdown_reduction": score}, prefix="grid/")
        trials.append({**overrides, "trial": index, "median_max_drawdown_reduction": score})
        if score > best_score:
            best_params, best_score = params, score
    logger.info("Grid search scored %d trials; best median drawdown reduction %.4f", len(trials), best_score)
    return best_params, pd.DataFrame(trials)


def frequency_comparison(market_data: tuple, cfg: dict, model_params: dict) -> pd.DataFrame:
    """Compare daily, weekly and monthly review for today's rule and the model, on the train years only.

    Inputs: Simulator market frames, project configuration and the model parameters to fit.
    Outputs: One row per review frequency (config.ALLOWED_VALUES): the number of scoring blocks, the median
        drawdown reduction, and the median drawdown, annual return and annual trading cost of the model and of
        today's rule. Every number comes from the simulator, so it is net of trading costs.
    Why: The labels depend on the review frequency, so each one gets its own decision table and walk-forward
        fit. Validation and test years stay sealed, so the choice is not tuned on them.
    Raises: ValueError from the walk-forward windows or from a window with no labelled training rows.
    """
    train_end = cfg["split"]["train_end"]
    training_market = tuple(frame.loc[:train_end] for frame in market_data)
    target = allocation.stored_target(cfg["target_weights"].get("method", "policy"), cfg)
    rows = []
    for review in config.ALLOWED_VALUES[("rebalancing", "review")]:
        settings = {**cfg, "rebalancing": {**cfg["rebalancing"], "review": review}}
        table = pd.DataFrame()
        if cfg["target_weights"].get("method", "policy") == "policy":
            table = portfolio.build_decision_table(
                *training_market, target, cfg["portfolio"]["first_decision_date"], train_end, settings)
        scores = walk_forward_scores(table, training_market, settings, model_params)
        reduction = scores["max_drawdown_reduction"]
        rows.append({
            "review": review,
            "blocks": len(scores),
            "blocks_scored": int(reduction.notna().sum()),
            "median_max_drawdown_reduction": float(reduction.median()),
            "model_median_max_drawdown": float(scores["agent_mdd"].median()),
            "rule_median_max_drawdown": float(scores["baseline_mdd"].median()),
            "model_median_annual_return": float(scores["agent_return"].median()),
            "rule_median_annual_return": float(scores["baseline_return"].median()),
            "model_median_annual_trading_cost": float(scores["agent_cost"].median()),
            "rule_median_annual_trading_cost": float(scores["baseline_cost"].median()),
        })
    logger.info("Compared %d review frequencies on the train years", len(rows))
    return pd.DataFrame(rows)

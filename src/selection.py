"""Date windows for validating candidate portfolio models within the training period."""

import logging
from collections.abc import Callable
from itertools import product
from math import isfinite

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src import config, evaluate, features, portfolio, tracking, train

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
    target_weights = pd.Series(cfg["target_weights"]["policy"], dtype=float)
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


def walk_forward_scores(table: pd.DataFrame, market_data: tuple, cfg: dict, model_params: dict) -> pd.DataFrame:
    """Fit and score one candidate per chronological block inside the training period.

    Inputs: Decision table, simulator market frames, config and candidate model parameters.
    Outputs: One row per block with its dates, drawdown reduction, both drawdowns, annual returns and costs.
    Why: Use the gate's trading simulation while sealing the validation and test years.
    Raises: ValueError for invalid date windows or a window with no labelled training rows.
    """
    settings = cfg["selection"]
    last_end = cfg["split"]["train_end"]
    calendar = market_data[0].loc[:last_end].index
    windows = walk_forward_windows(
        calendar, settings["first_train_end"], last_end, settings["block"], settings["gap_days"],
    )
    train_table = table.loc[:last_end]
    columns = train_table.columns.drop([cfg["problem"]["target_column"], portfolio.RULE_ACTION_COLUMN],
                                       errors="ignore").tolist()
    reward_end = portfolio.execution_schedule(calendar, train_table.index)["reward_end"]
    rows = []
    for train_dates, score_dates in windows:
        # Both the decision and its forward-looking label must precede the scoring period.
        model = _fit_candidate(
            train_table.loc[train_table.index.isin(train_dates) & reward_end.lt(score_dates[0])],
            columns, cfg, model_params,
        )
        rows.append(_score_window(model, columns, market_data, score_dates, cfg))
    return pd.DataFrame(rows)


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
    defaults = cfg["model"]["params"]
    trials = []
    best_params = defaults.copy()
    best_score = -float("inf")
    for index, combination in enumerate(product(*grid.values()), 1):
        overrides = dict(zip(grid, combination))
        params = {**defaults, **overrides}
        score = float(walk_forward_scores(table, market_data, cfg, params)[
            "max_drawdown_reduction"].median(skipna=False))
        if not isfinite(score):
            raise ValueError(f"Grid trial {index} produced no finite median walk-forward score.")
        metrics = {"trial": index, "median_max_drawdown_reduction": score}
        tracking.log_metrics(run, metrics, prefix="grid/")
        trials.append({**overrides, **metrics})
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
    target = pd.Series(cfg["target_weights"]["policy"], dtype=float)
    rows = []
    for review in config.ALLOWED_VALUES[("rebalancing", "review")]:
        settings = {**cfg, "rebalancing": {**cfg["rebalancing"], "review": review}}
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

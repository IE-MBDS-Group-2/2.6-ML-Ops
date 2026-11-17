"""Date windows for validating candidate portfolio models within the training period."""

import logging
from collections.abc import Callable

import pandas as pd
from sklearn.pipeline import Pipeline

from src import evaluate, features, portfolio, train

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


def _score_window(
    model: Pipeline, columns: list[str], market_data: tuple, score_dates: pd.DatetimeIndex, cfg: dict,
) -> dict:
    """Compare the model and current rule on a single scoring window.

    Inputs: Fitted model, feature order, market frames, scoring dates and configuration.
    Outputs: Block dates, both drawdowns and the relative drawdown reduction.
    Why: Score both policies with the same simulator used for the quality gate.
    """
    start, end = score_dates[0], score_dates[-1]
    period = start.date().isoformat(), end.date().isoformat()
    target_weights = pd.Series(cfg["target_weights"]["policy"], dtype=float)
    rule = portfolio.make_current_rule_policy(target_weights, portfolio.decision_dates(score_dates, *period), cfg)
    bounded_market = tuple(frame.loc[:end] for frame in market_data)
    agent, _ = portfolio.simulate(*bounded_market, target_weights, _model_policy(model, columns), *period, cfg)
    baseline, _ = portfolio.simulate(*bounded_market, target_weights, rule, *period, cfg)
    agent_mdd = evaluate.portfolio_metrics(agent["value"], agent["cost_paid"])["max_drawdown"]
    baseline_mdd = evaluate.portfolio_metrics(baseline["value"], baseline["cost_paid"])["max_drawdown"]
    return {
        "block_start": start,
        "block_end": end,
        "max_drawdown_reduction": evaluate.max_drawdown_reduction(agent_mdd, baseline_mdd),
        "agent_mdd": agent_mdd,
        "baseline_mdd": baseline_mdd,
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
    Outputs: One row per block with its dates, drawdown reduction and both drawdowns.
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

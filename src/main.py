"""
main.py  [Step 0: ORCHESTRATOR]
================================
Purpose   Run the whole pipeline, in order, with one command: python -m src.main
Owns      Step order, the date-based split, passing data between steps
Never     Config parsing (config.py), logging set-up details (logger.py), or any cleaning, feature or model
          logic. main.py only CALLS other modules

Flow
    config.yaml + .env -> tracking (start run) -> load_data -> clean_data -> validate -> to_wide -> features
      -> portfolio (decision table) -> _split_by_date (train / validation / test sealed)
      -> train (model.type: the classifier, or the reinforcement learning agent: 5 seeds, the median one)
      -> evaluate: quality gate, guards and rule challengers (validation)
           gate met:     test audit once -> save model -> register it (tracking)
           gate not met: nothing saved or registered; today's rule serves (decision D8)
      -> infer -> reports/ -> tracking (finish run)

Why it matters
- Every setting comes from config.yaml, read once (config.py) and passed to modules as arguments
- Split by DATE before anything is fitted, so no future prices leak into training
- The test years stay sealed until a model passes the gate on validation; then they are scored once
- Handled errors stop the run with exit code 1 and a message saying what to fix

Tests     tests/test_main.py
"""

import logging
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src import (benchmarks, clean_data, config, evaluate, features, infer,
                 load_data)
from src import logger as log_setup
from src import (portfolio, rl_agent, robustness, rules, selection, tracking,
                 trades, train, utils)

logger = logging.getLogger(__name__)

PROJECT_ROOT = config.PROJECT_ROOT
CONFIG_PATH = config.CONFIG_PATH


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float, val_size: float,
               random_state: int, stratify: bool) -> tuple:
    """Course contract: random split into train, validation and test.

    Inputs:  X, y; test_size and val_size as shares of the WHOLE dataset; random_state;
             stratify (keep class shares in every split, classification only).
    Outputs: (X_train, X_val, X_test, y_train, y_val, y_test).
    Why:     kept and tested for the course contract. Our pipeline uses _split_by_date instead,
             because a random split of weekly decisions would leak future prices
             (TEAM_BRIEF Decisions).
    Raises:  ValueError for sizes outside (0, 1), sizes that sum to 1 or more, or X and y of
             different lengths.
    """
    if not (0 < test_size < 1 and 0 < val_size < 1) or test_size + val_size >= 1:
        raise ValueError(
            f"test_size ({test_size}) and val_size ({val_size}) must each be between 0 and 1 "
            "and sum to less than 1."
        )
    if len(X) != len(y):
        raise ValueError(f"X and y must have the same number of rows, got {len(X)} and {len(y)}.")
    # Exact row counts, not fractions: 0.15 / 0.85 * 85 is 15.000000000000002, which sklearn
    # rounds up to 16.
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=round(test_size * len(X)), random_state=random_state,
        stratify=y if stratify else None,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=round(val_size * len(X)), random_state=random_state,
        stratify=y_rest if stratify else None,
    )
    logger.info("Split %d rows into train %d, validation %d, test %d",
                len(X), len(X_train), len(X_val), len(X_test))
    return X_train, X_val, X_test, y_train, y_val, y_test


def _split_by_date(table: pd.DataFrame, train_end: str, val_end: str) -> tuple:
    """Split the decision table by date: train to train_end, validation to val_end, test after.

    Inputs:  table indexed by decision date (DatetimeIndex); train_end and val_end as YYYY-MM-DD,
             both inclusive.
    Outputs: (train, validation, test) DataFrames; every row is in exactly one of them.
    Why:     the model is judged on later years than it learned from, as it would be in operation.
    Raises:  ValueError if the index is not dates, val_end is not after train_end, or a split
             is empty.
    """
    if not isinstance(table.index, pd.DatetimeIndex):
        raise ValueError("_split_by_date needs a table indexed by date (DatetimeIndex).")
    train_cut, val_cut = pd.Timestamp(train_end), pd.Timestamp(val_end)
    if val_cut <= train_cut:
        raise ValueError(
            f"val_end ({val_end}) must be after train_end ({train_end}) in config.yaml.")
    train_rows = table[table.index <= train_cut]
    val_rows = table[(table.index > train_cut) & (table.index <= val_cut)]
    test_rows = table[table.index > val_cut]
    for name, part in (("train", train_rows), ("validation", val_rows), ("test", test_rows)):
        if part.empty:
            raise ValueError(f"The {name} split is empty. Check the split dates in config.yaml.")
    logger.info("Date split: train %d, validation %d, test %d rows",
                len(train_rows), len(val_rows), len(test_rows))
    return train_rows, val_rows, test_rows


def _market_inputs(clean: pd.DataFrame, cfg: dict) -> tuple:
    """Turn the clean long table into close_wide, volume_wide and the market features.

    Inputs:  clean, the validated long table; cfg, the full config (reads data and features).
    Outputs: (close_wide, volume_wide, market_features), all date x column and on one calendar.
    """
    tickers, cols = cfg["data"]["tickers"], cfg["data"]["columns"]
    close_wide = clean_data.to_wide(clean, cols["close_column"], tickers, cols["date_column"], cols["ticker_column"])
    volume_wide = clean_data.to_wide(
        clean, cols["volume_column"], tickers, cols["date_column"], cols["ticker_column"]).reindex(close_wide.index)
    market = features.compute_market_features(
        close_wide, cfg["features"]["return_weeks"], cfg["features"]["vol_days"])
    return close_wide, volume_wide, market


def _target_weights(cfg: dict) -> pd.Series:
    """The model portfolio's target weights from config.yaml."""
    return pd.Series(cfg["target_weights"]["policy"], dtype=float)


def _feature_columns(table: pd.DataFrame, cfg: dict) -> list[str]:
    """The model inputs: every decision-table column except today's rule and the label.

    Inputs:  table, the decision table; cfg, the full config (reads problem.target_column).
    Outputs: the feature column names, in table order.
    Why:     the classifier and the RL agent must see exactly the same inputs, so both get them from here.
    """
    return [name for name in table.columns
            if name not in (portfolio.RULE_ACTION_COLUMN, cfg["problem"]["target_column"])]


def _train_and_score(table: pd.DataFrame, cfg: dict) -> tuple:
    """Split by date, fit on the train years only, and score on validation.

    Inputs:  table, the decision table with best_action; cfg, the full config.
    Outputs: (model, feature_columns, validation_rows, validation_scores).
    Why:     the test years are split off here and never touched again: they stay sealed.
    """
    target_column = cfg["problem"]["target_column"]
    labelled = table.dropna(subset=[target_column])
    train_rows, val_rows, _sealed_test = _split_by_date(
        labelled, cfg["split"]["train_end"], cfg["split"]["val_end"])
    columns = _feature_columns(table, cfg)
    model = train.train_model(
        train_rows[columns], train_rows[target_column],
        features.get_feature_preprocessor(columns), cfg["problem"]["problem_type"],
        cfg["model"]["params"], cfg["model"]["random_state"],
    )
    scores = evaluate.evaluate_model(
        model, val_rows[columns], val_rows[target_column], cfg["problem"]["problem_type"])
    return model, columns, val_rows, scores


def _model_policy(model: Pipeline, columns: list[str]) -> Callable[[pd.Series], float]:
    """Wrap the fitted model as a policy: a function from a state row to an action."""
    def policy(state_row: pd.Series) -> float:
        """The model's action for one decision date."""
        return float(model.predict(state_row[columns].to_frame().T.astype(float))[0])
    return policy


def _validation_drawdown_reduction(model: Pipeline, columns: list[str], market_data: tuple, cfg: dict) -> float:
    """The gate metric of one model on validation: how much smaller its worst loss is than today's rule's.

    Inputs:  model and its feature columns; market_data; cfg, the full config.
    Outputs: max_drawdown_reduction (0.20 means the worst loss is 20% smaller than today's rule's).
    Why:     the seeds of the reinforcement learning agent are ranked by the same number the gate uses.
    """
    agent_daily, baseline_daily = _validation_paths(model, columns, market_data, cfg)
    drawdowns = [evaluate.portfolio_metrics(daily["value"], daily["cost_paid"])["max_drawdown"]
                 for daily in (agent_daily, baseline_daily)]
    return evaluate.max_drawdown_reduction(*drawdowns)


def _train_rl_agent(table: pd.DataFrame, market_data: tuple, cfg: dict) -> tuple:
    """Fit the reinforcement learning agent once per seed and keep the median one.

    Inputs:  table, the decision table (for the feature columns and the validation states); market_data;
             cfg, the full config (reads rl, portfolio, split and fallback).
    Outputs: (agent, report). agent is the QAgent of the median seed. report holds the validation
             max_drawdown_reduction of every seed, the median seed, and fallback_share: how often the
             service would set that agent aside for today's rule on the validation states.
    Why:     with about 640 weekly decisions one lucky seed proves nothing (README: the median of 5).
             Experience is collected on the train years only; validation is used to rank the seeds and for
             the gate, never to choose settings.
    """
    columns, settings = _feature_columns(table, cfg), cfg["rl"]
    transitions = rl_agent.collect_transitions(
        market_data, _target_weights(cfg), cfg, cfg["portfolio"]["first_decision_date"],
        cfg["split"]["train_end"], settings["n_episodes"], settings["seeds"][0])
    agents, scores = {}, {}
    for seed in settings["seeds"]:
        agents[seed] = rl_agent.fit_q_agent(transitions, columns, cfg["portfolio"]["actions"], settings, seed)
        scores[seed] = _validation_drawdown_reduction(agents[seed], columns, market_data, cfg)
    median = rl_agent.median_seed(scores)
    share = rl_agent.fallback_share(agents[median], table.loc[slice(*_period(cfg, "validation")), columns],
                                    cfg["fallback"]["min_probability"])
    logger.info("RL agent: %d seeds, %s per seed %s, median seed %d; the fallback would replace it on %.0f%% "
                "of validation decisions", len(scores), cfg["quality_gate"]["metric_key"],
                {seed: round(score, 4) for seed, score in scores.items()}, median, 100 * share)
    report = {"seed_scores": {str(seed): score for seed, score in scores.items()}, "median_seed": median,
              "median_score": scores[median], "fallback_share": share}
    return agents[median], report


def _candidate(table: pd.DataFrame, market_data: tuple, cfg: dict) -> tuple:
    """Train the model that goes to the quality gate, as chosen by model.type.

    Inputs:  table, the decision table with best_action; market_data; cfg, the full config
             (reads model.type and rl.enabled).
    Outputs: (model, feature_columns, scores). scores always names the model_type. With "classifier" it
             holds the classifier's validation scores; with "fitted_q" it holds the agent's report under "rl".
             With "classifier" and rl.enabled the agent is trained too and reported under "rl", but the
             classifier stays the candidate.
    Why:     both kinds of model offer predict and predict_proba, so every later step treats them the same.
    """
    kind, columns = cfg["model"]["type"], _feature_columns(table, cfg)
    model, scores = None, {}
    if kind == "classifier":
        model, columns, _val_rows, scores = _train_and_score(table, cfg)
    if kind == "fitted_q" or cfg["rl"]["enabled"]:
        agent, scores["rl"] = _train_rl_agent(table, market_data, cfg)
        model = agent if kind == "fitted_q" else model
    return model, columns, {**scores, "model_type": kind}


def _period(cfg: dict, split: str) -> tuple[str, str]:
    """The first and last date of the validation or the test years, from the split section.

    Inputs:  cfg, the full config (reads split and data); split, "validation" or "test".
    Outputs: (start, end) as YYYY-MM-DD; start is the day after the previous period ends.
    Raises:  ValueError for any other split name.
    """
    ends = {"validation": ("train_end", "val_end"), "test": ("val_end", "test_end")}
    if split not in ends:
        raise ValueError(f"Unknown period {split!r}. Use 'validation' or 'test'.")
    previous_end, end = (cfg["split"][key] for key in ends[split])
    start = pd.Timestamp(previous_end) + pd.Timedelta(days=1)
    return str(start.date()), end


def _validation_paths(model: Pipeline, columns: list[str], market_data: tuple, cfg: dict,
                      split: str = "validation") -> tuple:
    """Run the model and today's rule over one period through the same simulate.

    Inputs:  model and its feature columns; market_data (close_wide, volume_wide, market);
             cfg, the full config (reads portfolio, split and costs); split, "validation" or "test".
    Outputs: (agent_daily, baseline_daily), the daily frames of the two runs.
    """
    close_wide = market_data[0]
    target = _target_weights(cfg)
    start, end = _period(cfg, split)
    start = str(close_wide.index[close_wide.index >= pd.Timestamp(start)][0].date())
    rule = portfolio.make_current_rule_policy(target, rules.review_dates(close_wide, start, end, cfg), cfg)
    agent_daily, _ = portfolio.simulate(
        *market_data, target, _model_policy(model, columns), start, end, cfg)
    baseline_daily, _ = portfolio.simulate(*market_data, target, rule, start, end, cfg)
    return agent_daily, baseline_daily


def _check_quality_gate(agent: dict, baseline: dict, cfg: dict) -> tuple:
    """Compare the model's worst loss with today's rule's and apply the agreed gate.

    Inputs:  agent and baseline, evaluate.portfolio_metrics results; cfg, the full config
             (reads quality_gate).
    Outputs: (metric value, True if the gate is met).
    Why:     one honest pass / fail on validation, logged whichever way it goes.
    """
    gate = cfg["quality_gate"]
    value = evaluate.max_drawdown_reduction(agent["max_drawdown"], baseline["max_drawdown"])
    met = evaluate.check_gate(value, gate["comparison"], gate["threshold"])
    logger.info("Quality gate %s: %s = %.4f on %s (needs %s %s)",
                "met" if met else "NOT met", gate["metric_key"], value, gate["split"],
                gate["comparison"], gate["threshold"])
    return value, met


def _portfolio_report(daily: pd.DataFrame, market_data: tuple, cfg: dict) -> dict:
    """Risk, return and cost of one simulated run, plus its tracking error to the target weights.

    Inputs:  daily, a daily frame from portfolio.simulate; market_data; cfg (reads target_weights).
    Outputs: evaluate.portfolio_metrics plus "tracking_error" (values on the review dates).
    Why:     the guards need the same numbers for the model and for today's rule.
    """
    close_wide = market_data[0]
    reviews = rules.review_dates(close_wide, str(daily.index[0].date()), str(daily.index[-1].date()), cfg)
    report = evaluate.portfolio_metrics(daily["value"], daily["cost_paid"])
    report["tracking_error"] = evaluate.tracking_error(
        daily["value"].reindex(reviews), rules.target_path(close_wide, _target_weights(cfg), reviews),
        portfolio.periods_per_year(portfolio.review_frequency(cfg)))
    return report


def _period_metrics(model: Pipeline, columns: list[str], market_data: tuple, cfg: dict, split: str,
                    challenger_values: dict[str, pd.Series] | None = None) -> tuple:
    """The model against today's rule and the simple-rule challengers over one period.

    Inputs:  model and its feature columns; market_data; cfg, the full config; split, "validation" or "test".
    Outputs: metrics, challenger table and the daily agent and baseline paths.
             When requested, challenger_values receives the daily challenger values.
    Why:     validation and the one-time test audit are measured by exactly the same code.
    """
    agent_daily, baseline_daily = _validation_paths(model, columns, market_data, cfg, split)
    agent, baseline = (_portfolio_report(daily, market_data, cfg) for daily in (agent_daily, baseline_daily))
    value, met = _check_quality_gate(agent, baseline, cfg)
    gate = cfg["quality_gate"]
    metrics = {
        "split": split, "agent": agent, "baseline": baseline, gate["metric_key"]: value,
        "gate": {"comparison": gate["comparison"], "threshold": gate["threshold"], "met": met},
        "guards": evaluate.guard_report(agent, baseline, cfg["guards"]),
    }
    challengers = rules.run_challengers(
        market_data, _target_weights(cfg), cfg, str(agent_daily.index[0].date()),
        str(agent_daily.index[-1].date()), agent_daily, challenger_values)
    return metrics, challengers, (agent_daily, baseline_daily)


def _test_audit(model: Pipeline, columns: list[str], market_data: tuple, cfg: dict, chosen: str) -> tuple[dict, dict]:
    """The one-time look at the test years, only for a model that passed the gate on validation.

    Inputs:  model and its feature columns; market_data; cfg; chosen, the simple rule selected on validation.
    Outputs: test metrics and release decision, plus the separate robustness diagnostics.
    Why:     the test years stay sealed until a model has earned the look; the decision is made once.
    """
    challenger_values = {}
    metrics, challengers, paths = _period_metrics(model, columns, market_data, cfg, "test", challenger_values)
    agent_drawdown = metrics["agent"]["max_drawdown"]
    rule_drawdown = float(challengers.loc[chosen, "max_drawdown"])
    metrics["guards"]["challenger"] = {"value": agent_drawdown, "limit": rule_drawdown,
                                       "passed": bool(agent_drawdown > rule_drawdown)}
    metrics["release"] = evaluate.release_decision(metrics["gate"]["met"], metrics["guards"])
    logger.info("Test audit done: %s", "release" if metrics["release"]["release"] else "do not release")
    return metrics, robustness.test_period_diagnostics(*paths, challenger_values, cfg)


def _numbers(metrics: dict, prefix: str = "") -> dict[str, float]:
    """Flatten nested metrics into name -> number, for the tracker ("agent/max_drawdown": -0.19).

    Inputs:  metrics, a nested dictionary; prefix, the path so far.
    Outputs: only the numeric leaves (True and False count as 1 and 0); text and lists are left out.
    """
    flat = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            flat.update(_numbers(value, f"{prefix}{key}/"))
        elif isinstance(value, (bool, int, float)):
            flat[f"{prefix}{key}"] = float(value)
    return flat


def _recommendations(model: Pipeline | None, columns: list[str], cfg: dict) -> pd.DataFrame:
    """Recommend an action for every decision date in the inference file.

    Inputs:  the fitted model and its feature columns, or None when no model passed the gate;
             cfg, the full config.
    Outputs: one row per decision date: prediction, probability, recommendation, fallback.
    Why:     a prediction below the confidence floor falls back to today's rule; with no approved model
             every row is today's rule (decision D8), and says so with fallback = True.
    Raises:  FileNotFoundError, ValueError (data checks, or too few days for the features).
    """
    clean = load_data.load_prices(PROJECT_ROOT / cfg["paths"]["inference_data"], cfg)
    close_wide, volume_wide, market = _market_inputs(clean, cfg)
    ready = market.dropna().index
    if ready.empty:
        raise ValueError(
            "The inference file is too short to compute the market features. "
            "Next: fetch more days (data.inference_days in config.yaml).")
    table = portfolio.build_decision_table(
        close_wide, volume_wide, market, _target_weights(cfg),
        str(ready[0].date()), str(close_wide.index[-1].date()), cfg)
    rule_actions = table[portfolio.RULE_ACTION_COLUMN]
    if model is None:
        predictions = pd.DataFrame({"prediction": rule_actions, "probability": 0.0}, index=table.index)
        return infer.apply_fallback(predictions, rule_actions, pd.Series(True, index=table.index),
                                    cfg["fallback"]["min_probability"])
    inputs = table[columns]
    predictions = infer.run_inference(model, inputs, include_proba=True)
    return infer.apply_fallback(
        predictions, rule_actions, inputs.isna().any(axis=1), cfg["fallback"]["min_probability"])


def _keep_or_drop_model(model: Pipeline, metrics: dict, paths: dict, cfg: dict, tracker: object) -> Pipeline | None:
    """Save and register a model that passed the gate; make sure a model that failed cannot be served.

    Inputs:  the fitted model; metrics (reads the gate result and the gate metric); paths; cfg (reads tracking
             and quality_gate); tracker, the tracking run or None.
    Outputs: the model if the gate is met, otherwise None.
    Why:     decision D8: a model that fails the gate is never saved or registered. A model file left over
             from an earlier run is removed too, so the API cannot load a model that did not pass.
    """
    if not metrics["gate"]["met"]:
        if paths["model"].exists():
            paths["model"].unlink()
            logger.warning("Removed %s: it is from an earlier run and no model passed the gate now", paths["model"])
        logger.warning("No model saved or registered: the quality gate is not met. Today's rule serves")
        return None
    utils.save_model(model, paths["model"])
    key = cfg["quality_gate"]["metric_key"]
    tracking.log_model(tracker, paths["model"], cfg["tracking"]["model_artifact"],
                       {key: metrics[key], "gate": metrics["gate"], "challenger": metrics["challenger"]["name"],
                        "model_type": metrics["model_type"]})
    return model


def _validation_report(fitted: tuple, market_data: tuple, cfg: dict, paths: dict, tracker: object) -> tuple:
    """Everything measured on validation: the gate, the guards and the selected simple-rule challenger.

    Inputs:  fitted, (model, feature columns, validation scores) from _candidate; market_data; cfg;
             paths; tracker, the tracking run or None.
    Outputs: (metrics, chosen): the metrics dictionary and the name of the selected rule. Also writes the
             challengers table and sends the numbers to the tracker.
    """
    model, columns, scores = fitted
    metrics, challengers, _daily_paths = _period_metrics(model, columns, market_data, cfg, "validation")
    metrics.update(scores)
    chosen = rules.select_challenger(challengers, challengers.attrs["guards"])
    metrics["challenger"] = {"name": chosen, "max_drawdown": float(challengers.loc[chosen, "max_drawdown"])}
    utils.save_csv(challengers.reset_index(), paths["challengers"])
    tracking.log_metrics(tracker, _numbers(metrics), prefix="val/")
    return metrics, chosen


def _write_recommendations(approved: Pipeline | None, columns: list[str], cfg: dict, paths: dict,
                           tracker: object) -> int:
    """Recommend on the latest inference data, save the file and send the table to the tracker if asked.

    Inputs:  approved, the model that passed the gate or None; its feature columns; cfg; paths; tracker.
    Outputs: the number of recommendations written. Also writes the proposed portfolio and the trades when a
             positions file exists (_write_trades).
    Why:     the batch run gives the same three answers as the API: recommendation, proposed portfolio, trades.
    """
    recommendations = _recommendations(approved, columns, cfg)
    utils.save_csv(recommendations.reset_index(), paths["predictions"])
    _write_trades(approved, columns, cfg, paths)
    if cfg["tracking"]["log_predictions_table"]:
        tracking.log_table(tracker, "predictions", recommendations.reset_index())
    logger.info("Pipeline finished: %d recommendations written to %s", len(recommendations), paths["predictions"])
    return len(recommendations)


def _write_robustness(model: Pipeline, columns: list[str], market_data: tuple, cfg: dict, path: Path,
                      tracker: object, test_diagnostics: dict, versions_tried: int) -> None:
    """Write historical and validation diagnostics and attach the test audit when available.

    Inputs: Fitted model and columns, market frames, config, report path, tracker, audit results and trial count.
    Outputs: Robustness JSON and tracking table.
    Why: Record diagnostics without using them to change the validation gate or release decision.
    """
    validation_end = cfg["split"]["val_end"]
    history = tuple(frame.loc[:validation_end] for frame in market_data)
    ceiling = robustness.feasibility_ceiling(
        history, _target_weights(cfg), cfg, cfg["portfolio"]["first_decision_date"], validation_end)

    def run_policy(settings: dict) -> pd.DataFrame:
        """Return aligned validation values for the selected trading cost settings."""
        agent, baseline = _validation_paths(model, columns, history, settings)
        return pd.DataFrame({"agent_value": agent["value"], "baseline_value": baseline["value"]})

    sensitivity = robustness.cost_sensitivity(run_policy, cfg).astype(object)
    results = {
        "feasibility_ceiling": ceiling,
        "validation_cost_sensitivity": sensitivity.where(sensitivity.notna(), None).to_dict(orient="records"),
        "test": test_diagnostics,
    }
    robustness.report_robustness(results, versions_tried, path, tracker)


def _select_model_params(table: pd.DataFrame, market_data: tuple, cfg: dict,
                         paths: dict, tracker: object) -> dict:
    """Select parameters using training years when selection is enabled.

    Inputs: Decision table, market frames, configuration, the resolved paths (selection and
        frequency_comparison are written) and tracker.
    Outputs: Configuration for the final fit; also writes the trial table and the daily, weekly and monthly
        comparison. The trials are scored on the train years only, so they are not counted as versions tried.
    Why: Keep holdout years out of parameter selection and preserve the original configuration.
    """
    if not cfg["selection"]["enabled"]:
        return cfg
    cutoff = cfg["split"]["train_end"]
    training_market = tuple(frame.loc[:cutoff] for frame in market_data)
    best, trials = selection.grid_search(table.loc[:cutoff], training_market, cfg, tracker)
    utils.save_csv(trials, paths["selection"])
    comparison = selection.frequency_comparison(market_data, cfg, best)
    utils.save_csv(comparison, paths["frequency_comparison"])
    tracking.log_table(tracker, "frequency_comparison", comparison)
    return {**cfg, "model": {**cfg["model"], "params": best}}


def _first_trading_day(close_wide: pd.DataFrame, start: str) -> str:
    """The first trading day on or after a date.

    Inputs:  close_wide, prices indexed by trading day; start, YYYY-MM-DD.
    Outputs: that trading day as YYYY-MM-DD.
    Why:     a period can start on a weekend or holiday, and the simulator starts on a trading day.
    """
    return str(close_wide.index[close_wide.index >= pd.Timestamp(start)][0].date())


def _write_benchmarks(market_data: tuple, cfg: dict, paths: dict, metrics: dict, tracker: object) -> None:
    """Report the four traditional portfolios over the validation years, next to the model and today's rule.

    Inputs:  market_data; cfg (reads benchmarks and split); paths (benchmarks); metrics, which gains a
             "benchmarks" entry; tracker, the run or None.
    Outputs: None. Writes paths.benchmarks and logs the table, when benchmarks.enabled is true.
    Why:     the business case reports these portfolios on the same dates as the gate; they are never gated.
    """
    start, end = _period(cfg, "validation")
    benchmarks.report_benchmarks(market_data, cfg, _first_trading_day(market_data[0], start), end,
                                 paths["benchmarks"], metrics, tracker)


def _positions_action(approved: Pipeline | None, columns: list[str], weights: pd.Series, cfg: dict) -> float:
    """The recommendation for the positions' own weights on the latest inference day.

    Inputs:  approved, the model that passed the gate or None; columns, its feature columns; weights, the
             current weights of the positions file; cfg, the full config.
    Outputs: 0.0, 0.5 or 1.0: the model's action with the 60% fallback to today's rule, or today's rule when no
             model passed the gate (D8).
    Why:     the same state the API builds (portfolio.live_state), so the batch answer for a book and the live
             answer for the same book agree. The positions file has no drawdown or weeks since the last
             rebalance, so both start at 0.
    """
    close_wide, volume_wide, market = _market_inputs(
        load_data.load_prices(PROJECT_ROOT / cfg["paths"]["inference_data"], cfg), cfg)
    state = portfolio.live_state(close_wide, volume_wide, market, weights, 0.0, 0,
                                 _target_weights(cfg).reindex(weights.index), cfg)
    rule = state[portfolio.RULE_ACTION_COLUMN]
    if approved is None:
        return float(rule.iloc[0])
    inputs = state[columns]
    answer = infer.apply_fallback(infer.run_inference(approved, inputs, include_proba=True), rule,
                                  inputs.isna().any(axis=1), cfg["fallback"]["min_probability"])
    return float(answer["recommendation"].iloc[0])


def _write_trades(approved: Pipeline | None, columns: list[str], cfg: dict, paths: dict) -> bool:
    """Recommend for the book in the positions file and write its proposed portfolio and trade list.

    Inputs:  approved, the model that passed the gate or None; columns, its feature columns; cfg (reads
             data.tickers, positions, target_weights and portfolio.min_trade); paths (positions,
             proposed_portfolio, trades).
    Outputs: True when the positions file exists and both reports were written, False when there is no file.
    Why:     the desk acts on orders, not on a 0 / 0.5 / 1.0 code; the batch run gives what the API gives.
    Raises:  ValueError from trades.positions_to_weights (unknown ticker, negative or missing value, total 0)
             and from portfolio.live_state.
    """
    if not paths["positions"].exists():
        logger.info("No positions file at %s: no proposed portfolio or trades written", paths["positions"])
        return False
    names = cfg["positions"]
    weights, total = trades.positions_to_weights(utils.load_csv(paths["positions"]), cfg["data"]["tickers"],
                                                 names["ticker_column"], names["value_column"])
    action = _positions_action(approved, columns, weights, cfg)
    proposed = trades.proposed_portfolio(weights, _target_weights(cfg).reindex(weights.index), action, total)
    utils.save_csv(proposed, paths["proposed_portfolio"])
    utils.save_csv(trades.trade_list(weights, proposed, total, cfg["portfolio"]["min_trade"]), paths["trades"])
    logger.info("Positions in %s: action %.1f, proposed portfolio and trades written", paths["positions"], action)
    return True


def _versions_tried(cfg: dict, scores: dict) -> int:
    """How many model versions were scored on validation, for the multiple-testing correction.

    Inputs:  cfg (reads model.type and rl.seeds); scores, from _candidate ("rl" is there when the agent was trained).
    Outputs: one for the classifier when it ran, plus one per RL seed when the agent ran.
    Why:     every version scored on validation counts, or the robustness report overstates the evidence. The
             grid-search trials do not count: they are scored on the train years only (walk-forward), never on
             validation. The simple-rule challengers are a separate family, reported with their own table.
    """
    tried = 1 if cfg["model"]["type"] == "classifier" else 0
    return tried + (len(cfg["rl"]["seeds"]) if "rl" in scores else 0)


def run(cfg: dict, tracker: object = None) -> dict:
    """The pipeline: data -> decision table -> model -> gate, guards and challengers -> recommendations.

    Inputs:  cfg, the full config dictionary (see config.load_config); tracker, the run from
             tracking.start_run, or None when tracking is off (every tracking call then does nothing).
    Outputs: the metrics dictionary, also written to paths.metrics. Also writes the clean data, the
             challengers table, robustness report and predictions, and the model only if it passed the gate.
    Why:     one function that calls every module in order; no cleaning or model logic of its own.
    Raises:  FileNotFoundError, OSError, KeyError, ValueError from the steps it calls.
    """
    paths = config.resolve_paths(cfg, PROJECT_ROOT)
    # 1. Load, clean and validate the raw prices; keep the clean table and record it as the data version.
    clean = load_data.load_prices(paths["raw_data"], cfg)
    utils.save_csv(clean, paths["processed_data"])
    tracking.log_dataset(tracker, paths["processed_data"], cfg["tracking"]["dataset_artifact"])
    # 2. Wide prices and market features, then the decision table with the best_action label.
    market_data = _market_inputs(clean, cfg)
    table = portfolio.build_decision_table(
        *market_data, _target_weights(cfg),
        cfg["portfolio"]["first_decision_date"], cfg["data"]["end"], cfg)
    # 3. Select the settings on the train years (when enabled), fit the candidate (model.type) on them,
    #    and score it on validation.
    cfg = _select_model_params(table, market_data, cfg, paths, tracker)
    model, columns, scores = _candidate(table, market_data, cfg)
    # 4. On validation: the quality gate, the guards and the simple-rule challengers.
    metrics, chosen = _validation_report((model, columns, scores), market_data, cfg, paths, tracker)
    # 5. Only a model that passed the gate gets the one-time test audit, and is saved and registered.
    test_diagnostics = {"status": "sealed", "reason": "validation_gate_not_met"}
    if metrics["gate"]["met"]:
        metrics["test"], test_diagnostics = _test_audit(model, columns, market_data, cfg, chosen)
        tracking.log_metrics(tracker, _numbers(metrics["test"]), prefix="test/")
    _write_robustness(model, columns, market_data, cfg, paths["robustness"], tracker, test_diagnostics,
                      _versions_tried(cfg, scores))
    approved = _keep_or_drop_model(model, metrics, paths, cfg, tracker)
    # Reported, never gated: the traditional portfolios on the validation years.
    _write_benchmarks(market_data, cfg, paths, metrics, tracker)
    utils.save_json(metrics, paths["metrics"])
    # 6. Recommend on the latest inference data: the approved model, or today's rule if there is none.
    _write_recommendations(approved, columns, cfg, paths, tracker)
    return metrics


def main() -> None:
    """Course contract entry point: python -m src.main.

    Inputs:  none (reads CONFIG_PATH and the environment).
    Outputs: none. Runs the pipeline and records it; on a handled error it logs the cause and the next
             action, marks the tracked run as failed, then exits with code 1.
    Why:     one command that CI, teammates and the grader all run the same way.
    """
    tracker = None
    try:
        cfg = config.load_config(CONFIG_PATH)
        log_setup.setup_logging(cfg["logging"]["level"], PROJECT_ROOT / cfg["paths"]["log_file"])
        logger.info("Config loaded from %s: %d sections", CONFIG_PATH, len(cfg))
        tracker = tracking.start_run(cfg, config.load_env(PROJECT_ROOT / ".env"))
        run(cfg, tracker)
    except (ValueError, OSError, KeyError) as exc:
        tracking.finish_run(tracker, failed=True)
        logger.error("Pipeline stopped: %s. Next: check config.yaml and the input files.",
                     str(exc).rstrip(". "))
        raise SystemExit(1) from None
    tracking.finish_run(tracker)


if __name__ == "__main__":
    main()

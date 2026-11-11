"""
main.py  [Step 0: ORCHESTRATOR]
================================
Purpose   Run the whole pipeline, in order, with one command: python -m src.main
Owns      Loading config.yaml once, logging set-up, the date-based split, passing data between steps
Never     Cleaning, feature or model logic. main.py only CALLS other modules

Flow
    config.yaml -> load_data -> clean_data -> validate -> to_wide -> features
      -> portfolio (decision table) -> _split_by_date (train / validation / test sealed)
      -> train -> evaluate + quality gate (validation) -> save model -> infer -> reports/

Why it matters
- Every setting comes from config.yaml, read once here and passed to modules as arguments
- Split by DATE before anything is fitted, so no future prices leak into training
- The test years are never scored here: the split keeps them sealed
- Handled errors stop the run with exit code 1 and a message saying what to fix

Tests     tests/test_main.py
"""

import logging
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src import (clean_data, evaluate, features, infer, load_data, portfolio,
                 train, utils, validate)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
REQUIRED_SECTIONS = (
    "paths", "data", "portfolio", "problem", "split", "features",
    "model", "quality_gate", "fallback", "costs", "logging",
)
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def load_config(path: Path) -> dict:
    """Read config.yaml once and fail fast if a section is missing.

    Inputs:  path, the YAML file (normally CONFIG_PATH).
    Outputs: the settings as a dictionary.
    Why:     one source of truth for every setting; a missing section stops the run before any work.
    Raises:  FileNotFoundError if the file is missing; ValueError if it is not a mapping;
             KeyError naming every missing section.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}. Run from the repository root.")
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} must contain a YAML mapping of sections.")
    missing = [section for section in REQUIRED_SECTIONS if section not in cfg]
    if missing:
        raise KeyError(f"config.yaml is missing sections: {missing}")
    return cfg


def configure_logging(level: str, log_file: Path) -> None:
    """Send every module's log lines to the terminal and to one log file.

    Inputs:  level, e.g. "INFO" (from config.yaml); log_file, e.g. logs/pipeline.log.
    Outputs: None. The root logger gets a console and a file handler with one format.
    Why:     modules only call logging.getLogger(__name__); where lines go is decided once, here.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level.upper(),
        format=LOG_FORMAT,
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")],
        force=True,
    )


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


def _load_prices(path: Path, cfg: dict) -> pd.DataFrame:
    """Read one long price file, clean it, and stop here if the data is not trustworthy.

    Inputs:  path, a long file (date, ticker, close, volume); cfg, the full config (reads data).
    Outputs: the clean long table.
    Why:     training and inference data go through the same three steps, so they cannot differ.
    Raises:  FileNotFoundError, ValueError (from load_data and validate).
    """
    cols = cfg["data"]["columns"]
    clean = clean_data.clean_dataframe(
        load_data.load_raw_data(path), date_column=cols["date"], sort_by=[cols["date"], cols["ticker"]])
    validate.validate_dataframe(
        clean, cfg["data"]["required_columns"], tickers=cfg["data"]["tickers"],
        max_abs_daily_return=cfg["data"]["max_abs_daily_return"], price_columns=cols,
    )
    return clean


def _market_inputs(clean: pd.DataFrame, cfg: dict) -> tuple:
    """Turn the clean long table into close_wide, volume_wide and the market features.

    Inputs:  clean, the validated long table; cfg, the full config (reads data and features).
    Outputs: (close_wide, volume_wide, market_features), all date x column and on one calendar.
    """
    tickers, cols = cfg["data"]["tickers"], cfg["data"]["columns"]
    close_wide = clean_data.to_wide(clean, cols["close"], tickers, cols["date"], cols["ticker"])
    volume_wide = clean_data.to_wide(
        clean, cols["volume"], tickers, cols["date"], cols["ticker"]).reindex(close_wide.index)
    market = features.compute_market_features(
        close_wide, cfg["features"]["return_weeks"], cfg["features"]["vol_days"])
    return close_wide, volume_wide, market


def _target_weights(cfg: dict) -> pd.Series:
    """The model portfolio's target weights from config.yaml."""
    return pd.Series(cfg["portfolio"]["target_weights"], dtype=float)


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
    columns = [name for name in table.columns if name not in (portfolio.RULE_ACTION_COLUMN, target_column)]
    model = train.train_model(
        train_rows[columns], train_rows[target_column],
        features.get_feature_preprocessor(columns), cfg["problem"]["problem_type"],
        cfg["model"]["params"], cfg["model"]["random_state"],
    )
    scores = evaluate.evaluate_model(
        model, val_rows[columns], val_rows[target_column], cfg["problem"]["problem_type"])
    return model, columns, val_rows, scores


def _model_policy(model, columns: list[str]):
    """Wrap the fitted model as a policy: a function from a state row to an action."""
    def policy(state_row: pd.Series) -> float:
        return float(model.predict(state_row[columns].to_frame().T.astype(float))[0])
    return policy


def _validation_paths(model, columns: list[str], market_data: tuple, cfg: dict) -> tuple:
    """Run the model and today's rule over the validation years through the same simulate.

    Inputs:  model and its feature columns; market_data (close_wide, volume_wide, market);
             cfg, the full config (reads portfolio, split and costs).
    Outputs: (agent_daily, baseline_daily), the daily frames of the two runs.
    """
    close_wide = market_data[0]
    target = _target_weights(cfg)
    end = cfg["split"]["val_end"]
    after_train = close_wide.index[close_wide.index > pd.Timestamp(cfg["split"]["train_end"])]
    start = str(after_train[0].date())
    days = close_wide.index[close_wide.index <= pd.Timestamp(end)]
    rule = portfolio.make_current_rule_policy(
        target, portfolio.decision_dates(days, start, end), cfg)
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


def _validation_metrics(scores: dict, daily_frames: tuple, cfg: dict) -> dict:
    """Everything reported for the validation years, with the quality gate result.

    Inputs:  scores, evaluate_model results; daily_frames, (agent_daily, baseline_daily);
             cfg, the full config (reads quality_gate).
    Outputs: the metrics dictionary written to paths.metrics.
    """
    agent, baseline = (
        evaluate.portfolio_metrics(daily["value"], daily["cost_paid"]) for daily in daily_frames)
    value, met = _check_quality_gate(agent, baseline, cfg)
    gate = cfg["quality_gate"]
    return {
        "split": gate["split"], **scores, "agent": agent, "baseline": baseline,
        gate["metric_key"]: value,
        "gate": {"comparison": gate["comparison"], "threshold": gate["threshold"], "met": met},
    }


def _recommendations(model, columns: list[str], cfg: dict) -> pd.DataFrame:
    """Recommend an action for every decision date in the inference file.

    Inputs:  the fitted model and its feature columns; cfg, the full config.
    Outputs: one row per decision date: prediction, probability, recommendation, fallback.
    Why:     a prediction below the confidence floor falls back to today's rule.
    Raises:  FileNotFoundError, ValueError (data checks, or too few days for the features).
    """
    clean = _load_prices(PROJECT_ROOT / cfg["paths"]["inference_data"], cfg)
    close_wide, volume_wide, market = _market_inputs(clean, cfg)
    ready = market.dropna().index
    if ready.empty:
        raise ValueError(
            "The inference file is too short to compute the market features. "
            "Next: fetch more days (data.inference_days in config.yaml).")
    table = portfolio.build_decision_table(
        close_wide, volume_wide, market, _target_weights(cfg),
        str(ready[0].date()), str(close_wide.index[-1].date()), cfg)
    inputs = table[columns]
    predictions = infer.run_inference(model, inputs, include_proba=True)
    return infer.apply_fallback(
        predictions, table[portfolio.RULE_ACTION_COLUMN], inputs.isna().any(axis=1),
        cfg["fallback"]["min_probability"])


def run(cfg: dict) -> dict:
    """The pipeline: data -> decision table -> model -> gate on validation -> recommendations.

    Inputs:  cfg, the full config dictionary (see load_config).
    Outputs: the metrics dictionary, also written to paths.metrics. Also writes the clean data,
             the model and the predictions, at the paths in config.yaml.
    Why:     one function that calls every module in order; no cleaning or model logic of its own.
    Raises:  FileNotFoundError, OSError, KeyError, ValueError from the steps it calls.
    """
    paths = {name: PROJECT_ROOT / value for name, value in cfg["paths"].items()}
    # 1. Load, clean and validate the raw prices; keep the clean table.
    clean = _load_prices(paths["raw_data"], cfg)
    utils.save_csv(clean, paths["processed_data"])
    # 2. Wide prices and market features, then the decision table with the best_action label.
    market_data = _market_inputs(clean, cfg)
    table = portfolio.build_decision_table(
        *market_data, _target_weights(cfg),
        cfg["portfolio"]["first_decision_date"], cfg["data"]["end"], cfg)
    # 3. Split by date, fit on the train years, score on validation.
    model, columns, _val_rows, scores = _train_and_score(table, cfg)
    # 4. The model against today's rule over the validation years, and the quality gate.
    metrics = _validation_metrics(
        scores, _validation_paths(model, columns, market_data, cfg), cfg)
    # 5. Save the model and the metrics, then recommend on the latest inference data.
    utils.save_model(model, paths["model"])
    utils.save_json(metrics, paths["metrics"])
    recommendations = _recommendations(model, columns, cfg)
    utils.save_csv(recommendations.reset_index(), paths["predictions"])
    logger.info("Pipeline finished: %d recommendations written to %s",
                len(recommendations), paths["predictions"])
    return metrics


def main() -> None:
    """Course contract entry point: python -m src.main.

    Inputs:  none (reads CONFIG_PATH).
    Outputs: none. Runs the pipeline; on a handled error it logs the cause and the next action,
             then exits with code 1.
    Why:     one command that CI, teammates and the grader all run the same way.
    """
    try:
        cfg = load_config(CONFIG_PATH)
        configure_logging(cfg["logging"]["level"], PROJECT_ROOT / cfg["paths"]["log_file"])
        logger.info("Config loaded from %s: %d sections", CONFIG_PATH, len(cfg))
        run(cfg)
    except (ValueError, OSError, KeyError) as exc:
        logger.error("Pipeline stopped: %s. Next: check config.yaml and the input files.",
                     str(exc).rstrip(". "))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()

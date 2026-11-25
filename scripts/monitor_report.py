"""
monitor_report.py  [Stage 3: THE WEEKLY MONITORING RUN]
=======================================================
Purpose   Measure the two monitored retrain triggers and show them in W&B with their thresholds drawn in:
          the input drift (largest PSI of the daily returns, per week) and the fallback rate (per week)
Input     config.yaml (monitoring, paths, data, split, portfolio); paths.raw_data (the train years are the
          reference); paths.inference_data (the recent days); the answers the API logged to W&B (serve runs),
          or --answers, a CSV with the same columns (as_of, fallback) when W&B is off
Output    paths.monitoring (reports/monitoring.json): the latest drift, the weekly series and both alerts; with
          tracking on, a W&B run of job type "monitor" with one chart per trigger and its retrain limit
Run       python -m scripts.monitor_report                          (from the repository root)
          python -m scripts.monitor_report --answers answers.csv    (W&B off: fallback rate from a local file)
Used by   the person who runs the service, once a week after Friday's close (model card, "What triggers a
          retrain")

Why it matters
- Open gap 12: the drift alert now has a number (PSI above monitoring.psi_limit, 0.2) computed by code
- The prices go through the same load, clean and validate steps as training, so the comparison is like for like
- Why daily returns and not the 15 market features, why 180 days and why BIL is left out: src/monitoring.py and
  the comments under monitoring in config.yaml

Tests     tests/test_monitor_report.py
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src import (clean_data, config, load_data, monitoring, tracking, utils,
                 validate)
from src.logger import setup_logging

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Read the command line: --answers, an optional CSV of logged answers used instead of W&B.

    Inputs:  argv, the arguments after the script name (always passed in, never read from sys.argv here).
    Outputs: the parsed options: answers (a Path, or None when --answers was not given).
    Why:     an explicit argv keeps pytest's own arguments away from the parser and keeps main() testable.
    """
    parser = argparse.ArgumentParser(description="Measure input drift and the fallback rate, and log them to W&B.")
    parser.add_argument("--answers", type=Path, default=None,
                        help="CSV of API answers (as_of, fallback); default: the serve runs in W&B")
    return parser.parse_args(argv)


def daily_returns(path: Path, cfg: dict) -> pd.DataFrame:
    """Load, clean and check one long price file, and return the daily returns of the monitored tickers.

    Inputs:  path, a long price file (date, ticker, close, volume); cfg, the full config (data, monitoring).
    Outputs: date x ticker simple daily returns for monitoring.tickers; the first day is NaN.
    Why:     the same steps main.py runs before training, so reference and recent returns are built alike.
    Raises:  FileNotFoundError, ValueError (from load_data and validate).
    """
    data, cols = cfg["data"], cfg["data"]["columns"]
    clean = clean_data.clean_dataframe(load_data.load_raw_data(path), date_column=cols["date_column"],
                                       sort_by=[cols["date_column"], cols["ticker_column"]])
    validate.validate_dataframe(clean, data["required_columns"], tickers=data["tickers"],
                                max_abs_daily_return=data["max_abs_daily_return"], price_columns=cols)
    close_wide = clean_data.to_wide(clean, cols["close_column"], data["tickers"], cols["date_column"],
                                    cols["ticker_column"])
    return close_wide[cfg["monitoring"]["tickers"]].pct_change(fill_method=None)


def _answers(path: Path | None, cfg: dict, env: dict[str, str]) -> pd.DataFrame:
    """The logged API answers: from --answers when given, else from W&B; empty when W&B is off.

    Inputs:  path, the --answers CSV or None; cfg, the full config (tracking.enabled, tracking.project);
             env, the loaded environment settings (WANDB_MODE and the W&B login).
    Outputs: a DataFrame of logged answers; empty when there is no file and tracking is off.
    Why:     the drift needs no W&B, so the run still reports it when there are no answers to read.
    """
    if path is not None:
        return utils.load_csv(path)
    if env["WANDB_MODE"] == "disabled" or not cfg["tracking"]["enabled"]:
        logger.warning("Tracking off and no --answers file: the fallback rate is not measured this run.")
        return pd.DataFrame()
    return tracking.fetch_prediction_batches(env, cfg["tracking"]["project"])


def measure(cfg: dict, reference: pd.DataFrame, current: pd.DataFrame, answers: pd.DataFrame) -> dict:
    """Both monitored triggers, as one summary.

    Inputs:  cfg, the full config (monitoring, split, portfolio); reference, daily returns over all raw dates
             (cut here to the train years); current, recent daily returns; answers, logged answers (may be empty).
    Outputs: a dictionary: limits, the weekly drift and fallback series, the PSI per ticker of the latest
             window, and drift_alert / fallback_alert.
    Why:     drift and fallback are the two retrain triggers, so one summary holds both for the report and the W&B run.
    Raises:  ValueError if the recent data is shorter than monitoring.window_days.
    """
    settings = cfg["monitoring"]
    train = reference.loc[cfg["portfolio"]["first_decision_date"]:cfg["split"]["train_end"]]
    columns = list(current.columns)
    drift = monitoring.weekly_drift(train, current, columns, settings["psi_bins"], settings["window_days"])
    if drift.empty:
        raise ValueError(f"monitor_report: the recent data has no Friday with {settings['window_days']} days of "
                         "daily returns. Fetch more days (data.inference_days) or lower monitoring.window_days.")
    # The breakdown is for the same window as the latest week (ending on its Friday), not the file's last rows.
    latest_window = current.dropna(subset=columns).loc[:drift.index[-1]].tail(settings["window_days"])
    latest = monitoring.feature_drift(train, latest_window, columns, settings["psi_bins"])
    rates = (monitoring.weekly_fallback_rate(answers, settings["date_column"], settings["fallback_column"])
             if not answers.empty else pd.DataFrame(columns=["fallback_rate", "answers"]))
    return {
        "psi_limit": settings["psi_limit"],
        "fallback_limit": settings["fallback_limit"],
        "fallback_weeks": settings["fallback_weeks"],
        "latest_week": str(drift.index[-1].date()),
        "latest_max_psi": float(drift["max_psi"].iloc[-1]),
        "latest_psi_by_ticker": {name: round(float(value), 4) for name, value in latest.items()},
        "drift_alert": monitoring.drift_alert(drift, settings["psi_limit"]),
        "fallback_alert": monitoring.fallback_alert(rates, settings["fallback_limit"], settings["fallback_weeks"]),
        "weekly_drift": drift,
        "weekly_fallback_rate": rates,
    }


def _report(summary: dict) -> dict:
    """The summary with its two weekly tables turned into plain lists, ready for JSON.

    Inputs:  summary, the dictionary from measure (its weekly_drift and weekly_fallback_rate are tables).
    Outputs: a copy of the summary whose two weekly tables are lists of records, with "week" as YYYY-MM-DD text.
    Why:     DataFrames cannot be written as JSON, and the saved report must be readable without pandas.
    """
    def rows(table: pd.DataFrame) -> list[dict]:
        """One weekly table as a list of records.

        Inputs:  table, a table indexed by week.
        Outputs: one dictionary per week, with the week as its first 10 characters (YYYY-MM-DD).
        Why:     the saved report holds plain values only.
        """
        out = table.reset_index()
        out["week"] = out["week"].astype(str).str[:10]
        return out.to_dict(orient="records")
    return {**summary, "weekly_drift": rows(summary["weekly_drift"]),
            "weekly_fallback_rate": rows(summary["weekly_fallback_rate"].rename_axis("week"))}


def _log_to_wandb(summary: dict, cfg: dict, env: dict[str, str]) -> None:
    """One W&B run of job type "monitor": the two charts with their limits, the alerts and the latest PSI.

    Inputs:  summary, the dictionary from measure (weekly tables, limits, alerts); cfg, the full config;
             env, the loaded environment settings.
    Outputs: None. Starts, fills and finishes one W&B run (the fallback chart is skipped when there are no answers).
    Why:     the team watches drift and fallback over time in W&B, with the limit drawn on each chart.
    """
    run = tracking.start_run(cfg, env, job_type="monitor")
    tracking.log_limit_chart(run, "monitor/drift_psi", summary["weekly_drift"]["max_psi"].rename("max PSI"),
                             summary["psi_limit"], "Input drift: largest PSI of the daily returns")
    rates = summary["weekly_fallback_rate"]
    if not rates.empty:
        tracking.log_limit_chart(run, "monitor/fallback_rate", rates["fallback_rate"].rename("fallback rate"),
                                 summary["fallback_limit"], "Share of answers that fell back to today's rule")
    tracking.log_metrics(run, {"latest_max_psi": summary["latest_max_psi"],
                               "drift_alert": int(summary["drift_alert"]),
                               "fallback_alert": int(summary["fallback_alert"])}, prefix="monitor/")
    tracking.finish_run(run)


def main(argv: list[str] | None = None) -> dict:
    """Measure, save paths.monitoring, log the W&B run and say whether a retrain is due.

    Inputs:  argv, the command-line arguments (None means none, so a test never reads sys.argv).
    Outputs: the JSON-ready summary that was saved.
    Why:     one command checks both retrain triggers, so the weekly job and a person run the same thing.
    """
    args = _parse_args([] if argv is None else argv)
    cfg, env = config.load_config(), config.load_env()
    paths = config.resolve_paths(cfg)
    summary = measure(cfg, daily_returns(paths["raw_data"], cfg), daily_returns(paths["inference_data"], cfg),
                      _answers(args.answers, cfg, env))
    report = _report(summary)
    utils.save_json(report, paths["monitoring"])
    _log_to_wandb(summary, cfg, env)
    logger.info("Week %s: largest PSI %.3f (limit %.2f), drift alert %s, fallback alert %s. Written to %s",
                report["latest_week"], report["latest_max_psi"], report["psi_limit"], report["drift_alert"],
                report["fallback_alert"], paths["monitoring"])
    if report["drift_alert"] or report["fallback_alert"]:
        logger.warning("A retrain trigger fired: retrain and run the quality gate again (model card).")
    return report


if __name__ == "__main__":
    setup_logging("INFO")
    main(sys.argv[1:])

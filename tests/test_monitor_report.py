"""Tests for scripts/monitor_report.py: the weekly monitoring run, end to end on synthetic prices, offline."""

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import make_sample_prices as sample
from scripts import monitor_report
from src import config, tracking


def _setup(tmp_path: Path, monkeypatch, inference_days: int = 300) -> dict:
    """Synthetic raw and inference files under tmp_path, and the team config pointed at them (tracking off)."""
    cfg = config.load_config()
    data = cfg["data"]
    prices = sample.make_sample_prices(data["tickers"], data["start"], data["end"], 42, data["columns"])
    recent = prices[prices[data["columns"]["date_column"]].isin(
        sorted(prices[data["columns"]["date_column"]].unique())[-inference_days:])]
    raw, inference = tmp_path / "prices.csv", tmp_path / "prices_latest.csv"
    prices.to_csv(raw, index=False)
    recent.to_csv(inference, index=False)
    cfg["paths"] = {**cfg["paths"], "raw_data": str(raw), "inference_data": str(inference),
                    "monitoring": str(tmp_path / "monitoring.json")}
    monkeypatch.setattr(monitor_report.config, "load_config", lambda: cfg)
    monkeypatch.setattr(monitor_report.config, "load_env", lambda: {**config.ENV_DEFAULTS, "WANDB_MODE": "disabled"})
    return cfg


def test_integration_main_writes_the_report_without_alerts_on_steady_data(tmp_path, monkeypatch):
    """Synthetic prices follow one process throughout: no drift alert, and no answers means no fallback alert."""
    cfg = _setup(tmp_path, monkeypatch)
    report = monitor_report.main([])
    saved = json.loads(Path(cfg["paths"]["monitoring"]).read_text(encoding="utf-8"))
    assert saved["drift_alert"] is False
    assert saved["fallback_alert"] is False
    assert saved["latest_max_psi"] < cfg["monitoring"]["psi_limit"]
    assert saved["weekly_fallback_rate"] == []
    assert set(saved["latest_psi_by_ticker"]) == {"SPY", "EFA", "IEF", "GLD"}   # BIL is left out
    assert saved["weekly_drift"][-1]["week"] == saved["latest_week"] == report["latest_week"]


def test_integration_main_reads_answers_from_a_file(tmp_path, monkeypatch):
    """Four weeks with 60 to 80% of answers on today's rule (above the 50% limit): the fallback alert fires."""
    cfg = _setup(tmp_path, monkeypatch)
    days = pd.bdate_range("2025-12-01", "2025-12-26")
    answers = pd.DataFrame({"as_of": days, "fallback": [index % 4 != 0 for index in range(len(days))]})
    answers.to_csv(tmp_path / "answers.csv", index=False)
    report = monitor_report.main(["--answers", str(tmp_path / "answers.csv")])
    assert [row["week"] for row in report["weekly_fallback_rate"]] == [
        "2025-12-05", "2025-12-12", "2025-12-19", "2025-12-26"]
    assert report["fallback_alert"] is True
    assert json.loads(Path(cfg["paths"]["monitoring"]).read_text(encoding="utf-8"))["fallback_alert"] is True


def test_unit_measure_flags_a_shifted_market(tmp_path, monkeypatch):
    """Recent daily moves twice as large on one ticker (volatility doubled): the drift alert fires, naming it."""
    cfg = _setup(tmp_path, monkeypatch)
    reference = monitor_report.daily_returns(Path(cfg["paths"]["raw_data"]), cfg)
    current = monitor_report.daily_returns(Path(cfg["paths"]["inference_data"]), cfg)
    current["GLD"] = current["GLD"] * 2
    summary = monitor_report.measure(cfg, reference, current, pd.DataFrame())
    assert summary["drift_alert"] is True
    assert summary["latest_max_psi"] > cfg["monitoring"]["psi_limit"]
    assert next(iter(summary["latest_psi_by_ticker"])) == "GLD"


def test_unit_latest_breakdown_uses_the_same_window_as_the_latest_week(tmp_path, monkeypatch):
    """The per-ticker PSI is for the window ending on the latest Friday, so its largest value is latest_max_psi.

    The sample data ends on a Wednesday (2025-12-31): the last rows of the file are not the latest week's window.
    """
    cfg = _setup(tmp_path, monkeypatch)
    reference = monitor_report.daily_returns(Path(cfg["paths"]["raw_data"]), cfg)
    current = monitor_report.daily_returns(Path(cfg["paths"]["inference_data"]), cfg)
    current["GLD"] = current["GLD"] * 2
    assert current.index[-1].day_of_week != 4                 # the file does not end on a Friday
    summary = monitor_report.measure(cfg, reference, current, pd.DataFrame())
    assert max(summary["latest_psi_by_ticker"].values()) == pytest.approx(summary["latest_max_psi"], abs=1e-4)


def test_unit_measure_stops_when_recent_data_is_too_short(tmp_path, monkeypatch):
    cfg = _setup(tmp_path, monkeypatch, inference_days=100)
    reference = monitor_report.daily_returns(Path(cfg["paths"]["raw_data"]), cfg)
    current = monitor_report.daily_returns(Path(cfg["paths"]["inference_data"]), cfg)
    with pytest.raises(ValueError, match="window_days"):
        monitor_report.measure(cfg, reference, current, pd.DataFrame())


def test_unit_answers_come_from_wandb_when_tracking_is_on(monkeypatch):
    calls = []
    monkeypatch.setattr(tracking, "fetch_prediction_batches",
                        lambda env, project: calls.append(project) or pd.DataFrame({"as_of": [], "fallback": []}))
    cfg = {"tracking": {"enabled": True, "project": "test-project"}}
    monitor_report._answers(None, cfg, {"WANDB_MODE": "offline"})
    assert calls == ["test-project"]


def test_unit_charts_and_alerts_go_to_one_monitor_run(monkeypatch):
    """With tracking on: a "monitor" run, a chart for each trigger with its limit, the alerts, then finish."""
    logged = []
    monkeypatch.setattr(tracking, "start_run", lambda cfg, env, job_type: logged.append(job_type) or "run")
    monkeypatch.setattr(tracking, "log_limit_chart", lambda run, name, values, limit, title: logged.append(
        (name, limit)))
    monkeypatch.setattr(tracking, "log_metrics", lambda run, metrics, prefix: logged.append(sorted(metrics)))
    monkeypatch.setattr(tracking, "finish_run", lambda run: logged.append("finished"))
    week = pd.to_datetime(["2026-11-20"])
    summary = {"weekly_drift": pd.DataFrame({"max_psi": [0.1]}, index=week), "psi_limit": 0.2,
               "weekly_fallback_rate": pd.DataFrame({"fallback_rate": [0.4]}, index=week), "fallback_limit": 0.5,
               "latest_max_psi": 0.1, "drift_alert": False, "fallback_alert": False}
    monitor_report._log_to_wandb(summary, {}, {})
    assert logged == ["monitor", ("monitor/drift_psi", 0.2), ("monitor/fallback_rate", 0.5),
                      ["drift_alert", "fallback_alert", "latest_max_psi"], "finished"]

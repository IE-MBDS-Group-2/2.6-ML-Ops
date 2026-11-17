"""Known-value and reporting checks for the robustness diagnostics."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config, robustness, tracking


def test_unit_holm_textbook_example() -> None:
    """Apply the Holm multipliers in ascending p-value order."""
    assert robustness.holm_adjust({"a": .01, "b": .04, "c": .03}) == pytest.approx(
        {"a": .03, "b": .06, "c": .06})


def test_unit_bootstrap_is_reproducible() -> None:
    """The same seed repeats the same paired block draws."""
    baseline = pd.Series(np.linspace(-.04, .04, 20))
    agent = baseline + pd.Series(np.cos(np.arange(20))) / 100
    assert robustness.bootstrap_p_value(agent, baseline, 199, 4, 8) == robustness.bootstrap_p_value(
        agent, baseline, 199, 4, 8)


def test_unit_bootstrap_near_half_for_identical_series() -> None:
    """Identical paths give half weight to every tie."""
    returns = pd.Series([.02, -.01, .03, -.025] * 3)
    assert robustness.bootstrap_p_value(returns, returns, 99, 3, 8) == pytest.approx(.5)


def test_unit_bootstrap_small_for_a_clearly_better_series() -> None:
    """A fixed weekly advantage leaves every bootstrap draw above zero."""
    baseline = pd.Series([.02, -.01, .03, -.025] * 3)
    assert robustness.bootstrap_p_value(baseline + .01, baseline, 199, 3, 8) < .05


def test_unit_ceiling_is_at_least_todays_rule() -> None:
    """Run real labels and the shared simulator on a declining toy portfolio."""
    cfg = config.load_config()
    dates = pd.bdate_range("2019-12-27", periods=35)
    closes = pd.DataFrame({"A": np.linspace(102, 75, len(dates)),
                           "B": np.linspace(100, 99, len(dates))}, index=dates)
    volumes = pd.DataFrame(1_000_000.0, index=dates, columns=closes.columns)
    target = pd.Series({"A": .5, "B": .5})
    result = robustness.feasibility_ceiling(
        (closes, volumes, pd.DataFrame(index=dates)), target, cfg, "2020-01-02", "2020-02-13")
    assert result["baseline_mdd"] < 0
    assert result["max_drawdown_reduction"] >= 0
    assert result["gate_target"] == cfg["quality_gate"]["threshold"]


def test_unit_largest_episode_removed() -> None:
    """A baseline crash creates a pass that disappears after removing its returns."""
    days = pd.date_range("2022-01-01", periods=7)
    baseline = pd.DataFrame({"value": [100, 110, 70, 110, 109, 107, 110]}, index=days)
    agent = pd.DataFrame({"value": [100, 110, 107, 110, 107, 105, 110]}, index=days)
    result = robustness.without_largest_episode(agent, baseline)
    assert result["episode_start"] == result["episode_end"] == "2022-01-03"
    assert result["full_reduction"] > .2
    assert result["without_episode_reduction"] < .2
    assert result["fragile"] is True


def test_unit_cost_sensitivity_two_rows() -> None:
    """Rerun each cost assumption without mutating the caller's config."""
    cfg = config.load_config()

    def run(settings: dict) -> pd.DataFrame:
        """Return deterministic portfolio values under the specified cost model."""
        finish = 90 if settings["costs"]["model"] == "flat" else 85
        return pd.DataFrame({"agent_value": [100, finish], "baseline_value": [100, 80]})

    result = robustness.cost_sensitivity(run, cfg)
    assert result["cost_model"].tolist() == ["flat", "realistic"]
    assert result["max_drawdown_reduction"].tolist() == pytest.approx([.5, .25])
    assert cfg["costs"]["model"] == "flat"


@pytest.mark.parametrize("values", [{}, {"low": -.1}, {"high": 1.1}, {"nan": float("nan")}])
def test_unit_robustness_raises_for_invalid_probabilities(values: dict[str, float]) -> None:
    """Holm requires a nonempty family with valid probabilities."""
    with pytest.raises(ValueError, match="p-value"):
        robustness.holm_adjust(values)


def test_unit_robustness_raises_for_mismatched_weekly_returns() -> None:
    """Unpaired series cannot be drawn in matched blocks."""
    with pytest.raises(ValueError, match="same length"):
        robustness.bootstrap_p_value(pd.Series([.02]), pd.Series([.01, -.02]), 10, 1, 1)


def test_report_persists_counts_and_logs_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The report records the same candidate count in JSON and tracking."""
    captured = []
    monkeypatch.setattr(tracking, "log_table", lambda run, name, table: captured.append((run, name, table)))
    run = object()
    path = tmp_path / "reports" / "robustness.json"
    robustness.report_robustness({"holm": {"rule": .06}}, 12, path, run)
    assert json.loads(path.read_text()) == {"holm": {"rule": .06}, "versions_tried": 12}
    assert captured[0][:2] == (run, "robustness")
    assert captured[0][2]["result"].map(json.loads).tolist() == [{"rule": .06}, 12]

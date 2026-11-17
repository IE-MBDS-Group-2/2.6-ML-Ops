"""Known-value and reporting checks for the robustness diagnostics."""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config, robustness, tracking

GATE = {"threshold": 0.20, "comparison": ">="}  # the gate settings the caller passes; the module has no defaults


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
    assert result["hindsight_reduction"] >= 0  # the raw value: the reported one is clamped at 0 and cannot fail
    assert result["max_drawdown_reduction"] == max(0.0, result["hindsight_reduction"])
    assert result["status"] == "measured"
    assert result["gate_target"] == cfg["quality_gate"]["threshold"]


def test_unit_largest_episode_removed() -> None:
    """A baseline crash creates a pass that disappears after removing its returns."""
    days = pd.date_range("2022-01-01", periods=7)
    baseline = pd.DataFrame({"value": [100, 110, 70, 110, 109, 107, 110]}, index=days)
    agent = pd.DataFrame({"value": [100, 110, 107, 110, 107, 105, 110]}, index=days)
    result = robustness.without_largest_episode(agent, baseline, **GATE)
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


def test_unit_robustness_raises_when_no_version_was_evaluated(tmp_path: Path) -> None:
    """A report with zero versions tried would hide the multiple-testing count, so nothing is written."""
    path = tmp_path / "robustness.json"
    with pytest.raises(ValueError, match="at least one evaluated version"):
        robustness.report_robustness({}, 0, path)
    assert not path.exists()


# ---------- the raises and "inconclusive" branches ----------

def _path(values: list[float]) -> pd.DataFrame:
    """A daily value path on consecutive dates."""
    return pd.DataFrame({"value": values}, index=pd.date_range("2022-01-01", periods=len(values)))


def _runner(agent: list[float], baseline: list[float]) -> Callable[[dict], pd.DataFrame]:
    """A cost-sensitivity runner that returns the same two paths for any cost model."""
    def run(settings: dict) -> pd.DataFrame:
        """Return the fixed agent and baseline values, ignoring the cost model in the settings."""
        return pd.DataFrame({"agent_value": agent, "baseline_value": baseline})
    return run


@pytest.mark.parametrize("values", [[100.0], [100.0, float("nan")], [100.0, -5.0], [100.0, 0.0]],
                         ids=["one value", "not finite", "negative", "zero"])
def test_unit_robustness_raises_for_an_unusable_portfolio_path(values: list[float]) -> None:
    """A drawdown needs at least two finite, positive values."""
    with pytest.raises(ValueError, match="at least two finite, positive"):
        robustness.cost_sensitivity(_runner(values, values), config.load_config())


def test_unit_cost_sensitivity_flat_baseline_has_no_score() -> None:
    """A baseline without any loss has no drawdown to reduce: the score is empty, not a division by zero."""
    result = robustness.cost_sensitivity(_runner([100, 95], [100, 100]), config.load_config())
    assert result["max_drawdown_reduction"].isna().all()


@pytest.mark.parametrize("agent_dates, baseline_dates", [
    (["2022-01-01", "2022-01-02", "2022-01-03"], ["2022-01-02", "2022-01-03", "2022-01-04"]),
    (["2022-01-01", "2022-01-01", "2022-01-03"], ["2022-01-01", "2022-01-01", "2022-01-03"]),
    (["2022-01-03", "2022-01-02", "2022-01-01"], ["2022-01-03", "2022-01-02", "2022-01-01"]),
], ids=["different dates", "duplicate dates", "unsorted dates"])
def test_unit_robustness_raises_for_a_mismatched_calendar(agent_dates: list[str], baseline_dates: list[str]) -> None:
    """The two paths must share one unique, sorted calendar before returns can be removed from both."""
    agent = pd.DataFrame({"value": [100, 90, 95]}, index=pd.DatetimeIndex(agent_dates))
    baseline = pd.DataFrame({"value": [100, 80, 90]}, index=pd.DatetimeIndex(baseline_dates))
    with pytest.raises(ValueError, match="unique, sorted calendar"):
        robustness.without_largest_episode(agent, baseline, **GATE)


def test_unit_flat_baseline_makes_the_episode_check_inconclusive() -> None:
    """With no baseline decline there is no episode to remove."""
    result = robustness.without_largest_episode(_path([100, 99, 98]), _path([100, 100, 100]), **GATE)
    assert result["status"] == "inconclusive"
    assert result["full_reduction"] is None
    assert result["episode_start"] is None and result["fragile"] is None


def test_unit_episode_covering_the_whole_sample_is_inconclusive() -> None:
    """When the largest decline is the whole sample, nothing is left to compare."""
    result = robustness.without_largest_episode(_path([100, 99, 95]), _path([100, 90, 80]), **GATE)
    assert result["status"] == "inconclusive"
    assert result["full_reduction"] == pytest.approx(.75)
    assert result["without_episode_reduction"] is None
    assert (result["episode_start"], result["episode_end"]) == ("2022-01-02", "2022-01-03")


def test_unit_baseline_flat_after_removal_is_inconclusive() -> None:
    """If the baseline's only loss is the removed episode, the reduced score has no baseline drawdown."""
    result = robustness.without_largest_episode(_path([100, 100, 90, 90, 90]), _path([100, 100, 50, 50, 50]), **GATE)
    assert result["full_reduction"] == pytest.approx(.8)
    assert result["without_episode_reduction"] is None
    assert result["status"] == "inconclusive" and result["fragile"] is None


@pytest.mark.parametrize("draws, block, agent", [
    (0, 2, [.01, .02, -.01, .0]),
    (10, 0, [.01, .02, -.01, .0]),
    (10, 5, [.01, .02, -.01, .0]),
    (10, 2, [.01, float("nan"), -.01, .0]),
], ids=["no draws", "block of zero", "block longer than the sample", "not finite"])
def test_unit_robustness_raises_for_invalid_bootstrap_settings(draws: int, block: int, agent: list[float]) -> None:
    """Draws must be positive, the block must fit in the sample and the returns must be finite."""
    other = pd.Series([.01, .02, -.01, .0])
    with pytest.raises(ValueError, match="positive draws and a block"):
        robustness.bootstrap_p_value(pd.Series(agent), other, draws, block, 1)


def _diagnostic_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.Series]]:
    """Sixty business days of two wavy paths and one challenger, enough for about twelve weekly returns."""
    days = pd.bdate_range("2022-01-03", periods=60)
    steps = np.arange(60)
    agent = pd.DataFrame({"value": 100 + steps * .1 + 3 * np.sin(steps / 4)}, index=days)
    baseline = pd.DataFrame({"value": 100 + steps * .05 + 6 * np.sin(steps / 5)}, index=days)
    return agent, baseline, {"rule": baseline["value"]}


def test_unit_period_diagnostics_read_the_bootstrap_settings_from_config() -> None:
    """Draws and block length come from the robustness section, and the block never exceeds the sample."""
    agent, baseline, challengers = _diagnostic_inputs()
    cfg = config.load_config()
    cfg["robustness"].update(bootstrap_draws=50, bootstrap_block_weeks=2)
    result = robustness.test_period_diagnostics(agent, baseline, challengers, cfg)
    significance = result["challenger_significance"]
    assert significance["status"] == "measured"
    assert (significance["draws"], significance["block_weeks"]) == (50, 2)
    assert set(significance["p_values"]) == {"rule"} == set(significance["holm_adjusted"])
    assert "largest_episode" in result
    cfg["robustness"]["bootstrap_block_weeks"] = 1000
    clipped = robustness.test_period_diagnostics(agent, baseline, challengers, cfg)["challenger_significance"]
    assert clipped["block_weeks"] == clipped["weekly_returns"]


def test_unit_period_diagnostics_without_challengers_is_inconclusive() -> None:
    """With no challenger there is nothing to compare, so no p-value is invented."""
    agent, baseline, _ = _diagnostic_inputs()
    result = robustness.test_period_diagnostics(agent, baseline, {}, config.load_config())
    assert result["challenger_significance"] == {"status": "inconclusive"}

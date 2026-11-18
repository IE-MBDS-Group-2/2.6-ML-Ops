"""Acceptance tests for walk-forward model selection."""

from unittest.mock import Mock

import pandas as pd
import pytest

from src import portfolio, selection, train


def test_unit_windows_known_values() -> None:
    """Check three annual scoring blocks, each separated by five trading dates."""
    dates = pd.date_range("2010-01-01", "2014-12-31", freq="B")

    windows = selection.walk_forward_windows(
        dates, first_train_end="2011-12-31", last_end="2014-12-31", block="YE", gap_days=5
    )

    assert len(windows) == 3
    for year, (train_dates, score_dates) in zip((2012, 2013, 2014), windows):
        assert score_dates.equals(dates[dates.year == year])
        assert train_dates[0] == dates[0]
        start = dates.get_loc(score_dates[0])
        assert train_dates.equals(dates[:start - 5])


def test_unit_no_window_reaches_past_the_last_end() -> None:
    """Keep later dates out when the final scoring block ends midyear."""
    dates = pd.date_range("2010-01-01", "2014-12-31", freq="B")
    last_end = "2013-06-28"

    windows = selection.walk_forward_windows(dates, "2011-12-31", last_end, "YE", 5)

    assert len(windows) == 2
    assert windows[-1][1][-1] == pd.Timestamp(last_end)
    assert all(score.max() <= pd.Timestamp(last_end) for _, score in windows)


def test_unit_train_and_block_never_overlap() -> None:
    """Count the gap in supplied trading dates, even when a weekday is absent."""
    dates = pd.date_range("2011-01-01", "2013-12-31", freq="B").difference(
        pd.DatetimeIndex(["2011-12-26", "2012-12-25"])
    )

    windows = selection.walk_forward_windows(dates, "2011-12-31", "2013-12-31", "YE", 5)

    for train_days, score_days in windows:
        assert train_days.max() < score_days.min()
        assert dates.get_loc(score_days.min()) - dates.get_loc(train_days.max()) - 1 == 5


@pytest.mark.parametrize(
    ("first_end", "last_end", "block", "gap", "message"),
    [
        ("2011-12-31", "2012-12-31", "YE", -1, "gap_days"),
        ("2012-12-31", "2011-12-31", "YE", 0, "first_train_end"),
        ("2011-12-31", "2012-12-31", "0YE", 0, "positive pandas offset"),
        ("2010-12-31", "2012-12-31", "YE", 5, "no training or scoring dates"),
        ("2011-12-31", "2014-12-31", "YE", 5, "no training or scoring dates"),
    ],
)
def test_unit_windows_reject_invalid_splits(
    first_end: str, last_end: str, block: str, gap: int, message: str,
) -> None:
    """Fail clearly for invalid settings and for empty training or scoring windows."""
    dates = pd.date_range("2011-01-01", "2012-12-31", freq="B")

    with pytest.raises(ValueError, match=message):
        selection.walk_forward_windows(dates, first_end, last_end, block, gap)


def test_unit_windows_reject_unsorted_dates() -> None:
    """Reject an unsorted calendar before counting the trading-day gap."""
    dates = pd.DatetimeIndex(["2012-01-02", "2011-12-30"])

    with pytest.raises(ValueError, match="sorted, unique trading dates"):
        selection.walk_forward_windows(dates, "2011-12-31", "2012-01-02", "YE", 0)


def test_unit_walk_forward_scores_one_row_per_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fit on each window's training rows and compare both simulated paths on its scoring dates."""
    days = pd.date_range("2017-01-02", "2020-12-31", freq="B")
    windows = selection.walk_forward_windows(days, "2017-12-31", "2019-12-31", "YE", 5)
    decision_days = days[days.dayofweek == 4]
    table = pd.DataFrame({"signal": 0.2, "target_action": 0.5, portfolio.RULE_ACTION_COLUMN: 0.0},
                         index=decision_days)
    close = pd.DataFrame({"ETF": 100.0}, index=days)
    volume = pd.DataFrame({"ETF": 1000.0}, index=days)
    cfg = {
        "target_weights": {"policy": {"ETF": 1.0}},
        "portfolio": {"current_rule": {"quarter_months": [1, 4, 7, 10], "drift_band": 0.05}},
        "problem": {"target_column": "target_action", "problem_type": "classification"},
        "model": {"random_state": 42},
        "selection": {"first_train_end": "2017-12-31", "block": "YE", "gap_days": 5},
        "split": {"train_end": "2019-12-31", "val_end": "2020-12-31"},
        "rebalancing": {"review": "weekly"},
    }
    model_params = {"C": 1.0}
    fit = Mock(return_value=Mock())
    monkeypatch.setattr(train, "train_model", fit)
    agent = pd.DataFrame({"value": [1.0, 0.9], "cost_paid": [0.0, 0.0]})
    baseline = pd.DataFrame({"value": [1.0, 0.8], "cost_paid": [0.0, 0.0]})
    simulate = Mock(side_effect=[(agent, pd.DataFrame()), (baseline, pd.DataFrame())] * len(windows))
    monkeypatch.setattr(portfolio, "simulate", simulate)

    result = selection.walk_forward_scores(table, (close, volume, pd.DataFrame(index=days)), cfg, model_params)

    assert list(result.columns) == [
        "block_start", "block_end", "max_drawdown_reduction", "agent_mdd", "baseline_mdd",
        "agent_return", "baseline_return", "agent_cost", "baseline_cost"]
    assert len(result) == fit.call_count == len(windows)
    assert simulate.call_count == 2 * len(windows)
    assert list(pd.to_datetime(result["block_start"])) == [score[0] for _, score in windows]
    assert list(pd.to_datetime(result["block_end"])) == [score[-1] for _, score in windows]
    assert result["agent_mdd"].tolist() == pytest.approx([-0.10, -0.10])
    assert result["baseline_mdd"].tolist() == pytest.approx([-0.20, -0.20])
    assert result["max_drawdown_reduction"].tolist() == pytest.approx([0.50, 0.50])
    # The 2017-12-22 decision is outside the gap, but its label reaches 2018.
    safe_dates = pd.to_datetime(["2017-12-15", "2018-12-21"])
    for call, safe_last in zip(fit.call_args_list, safe_dates, strict=True):
        fitted_features, fitted_labels = call.args[:2]
        assert list(fitted_features.columns) == ["signal"]
        assert fitted_features.index.equals(table.loc[:safe_last].index)
        assert fitted_labels.index.equals(fitted_features.index)
    for call in simulate.call_args_list:
        assert call.args[0].index[-1] == pd.Timestamp(call.args[6])
    model = fit.return_value
    model.predict.return_value = [0.5]
    assert simulate.call_args_list[0].args[4](pd.Series({"signal": 0.2})) == 0.5
    with pytest.raises(ValueError, match="No labelled training rows"):
        selection.walk_forward_scores(
            table.assign(target_action=float("nan")), (close, volume, pd.DataFrame(index=days)), cfg, model_params,
        )


def _grid_config() -> dict:
    """Return a two-parameter grid and a fixed model setting."""
    return {"problem": {"problem_type": "classification"},
            "model": {"params": {"max_iter": 1000}},
            "selection": {"grid": {"C": [.1, 1], "class_weight": [None, "balanced"]}}}


def test_unit_grid_picks_the_highest_median(monkeypatch: pytest.MonkeyPatch) -> None:
    """Try exactly the Cartesian product and rank by median, not by the largest score."""
    scores = Mock(side_effect=[
        pd.DataFrame({"max_drawdown_reduction": [0.0, 0.0, 1.0]}),
        pd.DataFrame({"max_drawdown_reduction": [.1, .1, .1]}),
        pd.DataFrame({"max_drawdown_reduction": [.2, .2, .2]}),
        pd.DataFrame({"max_drawdown_reduction": [.3, .3, -.9]}),
    ])
    monkeypatch.setattr(selection, "walk_forward_scores", scores)
    best, trials = selection.grid_search(pd.DataFrame(), (), _grid_config())
    # pandas stores a missing class_weight (None) as NaN in the trial table, so compare it as missing, not as None.
    class_weights = trials["class_weight"].astype(object).where(trials["class_weight"].notna(), None)
    assert list(zip(trials["C"], class_weights)) == [(.1, None), (.1, "balanced"), (1, None), (1, "balanced")]
    assert trials["median_max_drawdown_reduction"].tolist() == pytest.approx([0, .1, .2, .3])
    assert best == {"max_iter": 1000, "C": 1, "class_weight": "balanced"}
    assert scores.call_count == 4


def test_unit_grid_logs_every_try(monkeypatch: pytest.MonkeyPatch) -> None:
    """Record every grid trial under the tracking prefix and keep the first tie."""
    scores = pd.DataFrame({"max_drawdown_reduction": [.02, .01]})
    monkeypatch.setattr(selection, "walk_forward_scores", Mock(return_value=scores))
    log = Mock()
    monkeypatch.setattr(selection.tracking, "log_metrics", log)
    run = object()
    best, _ = selection.grid_search(pd.DataFrame(), (), _grid_config(), run)
    assert best["C"] == .1 and best["class_weight"] is None
    assert [call.args[1]["trial"] for call in log.call_args_list] == [1, 2, 3, 4]
    assert all(call.args[0] is run and call.kwargs["prefix"] == "grid/" for call in log.call_args_list)


@pytest.mark.parametrize("grid,message", [
    ({}, "nonempty"), ({"C": []}, "nonempty"), ({"typo": [1]}, "Unsupported"),
    ({"random_state": [0]}, "Unsupported"),
])
def test_unit_grid_raises_for_bad_parameters(grid: dict, message: str) -> None:
    """Reject invalid grids before attempting any candidate."""
    cfg = _grid_config()
    cfg["selection"]["grid"] = grid
    with pytest.raises(ValueError, match=message):
        selection.grid_search(pd.DataFrame(), (), cfg)


def test_unit_frequency_comparison_one_row_per_review_on_train_years(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each review frequency gets its own decision table and walk-forward fit, on the train years only."""
    days = pd.date_range("2017-01-02", "2020-12-31", freq="B")
    market = (pd.DataFrame({"ETF": 100.0}, index=days), pd.DataFrame({"ETF": 1000.0}, index=days),
              pd.DataFrame(index=days))
    cfg = {"split": {"train_end": "2019-12-31"}, "portfolio": {"first_decision_date": "2017-02-03"},
           "target_weights": {"policy": {"ETF": 1.0}}, "rebalancing": {"review": "weekly", "method": "smart"}}
    seen = []

    def build(close, volume, features_frame, target, start, end, settings):
        seen.append((settings["rebalancing"]["review"], close.index.max(), start, end))
        return pd.DataFrame({"signal": 0.0}, index=days[:5])

    def scores(table, market_frames, settings, params):
        factor = {"daily": 1.0, "weekly": 2.0, "monthly": 3.0}[settings["rebalancing"]["review"]]
        return pd.DataFrame({
            "max_drawdown_reduction": [0.1 * factor, 0.3 * factor, float("nan")],
            "agent_mdd": [-0.1, -0.2, -0.3], "baseline_mdd": [-0.2, -0.4, -0.6],
            "agent_return": [0.05, 0.06, 0.07], "baseline_return": [0.04, 0.05, 0.06],
            "agent_cost": [0.001, 0.002, 0.003], "baseline_cost": [0.0, 0.0, 0.0]})

    monkeypatch.setattr(portfolio, "build_decision_table", build)
    monkeypatch.setattr(selection, "walk_forward_scores", scores)

    result = selection.frequency_comparison(market, cfg, {"C": 1.0})

    assert result["review"].tolist() == ["daily", "weekly", "monthly"]
    assert [item[0] for item in seen] == ["daily", "weekly", "monthly"]
    assert all(item[1] == pd.Timestamp("2019-12-31") and item[3] == "2019-12-31" for item in seen)
    assert result["blocks"].tolist() == [3, 3, 3]
    assert result["blocks_scored"].tolist() == [2, 2, 2]
    assert result["median_max_drawdown_reduction"].tolist() == pytest.approx([0.2, 0.4, 0.6])
    assert result["model_median_max_drawdown"].tolist() == pytest.approx([-0.2] * 3)
    assert result["rule_median_max_drawdown"].tolist() == pytest.approx([-0.4] * 3)
    assert result["model_median_annual_return"].tolist() == pytest.approx([0.06] * 3)
    assert result["rule_median_annual_trading_cost"].tolist() == pytest.approx([0.0] * 3)
    assert cfg["rebalancing"]["review"] == "weekly"          # the caller's config is not changed

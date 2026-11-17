"""Exercise long-only allocation, split boundaries and the real simulator."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src import benchmarks, config, tracking


def _uncorrelated_returns() -> pd.DataFrame:
    """Return two equal-variance assets with exactly zero covariance."""
    return pd.DataFrame({"B": [0.01, -0.01, 0.01, -0.01],
                         "A": [0.01, 0.01, -0.01, -0.01]})


@pytest.mark.parametrize("method", ["min_variance", "risk_parity"])
def test_unit_two_equal_uncorrelated_assets_give_half_half(method: str) -> None:
    """Symmetric minimum-variance and risk-parity allocations should be equal weight."""
    assert benchmarks.fit_weights(_uncorrelated_returns(), method, {}).tolist() == pytest.approx(
        [0.5, 0.5], abs=0.01,
    )


def test_unit_min_variance_prefers_the_calmer_asset() -> None:
    """With zero covariance, the lower-variance asset receives most of the portfolio."""
    returns = _uncorrelated_returns()
    returns["A"] /= 3
    weights = benchmarks.fit_weights(returns, "min_variance", {})
    assert weights["A"] > 0.5


@pytest.mark.parametrize("method", benchmarks.METHODS)
def test_unit_weights_are_long_only_and_sum_to_one(method: str) -> None:
    """Keep the caller's ticker order and invest fully without shorting."""
    rng = np.random.default_rng(12)
    columns = ["C", "A", "B"]
    frame = pd.DataFrame(rng.normal(0.001, 0.01, size=(60, len(columns))), columns=columns)
    weights = benchmarks.fit_weights(frame, method, {"risk_aversion": 3.0})
    assert list(weights.index) == columns
    assert (weights >= 0).all()
    assert weights.sum() == pytest.approx(1.0)


def test_unit_unknown_method_raises() -> None:
    """Name the supported methods when the requested benchmark does not exist."""
    with pytest.raises(ValueError, match="min_variance, max_sharpe, mean_variance, risk_parity"):
        benchmarks.fit_weights(_uncorrelated_returns(), "equal_weight", {})


def test_unit_invalid_optimizer_inputs_raise() -> None:
    """Reject insufficient or nonfinite returns and invalid risk aversion."""
    returns = _uncorrelated_returns()
    with pytest.raises(ValueError, match="at least two"):
        benchmarks.fit_weights(returns.iloc[:1], "min_variance", {})
    with pytest.raises(ValueError, match="must be finite"):
        benchmarks.fit_weights(returns * float("inf"), "min_variance", {})
    with pytest.raises(ValueError, match="risk_aversion"):
        benchmarks.fit_weights(returns, "mean_variance", {"risk_aversion": -1})


def test_unit_unsuccessful_fit_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop when SLSQP cannot find a valid result."""
    monkeypatch.setattr(benchmarks.optimize, "minimize", Mock(return_value=Mock(success=False, message="failed")))
    with pytest.raises(ValueError, match="could not fit weights"):
        benchmarks.fit_weights(_uncorrelated_returns(), "min_variance", {})


def _market_fixture() -> tuple[tuple, dict]:
    """Build deterministic price and volume frames with enough history for both periods."""
    cfg = config.load_config()
    cfg["data"]["tickers"] = ["B", "A"]
    cfg["portfolio"]["first_decision_date"] = "2019-01-04"
    cfg["split"]["train_end"] = "2019-12-31"
    days = pd.date_range("2019-01-01", "2020-03-31", freq="B")
    rng = np.random.default_rng(32)
    changes = pd.DataFrame(rng.normal(0.0004, 0.01, (len(days), 2)), index=days, columns=["B", "A"])
    close = 100.0 * (1.0 + changes).cumprod()
    volume = pd.DataFrame(100_000.0, index=days, columns=close.columns)
    return (close, volume, pd.DataFrame(index=days)), cfg


def test_unit_only_train_rows_are_used() -> None:
    """Changing every post-training close leaves fitted allocations identical."""
    market_data, cfg = _market_fixture()
    close, volume, features = market_data
    poisoned = close.copy()
    poisoned.loc["2020":, "A"] *= 100.0
    initial = benchmarks.run_benchmarks(market_data, cfg, "2020-01-01", "2020-03-31")
    updated = benchmarks.run_benchmarks((poisoned, volume, features), cfg, "2020-01-01", "2020-03-31")
    assert initial["weights"].tolist() == updated["weights"].tolist()


def test_integration_run_benchmarks_one_row_per_method() -> None:
    """Exercise all four allocations through the real, cost-aware trading simulator."""
    market_data, cfg = _market_fixture()
    table = benchmarks.run_benchmarks(market_data, cfg, "2020-01-01", "2020-03-31")
    assert table["method"].tolist() == cfg["benchmarks"]["methods"]
    assert set(table) == {
        "method", "weights", "max_drawdown", "annual_return", "annual_volatility", "sharpe", "annual_trading_cost",
    }
    assert np.isfinite(table.select_dtypes(include="number").to_numpy()).all()


def test_integration_reports_csv_metrics_and_tracking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Write the configured CSV and publish the same rows for metrics and W&B."""
    market_data, cfg = _market_fixture()
    path = tmp_path / "reports" / "benchmarks.csv"
    run = Mock()
    log_table = Mock()
    monkeypatch.setattr(tracking, "log_table", log_table)
    metrics = {}
    benchmarks.report_benchmarks(market_data, cfg, "2020-01-01", "2020-03-31", path, metrics, run)
    owner, name, table = log_table.call_args.args
    assert owner is run and name == "benchmarks"
    assert pd.read_csv(path)["method"].tolist() == table["method"].tolist()
    assert metrics["benchmarks"] == table.to_dict(orient="records")


def test_unit_disabled_benchmarks_leave_no_report(tmp_path: Path) -> None:
    """When disabled, benchmark reporting needs no market frames or tracker."""
    _, cfg = _market_fixture()
    cfg["benchmarks"]["enabled"] = False
    path = tmp_path / "benchmarks.csv"
    metrics = {}
    benchmarks.report_benchmarks((), cfg, "", "", path, metrics, None)
    assert not path.exists()
    assert not metrics

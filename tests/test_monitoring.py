"""Tests for src/monitoring.py: the drift measure (PSI) and the weekly fallback rate, on small known tables."""

import math

import pandas as pd
import pytest

from src import monitoring

REFERENCE = pd.Series(range(10), dtype=float)   # 0..9: with 2 bins the edge is the median, 4.5


# ---------------------------------------------------------------- population_stability_index


def test_unit_psi_is_zero_for_the_same_distribution():
    assert monitoring.population_stability_index(REFERENCE, REFERENCE, bins=5) == pytest.approx(0.0)


def test_unit_psi_known_value_two_bins():
    """Reference shares 50/50, current 75/25: PSI = 0.25 ln(1.5) + (-0.25) ln(0.5) = 0.25 ln(3) = 0.2747."""
    current = pd.Series([0.0, 1.0, 2.0, 9.0])
    psi = monitoring.population_stability_index(REFERENCE, current, bins=2)
    assert psi == pytest.approx(0.25 * math.log(3))


def test_unit_psi_empty_bin_uses_the_floor_not_log_zero():
    """Every current value below the median: the empty upper bin counts as SHARE_FLOOR, so PSI stays finite."""
    floor = monitoring.SHARE_FLOOR
    expected = 0.5 * math.log(1 / 0.5) + (floor - 0.5) * math.log(floor / 0.5)
    psi = monitoring.population_stability_index(REFERENCE, pd.Series([0.0, 1.0, 2.0]), bins=2)
    assert psi == pytest.approx(expected)


def test_unit_psi_ignores_missing_values():
    with_gaps = pd.Series([None, 0.0, 1.0, 2.0, 9.0, None])
    assert monitoring.population_stability_index(REFERENCE, with_gaps, bins=2) == pytest.approx(0.25 * math.log(3))


def test_unit_psi_repeated_values_merge_bins():
    """A constant reference has one quantile edge repeated: the bins merge, and the same constant gives 0."""
    constant = pd.Series([1.0] * 10)
    assert monitoring.population_stability_index(constant, constant, bins=10) == pytest.approx(0.0)


@pytest.mark.parametrize("bins", [0, 1])
def test_unit_psi_rejects_fewer_than_two_bins(bins):
    with pytest.raises(ValueError, match="bins must be at least 2"):
        monitoring.population_stability_index(REFERENCE, REFERENCE, bins=bins)


def test_unit_psi_rejects_an_empty_side():
    with pytest.raises(ValueError, match="at least one value"):
        monitoring.population_stability_index(REFERENCE, pd.Series([None], dtype=float), bins=2)


# ---------------------------------------------------------------- feature_drift and weekly_drift


def test_unit_feature_drift_sorts_largest_first():
    reference = pd.DataFrame({"steady": REFERENCE, "moved": REFERENCE})
    current = pd.DataFrame({"steady": REFERENCE, "moved": REFERENCE + 100})
    psi = monitoring.feature_drift(reference, current, ["steady", "moved"], bins=2)
    assert list(psi.index) == ["moved", "steady"]
    assert psi["steady"] == pytest.approx(0.0)
    assert psi.name == "psi"


def test_unit_feature_drift_names_missing_columns():
    with pytest.raises(KeyError, match="absent"):
        monitoring.feature_drift(pd.DataFrame({"x": REFERENCE}), pd.DataFrame({"x": REFERENCE}), ["x", "absent"], 2)


def _business_days(values: list[float]) -> pd.DataFrame:
    """One feature "x" on business days from Monday 2 Nov 2026."""
    return pd.DataFrame({"x": values}, index=pd.bdate_range("2026-11-02", periods=len(values)))


def test_unit_weekly_drift_one_row_per_friday_with_a_full_window():
    """Ten business days, a 5-day window: Fridays 6 and 13 Nov both have five days behind them."""
    reference = pd.DataFrame({"x": pd.Series(range(100), dtype=float)})
    drift = monitoring.weekly_drift(reference, _business_days([50.0] * 10), ["x"], bins=10, window_days=5)
    assert list(drift.index) == list(pd.to_datetime(["2026-11-06", "2026-11-13"]))
    assert (drift["feature"] == "x").all()
    assert (drift["max_psi"] > 0.2).all()   # every value in one decile: a clear shift


def test_unit_weekly_drift_skips_fridays_without_a_full_window():
    reference = pd.DataFrame({"x": pd.Series(range(100), dtype=float)})
    drift = monitoring.weekly_drift(reference, _business_days([50.0] * 10), ["x"], bins=10, window_days=6)
    assert list(drift.index) == [pd.Timestamp("2026-11-13")]


def test_unit_weekly_drift_empty_when_data_is_short():
    reference = pd.DataFrame({"x": REFERENCE})
    drift = monitoring.weekly_drift(reference, _business_days([1.0] * 3), ["x"], bins=2, window_days=5)
    assert drift.empty
    assert list(drift.columns) == ["max_psi", "feature"]


def test_unit_weekly_drift_rejects_a_window_below_one():
    with pytest.raises(ValueError, match="window_days"):
        monitoring.weekly_drift(pd.DataFrame({"x": REFERENCE}), _business_days([1.0]), ["x"], 2, window_days=0)


# ---------------------------------------------------------------- weekly_fallback_rate


def test_unit_weekly_fallback_rate_known_values():
    """Mon 16 and Fri 20 Nov fall in the week ending Friday 20 Nov; Sat 21 Nov in the week ending 27 Nov."""
    answers = pd.DataFrame({
        "as_of": ["2026-11-16", "2026-11-20", "2026-11-20", "2026-11-20", "2026-11-21"],
        "fallback": [True, False, "True", "false", 1],
    })
    rates = monitoring.weekly_fallback_rate(answers, "as_of", "fallback")
    assert list(rates.index) == list(pd.to_datetime(["2026-11-20", "2026-11-27"]))
    assert rates["fallback_rate"].tolist() == [0.5, 1.0]
    assert rates["answers"].tolist() == [4, 1]
    assert rates.index.name == "week"


def test_unit_weekly_fallback_rate_names_missing_columns():
    with pytest.raises(KeyError, match="fallback"):
        monitoring.weekly_fallback_rate(pd.DataFrame({"as_of": ["2026-11-20"]}), "as_of", "fallback")


def test_unit_weekly_fallback_rate_rejects_no_answers():
    with pytest.raises(ValueError, match="no answers"):
        monitoring.weekly_fallback_rate(pd.DataFrame({"as_of": [], "fallback": []}), "as_of", "fallback")


# ---------------------------------------------------------------- alerts


def _rates(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"fallback_rate": values})


@pytest.mark.parametrize("values, expected", [
    ([0.6, 0.7, 0.6, 0.55], True),          # above 50% in each of the last 4 weeks
    ([0.9, 0.6, 0.5, 0.6, 0.7], False),     # 50% exactly is not above the limit
    ([0.6, 0.7, 0.8], False),               # only 3 weeks measured yet
    ([0.1, 0.6, 0.7, 0.6, 0.55], True),     # only the last 4 count
])
def test_unit_fallback_alert(values, expected):
    assert monitoring.fallback_alert(_rates(values), limit=0.5, weeks=4) is expected


@pytest.mark.parametrize("values, expected", [([0.3, 0.1], False), ([0.1, 0.25], True), ([0.1, 0.2], False)])
def test_unit_drift_alert_reads_the_latest_week(values, expected):
    assert monitoring.drift_alert(pd.DataFrame({"max_psi": values}), limit=0.2) is expected


def test_unit_drift_alert_false_without_weeks():
    assert monitoring.drift_alert(pd.DataFrame({"max_psi": []}), limit=0.2) is False

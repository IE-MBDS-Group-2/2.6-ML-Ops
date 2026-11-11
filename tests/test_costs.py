"""Tests for src/costs.py. Expected values are worked out by hand."""
import math

import numpy as np
import pandas as pd
import pytest

from src import costs

CFG = {
    "aum": 100_000_000,
    "model": "realistic",
    "flat": {"rate": 0.001},
    "commission": {"rate": 0.0005, "min_fee": 1.25, "cap": 0.01},
    "slippage": {"entry_bps": 5.0, "exit_bps": 10.0, "impact_factor": 2.0, "adv_window": 20},
}


def cfg(model):
    return {**CFG, "model": model}


def test_flat_is_rate_times_absolute_weight_change():
    out = costs.trade_costs([0.10, -0.05], aum=1e8, cfg=cfg("flat"))
    np.testing.assert_allclose(out, [1e-4, 5e-5])


def test_flat_ignores_aum_and_volume():
    a = costs.trade_costs([0.1], aum=1e3, adv=[1.0], cfg=cfg("flat"))
    b = costs.trade_costs([0.1], aum=1e12, cfg=cfg("flat"))
    np.testing.assert_allclose(a, b)


def test_commission_rate_floor_and_cap():
    out = costs.commission([1e7, 1000.0, 100.0, 0.0], rate=0.0005, min_fee=1.25, cap=0.01)
    # 0.05% of 1e7; floor 1.25 beats 0.5; cap 1% of 100 = 1.0 beats floor; zero trade is free
    np.testing.assert_allclose(out, [5000.0, 1.25, 1.0, 0.0])


def test_sqrt_impact_values():
    out = costs.sqrt_impact_bps([1e6, 4e6], adv=[1e8, 1e8], impact_factor=2.0)
    np.testing.assert_allclose(out, [2.0, 4.0])  # 1% ADV -> 2 bps, 4% ADV -> 4 bps


def test_impact_is_zero_without_usable_adv_or_trade():
    out = costs.sqrt_impact_bps([1e6, 1e6, 1e6, 0.0], adv=[np.nan, 0.0, -5.0, 1e8], impact_factor=2.0)
    np.testing.assert_array_equal(out, [0.0, 0.0, 0.0, 0.0])


def test_realistic_cost_hand_computed():
    aum, adv = 1e8, np.array([1e10, 1e10])
    out = costs.trade_costs([0.05, -0.05], aum=aum, adv=adv, cfg=cfg("realistic"))
    tv = 5e6
    impact = 2.0 * math.sqrt(tv / 1e10 * 100)  # 0.05% of ADV
    buy = (0.0005 * tv + tv * (5.0 + impact) * 1e-4) / aum
    sell = (0.0005 * tv + tv * (10.0 + impact) * 1e-4) / aum
    np.testing.assert_allclose(out, [buy, sell])


def test_sells_cost_more_than_equal_buys():
    out = costs.trade_costs([0.05, -0.05], aum=1e8, adv=[1e10, 1e10], cfg=cfg("realistic"))
    assert out[1] > out[0]


def test_no_trade_costs_nothing():
    for model in ("flat", "realistic"):
        out = costs.trade_costs([0.0, 0.0], aum=1e8, adv=[1e10, 1e10], cfg=cfg(model))
        np.testing.assert_array_equal(out, [0.0, 0.0])


def test_realistic_without_adv_has_no_impact_term():
    with_adv = costs.trade_costs([0.05], aum=1e8, adv=[1e6], cfg=cfg("realistic"))
    without = costs.trade_costs([0.05], aum=1e8, cfg=cfg("realistic"))
    expected = (0.0005 * 5e6 + 5e6 * 5.0 * 1e-4) / 1e8
    np.testing.assert_allclose(without, [expected])
    assert with_adv[0] > without[0]


def test_fixed_minimum_fee_matters_only_at_small_aum():
    small = costs.trade_costs([0.001], aum=1e3, cfg=cfg("realistic"))  # trade value 1.0
    large = costs.trade_costs([0.001], aum=1e9, cfg=cfg("realistic"))
    # at small AUM the 1% cap binds the 1.25 floor; fee fraction is not 0.05%
    assert small[0] != pytest.approx(large[0])


def test_impact_share_grows_with_aum():
    adv = [1e10]
    low = costs.trade_costs([0.05], aum=1e7, adv=adv, cfg=cfg("realistic"))[0]
    high = costs.trade_costs([0.05], aum=1e10, adv=adv, cfg=cfg("realistic"))[0]
    assert high > low


def test_unknown_model_and_missing_cfg_raise():
    with pytest.raises(ValueError):
        costs.trade_costs([0.1], aum=1e8, cfg=cfg("mystery"))
    with pytest.raises(ValueError):
        costs.trade_costs([0.1], aum=1e8)


def test_adv_uses_only_past_and_current_days():
    idx = pd.date_range("2020-01-01", periods=5)
    close = pd.DataFrame({"A": [10.0] * 5}, index=idx)
    volume = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
    adv = costs.average_daily_dollar_volume(close, volume, window=3)
    assert adv["A"].iloc[:2].isna().all()  # not enough history yet
    assert adv["A"].iloc[2] == pytest.approx(20.0)  # mean(10,20,30)
    changed = volume.copy()
    changed.iloc[4, 0] = 1e9  # a future day must not move earlier values
    adv2 = costs.average_daily_dollar_volume(close, changed, window=3)
    assert adv2["A"].iloc[3] == pytest.approx(adv["A"].iloc[3])

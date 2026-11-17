"""Tests for src/trades.py: positions to weights, the proposed portfolio and the trade list.

Tiny hand-made positions only. Never the real data/ folder.
"""
import numpy as np
import pandas as pd
import pytest

from src import portfolio, trades

TICKERS = ["SPY", "EFA", "IEF", "GLD", "BIL"]
TICKER, VALUE = "ticker", "market_value"


def positions(rows: dict) -> pd.DataFrame:
    """A positions table with the config's column names, one row per (ticker, value) pair."""
    return pd.DataFrame({TICKER: list(rows), VALUE: list(rows.values())})


def two_assets(spy: float, ief: float) -> pd.Series:
    """Weights over the five tickers with only SPY and IEF held."""
    return pd.Series([spy, 0.0, ief, 0.0, 0.0], index=TICKERS)


# ---------- positions_to_weights ----------

def test_unit_positions_to_weights_known_values():
    weights, total = trades.positions_to_weights(positions({"IEF": 350.0, "SPY": 650.0}), TICKERS, TICKER, VALUE)
    assert list(weights.index) == TICKERS
    assert weights["SPY"] == pytest.approx(0.65)
    assert weights["IEF"] == pytest.approx(0.35)
    assert total == pytest.approx(1000.0)


def test_unit_missing_ticker_gets_zero_weight():
    held = positions({"SPY": 400.0, "EFA": 200.0, "IEF": 300.0, "GLD": 100.0})
    weights, _ = trades.positions_to_weights(held, TICKERS, TICKER, VALUE)
    assert weights["BIL"] == 0.0
    assert weights.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("held, message", [
    (positions({"SPY": 500.0, "QQQ": 500.0}), "QQQ"),
    (positions({"SPY": 500.0, "IEF": -10.0}), "Negative market value for \\['IEF'\\]"),
    (positions({"SPY": 500.0, "IEF": None}), "No market value for \\['IEF'\\]"),
    (pd.DataFrame({TICKER: ["SPY", "SPY"], VALUE: [1.0, 2.0]}), "more than once.*SPY"),
    (positions({"SPY": 0.0, "IEF": 0.0}), "total market value of 0"),
])
def test_unit_positions_to_weights_raises(held, message):
    with pytest.raises(ValueError, match=message):
        trades.positions_to_weights(held, TICKERS, TICKER, VALUE)


def test_unit_positions_to_weights_raises_on_missing_column():
    with pytest.raises(ValueError, match="market_value"):
        trades.positions_to_weights(pd.DataFrame({TICKER: ["SPY"], "value": [1.0]}), TICKERS, TICKER, VALUE)


def test_unit_column_names_come_from_the_arguments():
    held = pd.DataFrame({"symbol": ["SPY", "IEF"], "mv": [650.0, 350.0]})
    weights, total = trades.positions_to_weights(held, TICKERS, "symbol", "mv")
    assert weights["SPY"] == pytest.approx(0.65)
    assert total == pytest.approx(1000.0)


# ---------- proposed_portfolio ----------

@pytest.mark.parametrize("action, spy, ief", [(1.0, 0.60, 0.40), (0.5, 0.625, 0.375), (0.0, 0.65, 0.35)])
def test_unit_proposed_portfolio_known_values(action, spy, ief):
    proposed = trades.proposed_portfolio(two_assets(0.65, 0.35), two_assets(0.60, 0.40), action, 1000.0)
    assert list(proposed.columns) == ["ticker", "weight", "value"]
    by_ticker = proposed.set_index("ticker")
    assert by_ticker.loc["SPY", "weight"] == pytest.approx(spy)
    assert by_ticker.loc["IEF", "weight"] == pytest.approx(ief)
    assert np.allclose(by_ticker["value"], by_ticker["weight"] * 1000.0)
    assert by_ticker["weight"].sum() == pytest.approx(1.0)


def test_unit_proposed_portfolio_raises_on_action_outside_0_1():
    with pytest.raises(ValueError, match="between 0 and 1"):
        trades.proposed_portfolio(two_assets(0.65, 0.35), two_assets(0.60, 0.40), 1.5, 1000.0)


# ---------- trade_list ----------

def test_unit_trade_list_nets_to_zero():
    before = two_assets(0.65, 0.35)
    proposed = trades.proposed_portfolio(before, two_assets(0.60, 0.40), 1.0, 1000.0)
    orders = trades.trade_list(before, proposed, 1000.0, min_trade=0.0)
    assert list(orders.columns) == ["ticker", "side", "value", "weight_before", "weight_after"]
    by_ticker = orders.set_index("ticker")
    assert by_ticker.loc["SPY", "side"] == "sell"
    assert by_ticker.loc["SPY", "value"] == pytest.approx(50.0)
    assert by_ticker.loc["IEF", "side"] == "buy"
    assert by_ticker.loc["IEF", "value"] == pytest.approx(50.0)
    assert len(orders) == 2
    signed = np.where(orders["side"] == "buy", orders["value"], -orders["value"])
    assert signed.sum() == pytest.approx(0.0)


def test_unit_min_trade_drops_small_trades():
    before = two_assets(0.601, 0.399)
    proposed = trades.proposed_portfolio(before, two_assets(0.60, 0.40), 1.0, 1000.0)
    orders = trades.trade_list(before, proposed, 1000.0, min_trade=0.005)
    assert orders.empty
    assert list(orders.columns) == ["ticker", "side", "value", "weight_before", "weight_after"]


def test_unit_no_action_gives_no_trades():
    before = two_assets(0.65, 0.35)
    proposed = trades.proposed_portfolio(before, two_assets(0.60, 0.40), 0.0, 1000.0)
    assert trades.trade_list(before, proposed, 1000.0, min_trade=0.0).empty


# ---------- integration ----------

def test_integration_matches_apply_action():
    rng = np.random.default_rng(7)
    weights = pd.Series(rng.dirichlet(np.ones(len(TICKERS))), index=TICKERS)
    target = pd.Series(rng.dirichlet(np.ones(len(TICKERS))), index=TICKERS)
    for action in (0.0, 0.5, 1.0):
        proposed = trades.proposed_portfolio(weights, target, action, 1.0).set_index("ticker")["weight"]
        expected = portfolio.apply_action(weights, target, action)
        assert np.allclose(proposed.to_numpy(), expected.to_numpy())

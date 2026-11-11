"""
costs.py  [Execution costs: commission and slippage]
=====================================================
Purpose   Turn a rebalance (change in portfolio weights) into a trading cost
Used by   the shared execution function, so the agent and every baseline rule
          are charged by exactly the same code
Ported    from the Argus backtester cost models (asymmetric slippage with
          square-root market impact, IBKR Pro EU commission), rewritten for
          weights instead of share counts

Two models, chosen by config.yaml (costs.model)
- flat       cost = |weight change| x flat.rate. The README baseline (0.1%).
             Does not depend on AUM or volume.
- realistic  cost = commission + slippage, per ETF, per trade.
             Commission: rate x trade value, floored at min_fee, capped at
             cap x trade value (a trade of zero costs zero).
             Slippage: (entry_bps or exit_bps + impact_factor x sqrt(% of ADV))
             x trade value. Buys count as entries, sells as exits.

Units
- Weights are fractions of portfolio value. aum is the capital allocated to
  the model portfolio, in the same currency as min_fee. No FX conversion.
- Every cost function returns cost as a FRACTION of portfolio value, one value
  per ETF, so the reward is simply: portfolio return minus costs.sum().

Look-ahead
- ADV for a decision at the close of day t uses days up to and including t.

Tests     tests/test_costs.py
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_BPS = 1e-4


def average_daily_dollar_volume(
    close: pd.DataFrame, volume: pd.DataFrame, window: int
) -> pd.DataFrame:
    """Trailing mean of close x volume over `window` days, ending on each day.

    The value on day t uses days t-window+1 .. t only, so it is known at the
    close of day t. The first window-1 rows are NaN.
    """
    return (close * volume).rolling(window, min_periods=window).mean()


def sqrt_impact_bps(trade_value, adv, impact_factor: float):
    """Square-root market impact in bps: impact_factor x sqrt(trade_value / adv x 100).

    Zero where adv is missing, NaN or non-positive, or where nothing is traded.
    """
    trade_value = np.asarray(trade_value, dtype=float)
    adv = np.asarray(adv, dtype=float)
    valid = np.isfinite(adv) & (adv > 0) & (trade_value > 0)
    pct_adv = np.divide(trade_value, adv, out=np.zeros_like(trade_value), where=valid) * 100.0
    return impact_factor * np.sqrt(pct_adv)


def commission(trade_value, rate: float, min_fee: float, cap: float):
    """Commission in currency units per trade: rate x value, floored then capped."""
    trade_value = np.asarray(trade_value, dtype=float)
    fee = np.maximum(rate * trade_value, min_fee)
    fee = np.minimum(fee, cap * trade_value)
    return np.where(trade_value > 0, fee, 0.0)


def slippage_bps(delta_w, trade_value, adv, entry_bps: float, exit_bps: float, impact_factor: float):
    """Adverse slippage in bps per ETF: base (entry for buys, exit for sells) + impact."""
    delta_w = np.asarray(delta_w, dtype=float)
    base = np.where(delta_w < 0, exit_bps, entry_bps)
    return base + sqrt_impact_bps(trade_value, adv, impact_factor)


def trade_costs(delta_w, aum: float, adv=None, cfg: dict | None = None):
    """Cost of one rebalance as a fraction of portfolio value, per ETF.

    Args:
        delta_w: signed weight change per ETF (positive = buy, negative = sell).
        aum: capital allocated to the model portfolio. Passed in, never read
            from a global, so a sweep over AUM can override it.
        adv: trailing average daily dollar volume per ETF, in the same currency
            as aum. Only used by the realistic model; None means no impact term.
        cfg: the `costs` section of config.yaml.
    """
    if cfg is None:
        raise ValueError("cfg (the costs section of config.yaml) is required")
    delta_w = np.asarray(delta_w, dtype=float)
    model = cfg["model"]
    logger.debug("Trade costs: %s model, %d assets, aum %.0f", model, delta_w.size, aum)

    if model == "flat":
        return np.abs(delta_w) * cfg["flat"]["rate"]

    if model == "realistic":
        trade_value = np.abs(delta_w) * aum
        com, slip = cfg["commission"], cfg["slippage"]
        adv_arr = np.full_like(trade_value, np.nan) if adv is None else adv
        fee = commission(trade_value, com["rate"], com["min_fee"], com["cap"])
        bps = slippage_bps(
            delta_w, trade_value, adv_arr, slip["entry_bps"], slip["exit_bps"], slip["impact_factor"]
        )
        return (fee + trade_value * bps * _BPS) / aum

    raise ValueError(f"costs.model must be 'flat' or 'realistic', got {model!r}")

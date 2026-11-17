"""
trades.py  [Stage 3: POSITIONS IN, PROPOSED PORTFOLIO AND TRADES OUT]
=====================================================================
Purpose   Turn one portfolio's holdings (ticker, market value) into weights, apply the recommended action, and
          list the trades that get there
Input     A positions table (column names from the positions section of config.yaml), the configured tickers,
          the target weights, the recommended action and portfolio.min_trade
Output    positions_to_weights: weights per ticker and the total value; proposed_portfolio: ticker, weight,
          value; trade_list: ticker, side, value, weight_before, weight_after
Called by api.py (each live request) and main.py (batch, when paths.positions exists)
Calls     portfolio.apply_action

Why it matters
- A portfolio manager thinks in holdings and orders, not in weights and actions: this is the translation
- No new maths: the proposed weights come from portfolio.apply_action, the same formula the simulator uses
- Buys and sells net to zero, so the proposal never needs new cash or leaves cash idle

Tests     tests/test_trades.py
"""

import logging

import pandas as pd

from src import portfolio

logger = logging.getLogger(__name__)

BUY, SELL = "buy", "sell"
PORTFOLIO_COLUMNS = ["ticker", "weight", "value"]
TRADE_COLUMNS = ["ticker", "side", "value", "weight_before", "weight_after"]
_NO_TRADE = 1e-12   # a weight change this small is rounding, not a trade


def _check_positions(positions: pd.DataFrame, tickers: list[str], ticker_column: str, value_column: str) -> None:
    """Raise on the first problem in a positions table, before any maths.

    Inputs:  positions, one row per holding; tickers, data.tickers from config; the two column names.
    Outputs: None.
    Why:     positions come from outside the pipeline (a file or an API request), so nothing can be assumed.
    Raises:  ValueError for a missing column, an unknown or repeated ticker, a missing or negative value.
    """
    missing_columns = [name for name in (ticker_column, value_column) if name not in positions.columns]
    if missing_columns:
        raise ValueError(f"The positions have no column {missing_columns}. Expected {ticker_column!r} and "
                         f"{value_column!r} (the positions section of config.yaml).")
    unknown = sorted(set(positions[ticker_column]) - set(tickers))
    if unknown:
        raise ValueError(f"Unknown tickers in the positions: {unknown}. Allowed: {', '.join(tickers)}.")
    repeated = sorted(positions.loc[positions[ticker_column].duplicated(), ticker_column].unique())
    if repeated:
        raise ValueError(f"Tickers listed more than once in the positions: {repeated}. Send one row per holding.")
    values = pd.to_numeric(positions[value_column], errors="coerce")
    no_value = sorted(positions.loc[values.isna(), ticker_column])
    if no_value:
        raise ValueError(f"No market value for {no_value}. Send a number for every holding (0 if sold out).")
    negative = sorted(positions.loc[values < 0, ticker_column])
    if negative:
        raise ValueError(f"Negative market value for {negative}. Positions must be long only (0 or more).")


def positions_to_weights(positions: pd.DataFrame, tickers: list[str], ticker_column: str,
                         value_column: str) -> tuple[pd.Series, float]:
    """Weights per ticker, in config ticker order, and the portfolio's total market value.

    Inputs:  positions, one row per holding; tickers, data.tickers from config; ticker_column and value_column,
             the column names (positions.ticker_column and positions.value_column in config.yaml).
    Outputs: (weights, total): weights sum to 1, 0.0 for a configured ticker that is not held.
    Why:     the model and today's rule work in weights; the trades are sized from the total.
    Raises:  ValueError for an unknown ticker, a negative or missing value, a repeated ticker, or a total of 0.
    """
    _check_positions(positions, tickers, ticker_column, value_column)
    values = pd.to_numeric(positions[value_column]).set_axis(positions[ticker_column])
    values = values.reindex(tickers, fill_value=0.0).astype(float)
    total = float(values.sum())
    if total <= 0:
        raise ValueError("The positions add up to a total market value of 0. Send at least one holding above 0.")
    logger.info("Positions: %d holdings, total value %.2f", int((values > 0).sum()), total)
    return values / total, total


def proposed_portfolio(weights: pd.Series, target: pd.Series, action: float, total_value: float) -> pd.DataFrame:
    """The portfolio after the recommended action: ticker, weight and value.

    Inputs:  weights, current weights per ticker; target, target weights; action, share of the way back to
             target (0.0, 0.5 or 1.0); total_value, the portfolio's market value.
    Outputs: one row per ticker in the order of weights, columns ticker, weight, value (= weight x total).
    Why:     the same formula as the simulator (portfolio.apply_action), so the live answer and the backtest agree.
    Raises:  ValueError if action is outside [0, 1] (from portfolio.apply_action).
    """
    after = portfolio.apply_action(weights, target, action)
    proposed = pd.DataFrame({"ticker": after.index, "weight": after.to_numpy(),
                             "value": after.to_numpy() * total_value})
    logger.info("Proposed portfolio for action %.1f: %d tickers", action, len(proposed))
    return proposed[PORTFOLIO_COLUMNS]


def trade_list(weights: pd.Series, proposed: pd.DataFrame, total_value: float, min_trade: float) -> pd.DataFrame:
    """The orders that turn the current weights into the proposed portfolio.

    Inputs:  weights, current weights per ticker; proposed, from proposed_portfolio; total_value, the portfolio's
             market value; min_trade, the smallest trade kept, as a share of the portfolio (portfolio.min_trade).
    Outputs: one row per trade, columns ticker, side ("buy" or "sell"), value (always positive), weight_before,
             weight_after. Empty (same columns) when nothing needs to trade.
    Why:     the desk executes orders; trades too small to be worth their fixed costs are dropped.
    """
    after = proposed.set_index("ticker")["weight"]
    before = weights.reindex(after.index).astype(float)
    change = after - before
    keep = change.abs() >= max(min_trade, _NO_TRADE)
    trades = pd.DataFrame({
        "ticker": change.index[keep],
        "side": [BUY if delta > 0 else SELL for delta in change[keep]],
        "value": change[keep].abs().to_numpy() * total_value,
        "weight_before": before[keep].to_numpy(),
        "weight_after": after[keep].to_numpy(),
    }, columns=TRADE_COLUMNS)
    logger.info("Trade list: %d trades (min_trade %.4f)", len(trades), min_trade)
    return trades

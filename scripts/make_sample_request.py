"""
make_sample_request.py  [Stage 3: THE SAMPLE /predict BODY]
===========================================================
Purpose   Write scripts/sample_request.json: a valid /predict body built from SYNTHETIC prices
Input     config.yaml: data.tickers, data.columns, data.end, target_weights.policy
Output    scripts/sample_request.json: the last --days trading days of seeded random-walk prices, positions at
          the target weights, drawdown 0, weeks_since_rebalance 0
Run       python scripts/make_sample_request.py            (from the repository root)
Used by   scripts/smoke_test.py (after each deploy), the README curl example, tests/test_smoke_test.py

Why it matters
- The sample never contains the real price snapshot (Yahoo's data is for personal use only): the prices are a
  seeded random walk, so the same seed always writes the same file
- Column names, tickers and target weights come from config.yaml, so the sample always fits the API

Tests     tests/test_smoke_test.py
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger("make_sample_request")

DEFAULT_OUT = Path(__file__).with_name("sample_request.json")
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"
DAILY_DRIFT, DAILY_VOLATILITY = 0.0003, 0.01   # a plain random walk: realistic enough, clearly not real data
START_PRICE, VOLUME = 100.0, 1_000_000.0


def synthetic_prices(tickers: list[str], end: str, days: int, seed: int) -> pd.DataFrame:
    """Seeded random-walk closes and constant volumes, one column per ticker, on the last `days` business days.

    Inputs:  tickers; end, the last date (YYYY-MM-DD); days, how many business days; seed, the random seed.
    Outputs: a frame indexed by date with columns ("close" or "volume", ticker).
    Raises:  ValueError if days is below 1.
    """
    if days < 1:
        raise ValueError(f"days must be 1 or more, got {days}.")
    dates = pd.bdate_range(end=end, periods=days)
    returns = np.random.default_rng(seed).normal(DAILY_DRIFT, DAILY_VOLATILITY, size=(days, len(tickers)))
    close = pd.DataFrame(START_PRICE * np.cumprod(1 + returns, axis=0), index=dates, columns=tickers)
    return pd.concat({"close": close, "volume": pd.DataFrame(VOLUME, index=dates, columns=tickers)}, axis=1)


def build_request(cfg: dict, days: int, seed: int, total_value: float) -> dict:
    """The /predict body for synthetic prices, with positions at the target weights.

    Inputs:  cfg, the loaded config; days, trading days of prices; seed; total_value, the portfolio's value.
    Outputs: a dictionary ready to dump as JSON.
    """
    tickers, cols = cfg["data"]["tickers"], cfg["data"]["columns"]
    target = cfg["target_weights"]["policy"]
    prices = synthetic_prices(tickers, cfg["data"]["end"], days, seed)
    records = [
        {cols["date_column"]: str(day.date()), cols["ticker_column"]: ticker,
         cols["close_column"]: round(float(prices.at[day, ("close", ticker)]), 4),
         cols["volume_column"]: float(prices.at[day, ("volume", ticker)])}
        for day in prices.index for ticker in tickers
    ]
    return {
        "as_of": str(prices.index[-1].date()),
        "portfolio_id": "sample",
        "positions": [{"ticker": ticker, "market_value": round(target[ticker] * total_value, 2)} for ticker in tickers],
        "drawdown": 0.0,
        "weeks_since_rebalance": 0,
        "prices": records,
    }


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """The command-line options."""
    parser = argparse.ArgumentParser(description="Write the sample /predict request from synthetic prices.")
    parser.add_argument("--days", type=int, default=70, help="trading days of prices (default 70)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    parser.add_argument("--total", type=float, default=1_000_000, help="portfolio value (default 1,000,000)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="where to write the JSON")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="the config file")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Write the sample request.

    Inputs:  argv, the command-line arguments (None reads sys.argv).
    Outputs: None; writes --out.
    """
    args = parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    body = build_request(cfg, args.days, args.seed, args.total)
    args.out.write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8")
    logger.warning("Wrote %s: %d SYNTHETIC price rows (seed %d), not real prices", args.out, len(body["prices"]),
                   args.seed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()

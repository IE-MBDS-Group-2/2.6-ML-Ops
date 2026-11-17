"""
make_sample_request.py  [Stage 3: THE SAMPLE /predict BODY]
===========================================================
Purpose   Write scripts/sample_request.json: a valid /predict body built from SYNTHETIC prices
Input     config.yaml: data.tickers, data.columns, data.end, target_weights.policy, model.random_state,
          api.min_history_days
Output    scripts/sample_request.json: the last --days trading days of the same synthetic prices CI uses
          (scripts/make_sample_prices.py), positions at the target weights, drawdown 0, weeks_since_rebalance 0
Run       python -m scripts.make_sample_request            (from the repository root)
Used by   scripts/smoke_test.py (after each deploy), the README curl example, tests/test_smoke_test.py

Why it matters
- The sample never contains the real price snapshot (Yahoo's data is for personal use only): it is the seeded
  synthetic generator CI already uses, so there is one generator, and the same config always writes the same file
- Column names, tickers, target weights and the seed come from config.yaml, so the sample always fits the API

Tests     tests/test_smoke_test.py
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from scripts.make_sample_prices import make_sample_prices
from src import config

logger = logging.getLogger("make_sample_request")

DEFAULT_OUT = Path(__file__).with_name("sample_request.json")
DEFAULT_DAYS = 70              # the spec's sample length: api.min_history_days (61) plus a margin for holidays
CLOSE_DECIMALS = 4             # keeps the JSON short; the API needs no more precision than this


def build_request(cfg: dict, days: int, total_value: float) -> dict:
    """The /predict body for synthetic prices, with positions at the target weights.

    Inputs:  cfg, the loaded config; days, trading days of prices; total_value, the portfolio's value.
    Outputs: a dictionary ready to dump as JSON: as_of (the last price date), portfolio_id, positions, drawdown,
             weeks_since_rebalance and prices (records with the data.columns names).
    Why:     the smoke test and the README example need a body the API accepts, built only from synthetic data.
    Raises:  ValueError if days is below api.min_history_days (the API would reject the request).
    """
    needed = cfg["api"]["min_history_days"]
    if days < needed:
        raise ValueError(f"days is {days}, but the API needs at least {needed} trading days "
                         "(api.min_history_days in config.yaml).")
    data = cfg["data"]
    end = pd.Timestamp(data["end"])
    start = pd.bdate_range(end=end, periods=days)[0]
    prices = make_sample_prices(data["tickers"], str(start.date()), str(end.date()), cfg["model"]["random_state"],
                                data["columns"])
    close = data["columns"]["close_column"]
    prices[close] = prices[close].round(CLOSE_DECIMALS)
    target = cfg["target_weights"]["policy"]
    return {
        "as_of": str(prices[data["columns"]["date_column"]].iloc[-1]),
        "portfolio_id": "sample",
        "positions": [{"ticker": ticker, "market_value": round(target[ticker] * total_value, 2)}
                      for ticker in data["tickers"]],
        "drawdown": 0.0,
        "weeks_since_rebalance": 0,
        "prices": prices.to_dict(orient="records"),
    }


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """The command-line options.

    Inputs:  argv, the arguments (None reads sys.argv).
    Outputs: days, total, out and config.
    Why:     the defaults write the committed sample; the options let a test write a smaller one elsewhere.
    """
    parser = argparse.ArgumentParser(description="Write the sample /predict request from synthetic prices.")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS,
                        help=f"trading days of prices (default {DEFAULT_DAYS})")
    parser.add_argument("--total", type=float, default=1_000_000, help="portfolio value (default 1,000,000)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="where to write the JSON")
    parser.add_argument("--config", type=Path, default=config.CONFIG_PATH, help="the config file")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Write the sample request.

    Inputs:  argv, the command-line arguments (None reads sys.argv).
    Outputs: None; writes --out.
    Why:     one command regenerates the committed sample whenever the config changes.
    Raises:  ValueError from build_request; the errors of config.load_config.
    """
    args = parse_args(argv)
    cfg = config.load_config(args.config)
    body = build_request(cfg, args.days, args.total)
    args.out.write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8")
    logger.warning("Wrote %s: %d SYNTHETIC price rows, not real prices", args.out, len(body["prices"]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()

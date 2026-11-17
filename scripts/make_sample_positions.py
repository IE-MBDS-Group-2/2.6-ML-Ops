"""
make_sample_positions.py  [Demo data: a deterministic positions file for the API, the trades and the demo]
===========================================================================================================
Purpose   Build one demo portfolio (ticker, market value) that has drifted away from the policy mix, so the
          demo shows a real recommendation and a real trades list
Input     config.yaml: target_weights.policy, data.tickers, costs.aum (the portfolio size),
          model.random_state (the seed), positions.ticker_column, positions.value_column, paths.positions
Output    paths.positions (a CSV, one row per ticker), the file src/trades.py reads
Run       python -m scripts.make_sample_positions          (from the repository root)
          python -m scripts.make_sample_positions --force  (replace a file that already exists)
Safety    An existing positions file is never overwritten unless --force is passed: it may be a real one
Not for   real results. The file is demo data and is never committed (data/ is ignored by Git)

Why it matters
- Same seed, same file: the rehearsal and the live demo show the same numbers
- The biggest holding drifts by DRIFT_SHARE of the portfolio, more than today's 5-point band, so today's rule
  and the model both have something to decide

Tests     tests/test_make_sample_positions.py
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import config, utils

logger = logging.getLogger(__name__)

DRIFT_SHARE = 0.08      # share of the portfolio moved away from the biggest holding (above the 5-point band)
NOISE = 0.02            # relative noise on every weight, so the table looks like real holdings


def make_sample_positions(target: pd.Series, total_value: float, seed: int,
                          ticker_column: str, value_column: str) -> pd.DataFrame:
    """Build a demo positions table: the target mix after a deliberate drift, as market values.

    Inputs:  target, target weights per ticker (the policy mix; at least two tickers); total_value, the portfolio's
             market value (config costs.aum); seed, the random seed; ticker_column and value_column, the column
             names (config positions.ticker_column and positions.value_column).
    Outputs: a DataFrame with exactly those two columns, one row per target ticker in target order, values above
             0 in cents that add up to total_value. The biggest holding is underweight by about DRIFT_SHARE.
    Why:     the demo needs a portfolio that is off target; a perfectly balanced one would produce no trades.
    Raises:  ValueError if total_value is not above 0 (it says to check costs.aum in config.yaml).
    """
    if total_value <= 0:
        raise ValueError(f"total_value must be above 0, got {total_value}. Check costs.aum in config.yaml.")
    weights = _drifted_weights(target, np.random.default_rng(seed))
    return pd.DataFrame({ticker_column: list(target.index),
                         value_column: np.round(weights.to_numpy() * total_value, 2)})


def _drifted_weights(target: pd.Series, rng: np.random.Generator) -> pd.Series:
    """Move part of the biggest holding to one other ticker, add a little noise, and rescale to sum to 1.

    Inputs:  target, target weights per ticker; rng, the seeded random generator.
    Outputs: weights per ticker, all above 0, summing to 1.
    Why:     the move is capped at half the biggest holding, so no position can go to zero or below, whatever
             the target. The receiving ticker is the only random choice; the noise makes the rest look real.
    """
    biggest = target.idxmax()
    shift = min(DRIFT_SHARE, float(target[biggest]) / 2)
    others = [ticker for ticker in target.index if ticker != biggest]
    receiver = others[int(rng.integers(len(others)))]
    weights = target.astype(float).copy()
    weights[biggest] -= shift
    weights[receiver] += shift
    weights = weights * np.exp(NOISE * rng.standard_normal(len(weights)))
    return weights / weights.sum()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Read the command line.

    Inputs:  argv, the arguments after the script name (always passed in, never read from sys.argv here).
    Outputs: the parsed options: force (True when --force was given).
    Why:     an explicit argv keeps pytest's own arguments away from the parser and keeps main() testable.
    """
    parser = argparse.ArgumentParser(description="Write a demo positions file for the API and the demo.")
    parser.add_argument("--force", action="store_true", help="overwrite the positions file if it already exists")
    return parser.parse_args(argv)


def _refuse_to_overwrite(path: Path, force: bool) -> None:
    """Stop before writing if the positions file already exists and --force was not given.

    Inputs:  path, the file main() is about to write; force, True when the user passed --force.
    Outputs: None.
    Why:     a file at paths.positions may be a real portfolio; demo data must never replace it by accident.
    Raises:  FileExistsError naming the file and saying to pass --force.
    """
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {path}. "
                              "Pass --force to replace the existing file, or move it first.")


def main(argv: list[str] | None = None) -> None:
    """Write the demo positions file, using only the settings in config.yaml.

    Inputs:  argv, the command-line arguments (None means none, so a call from a test never reads sys.argv);
             config.yaml through config.load_config (see the module docstring for the keys).
    Outputs: None. Writes paths.positions; a missing folder is created.
    Why:     the API demo and the batch run need a positions file, and none may be committed to Git.
    Raises:  FileExistsError if the file already exists and --force is not in argv; nothing is written.
    """
    args = _parse_args([] if argv is None else argv)
    cfg = config.load_config()
    paths = config.resolve_paths(cfg)
    _refuse_to_overwrite(paths["positions"], args.force)
    target = pd.Series(cfg["target_weights"]["policy"]).reindex(cfg["data"]["tickers"])
    positions = make_sample_positions(target, cfg["costs"]["aum"], cfg["model"]["random_state"],
                                      cfg["positions"]["ticker_column"], cfg["positions"]["value_column"])
    utils.save_csv(positions, paths["positions"])
    logger.warning("DEMO positions written to %s. This is sample data, never a real portfolio.", paths["positions"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main(sys.argv[1:])

"""Tests for scripts/make_sample_positions.py: a deterministic demo positions file for the API and the demo."""

import runpy
import sys

import pandas as pd
import pytest

from scripts import make_sample_positions as sample
from src import config, trades

TARGET = pd.Series({"SPY": 0.40, "EFA": 0.20, "IEF": 0.30, "GLD": 0.05, "BIL": 0.05})
TICKER, VALUE = "ticker", "market_value"
TOTAL = 1_000_000.0


def test_unit_same_seed_gives_identical_positions():
    """The same target, total and seed give the same table; a different seed gives different values."""
    first = sample.make_sample_positions(TARGET, TOTAL, 42, TICKER, VALUE)
    second = sample.make_sample_positions(TARGET, TOTAL, 42, TICKER, VALUE)
    other = sample.make_sample_positions(TARGET, TOTAL, 43, TICKER, VALUE)

    pd.testing.assert_frame_equal(first, second)
    assert not first[VALUE].equals(other[VALUE])


def test_unit_values_add_up_to_total_value():
    """One row per target ticker, in target order, and the market values add up to the total."""
    positions = sample.make_sample_positions(TARGET, TOTAL, 42, TICKER, VALUE)

    assert list(positions[TICKER]) == list(TARGET.index)
    assert positions[VALUE].sum() == pytest.approx(TOTAL)


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 2024])
def test_unit_positions_drift_away_from_target(seed):
    """Every holding is above zero, and the portfolio has drifted at least 5 points from target.

    5 points is today's rebalancing band, so the demo always shows a real recommendation and trades.
    """
    positions = sample.make_sample_positions(TARGET, TOTAL, seed, TICKER, VALUE)
    weights = positions.set_index(TICKER)[VALUE] / TOTAL

    assert (positions[VALUE] > 0).all()
    assert (weights - TARGET).abs().max() >= 0.05


def test_unit_columns_come_from_arguments():
    """The two column names passed in (symbol, mv) are the only columns in the table."""
    positions = sample.make_sample_positions(TARGET, TOTAL, 42, "symbol", "mv")

    assert list(positions.columns) == ["symbol", "mv"]


def test_integration_output_passes_positions_to_weights():
    """The demo table is accepted by trades.positions_to_weights, which gives weights that sum to 1."""
    positions = sample.make_sample_positions(TARGET, TOTAL, 42, TICKER, VALUE)

    weights, total = trades.positions_to_weights(positions, list(TARGET.index), TICKER, VALUE)

    assert list(weights.index) == list(TARGET.index)
    assert weights.sum() == pytest.approx(1.0)
    assert total == pytest.approx(TOTAL)


def _use_config_in(tmp_path, monkeypatch) -> dict:
    """Load config.yaml, point paths.positions into tmp_path and make config.load_config return that copy."""
    cfg = config.load_config()
    cfg["paths"]["positions"] = str(tmp_path / "inference" / "positions.csv")
    monkeypatch.setattr(config, "load_config", lambda *args, **kwargs: cfg)
    return cfg


def test_integration_main_writes_positions_file(tmp_path, monkeypatch):
    """main() writes paths.positions with the configured columns, tickers and the configured portfolio size."""
    cfg = _use_config_in(tmp_path, monkeypatch)

    sample.main()

    positions = pd.read_csv(cfg["paths"]["positions"])
    assert list(positions.columns) == [cfg["positions"]["ticker_column"], cfg["positions"]["value_column"]]
    assert list(positions[cfg["positions"]["ticker_column"]]) == cfg["data"]["tickers"]
    assert positions[cfg["positions"]["value_column"]].sum() == pytest.approx(cfg["costs"]["aum"])


def test_unit_refuses_to_overwrite_existing_positions_file(tmp_path, monkeypatch):
    """With the file already there and no --force, main() raises FileExistsError and leaves the file as it was."""
    _use_config_in(tmp_path, monkeypatch)
    existing = tmp_path / "inference" / "positions.csv"
    existing.parent.mkdir()
    existing.write_text("keep me")

    with pytest.raises(FileExistsError, match=r"positions\.csv.*--force"):
        sample.main()

    assert existing.read_text() == "keep me"


def test_integration_force_overwrites_positions_file(tmp_path, monkeypatch):
    """With --force, main() replaces an existing positions file with the demo table."""
    cfg = _use_config_in(tmp_path, monkeypatch)
    existing = tmp_path / "inference" / "positions.csv"
    existing.parent.mkdir()
    existing.write_text("old")

    sample.main(["--force"])

    positions = pd.read_csv(existing)
    assert list(positions.columns) == [cfg["positions"]["ticker_column"], cfg["positions"]["value_column"]]
    assert len(positions) == len(cfg["data"]["tickers"])


def test_integration_running_as_a_script_writes_the_file(tmp_path, monkeypatch):
    """python -m scripts.make_sample_positions runs main() and writes the positions file."""
    _use_config_in(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["make_sample_positions"])  # pytest's own arguments must not reach the script
    monkeypatch.delitem(sys.modules, "scripts.make_sample_positions")  # run it fresh, as python -m would

    runpy.run_module("scripts.make_sample_positions", run_name="__main__")

    assert (tmp_path / "inference" / "positions.csv").exists()


@pytest.mark.parametrize("total_value", [0.0, -5.0])
def test_unit_raises_on_total_value_not_above_zero(total_value):
    """A portfolio size of zero or less stops with a ValueError that says what to fix."""
    with pytest.raises(ValueError, match="total_value"):
        sample.make_sample_positions(TARGET, total_value, 42, TICKER, VALUE)

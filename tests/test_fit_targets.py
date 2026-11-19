"""Check refitting through the same training prices and optimizer as the benchmark report."""

import hashlib
from pathlib import Path

import pandas as pd

from scripts import fit_targets
from scripts.make_sample_prices import make_sample_prices
from src import benchmarks, clean_data, config, load_data


def test_integration_refit_matches_benchmarks_and_updates_only_the_targets(tmp_path: Path) -> None:
    """Reproduce the capped weights, keep config comments and make repeat refits idempotent."""
    path = tmp_path / "config.yaml"
    source = config.CONFIG_PATH.read_text(encoding="utf-8")
    source = source.replace('train_end: "2019-12-31"', 'train_end: "2007-11-30"')
    path.write_text(source, encoding="utf-8")
    cfg = config.load_config(path)
    data = cfg["data"]
    cols = data["columns"]
    snapshot = tmp_path / "prices.csv"
    make_sample_prices(data["tickers"], data["start"], "2007-12-31", 12, cols).to_csv(snapshot, index=False)
    prices = load_data.load_prices(snapshot, cfg)
    close = clean_data.to_wide(prices, cols["close_column"], data["tickers"],
                               cols["date_column"], cols["ticker_column"])
    returns = close.loc["2007-09-07":"2007-11-30"].pct_change(fill_method=None).iloc[1:]
    fitted = fit_targets.fit_targets(path, snapshot)

    for method in config.CALCULATED_TARGETS:
        pd.testing.assert_series_equal(pd.Series(fitted[method]),
                                       benchmarks.fit_weights(returns, method, cfg["benchmarks"]))
    updated = path.read_text(encoding="utf-8")
    assert config.load_config(path)["target_weights"]["calculated"] == fitted
    assert "SHA-256: " + hashlib.sha256(snapshot.read_bytes()).hexdigest() in updated
    assert "training cut-off: 2007-11-30" in updated
    assert "what each price column is called" in updated
    assert fit_targets.fit_targets(path, snapshot) == fitted
    assert path.read_text(encoding="utf-8") == updated

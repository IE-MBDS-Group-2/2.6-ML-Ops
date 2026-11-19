"""Fit the four allocation targets on training prices and update the reviewed YAML configuration."""

import argparse
import hashlib
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

import yaml

from src import benchmarks, clean_data, config, load_data

logger = logging.getLogger(__name__)


def _yaml_with_targets(source: str, weights: dict, provenance: str) -> str:
    """Replace or insert the generated mapping and keep the other settings and comments intact."""
    mapping = yaml.compose(source)
    targets = next(value for key, value in mapping.value if key.value == "target_weights")
    stored = next(((key, value) for key, value in targets.value if key.value == "calculated"), None)
    if stored is None:
        last_value = targets.value[-1][1]
        start = end = last_value.end_mark.line + (last_value.end_mark.column != 0)
    else:
        key, value = stored
        start = key.start_mark.line
        end = value.end_mark.line + (value.end_mark.column != 0)
        if value.value and not value.flow_style:
            last_value = value.value[-1][1]
            end = last_value.end_mark.line + (last_value.end_mark.column != 0)
    lines = source.splitlines(keepends=True)
    block = yaml.safe_dump({"calculated": weights}, sort_keys=False).splitlines(keepends=True)
    block[1:1] = [f"  # {line}\n" for line in provenance.splitlines()]
    return "".join([*lines[:start], *("  " + line for line in block), *lines[end:]])


def fit_targets(config_path: Path, prices_path: Path) -> dict:
    """Fit train-window targets from one frozen snapshot and atomically update its config file."""
    cfg = config.load_config(config_path, validate_calculated=False)
    source = config_path.read_text(encoding="utf-8")
    checksum = hashlib.sha256(prices_path.read_bytes()).hexdigest()
    data = cfg["data"]
    cols = data["columns"]
    prices = load_data.load_prices(prices_path, cfg)
    close = clean_data.to_wide(prices, cols["close_column"], data["tickers"],
                               cols["date_column"], cols["ticker_column"])
    returns = benchmarks.training_returns(close, cfg)
    weights = {method: benchmarks.fit_weights(returns, method, cfg["benchmarks"]).to_dict()
               for method in config.CALCULATED_TARGETS}
    if hashlib.sha256(prices_path.read_bytes()).hexdigest() != checksum:
        raise ValueError("The snapshot changed during fitting. Retry with a frozen file.")
    provenance = (f"SHA-256: {checksum}\n"
                  f"Return dates: {returns.index[0]:%Y-%m-%d} to {returns.index[-1]:%Y-%m-%d}; "
                  f"training cut-off: {cfg['split']['train_end']}.")
    updated = _yaml_with_targets(source, weights, provenance)
    with NamedTemporaryFile("w", encoding="utf-8", dir=config_path.parent, delete=False) as handle:
        output = Path(handle.name)
        handle.write(updated)
    try:
        config.load_config(output)
        if config_path.read_text(encoding="utf-8") != source:
            raise ValueError("The config changed during fitting. Retry with the new config.")
        output.chmod(config_path.stat().st_mode & 0o777)
        output.replace(config_path)
    finally:
        output.unlink(missing_ok=True)
    return weights


def main() -> None:
    """Refit every allocation target from the configured snapshot or an explicit input path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=config.CONFIG_PATH)
    parser.add_argument("--prices", type=Path, help="override paths.raw_data with a local snapshot")
    options = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        cfg = config.load_config(options.config, validate_calculated=False)
        prices_path = options.prices or config.resolve_paths(cfg, options.config.parent)["raw_data"]
        fit_targets(options.config, prices_path)
    except (KeyError, OSError, ValueError) as error:
        parser.exit(1, f"Could not fit allocation targets: {error}\n")
    logger.info("Refitted the four allocation targets in %s.", options.config)


if __name__ == "__main__":
    main()

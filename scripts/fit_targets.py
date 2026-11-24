"""Fit the four allocation targets on training prices and update the reviewed YAML configuration."""

import argparse
import hashlib
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import yaml

from src import allocation, benchmarks, clean_data, config, load_data

logger = logging.getLogger(__name__)


def _last_scalar(node: yaml.Node) -> yaml.Node:
    """Locate the final scalar so following YAML comments remain outside the replacement.

    Inputs:
        node: a parsed YAML node (mapping, sequence or scalar).
    Outputs:
        The last scalar node reached by always following the last entry; the node itself if it is empty or a scalar.
    Why:
        Its end line marks where a replaced block stops, so comments after it are not overwritten.
    """
    if isinstance(node, yaml.MappingNode) and node.value:
        return _last_scalar(node.value[-1][1])
    if isinstance(node, yaml.SequenceNode) and node.value:
        return _last_scalar(node.value[-1])
    return node


def _generated_edits(targets: yaml.MappingNode, generated: dict) -> list:
    """Locate generated fields while leaving unrelated settings and their comments in place.

    Inputs:
        targets: the parsed target_weights mapping node of the config.
        generated: new top-level fields for that mapping, name -> data (here "calculated" and "fit").
    Outputs:
        A list of (first line, end line, YAML text) edits, with 0-based line numbers. Fields already
        present are replaced; missing ones are inserted together after the last line of the mapping.
    Why:
        Only the generated blocks change, so the reviewed settings and comments around them stay as written.
    """
    edits, missing = [], {}
    for name, data in generated.items():
        stored = next(((key, value) for key, value in targets.value if key.value == name), None)
        if stored is None:
            missing[name] = data
            continue
        key, value = stored
        edits.append((key.start_mark.line, _last_scalar(value).end_mark.line + 1,
                      yaml.safe_dump({name: data}, sort_keys=False)))
    if missing:
        insert = _last_scalar(targets).end_mark.line + 1
        edits.append((insert, insert, yaml.safe_dump(missing, sort_keys=False)))
    return edits


def _yaml_with_targets(source: str, generated: dict, provenance: str) -> str:
    """Replace generated mappings through their last scalar and preserve surrounding YAML.

    Inputs:
        source: the full text of the config file.
        generated: new fields for target_weights, name -> data.
        provenance: text written as comment lines under the "calculated:" line (checksum and date range).
    Outputs:
        The config text with the generated blocks replaced or added, indented under target_weights.
    Why:
        Rewriting by line keeps the comments and layout of the reviewed file, which a plain YAML dump would lose.
    """
    targets = next(value for key, value in yaml.compose(source).value if key.value == "target_weights")
    lines = source.splitlines(keepends=True)
    for start, end, text in sorted(_generated_edits(targets, generated), reverse=True):
        block = text.splitlines(keepends=True)
        if block[0].startswith("calculated:"):
            block[1:1] = [f"  # {line}\n" for line in provenance.splitlines()]
        lines[start:end] = ["  " + line for line in block]
    return "".join(lines)


def fit_targets(config_path: Path, prices_path: Path) -> dict:
    """Fit train-window targets from one frozen snapshot and atomically update its config file.

    Inputs:
        config_path: the YAML config to read and update (its target_weights section is rewritten).
        prices_path: the frozen price snapshot (CSV) the targets are fitted on.
    Outputs:
        The fitted weights, method -> {ticker: weight}. The config file is replaced with the new
        "calculated" and "fit" blocks, which carry the snapshot SHA-256.
    Why:
        The targets come from the training window only and record which snapshot made them. The write goes
        through a validated temporary file, and stops if the snapshot or the config changed meanwhile.

    Raises:
        ValueError: if the snapshot or the config changes during fitting, or the new config fails validation.
    """
    cfg = config.load_config(config_path, validate_calculated=False)
    source = config_path.read_text(encoding="utf-8")
    checksum = hashlib.sha256(prices_path.read_bytes()).hexdigest()
    cols = cfg["data"]["columns"]
    prices = load_data.load_prices(prices_path, cfg)
    close = clean_data.to_wide(prices, cols["close_column"], cfg["data"]["tickers"],
                               cols["date_column"], cols["ticker_column"])
    returns = benchmarks.training_returns(close, cfg)
    weights = {method: benchmarks.fit_weights(returns, method, cfg["benchmarks"]).to_dict()
               for method in config.CALCULATED_TARGETS}
    if hashlib.sha256(prices_path.read_bytes()).hexdigest() != checksum:
        raise ValueError("The snapshot changed during fitting. Retry with a frozen file.")
    provenance = (f"SHA-256: {checksum}\n"
                  f"Return dates: {returns.index[0]:%Y-%m-%d} to {returns.index[-1]:%Y-%m-%d}; "
                  f"training cut-off: {cfg['split']['train_end']}.")
    fit = {
        "snapshot_sha256": checksum,
        "train_start": cfg["portfolio"]["first_decision_date"],
        "train_end": cfg["split"]["train_end"],
        "return_start": str(returns.index[0].date()),
        "return_end": str(returns.index[-1].date()),
        "settings": config.fit_settings(cfg),
        "policy_weights": cfg["target_weights"]["policy"],
        "tracking_error": {
            method: allocation.target_fit(pd.Series(target), cfg, returns)["expected_tracking_error"]
            for method, target in weights.items()
        },
    }
    updated = _yaml_with_targets(source, {"calculated": weights, "fit": fit}, provenance)
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
    """Refit every allocation target from the configured snapshot or an explicit input path.

    Inputs:
        None. Reads --config (default config.CONFIG_PATH) and --prices (default: paths.raw_data in the config)
        from the command line.
    Outputs:
        None. Updates the config file; on a KeyError, OSError or ValueError it exits with status 1 and a message.
    Why:
        It is the command-line entry point, so a refit is one command and a failure is a clear message.
    """
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

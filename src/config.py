"""
config.py  [Settings loader]
============================
Purpose   Read config.yaml once, check it, and read the secrets and switches from .env
Input     Path to config.yaml; the .env file and the process environment
Output    Plain dictionaries of settings
Used by   main.py (training) and api.py (serving); never the other way round

Why it matters
- Fail fast: a typo in config.yaml stops the run at second one, not after training
- One table of allowed values (ALLOWED_VALUES): config, the API and the modules all check against it
- Secrets never live in config.yaml; load_env returns them without ever logging a value

Tests     tests/test_config.py
"""

import hashlib
import logging
import math
import os
from pathlib import Path

import yaml
from dotenv import dotenv_values

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

REQUIRED_SECTIONS = (
    "paths", "data", "portfolio", "target_weights", "rebalancing", "problem", "split", "features",
    "model", "quality_gate", "fallback", "costs", "logging",
)

# Settings that only accept listed values: (section, key) -> allowed values. The API and the
# modules read this same table, so the lists cannot drift apart.
ALLOWED_VALUES = {
    ("target_weights", "method"): ("policy", "min_variance", "max_sharpe", "mean_variance", "risk_parity"),
    ("rebalancing", "method"): ("smart", "calendar", "tolerance_band", "calendar_and_band"),
    ("rebalancing", "review"): ("daily", "weekly", "monthly"),
    ("rebalancing", "calendar"): ("monthly", "quarterly", "annually"),
    ("model", "type"): ("classifier", "fitted_q"),
    ("problem", "problem_type"): ("classification", "regression"),
    ("costs", "model"): ("flat", "realistic"),
}

CALCULATED_TARGETS = ALLOWED_VALUES[("target_weights", "method")][1:]
WEIGHT_SUM_TOLERANCE = 1e-9

# Names read from .env or the process environment, with the value used when a name is absent.
ENV_DEFAULTS = {
    "WANDB_API_KEY": "",
    "WANDB_ENTITY": "",
    "WANDB_MODE": "disabled",
    "MODEL_SOURCE": "local",
    "WANDB_MODEL_ALIAS": "prod",
    "APP_VERSION": "dev",
    "APP_COMMIT": "local",
    "MODEL_VERSION": "",
}
ALLOWED_ENV_VALUES = {
    "WANDB_MODE": ("online", "offline", "disabled"),
    "MODEL_SOURCE": ("local", "wandb"),
}


def load_config(path: Path = CONFIG_PATH, *, validate_calculated: bool = True) -> dict:
    """Read config.yaml once and fail fast if anything in it cannot work.

    Inputs:  path, the YAML file (normally CONFIG_PATH); validate_calculated, disabled only by the target
             refit script so outdated generated weights can be replaced.
    Outputs: the settings as a dictionary.
    Why:     one source of truth for every setting; a wrong value stops the run before any data is read.
    Raises:  FileNotFoundError if the file is missing; ValueError if it is not a mapping or a setting has a
             value that is not allowed; KeyError naming every missing section or key.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}. Run from the repository root.")
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} must contain a YAML mapping of sections.")
    missing = [section for section in REQUIRED_SECTIONS if section not in cfg]
    if missing:
        raise KeyError(f"config.yaml is missing sections: {missing}")
    _check_allowed_values(cfg)
    params = cfg.get("benchmarks", {})
    asset_cap(params.get("max_weight", 1.0), len(cfg["data"]["tickers"]) -
              (params.get("cash_ticker") is not None))
    _check_target_weights(cfg, validate_calculated)
    _check_rebalancing(cfg)
    _check_rl_seeds(cfg)
    logger.info("Config loaded from %s: %d sections", path, len(cfg))
    return cfg


def _check_allowed_values(cfg: dict) -> None:
    """Raise if a listed-value setting is absent (KeyError) or holds a value outside its list (ValueError).

    Inputs:  cfg, the loaded config.
    Outputs: None.
    Why:     every setting in ALLOWED_VALUES is checked against one table, so a typo stops the run at start-up.
    """
    absent = [f"{section}.{key}" for section, key in ALLOWED_VALUES if key not in (cfg[section] or {})]
    if absent:
        raise KeyError(f"config.yaml is missing keys: {absent}")
    for (section, key), allowed in ALLOWED_VALUES.items():
        value = cfg[section][key]
        if value not in allowed:
            raise ValueError(f"{section}.{key} is {value!r}. Allowed: {', '.join(allowed)}.")


def _check_target_weights(cfg: dict, validate_calculated: bool = True) -> None:
    """Validate the policy and every stored allocation against the configured tickers.

    Inputs:  cfg, the loaded config (reads target_weights, data.tickers, benchmarks); validate_calculated, False
             skips the stored calculated allocations (the refit script replaces them).
    Outputs: None.
    Why:     a wrong weight vector would silently mis-size every trade, so it stops the run before any data is read.
    """
    settings = cfg["target_weights"]
    policy = settings.get("policy")
    if not isinstance(policy, dict) or not policy:
        raise KeyError("config.yaml is missing keys: ['target_weights.policy'] (one weight per ticker).")
    tickers = cfg["data"].get("tickers", [])
    _check_weight_vector("target_weights.policy", policy, tickers)
    if not validate_calculated or ("calculated" not in settings and settings["method"] == "policy"):
        return
    calculated = settings.get("calculated")
    if not isinstance(calculated, dict) or calculated.keys() != set(CALCULATED_TARGETS):
        raise ValueError(f"target_weights.calculated must contain {', '.join(CALCULATED_TARGETS)}. "
                         "Run python -m scripts.fit_targets with the selected snapshot.")
    check_target_provenance(cfg)
    for method, weights in calculated.items():
        _check_weight_vector(f"target_weights.calculated.{method}", weights, tickers)
        _check_calculated_limits(method, weights, cfg)


def fit_settings(cfg: dict) -> dict:
    """Return the optimizer settings that affect fitted weights and require refitting when changed.

    Inputs:  cfg, the loaded config (reads benchmarks).
    Outputs: {"risk_aversion", "cash_ticker", "max_weight"} as configured (None when absent).
    Why:     stored allocations record these settings, so a change shows up as stale weights.
    """
    settings = cfg["benchmarks"]
    return {name: settings.get(name) for name in ("risk_aversion", "cash_ticker", "max_weight")}


def check_target_provenance(cfg: dict) -> None:
    """Require stored allocations and tracking risks to match the current fit window and settings.

    Inputs:  cfg, the loaded config (reads target_weights.fit, portfolio.first_decision_date, split.train_end).
    Outputs: None.
    Why:     weights fitted on another window, policy or optimizer setting must not be used as if they were current.
    Raises:  ValueError when the fit record is missing or stale, or the snapshot checksum is not a SHA-256.
    """
    fit = cfg["target_weights"].get("fit", {})
    if not isinstance(fit, dict):
        raise ValueError("target_weights.fit must contain the reviewed fit provenance.")
    expected = {
        "train_start": cfg["portfolio"]["first_decision_date"],
        "train_end": cfg["split"]["train_end"],
        "settings": fit_settings(cfg),
        "policy_weights": cfg["target_weights"]["policy"],
    }
    if any(fit.get(name) != value for name, value in expected.items()):
        raise ValueError("target_weights.fit is missing or stale. Re-run python -m scripts.fit_targets.")
    checksum = fit.get("snapshot_sha256", "")
    if not isinstance(checksum, str) or len(checksum) != 64 or any(char not in "0123456789abcdef" for char in checksum):
        raise ValueError("target_weights.fit.snapshot_sha256 must be a SHA-256 checksum.")
    _check_training_risk(fit.get("tracking_error"))


def _check_training_risk(errors: object) -> None:
    """Require a finite training-period tracking risk for every calculated allocation.

    Inputs:  errors, target_weights.fit.tracking_error: method -> annual tracking error.
    Outputs: None.
    Why:     serving reads these estimates instead of the training returns, so none may be missing or invalid.
    Raises:  ValueError when a method is missing or an estimate is not a finite nonnegative number.
    """
    if not isinstance(errors, dict) or errors.keys() != set(CALCULATED_TARGETS):
        raise ValueError("target_weights.fit.tracking_error must cover all four calculated methods.")
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in errors.values()):
        raise ValueError("target_weights.fit.tracking_error must contain finite nonnegative estimates.")


def check_target_snapshot(cfg: dict, path: Path) -> None:
    """Stop calculated-target training when the fit belongs to a different snapshot or split.

    Inputs:  cfg, the loaded config; path, the data snapshot file the run reads.
    Outputs: None. The policy target is not checked.
    Why:     stored weights are only valid for the data snapshot they were fitted on.
    Raises:  ValueError when the provenance is stale or the file's SHA-256 differs from the stored checksum.
    """
    if cfg["target_weights"]["method"] == "policy":
        return
    check_target_provenance(cfg)
    if hashlib.sha256(path.read_bytes()).hexdigest() != cfg["target_weights"]["fit"]["snapshot_sha256"]:
        raise ValueError("The stored targets belong to another snapshot. Re-run python -m scripts.fit_targets.")


def asset_cap(cap: object, size: int) -> float:
    """Validate that one cap permits a finite, fully invested risky allocation.

    Inputs:  cap, benchmarks.max_weight; size, the number of risky assets.
    Outputs: the cap as a float.
    Why:     a cap below 1 / size makes a fully invested allocation impossible.
    Raises:  ValueError when there is no risky asset, or the cap is not a number in [1 / size, 1].
    """
    if size < 1:
        raise ValueError("Calculated targets need at least one risky asset.")
    if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not 0 < cap <= 1 or cap < 1.0 / size:
        raise ValueError(f"benchmarks.max_weight {cap!r} must be between {1.0 / size:.6g} and 1 "
                         f"to invest fully across {size} risky assets.")
    return float(cap)


def _check_calculated_limits(method: str, weights: dict, cfg: dict) -> None:
    """Reject stored risky allocations that exceed the cap or allocate to the cash ticker.

    Inputs:  method, the calculated method's name; weights, its stored ticker -> weight; cfg, the loaded config
             (reads benchmarks and data.tickers).
    Outputs: None.
    Why:     stored weights must obey the same limits the optimizer was given.
    Raises:  ValueError when a weight exceeds benchmarks.max_weight or the cash ticker has a nonzero weight.
    """
    params = cfg["benchmarks"]
    cap = params.get("max_weight", 1.0)
    asset_cap(cap, len(cfg["data"]["tickers"]) - (params.get("cash_ticker") is not None))
    if max(weights.values()) > cap + WEIGHT_SUM_TOLERANCE:
        raise ValueError(f"target_weights.calculated.{method} exceeds benchmarks.max_weight.")
    cash = params.get("cash_ticker")
    if cash is not None and weights.get(cash) != 0.0:
        raise ValueError(f"target_weights.calculated.{method} must give the cash ticker {cash!r} zero weight.")


def _check_weight_vector(name: str, weights: object, tickers: list[str]) -> None:
    """Require a finite, long-only weight for each ticker and a total of one.

    Inputs:  name, the setting's path for the message; weights, ticker -> weight; tickers, data.tickers.
    Outputs: None.
    Why:     a vector that misses a ticker, goes short or does not sum to 1 is not a portfolio.
    Raises:  ValueError naming the setting and what is wrong.
    """
    if not isinstance(weights, dict) or weights.keys() != set(tickers):
        keys = list(weights) if isinstance(weights, dict) else []
        raise ValueError(f"{name} covers {keys} but data.tickers is {tickers}. "
                         "List one weight for every ticker, and no others.")
    if any(type(weight) not in (int, float) or not 0 <= weight <= 1 for weight in weights.values()):
        raise ValueError(f"{name} must contain finite weights between 0 and 1.")
    total = math.fsum(weights.values())
    if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
        raise ValueError(f"{name} sums to {total:.6f}. The weights must sum to 1.")


def _check_rebalancing(cfg: dict) -> None:
    """Raise if the tolerance band is not a number above 0.

    The calendar can never be more frequent than the review: the least frequent review (monthly) is the most
    frequent calendar, so the two allowed lists already guarantee it.

    Inputs:  cfg, the loaded config (reads rebalancing.tolerance_band, in percentage points).
    Outputs: None.
    Why:     a band of 0 or less would trade on every drift, and a non-number cannot be compared with one.
    Raises:  ValueError saying what to give.
    """
    band = cfg["rebalancing"].get("tolerance_band")
    if not isinstance(band, (int, float)) or isinstance(band, bool) or band <= 0:
        raise ValueError(f"rebalancing.tolerance_band is {band!r}. Give a number above 0, in percentage points "
                         "away from target (for example 5).")


def _check_rl_seeds(cfg: dict) -> None:
    """Raise if rl.seeds is present but is not a non-empty list of integers (the agent takes the median seed).

    Inputs:  cfg, the loaded config (reads rl.seeds when the rl section exists).
    Outputs: None.
    Why:     the agent trains once per seed and keeps the median one, so it needs at least one integer seed.
    Raises:  ValueError saying what to give.
    """
    if "rl" not in cfg:
        return
    seeds = (cfg["rl"] or {}).get("seeds")
    if (not isinstance(seeds, list) or not seeds
            or not all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds)):
        raise ValueError(f"rl.seeds is {seeds!r}. Give a non-empty list of integers (for example [0, 1, 2, 3, 4]).")


def resolve_paths(cfg: dict, root: Path = PROJECT_ROOT) -> dict[str, Path]:
    """Turn every entry of the paths section into a full path under the project root.

    Inputs:  cfg, the loaded config; root, the folder the relative paths start from.
    Outputs: name -> Path, same names as in config.yaml.
    Why:     main.py and api.py resolve paths the same way, wherever they are started from.
    """
    return {name: Path(root) / value for name, value in cfg["paths"].items()}


def load_env(env_file: Path | None = None) -> dict[str, str]:
    """Read the secrets and switches named in .env.example, from .env and the process environment.

    Inputs:  env_file, the .env file (default: PROJECT_ROOT / ".env"); it may be absent.
    Outputs: one entry per name in ENV_DEFAULTS. The process environment wins over the file, the file over
             the default, so the cloud host and CI can set values without a file.
    Why:     secrets stay out of config.yaml and out of Git; tests and CI run with the safe defaults.
    Raises:  ValueError if WANDB_MODE or MODEL_SOURCE holds a value outside its list.
    """
    env_file = PROJECT_ROOT / ".env" if env_file is None else Path(env_file)
    from_file = dotenv_values(env_file) if env_file.exists() else {}
    env = {name: os.environ.get(name) or from_file.get(name) or default for name, default in ENV_DEFAULTS.items()}
    for name, allowed in ALLOWED_ENV_VALUES.items():
        if env[name] not in allowed:
            raise ValueError(f"{name} is {env[name]!r}. Allowed: {', '.join(allowed)}. Check .env.")
    logger.info("Environment read: tracking mode %s, model source %s", env["WANDB_MODE"], env["MODEL_SOURCE"])
    return env

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

import logging
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

WEIGHT_SUM_TOLERANCE = 1e-9

# Names read from .env or the process environment, with the value used when a name is absent.
ENV_DEFAULTS = {
    "WANDB_API_KEY": "",
    "WANDB_ENTITY": "",
    "WANDB_MODE": "disabled",
    "MODEL_SOURCE": "local",
    "WANDB_MODEL_ALIAS": "prod",
    "APP_VERSION": "dev",
}
ALLOWED_ENV_VALUES = {
    "WANDB_MODE": ("online", "offline", "disabled"),
    "MODEL_SOURCE": ("local", "wandb"),
}


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Read config.yaml once and fail fast if anything in it cannot work.

    Inputs:  path, the YAML file (normally CONFIG_PATH).
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
    _check_target_weights(cfg)
    _check_rebalancing(cfg)
    logger.info("Config loaded from %s: %d sections", path, len(cfg))
    return cfg


def _check_allowed_values(cfg: dict) -> None:
    """Raise if a listed-value setting is absent (KeyError) or holds a value outside its list (ValueError)."""
    absent = [f"{section}.{key}" for section, key in ALLOWED_VALUES if key not in (cfg[section] or {})]
    if absent:
        raise KeyError(f"config.yaml is missing keys: {absent}")
    for (section, key), allowed in ALLOWED_VALUES.items():
        value = cfg[section][key]
        if value not in allowed:
            raise ValueError(f"{section}.{key} is {value!r}. Allowed: {', '.join(allowed)}.")


def _check_target_weights(cfg: dict) -> None:
    """Raise if the policy weights do not cover exactly the configured tickers or do not sum to 1."""
    policy = cfg["target_weights"].get("policy")
    if not isinstance(policy, dict) or not policy:
        raise KeyError("config.yaml is missing keys: ['target_weights.policy'] (one weight per ticker).")
    tickers = cfg["data"].get("tickers", [])
    if sorted(policy) != sorted(tickers):
        raise ValueError(f"target_weights.policy covers {sorted(policy)} but data.tickers is {sorted(tickers)}. "
                         "List one weight for every ticker, and no others.")
    total = float(sum(policy.values()))
    if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
        raise ValueError(f"target_weights.policy sums to {total:.6f}. The weights must sum to 1.")


def _check_rebalancing(cfg: dict) -> None:
    """Raise if the tolerance band is not a number above 0.

    The calendar can never be more frequent than the review: the least frequent review (monthly) is the most
    frequent calendar, so the two allowed lists already guarantee it.
    """
    band = cfg["rebalancing"].get("tolerance_band")
    if not isinstance(band, (int, float)) or isinstance(band, bool) or band <= 0:
        raise ValueError(f"rebalancing.tolerance_band is {band!r}. Give a number above 0, in percentage points "
                         "away from target (for example 5).")


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

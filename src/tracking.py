"""
tracking.py  [Experiment tracking with Weights & Biases]
========================================================
Purpose   Record each run's settings, data version, metrics and model in W&B, and fetch the promoted model
Input     The config (tracking section), the values from .env, metrics, tables and file paths
Output    A W&B run, or None when tracking is off; download_model returns the model file's path
Used by   main.py (training run) and api.py (download_model, monitoring); main.py never imports wandb itself

Why it matters
- Every training run can be compared and audited afterwards: same settings, same data, same metrics
- Tracking is optional (D13): with WANDB_MODE=disabled or tracking.enabled false, every function does nothing,
  so tests and CI run offline and never need a key
- Only a model that passed the quality gate is registered (D8); a person promotes it to "prod" in the W&B UI

Tests     tests/test_tracking.py
"""

import logging

import wandb

logger = logging.getLogger(__name__)


def _login(env: dict[str, str]) -> None:
    """Hand the API key from .env to W&B, when there is one.

    Inputs:  env, the values from config.load_env.
    Outputs: None.
    Why:     load_env reads .env into a dictionary, not into the process environment where W&B looks for the
             key; without this, an online run would ask for a key even though .env holds one.
    """
    if env["WANDB_API_KEY"]:
        wandb.login(key=env["WANDB_API_KEY"])


def start_run(cfg: dict, env: dict[str, str], job_type: str = "train") -> wandb.Run | None:
    """Start a W&B run with the project, team and names from the config, or return None when tracking is off.

    Inputs:  cfg, the loaded config (uses the tracking section and logs the whole config with the run);
             env, the values from config.load_env (WANDB_MODE, WANDB_ENTITY, WANDB_API_KEY);
             job_type, what the run does, for example "train" or "serve".
    Outputs: the W&B run, or None when WANDB_MODE is "disabled" or tracking.enabled is false.
    Why:     one place decides whether tracking is on; every other function accepts None and does nothing.
    """
    settings = cfg["tracking"]
    mode = env["WANDB_MODE"]
    if mode == "disabled" or not settings["enabled"]:
        logger.info("Tracking off (WANDB_MODE %s, tracking.enabled %s): no W&B run", mode, settings["enabled"])
        return None
    _login(env)
    run = wandb.init(
        project=settings["project"],
        entity=env["WANDB_ENTITY"] or None,
        name=settings["name"],
        group=settings["group"],
        tags=settings["tags"],
        job_type=job_type,
        config=cfg,
        mode=mode,
    )
    logger.info("W&B run started in project %s (mode %s, job type %s)", settings["project"], mode, job_type)
    return run

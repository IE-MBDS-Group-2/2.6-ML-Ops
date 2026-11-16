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
from pathlib import Path

import pandas as pd
import wandb

logger = logging.getLogger(__name__)

# W&B's own name for "the newest version". "prod" is never set in code: a person promotes a model in the W&B UI.
LATEST_ALIAS = "latest"


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


def log_metrics(run: wandb.Run | None, metrics: dict[str, float], prefix: str = "") -> None:
    """Log a set of metrics to the run, each name with the same prefix (for example "val/" or "test/").

    Inputs:  run, the W&B run or None; metrics, name -> value; prefix, put in front of every name.
    Outputs: None. With run=None nothing happens.
    Why:     the same metric names are logged for validation, test and the grid search; the prefix keeps them
             apart in W&B.
    """
    if run is None:
        return
    run.log({f"{prefix}{name}": value for name, value in metrics.items()})
    logger.info("Logged %d metrics to W&B with prefix %r", len(metrics), prefix)


def log_table(run: wandb.Run | None, name: str, table: pd.DataFrame) -> None:
    """Log a DataFrame to the run as a W&B table (for example the predictions).

    Inputs:  run, the W&B run or None; name, the table's name in W&B; table, the rows to log.
    Outputs: None. With run=None nothing happens.
    Why:     a table can be filtered and compared between runs in the W&B UI.
    """
    if run is None:
        return
    run.log({name: wandb.Table(dataframe=table)})
    logger.info("Logged table %s to W&B: %d rows", name, len(table))


def finish_run(run: wandb.Run | None, failed: bool = False) -> None:
    """Close the run and mark it failed when the pipeline stopped with an error.

    Inputs:  run, the W&B run or None; failed, True when the pipeline raised.
    Outputs: None. With run=None nothing happens.
    Why:     a failed run must not look like a finished one in W&B (exit code 1 shows it as failed).
    """
    if run is None:
        return
    exit_code = 1 if failed else 0
    run.finish(exit_code=exit_code)
    logger.info("W&B run finished with exit code %d", exit_code)


def _file_artifact(path: Path, name: str, artifact_type: str, **options: object) -> wandb.Artifact:
    """Build a W&B artifact that holds one file.

    Inputs:  path, the file to store; name, the artifact's name in W&B; artifact_type, "dataset" or "model";
             options, extra settings for wandb.Artifact (for example metadata).
    Outputs: the artifact, ready to be logged to a run.
    Why:     the dataset and the model are stored the same way; one helper keeps the two functions short.
    """
    artifact = wandb.Artifact(name, type=artifact_type, **options)
    artifact.add_file(str(path))
    return artifact


def log_dataset(run: wandb.Run | None, path: Path, name: str) -> None:
    """Store the data file the run trained on as a W&B dataset artifact, so every run names its data version.

    Inputs:  run, the W&B run or None; path, the data file (the cleaned prices); name, the artifact's name
             (tracking.dataset_artifact in config.yaml).
    Outputs: None. With run=None nothing happens.
    Why:     W&B keeps one version per distinct file, so two runs on the same data point to the same version (D20).
    """
    if run is None:
        return
    run.log_artifact(_file_artifact(path, name, "dataset"))
    logger.info("Logged dataset %s to W&B from %s", name, path)


def log_model(run: wandb.Run | None, path: Path, name: str, metadata: dict) -> None:
    """Register a model file in W&B as a model artifact with the alias "latest".

    Inputs:  run, the W&B run or None; path, the saved model file (.joblib); name, the artifact's name
             (tracking.model_artifact in config.yaml); metadata, facts stored with it (for example gate results).
    Outputs: None. With run=None nothing happens.
    Why:     only called after the quality gate passed (D8); a person then promotes it to "prod" in the W&B UI,
             and the API downloads that alias.
    """
    if run is None:
        return
    run.log_artifact(_file_artifact(path, name, "model", metadata=metadata), aliases=[LATEST_ALIAS])
    logger.info("Logged model %s to W&B from %s with alias %s", name, path, LATEST_ALIAS)

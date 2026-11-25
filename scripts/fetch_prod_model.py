"""
fetch_prod_model.py  [Stage 3: WHICH MODEL DOES THIS RELEASE SHIP?]
===================================================================
Purpose   Resolve the W&B alias "prod" on the model artifact to a concrete version (e.g. rebalance-model:v7) and
          download its .joblib, so the release image carries that exact model
Input     config.yaml (tracking.project, tracking.model_artifact, paths.model); WANDB_API_KEY and WANDB_ENTITY
          from the environment; optional --alias (default prod) and --output (default paths.model)
Output    models/model.joblib, MODEL_VERSION=<name>:<version> printed and appended to $GITHUB_OUTPUT
          No version has the alias: a notice, no file, MODEL_VERSION empty and exit 0, so the image serves today's
          rule (D8). Any other failure (no key or entity, wrong project, network, no .joblib): exit 1
Run       python -m scripts.fetch_prod_model
Used by   deploy.yml (publish job), before docker build --build-arg MODEL_VERSION=...

Why it matters
- "Promote with a label, ship with a release": a person moves prod in W&B, the release bakes THAT version in.
  The alias moves; the version does not, so /health can name exactly what serves
- Read-only: it never logs, tags or moves anything in W&B
- A missing prod is a decision (no approved model yet); a wrong key or entity is a mistake, so only the first
  lets the release go ahead without a model

Tests     tests/test_fetch_prod_model.py (W&B mocked)
"""

import argparse
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import wandb
import yaml

logger = logging.getLogger("fetch_prod_model")

ROOT = Path(__file__).resolve().parents[1]
NO_ALIAS = "artifact membership"   # wandb's ValueError when no version of the artifact has the alias


def _settings(config_path: Path) -> tuple[str, str, Path]:
    """The W&B project, the model artifact's name and the default output path, from config.yaml.

    Inputs:  config_path, the config file.
    Outputs: (tracking.project, tracking.model_artifact, paths.model resolved against the repository root).
    Why:     the names live in config.yaml only, the same ones main.py registers the model under.
    """
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return cfg["tracking"]["project"], cfg["tracking"]["model_artifact"], ROOT / cfg["paths"]["model"]


def fetch(api: Any, address: str, output: Path) -> str | None:
    """Download the model version an alias points to and copy its .joblib to output.

    Inputs:  api, a wandb.Api (or a test double); address, entity/project/name:alias; output, the file to write.
    Outputs: the concrete version, e.g. rebalance-model:v7; None when no version carries the alias.
    Why:     the release image must name and carry one fixed version, not the moving alias.
    Raises:  ValueError when W&B cannot find the entity or project; FileNotFoundError when the version holds no
             .joblib; wandb.errors.Error for network or key problems.
    """
    try:
        artifact = api.artifact(address, type="model")
    except ValueError as error:
        if NO_ALIAS in str(error):
            return None
        raise
    with tempfile.TemporaryDirectory() as folder:
        files = sorted(Path(artifact.download(root=folder)).glob("*.joblib"))
        if not files:
            raise FileNotFoundError(f"{address} holds no .joblib file. Check how it was registered.")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(files[0], output)
    name = address.rsplit("/", 1)[-1].split(":")[0]
    return f"{name}:{artifact.version}"


def write_github_output(version: str, github_output: str | None) -> None:
    """Append MODEL_VERSION=<version> to the $GITHUB_OUTPUT file, when there is one.

    Inputs:  version, the concrete version or ""; github_output, the file path (None outside GitHub Actions).
    Outputs: none; the file gains one line.
    Why:     the next workflow step passes it to docker build, and /health reports it.
    """
    if github_output:
        with open(github_output, "a", encoding="utf-8") as handle:
            handle.write(f"MODEL_VERSION={version}\n")


def main(argv: list[str] | None = None, api: Any = None) -> int:
    """Resolve the alias, download the model, report the version; return the exit code.

    Inputs:  argv, command line arguments (None reads sys.argv); api, an injected W&B client (tests).
    Outputs: 0 when a model was fetched or no version has the alias; 1 on any other failure.
    Why:     one entry point the release workflow can call and act on by exit code alone.
    """
    parser = argparse.ArgumentParser(description="Download the model version that carries a W&B alias")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--alias", default="prod")
    parser.add_argument("--output", type=Path, default=None, help="default: paths.model in config.yaml")
    args = parser.parse_args(argv)

    project, name, default_output = _settings(args.config)
    output = args.output or default_output
    entity = os.environ.get("WANDB_ENTITY", "")
    if not entity or (api is None and not os.environ.get("WANDB_API_KEY")):
        logger.error("WANDB_ENTITY and WANDB_API_KEY must be set (repository variable and secret).")
        return 1
    address = f"{entity}/{project}/{name}:{args.alias}"
    try:
        version = fetch(api if api is not None else wandb.Api(), address, output)
    except (ValueError, FileNotFoundError, wandb.errors.Error) as error:
        logger.error("Could not fetch %s: %s", address, error)
        return 1
    if version is None:
        logger.warning("No version of %s has the alias %r. The image ships without a model and serves today's "
                       "rule. Move %r in W&B to ship a model.", name, args.alias, args.alias)
    else:
        logger.info("Resolved %s -> %s, saved to %s", address, version, output)
    print(f"MODEL_VERSION={version or ''}")
    write_github_output(version or "", os.environ.get("GITHUB_OUTPUT"))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())

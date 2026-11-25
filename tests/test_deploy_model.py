"""The release image carries the prod model: deploy.yml fetches it and builds with MODEL_VERSION; Docker copies it."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _publish_steps() -> list[dict]:
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "deploy.yml").read_text(encoding="utf-8"))
    return workflow["jobs"]["publish"]["steps"]


def test_unit_publish_fetches_prod_with_the_wandb_secret_and_variable_before_building():
    steps = _publish_steps()
    names = [step.get("name", "") for step in steps]
    fetch = next(step for step in steps if step.get("id") == "model")
    build = next(step for step in steps if "docker build" in step.get("run", ""))
    assert fetch["run"] == "python -m scripts.fetch_prod_model"
    assert fetch["env"] == {"WANDB_API_KEY": "${{ secrets.WANDB_API_KEY }}", "WANDB_ENTITY": "${{ vars.WANDB_ENTITY }}"}
    assert names.index(fetch["name"]) < names.index(build["name"])
    assert build["env"]["MODEL_VERSION"] == "${{ steps.model.outputs.MODEL_VERSION }}"
    assert '--build-arg MODEL_VERSION="$MODEL_VERSION"' in build["run"]


def test_unit_the_image_copies_only_the_model_file_and_serves_it_locally():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    ignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert "COPY --chown=$MAMBA_USER:$MAMBA_USER model[s] /app/models/" in dockerfile
    assert "MODEL_SOURCE=local" in dockerfile and "MODEL_VERSION=$MODEL_VERSION" in dockerfile
    assert "!models/model.joblib" in ignore and "!models/" not in ignore

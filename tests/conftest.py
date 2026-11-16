"""Shared test set-up: every test runs offline, with tracking off and the local model source."""

import os

# Set before any src module is imported, so no test can reach W&B or the cloud by accident.
os.environ["WANDB_MODE"] = "disabled"
os.environ["MODEL_SOURCE"] = "local"

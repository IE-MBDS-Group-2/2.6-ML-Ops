# Retraining on the self-hosted runner

The `retrain` workflow (`.github/workflows/retrain.yml`) runs the full pipeline, `python -m src.main`, on the official price snapshot. It runs monthly (03:17 UTC on the 1st) and on demand.

## Run it by hand
1. GitHub > Actions > **retrain** > **Run workflow**, on the `main` branch.
2. Approve the run if the `retrain` environment asks for it.
3. Read the job summary: the gate result (`gate.met`, `gate.threshold`), the release decision and the W&B run link.

GitHub only lets you run a workflow by hand once its file is on `main`. The first manual run therefore comes after the release that brings `retrain.yml` to `main`.

## What a run does
1. Checks out the code (no stored credentials) and builds the environment from `conda-lock.yml`. The runner wipes its home after every job, so this is built fresh each time.
2. Copies the price snapshot from the runner (`ETF_PRICES_CSV`) to `data/raw/prices.csv` and checks its SHA-256 against the value in the README model card. A different file stops the run with a clear message.
3. Runs `python -m src.main` with W&B online (project `etf-rebalancing`, entity `ie-mbds-group-2`).
4. Writes the gate result and the W&B run link to the job summary, also when the run failed.

## When the gate passes
The run, the dataset artifact, the metrics and the model are in W&B. The model is registered with the alias `latest`. The workflow never moves `prod`.

## When the gate fails
The run, the dataset artifact, the metrics and the gate result are still in W&B, but no model is saved or registered (decision D8). The service keeps serving what `prod` points to, and the job summary says the gate failed with the reasons. The job itself ends green: a failed gate is a result, not an error. A red job means something broke (checksum, environment, pipeline).

## A person moves `prod`
Reading the run and promoting the model is a human decision. In W&B, open the model artifact `rebalance-model` and move the `prod` alias to the new version. Nothing in the workflow does it.

## Security set-up
- **Repository-level runner**, labelled `self-hosted` and `retrain`, not shared with other repositories.
- **No pull request triggers.** The workflow starts only from `workflow_dispatch` and `schedule`; there is no `pull_request`, `pull_request_target` or `push`, so no one can run code on the runner from a pull request or a fork.
- **Read-only token** (`permissions: contents: read`) and checkout with `persist-credentials: false`.
- **The W&B key** exists only as the `WANDB_API_KEY` secret of the `retrain` environment. The workflow never prints the environment and never turns on shell tracing.
- **The snapshot is never committed.** It stays on the runner, is copied into `data/raw/` for the run (ignored by Git) and is wiped with the runner's home.
- One retrain at a time (`concurrency: retrain`), 120 minutes at most.

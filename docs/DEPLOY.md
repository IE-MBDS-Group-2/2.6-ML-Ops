# Deploy runbook

How a release reaches the live service, and how to put the previous one back. The workflow is
`.github/workflows/deploy.yml`; this page is everything a person does around it.

```text
GitHub Release published -> deploy.yml: CI again on that commit -> Render deploy hook (?ref=<commit>)
                         -> Render builds the Dockerfile -> smoke_test.py waits until /health reports <commit>
```

## One-time set-up (admin)

1. **Render service.** Render Dashboard -> New -> Web Service -> connect GitHub and pick
   `IE-MBDS-Group-2/2.6-ML-Ops`. Language **Docker**, branch **`main`**, region Frankfurt, instance type
   **Free**. Under Advanced: health check path **`/health`**, Auto-Deploy **Off** (only the workflow deploys).
2. **Environment variables on Render** (the names in `.env.example`):

   | Name | Value now | Why |
   | --- | --- | --- |
   | `MODEL_SOURCE` | `local` | The image has no model file, so the API serves today's rule (decision D8) |
   | `WANDB_MODE` | `disabled` | Nothing to download while no model has passed the gate |
   | `APP_VERSION` | the release tag, e.g. `v1.1.0` | Shown on `/docs` and `/health`; a label only, the deploy checks the commit |

   When a model passes the gate and a person has moved the W&B alias `prod` to it: set `MODEL_SOURCE=wandb`,
   `WANDB_MODE=online`, `WANDB_ENTITY=ie-mbds-group-2`, `WANDB_MODEL_ALIAS=prod` and `WANDB_API_KEY` (as a
   Render secret value, never in the repo), then redeploy the current release (Rollback, below, with the same tag).
3. **GitHub.** Render service -> Settings -> Deploy Hook: copy the URL. GitHub repo -> Settings -> Secrets and
   variables -> Actions:
   - Secrets -> New repository secret `RENDER_DEPLOY_HOOK` = the hook URL. It contains a key: paste it only there.
   - Variables -> New repository variable `SERVICE_URL` = `https://<service-name>.onrender.com` (no trailing `/`).

## Dry run (once, before the first release)

1. Render service -> Settings -> Branch: `dev`.
2. GitHub -> Releases -> Draft a new release -> tag `v1.1.0-rc1` on `dev` -> tick "Set as a pre-release" ->
   Publish. That runs `deploy.yml` from the tag (Run workflow only works once `deploy.yml` is on the default
   branch, `main`).
3. Green means CI passed, Render built the image and the live `/health` reported that commit. Rule-only mode is
   expected (`policy: "rule"`).
4. Check a failure shows: rerun the deploy run with `SERVICE_URL` pointing at a wrong address; the smoke test
   step must go red. Put `SERVICE_URL` back.
5. Render service -> Settings -> Branch: back to `main`.
   - Delete the `v1.1.0-rc1` release and tag afterwards.

## Release

1. Release PR `dev` into `main`, approved, merge commit.
2. Render: set `APP_VERSION` to the new tag with **Save only** (not "Save, rebuild and deploy").
3. GitHub -> Releases -> Draft a new release -> tag `vX.Y.Z` on `main` -> notes -> **Publish**. That starts
   `deploy.yml`. Watch it in Actions; the run summary has the live links.
4. By hand: open `<SERVICE_URL>/docs`, call `/health`, and send `scripts/sample_request.json` to `/predict`.

## Rollback

Put the previous release back, code and all:

1. GitHub -> Actions -> deploy -> Run workflow -> Use workflow from -> **Tags** -> the previous release
   (e.g. `v1.1.0`) -> Run. It reruns CI on that commit, deploys that exact commit and smoke-tests it.
   (Run workflow appears once `deploy.yml` is on `main`, i.e. from v1.1.0 on.)
2. Render: set `APP_VERSION` back to that tag (Save only).

To roll back only the model (the code is fine): in W&B move the alias `prod` back to the previous model version,
then Render -> Manual Deploy -> Restart service. The API loads whatever `prod` points to at start-up.

## When it goes wrong

| Symptom | Likely cause | Next |
| --- | --- | --- |
| Hook step: `error: 404` | The commit is not on GitHub, or the hook URL is wrong | Copy the hook again into the secret |
| Smoke test times out on health | The build failed or is slow (free tier) | Render -> Events and Logs for that deploy |
| Smoke test: wrong commit | The new instance never became healthy, so Render kept the old one | Render deploy logs; fix, release a patch |
| Smoke test: `/predict` problem | The service runs but answers wrongly | `scripts/smoke_test.py` locally against `docker run` |

## Free tier

The service sleeps after about 15 minutes without requests; the first request then takes up to a minute. Before a
demo, open `/health` a few minutes early.

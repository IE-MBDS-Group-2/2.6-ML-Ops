# Deploy runbook

How a release reaches the live service, and how to put the previous one back. The workflow is
`.github/workflows/deploy.yml`; the service runs on a self-hosted machine that follows the image in GHCR.

```text
GitHub Release published -> deploy.yml: CI on that commit -> one docker build (commit and version baked in)
  -> push ghcr.io/ie-mbds-group-2/etf-rebalancing-api:<sha>, :<tag>, :release
  -> the host pulls :release (every minute), canary-tests it on a side port, swaps it in when /health is ok
  -> smoke_test.py waits until https://etf-api.mpholdings.co/health reports that commit and that version
```

**Why this design.** The image CI built and tested is the exact image that runs: it is built once, pushed with
immutable tags (`<sha>`, `<tag>`) and a moving tag (`release`) that the host follows. The host never needs
credentials from GitHub, CI never needs access to the host, and no port is open on it (Cloudflare tunnel).

## One-time set-up (admin)

1. **Repository variable** (Settings -> Secrets and variables -> Actions -> Variables): `SERVICE_URL` =
   `https://etf-api.mpholdings.co` (no trailing `/`). No secret is needed: the workflow pushes with its own token.
2. **Organisation:** Settings -> Packages: allow members to create public packages, and allow GitHub Actions to
   publish packages for this repository.
3. **After the first push** (the dry run below): Organisation -> Packages -> `etf-rebalancing-api` -> Package settings ->
   Change visibility -> **Public**, so the host can pull without a token. The image label links the package to
   this repository.
4. **The host** (operator side, already done): follows `:release` with a one-minute timer, canary-tests a new image
   on a side port, swaps only when `/health` is ok, restores the previous image if the new one turns unhealthy, and
   alerts the operator on deploy, refusal and rollback. It sets no `APP_VERSION` or commit: both come from the image.

## Dry run (once, before the first release)

1. GitHub -> Releases -> Draft a new release -> tag `v1.1.0-rc1` on `dev` -> tick **Set as a pre-release** -> Publish.
2. Actions -> deploy: `check-ref`, `ci`, `publish` and `smoke-test` go green. The smoke test passes when
   `/health` shows `"version": "v1.1.0-rc1"` and the dev commit (the host switches in about a minute).
3. Do step 3 of the set-up (package public) if the host could not pull.
4. Check a failure shows: re-run `smoke-test` with `SERVICE_URL` pointing at a wrong address; it must go red.
   Put `SERVICE_URL` back.
5. Delete the `v1.1.0-rc1` release and tag afterwards (its image stays in GHCR, which does no harm).

## Release

1. Release PR `dev` into `main`, approved, merge commit.
2. GitHub -> Releases -> Draft a new release -> tag `vX.Y.Z` on `main` -> notes -> **Publish**.
3. Watch Actions -> deploy; the run summary has the live links. By hand: `/health` shows the new version and commit;
   `/docs` opens; `scripts/sample_request.json` to `/predict` answers 200.

## Rollback

1. GitHub -> Actions -> deploy -> **Run workflow** -> Use workflow from -> **Tags** -> the release to put back
   (e.g. `v1.1.0`) -> Run. ("Run workflow" appears once `deploy.yml` is on `main`, i.e. from `v1.1.0` on.)
2. The run skips the build: it points `:release` at that tag's existing image (the exact image released then) and
   smoke-tests it. The host swaps it in like any release.

To roll back only the model (the code is fine): in W&B move the alias `prod` back to the previous model version,
then restart the container on the host. The API loads whatever `prod` points to at start-up (with
`MODEL_SOURCE=wandb`; today the service runs `MODEL_SOURCE=local` and serves today's rule).

## When it goes wrong

| Symptom | Likely cause | Next |
| --- | --- | --- |
| `publish`: denied pushing to ghcr.io | the organisation does not let Actions publish packages | set-up step 2 |
| Smoke test waits, commit never changes | the host could not pull (package private) or refused the canary | set-up step 3; host journal and alerts |
| Smoke test: wrong version | the image was built without `APP_VERSION` | check the `publish` step's build arguments |
| Rollback: no image for the tag | that tag was released before this workflow existed | release it again, or pick a newer tag |
| Smoke test: `HTTP 403: error code: 1010` | the Cloudflare edge (the smoke test sends `User-Agent: etf-smoke-test/1`) | check the Cloudflare status page, then re-run the failed smoke-test job |
| `/predict` answers but not from outside | Cloudflare or ISP blocking (e.g. during LaLiga broadcasts in Spain) | try another network; check the tunnel |

## Public paths

The tunnel serves only `/health`, `/predict`, `/docs` and `/openapi.json` (everything else 404), with a rate limit
of 20 requests a second per client (burst 100).

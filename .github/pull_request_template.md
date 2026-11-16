## What changed

## Why
<!-- Link the issue. PRs into dev do not close issues automatically: close it by hand after the merge. -->

## How it was tested
<!-- Paste the real output: pytest (passed, skipped, coverage), flake8, pylint, radon cc src -s -n C, isort. -->

## Notes for the modules that use this
<!-- Signatures, config keys read or added, errors raised, anything a caller must know. -->

## Skip ledger
<!-- Every skipped test and the issue it waits for, or "none". -->

## Checklist
- [ ] Base branch is `dev` (or `main` for a release); not stacked on another PR
- [ ] Branch updated from `dev` before asking for review
- [ ] Runs end to end: `python -m src.main`
- [ ] No secrets, no data, no models, no logs committed
- [ ] No `# noqa`, no new `# pylint: disable`, no hard-coded paths, columns or settings
- [ ] AI assistance: I reviewed every line and can explain it

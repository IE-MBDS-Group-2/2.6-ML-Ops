# Team brief

Business case: README.md, section 1 (merged into dev on 10 Nov 2026, PR #7)
Use case: the portfolio manager decides each Friday whether to rebalance the balanced model portfolio, using the recommended action (0%, 50% or 100% back to target)
Notebook: none; the pipeline is built from the README specification
Target: `best_action` (0.0, 0.5, 1.0) | Problem type: classification | ID column: `decision_date`
Feature groups:
- state: `w_<T>`, `drift_<T>`, `drawdown`, `weeks_since_rebalance`, `full_rebalance_cost` -> median impute + standard scale (built in `portfolio.py`)
- market: `ret_4w_<T>`, `ret_12w_<T>`, `vol_20d_<T>` -> median impute + standard scale (built in `features.py`, no look-ahead)
Excluded columns: `decision_date` (ID); future window returns (they would leak the label)
Model: LogisticRegression, multinomial, `class_weight="balanced"`, `max_iter=1000`, `solver="lbfgs"`, `random_state=42` (Ridge when `problem_type` is "regression")
Metric: `max_drawdown_reduction`, key in the evaluation output: `"max_drawdown_reduction"`, quality gate: >= 0.20 on validation (2020-2021). Also reported: `accuracy`, `f1_macro`, `max_drawdown`, `annual_return`, `annual_volatility`, `sharpe`, `annual_trading_cost`
Split: by date. Train 2007-09-07 to 2019-12-31 (includes 2008), validation 2020-2021, test 2022-2025 (sealed). The contract `split_data` is kept and tested; the pipeline uses `_split_by_date`
Validation rules: required columns present; no empty data; all 5 tickers present; every ticker has the same dates; close > 0; volume >= 0; |daily return| <= 0.25. Any failure stops the pipeline (bad data must not train a model)
README follow-ups: Stage 2 label and gate wording (D0, D2) -> Ricardo A.; snapshot owner placeholder -> Ricardo A.
features.py signature: `get_feature_preprocessor(numeric_columns)`; `compute_market_features(close_wide, return_weeks=[4, 12], vol_days=20)`
Agreed extra parameters: `validate_dataframe(..., tickers=None, max_abs_daily_return=None, price_columns=None)`; `clean_dataframe(..., date_column=None, sort_by=None)`; `clean_data.to_wide(df, value_column, tickers, date_column, ticker_column)`. Column names are never written in the code: `main.py` passes them from `data.columns` in `config.yaml`; `infer.apply_fallback(preds, rule_actions, missing_input, min_probability)` (the value comes from `fallback.min_probability`)
utils.py helpers: load_csv, save_csv, save_model, load_model, save_json
Environment: pandas, numpy, scikit-learn, joblib, pyyaml, python-dotenv, scipy, wandb, fastapi, uvicorn, pydantic, httpx, conda-lock, yfinance (fetch script only), pytest, pytest-cov, flake8, pylint, radon, black, isort, detect-secrets
Data snapshot: `data/raw/prices.csv` (made by `scripts/fetch_prices.py`), shared directly with the team (WhatsApp) by Ricardo A., not committed to Git (decision D4; Yahoo's API is for personal use only; how the professor receives it is to be confirmed with the professor). SHA-256: 3cce101c20eb1bcd5f453176affaa3c7f4a3f470a89973de47617134d6705526 (4,679 trading days per ticker, 2007-05-30 to 2025-12-31, downloaded 11 Nov 2026). Inference file `data/inference/prices_latest.csv` (300 trading days, 2024-10-21 to 2025-12-31): SHA-256 b5adf3d40afbf760afcb1527c07821064099a7587573b67a8d5ee3f02515e0f3
Decisions:
- Stage 2 model is a classifier of the best next-week action; the RL agent is Stage 3 (D0)
- Gate on `max_drawdown_reduction` on validation; guards reported, not gated (D2)
- Split by date, not random, because random splits leak future prices (D3)
- Flat 0.1% costs; realistic costs are a Stage 3 check (D1)
- Stage 2 model is a multinomial LogisticRegression with balanced class weights, and predictions below 0.60 confidence fall back to today's rule; the 10% information coefficient (IC) flag is reported only, never gated (D5). Why: the three actions are unevenly distributed, so balanced weights stop the model defaulting to the majority action; a linear model is fast, deterministic with a fixed seed and easy to explain to the portfolio manager; the fallback means a low-confidence call never replaces a rule the business already trusts; and a single gate (D2) keeps the pass/fail test simple and checkable
- Data snapshot is shared outside Git (D4): the grader's `git ls-files data models logs .env` check must print nothing, and Yahoo's API terms allow personal use only
- Why the classifier now (D0): the grader's contract is supervised, an RL agent cannot be tested properly by the deadline, and the classifier keeps the same inputs, actions, timing rule and gate, so the RL agent can join in Stage 3 as a challenger
- GitHub-hosted CI runners, because the repo is public (D7); a self-hosted runner is a later extension
Owners (team):
- utils -> Ricardo Arciniegas (@rarciniegascueva), branch feature/utils
- load_data -> Ricardo Arciniegas (@rarciniegascueva), branch feature/load-data
- clean_data -> Tate Bothner (@tatebothner), branch feature/clean-data
- validate -> Tate Bothner (@tatebothner), branch feature/validate
- features -> Farah (@faalnuseiratie), branch feature/features
- portfolio -> Riccardo Bellini (@riccbell), branch feature/portfolio
- train -> Riccardo Bellini (@riccbell), branch feature/train
- evaluate -> Raja Akel (@raja-akel), branch feature/evaluate
- infer -> Raja Akel (@raja-akel), branch feature/infer
- costs -> Miguel Palanca (@Smiggus), PR #9
- main -> Miguel Palanca (@Smiggus), branch feature/main

## Stage 3 (decisions D8 to D21, agreed 16 Nov 2026)

Milestone "Stage 3, CI/CD and live service", presentation Thu 19 Nov. Issues #55 to #79; specs and acceptance tests in
`docs/MODULE_SPECS.md` (Stage 3 section). The business case in README section 1 is not rewritten: Stage 3 delivers
what it commits to.

Product: the user sends a positions file (one portfolio or sleeve) and gets a recommendation, a proposed portfolio,
a trades list and one plain sentence saying why.

Settings a user chooses (all checked at start-up from `config.ALLOWED_VALUES`):
- `target_weights.method`: `policy` (the fund's own mix in `target_weights.policy`, the default) or calculated:
  `min_variance`, `max_sharpe`, `mean_variance`, `risk_parity`
- `rebalancing.method`: `smart` (our model or RL agent decides at each review, the default), `calendar`,
  `tolerance_band`, `calendar_and_band` (today's process with `quarterly` and `5`)
- `rebalancing.review`: `daily`, `weekly` (default), `monthly`; `rebalancing.calendar`; `rebalancing.tolerance_band`
  (percentage points from target)
- We never call this "tactical allocation": the tool only decides how to return to the targets.

Decisions:
- A model that fails the gate is not saved or registered; the API then serves today's rule and says so (D8)
- API contract: positions, drawdown, weeks since rebalance and 61 or more days of prices in; recommendation,
  reason, proposed portfolio and trades out (D9)
- CI runs the pipeline on synthetic prices from `scripts/make_sample_prices.py`; no secrets in CI (D10)
- Render hosts the API, deployed by GitHub Release; the API loads the W&B model alias `prod` (D11, D12)
  (19 Nov 2026: Render replaced by the self-hosted GHCR deploy, Stage 3, PR #127 and the infra work; see docs/DEPLOY.md)
- W&B team (entity) `ie-mbds-group-2`, project `etf-rebalancing`; everyone uses their own API key in `.env`;
  tracking is optional and off in tests and CI (D13)
- Scope tiers: Must = the CI/CD chain plus the README's commitments (RL agent, rule challengers, benchmarks,
  guards and release decision, robustness checks); Should = positions and trades, review frequency, choices (D14)
- Release `v1.1.0` (D15); dates: build Tue 17, freeze Wed 18 14:00, release Wed evening (D16)
- RL agent: fitted Q iteration with scikit-learn, same gate as the classifier, median of 5 seeds (D18)
- Benchmarks reported, never gated; data versioned as a W&B dataset artifact (D19, D20)
- Everyone builds their own issues against the acceptance tests in their walkthrough; every PR is reviewed (D21)
- Tracking-error guard X = 0.012 (`guards.max_tracking_error`): 2 x today's rule on validation 2020-21 (0.0059),
  like the 2 x cost guard; never rebalancing (0.0127) fails it. Agreed with Miguel (#64)

Secrets and switches (`.env`, never in Git): `WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_MODE`, `MODEL_SOURCE`,
`WANDB_MODEL_ALIAS`, `APP_VERSION`.

| Stage 3 role | Owner | Issues |
|---|---|---|
| Coordinator, wiring and RL | Miguel Palanca (@Smiggus) | #55 decisions, #56 setup, #62 RL agent, #71 main wiring, #76 deploy, #78 release |
| Tracking, containers, CI and allocation choices | Tate Bothner (@tatebothner) | #58 tracking, #61 Docker, #69 CI, #74 choices |
| API, smoke test and monitoring | Raja Akel (@raja-akel) | #67 API, #68 smoke test, #77 monitoring |
| Live state, guards, rules and frequency | Riccardo Bellini (@riccbell) | #59 live_state, #64 guards, #66 rule challengers, #72 review frequency |
| Model selection, benchmarks and robustness | Farah (@faalnuseiratie) | #60 walk-forward, #63 benchmarks, #70 grid search, #73 robustness |
| Data, trades and story | Ricardo Arciniegas (@rarciniegascueva) | #57 sample prices, #65 trades, #75 model card, #79 deck |

## Roles and contributions

RL rebalancing proposal (PR #7). Module owners work on `feature/<module>` branches; `utils.py` merges first and `main.py` last.
Work tracked as GitHub issues in the milestone "Stage 2, modular pipeline", ordered by `wave-0` to `wave-3` labels.

| Role | Name | Contribution | Where it lives |
|---|---|---|---|
| Admin | Miguel Palanca (@Smiggus) | Repo admin: branch protection, reviews, this brief, `config.yaml`. Owns `main.py`, the pipeline test, README (outside section 1), CI on GitHub-hosted runners, integration and the release PR. Wrote the trading cost model | `TEAM_BRIEF.md`, `config.yaml`, `src/main.py`, `tests/test_main.py`, `src/costs.py`, `.github/workflows/ci.yml` |
| Collaborator 1 | Ricardo Arciniegas (@rarciniegascueva) | Business case owner (PR #7). Shared helpers, loading the data, the frozen price snapshot | `README.md` section 1, `src/utils.py`, `src/load_data.py`, `scripts/fetch_prices.py` |
| Collaborator 2 | Tate Bothner (@tatebothner) | Cleaning and the data quality gate | `src/clean_data.py`, `src/validate.py` |
| Collaborator 3 | Farah (@faalnuseiratie) | Market features and the unfitted feature recipe | `src/features.py` |
| Collaborator 4 | Riccardo Bellini (@riccbell) | Portfolio simulator, today's rule and labels; fitting the model | `src/portfolio.py`, `src/train.py` |
| Collaborator 5 | Raja Akel (@raja-akel) | Metrics, the quality gate, and predictions with fallback | `src/evaluate.py`, `src/infer.py` |

Moved to Stage 3 (decision D6): traditional portfolio baselines, the market-stress (VIX) input, realistic-cost and AUM studies, the self-hosted CI runner, and the RL agent challenger.

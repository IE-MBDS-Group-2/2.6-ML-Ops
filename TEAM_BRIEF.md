# Team brief

Business case: README.md, section 1 (merged into dev on 10 Nov 2026, PR #7)
Use case: the portfolio manager decides each Friday whether to rebalance the balanced model portfolio, using the recommended action (0%, 50% or 100% back to target)
Notebook: none; the pipeline is built from the README specification
Target: `best_action` (0.0, 0.5, 1.0) | Problem type: classification | ID column: `decision_date`
Feature groups:
- state: `w_<T>`, `drift_<T>`, `drawdown`, `weeks_since_rebalance`, `full_rebalance_cost` -> median impute + standard scale (built in `portfolio.py`)
- market: `ret_4w_<T>`, `ret_12w_<T>`, `vol_20d_<T>` -> median impute + standard scale (built in `features.py`, no look-ahead)
Excluded columns: `decision_date` (ID); future window returns (they would leak the label)
Model: LogisticRegression, multinomial, `class_weight="balanced"`, `max_iter=1000`, `random_state=42`
Metric: `max_drawdown_reduction`, key in the evaluation output: `"max_drawdown_reduction"`, quality gate: >= 0.20 on validation (2020-2021). Also reported: `accuracy`, `f1_macro`, `max_drawdown`, `annual_return`, `annual_volatility`, `sharpe`, `annual_trading_cost`
Split: by date. Train 2007-09-07 to 2019-12-31 (includes 2008), validation 2020-2021, test 2022-2025 (sealed). The contract `split_data` is kept and tested; the pipeline uses `_split_by_date`
Validation rules: required columns present; no empty data; all 5 tickers present; every ticker has the same dates; close > 0; volume >= 0; |daily return| <= 0.25. Any failure stops the pipeline (bad data must not train a model)
README follow-ups: Stage 2 label and gate wording (D0, D2) -> Ricardo A.; snapshot owner placeholder -> Ricardo A.
features.py signature: `get_feature_preprocessor(numeric_columns)`; `compute_market_features(close_wide, return_weeks=[4, 12], vol_days=20)`
Agreed extra parameters: `validate_dataframe(..., tickers=None, max_abs_daily_return=None)`; `clean_data.to_wide(df, value_column, tickers)`; `infer.apply_fallback(preds, rule_actions, missing_input, min_probability=0.60)`
utils.py helpers: load_csv, save_csv, save_model, load_model, save_json
Environment: pandas, numpy, scikit-learn, joblib, pyyaml, yfinance (fetch script only), pytest, pytest-cov, flake8, pylint, radon, black, isort, detect-secrets
Data snapshot: `data/raw/prices.csv` (made by `scripts/fetch_prices.py`), shared directly with the team (WhatsApp) by Ricardo A., not committed to Git (decision D4; Yahoo's API is for personal use only; how the professor receives it is to be confirmed with the professor). SHA-256: 3ada61c6a3cc037e74537f9fa8a739eb2c0c67cc866e95eeb039fad7bf0c6c71 (4,679 trading days per ticker, 2007-05-30 to 2025-12-31, downloaded 11 Nov 2026)
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

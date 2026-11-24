# Module specs (Stage 2)

These are the names, inputs and outputs every module must keep, so they fit together on `dev`. The names marked
**contract** are fixed by the course, and the grader's probes call them. Everything else is ours. Every setting comes
from `config.yaml`, passed in by `main.py`; modules never read the config file themselves.

## How data flows

```text
data/raw/prices.csv (date, ticker, close, volume; long format)
  -> load_data.load_raw_data            (read only)
  -> clean_data.clean_dataframe         (types, sort, duplicates)
  -> validate.validate_dataframe        (stop on bad data)
  -> clean_data.to_wide                 (close_wide, volume_wide: date x ticker)
  -> features.compute_market_features   (4w and 12w returns, 20d vol, per ETF, no look-ahead)
  -> portfolio.build_decision_table     (one row per Friday: state + market features + label best_action,
                                          simulated along the current rule's path)
  -> split.split_by_date               (train 2007-09 to 2019 / validation 2020-21 / test 2022-25 sealed)
  -> features.get_feature_preprocessor  (unfitted imputer + scaler)
  -> train.train_model                  (Pipeline: preprocess + LogisticRegression)
  -> evaluate.evaluate_model            (accuracy, f1_macro on validation)
  -> portfolio.simulate (agent policy vs current rule, validation) -> evaluate.portfolio_metrics
  -> evaluate.max_drawdown_reduction + check_gate  -> "gate met / not met" in the log and reports/metrics.json
  -> utils.save_model                   (models/model.joblib)
  -> infer.run_inference + apply_fallback on data/inference/prices_latest.csv -> reports/predictions.csv
```

Columns and units used everywhere:
- `date`: datetime.
- `ticker`: one of the config tickers.
- Weights and costs: fractions of portfolio value.
- Actions: `0.0`, `0.5`, `1.0`, the share of the way back to target.

Every module starts like this:

```python
import logging
logger = logging.getLogger(__name__)
```

Required tests per module: at least 2 `test_unit_` tests (one with a known expected value) and 1
`test_integration_` test. Every `raise` must be tested. If a neighbour module isn't on `dev` yet, mark the integration
test `@pytest.mark.skip(reason="Awaiting <module> in dev")`, and remove the skip once it lands.

---

## utils.py (Ricardo A., merges first)

```python
def load_csv(filepath: Path) -> pd.DataFrame            # contract
def save_csv(df: pd.DataFrame, filepath: Path) -> None   # contract: creates parent folders, index=False
def save_model(model, filepath: Path) -> None            # contract: joblib, creates parent folders
def load_model(filepath: Path)                           # contract
def save_json(data: dict, filepath: Path) -> None        # ours: for reports/metrics.json
```

Generic only: no column names, no business rules.

Tests:
- `save_csv` then `load_csv` gives the same frame.
- A missing file raises `FileNotFoundError`, for both CSV and model.
- Parent folders are created.
- Integration: a tiny `LogisticRegression` saved and reloaded gives the same predictions.

## load_data.py (Ricardo A.)

```python
def load_raw_data(raw_data_path: Path) -> pd.DataFrame   # contract
```

- Missing path: `FileNotFoundError`. Folder: `IsADirectoryError`. Zero bytes: `ValueError("... is empty: ...")`.
  No rows: `ValueError("... has no rows: ...")`.
- Reads through `utils.load_csv` and logs the row count. Never downloads.
- A 7-test file for this already exists (`test_load_data.py`, generic columns). Use it.

`scripts/fetch_prices.py` (outside `src/`, so it isn't in coverage): yfinance, `auto_adjust=True`, tickers and dates
from `config.yaml`. It writes `data/raw/prices.csv` and `data/inference/prices_latest.csv`, and prints the SHA-256.

## clean_data.py (Tate)

```python
def clean_dataframe(df_raw: pd.DataFrame, target_column: str | None = None, date_column: str | None = None,
                    sort_by: list[str] | None = None) -> pd.DataFrame                         # contract + extras
def to_wide(df: pd.DataFrame, value_column: str, tickers: list[str], date_column: str,
            ticker_column: str) -> pd.DataFrame                                               # ours
```

Column names are arguments, never literals: `main.py` passes them from `data.columns` in `config.yaml`.

- `clean_dataframe`: lowercase column names; parse `date`; sort by `date, ticker`; drop duplicate `(date, ticker)`
  rows (log how many). Never modifies the input. `target_column` is unused here (the raw file has no target); keep
  the parameter for the contract.
- `to_wide`: a `date` x `ticker` table in config ticker order. Rows are only dates where **every** ticker has a value
  (the common calendar), and it logs dropped dates.

Tests:
- Known value: 5 rows with 1 duplicate gives 4.
- The input is unchanged.
- `to_wide` drops a date missing one ticker.
- Integration: the output passes `validate_dataframe`.

## validate.py (Tate)

```python
def validate_dataframe(df: pd.DataFrame, required_columns: list[str], target_column: str | None = None,
                       tickers: list[str] | None = None, max_abs_daily_return: float | None = None,
                       price_columns: dict[str, str] | None = None) -> bool                  # contract + extras
```

`price_columns` is `data.columns` from the config. The price-table rules below run only when it is given; without
it only the general rules run (required columns, empty frame, empty values, target). Passing `tickers` or
`max_abs_daily_return` without `price_columns` raises a `ValueError`, so a rule is never skipped silently.

Returns `True` or raises `ValueError` with a clear message. It never fixes data. It stops on:
- missing required columns, or an empty frame;
- a configured ticker that's absent;
- a ticker with fewer dates than the others;
- close <= 0, or volume < 0;
- any \|daily return\| > `max_abs_daily_return` (config: 0.25).

Tests: valid data returns `True`; one test per raise; integration: `clean_dataframe` output passes.

## features.py (Farah)

```python
def compute_market_features(close_wide: pd.DataFrame, return_weeks: list[int], vol_days: int) -> pd.DataFrame  # ours
def get_feature_preprocessor(numeric_columns: list[str]) -> ColumnTransformer                                 # contract
```

- `compute_market_features`: same index as `close_wide`. Columns `ret_4w_<T>`, `ret_12w_<T>` (a week is 5 trading
  days) and `vol_20d_<T>` (annualised std of daily returns). Each value uses data up to and including that date only.
- `get_feature_preprocessor`: an **unfitted** `ColumnTransformer`, with `SimpleImputer(median)` then `StandardScaler`
  on `numeric_columns`. An empty list raises `ValueError`.

Tests:
- Known value: a 12-week return on a price that doubles is `1.0`.
- **Look-ahead test**: change prices after date t, and the features at t don't change.
- The preprocessor is unfitted; an empty list raises.
- Integration: `fit_transform` on tiny data gives no NaNs and the expected column count.

## portfolio.py (Riccardo B., critical path)

```python
def decision_dates(trading_days: pd.DatetimeIndex, start: str, end: str) -> pd.DatetimeIndex
def apply_action(weights: pd.Series, target: pd.Series, action: float) -> pd.Series
def current_rule_action(weights: pd.Series, target: pd.Series, date: pd.Timestamp, is_quarter_start: bool,
                        band: float) -> float
def simulate(close_wide: pd.DataFrame, volume_wide: pd.DataFrame, market_features: pd.DataFrame,
             target: pd.Series, policy, start: str, end: str, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]
def build_decision_table(close_wide, volume_wide, market_features, target, start, end, cfg) -> pd.DataFrame
```

- `decision_dates`: the last trading day of each week within `[start, end]`.
- `apply_action`: weights moved `action` of the way to target. Fractions sum to 1.
- `current_rule_action`: `1.0` if it's the first decision date of a quarter, or any \|weight − target\| > `band`
  (0.05); otherwise `0.0`.
- **Timing rule (README):** decide at Friday close t, trade at the next trading day's close e, then earn returns from
  e+1 to the next execution day inclusive. Weights drift with prices between trades.
- `simulate`: starts at target weights on the first decision date. At each decision date it builds the state row,
  calls `policy(state_row) -> action`, applies the action at e, and charges `costs.trade_costs(delta_w, aum, adv, cost_cfg)`.
  It returns:
  - a **daily** frame with `value`, `cost_paid`, and the weights;
  - a **decision** frame, one row per Friday, with the state features and the action taken.
- State features at Friday close:
  - `w_<T>` and `drift_<T>` (= w − target), the pre-trade weights;
  - `drawdown` (value / running peak − 1);
  - `weeks_since_rebalance`;
  - `full_rebalance_cost` (`costs.trade_costs(target − w).sum()`).
- `build_decision_table`: runs `simulate` with the current rule as the policy. For every decision row it also tries
  all 3 actions one window ahead from that state, with reward = window return − costs − `penalty_lambda` × increase
  in drawdown. The label `best_action` is the argmax; ties go to the smaller action. The last Friday has no full
  window, so its label is NaN, and `main` drops it for training.

Tests (they are the README's own):
- the first reward date is after the execution date;
- reward windows join with no gaps or overlaps;
- the agent and the baseline go through the same `simulate`;
- known value: with flat 0.1% cost, a full rebalance of weights that are off by +0.05 and −0.05 costs
  0.05 × 0.001 × 2 = 0.0001 in total;
- the current rule fires on the quarter start and on a 6-point drift, and not on a 4-point drift.

## train.py (Riccardo B.)

```python
def train_model(X_train: pd.DataFrame, y_train: pd.Series, preprocessor: ColumnTransformer, problem_type: str,
                model_params: dict | None = None, random_state: int = 42) -> Pipeline                        # contract
```

- It returns `Pipeline([("preprocess", preprocessor), ("model", LogisticRegression(**model_params))])`, fitted.
  This is the only `.fit()` in the project.
- `problem_type` must be `"classification"`; anything else raises `ValueError`. It logs the class counts.

Tests:
- Returns a fitted Pipeline with one prediction per row.
- A bad `problem_type` raises.
- Integration: `get_feature_preprocessor`, then `train_model`, then predict; a saved and reloaded model predicts the
  same.

## evaluate.py (Raja)

```python
def evaluate_model(model: Pipeline, X_eval: pd.DataFrame, y_eval: pd.Series, problem_type: str) -> dict[str, float]  # contract
def portfolio_metrics(values: pd.Series, costs_paid: pd.Series) -> dict[str, float]
def max_drawdown_reduction(agent_mdd: float, baseline_mdd: float) -> float
def check_gate(value: float, comparison: str, threshold: float) -> bool
```

- `evaluate_model`: plain floats `{"accuracy": ..., "f1_macro": ...}`. Empty or mismatched inputs raise `ValueError`.
- `portfolio_metrics`:
  - `max_drawdown`: the minimum of value / running peak − 1, so it's negative;
  - `annual_return`: CAGR, 252 days;
  - `annual_volatility`;
  - `sharpe`: return / volatility, risk-free rate 0;
  - `annual_trading_cost`: sum of costs / years.
- `max_drawdown_reduction` = 1 − \|agent\| / \|baseline\|. A baseline of 0 raises `ValueError`.
- `check_gate`: supports `>=` and `<=`; anything else raises `ValueError`.

Tests:
- Known values: values 100, 120, 90, 110 give `max_drawdown = -0.25`.
- A reduction of agent −0.16 against baseline −0.20 is `0.20`.
- `check_gate(0.20, ">=", 0.20)` is `True`.
- Integration: `train_model` output scored by `evaluate_model`.

## infer.py (Raja)

```python
def run_inference(model: Pipeline, X_infer: pd.DataFrame, include_proba: bool = False) -> pd.DataFrame   # contract
def apply_fallback(preds: pd.DataFrame, rule_actions: pd.Series, missing_input: pd.Series,
                   min_probability: float) -> pd.DataFrame
```

- `run_inference`: same index as `X_infer`; columns `prediction`, plus `probability` (the top-class probability) if
  `include_proba`. The input is unchanged.
- `apply_fallback`: where `probability < min_probability` (0.60), or any input is missing, set `recommendation` to the
  current rule's action and `fallback=True`; otherwise `recommendation = prediction` and `fallback=False`.

Tests:
- Same index and the prediction column.
- The input is unchanged.
- Fallback on low probability and on missing input.
- Integration: `save_model`, then `load_model`, then `run_inference`.

## main.py (Miguel)

```python
def split_data(X, y, test_size: float, val_size: float, random_state: int, stratify: bool) -> tuple  # contract; calls split.split_data
# the split itself lives in src/split.py: split_data (random, contract), split_by_date (the pipeline), up_to_date
def main() -> None                                                                                   # contract
```

- It loads `config.yaml` once and fails fast on a missing section (`KeyError`). It configures logging from the config.
- It runs the flow above, logging `Quality gate met` or `NOT met`, with `max_drawdown_reduction` on validation.
- It writes `data/processed/clean.csv`, `models/model.joblib`, `reports/predictions.csv` (decision date first) and
  `reports/metrics.json`.
- It catches `ValueError`, `OSError` and `KeyError`, logs the cause and the next action, and exits 1.

Tests:
- `split_data` sizes with no overlap.
- `split.split_by_date` boundaries (tests/test_split.py).
- A missing raw file exits 1.
- **Pipeline test:** synthetic prices for 5 tickers over about 3 years in `tmp_path`, with config paths and dates
  patched. Then the 4 outputs exist, the model reloads, and there's one prediction row per inference decision date.

---

# Module specs (Stage 3)

The contract for every Stage 3 issue. The Stage 2 contracts above stay as they are, with one change: `main.py` no
longer has `load_config` or `configure_logging`; they moved to `src/config.py` and `src/logger.py`.
Every rule from Stage 2 still applies: typed and documented functions, a logger per module, no `print`, no hard-coded paths, columns or settings (read them from `config.yaml`), every `raise` tested, 90% branch coverage or more, no function above complexity 10.

```text
 TRAINING (python -m src.main)                         SERVING (uvicorn src.api:app)
 config.yaml + .env -> config.py, logger.py, tracking.py      client --POST /predict--> api.py
 load -> clean -> validate -> to_wide -> features            clean -> validate -> to_wide -> features
   -> portfolio (decision table) -> [selection: grid]          -> portfolio.live_state -> infer + fallback
   -> train -> evaluate -> QUALITY GATE                        ^ model from W&B alias prod (or local file)
        pass: test audit, save, register (tracking) ----------+
        fail: nothing saved; today's rule serves (D8)

 DELIVERY  feature --PR--> dev --PR--> main --GitHub Release--> deploy.yml --> GHCR :release --> host --> smoke_test.py
                   ci.yml          ci.yml   (lint, sample-data pipeline, tests, complexity, Docker build)
```

### config.py and logger.py (Miguel, setup)

```python
PROJECT_ROOT: Path            # repository root
CONFIG_PATH: Path             # PROJECT_ROOT / "config.yaml"
def load_config(path: Path = CONFIG_PATH) -> dict                       # config.py
def resolve_paths(cfg: dict, root: Path = PROJECT_ROOT) -> dict[str, Path]   # config.py
def load_env(env_file: Path | None = None) -> dict[str, str]           # config.py
def setup_logging(level: str, log_file: Path | None = None) -> None     # logger.py
```

- `load_config`: reads YAML once; raises `FileNotFoundError` (no file), `ValueError` (not a mapping, or a key has
  value that is not allowed) and `KeyError` (names every missing section and key).
- **Allowed values, checked at start-up.** `config.ALLOWED_VALUES` is the one table of settings that only accept
  listed values, and `load_config` checks each one: `target_weights.method` (`policy`, `min_variance`,
  `max_sharpe`, `mean_variance`, `risk_parity`); `rebalancing.method` (`smart`, `calendar`, `tolerance_band`,
  `calendar_and_band`); `rebalancing.review` (`daily`, `weekly`, `monthly`); `rebalancing.calendar` (`monthly`,
  `quarterly`, `annually`); `model.type` (`classifier`, `fitted_q`); `problem.problem_type`; `costs.model`.
  A wrong value stops the run before any data is read, with a message that names the key, the value found and
  the allowed values, for example: `rebalancing.method is 'smrt'. Allowed: smart, calendar, tolerance_band,
  calendar_and_band.` Also checked: `target_weights.policy` covers exactly `data.tickers` and sums to 1;
  `rebalancing.tolerance_band` is a number above 0.
- `resolve_paths(cfg, root)`: every `paths.*` entry as a full path; `main.py` and `api.py` both use it.
- `load_env`: reads `.env` (python-dotenv) and the process environment; returns only the names in `.env.example`.
  Defaults: `WANDB_MODE=disabled`, `MODEL_SOURCE=local`, `WANDB_MODEL_ALIAS=prod`. Never logs a value.
- `setup_logging`: console plus optional file, one format, `force=True`. Only `main.py` and `api.py` call it.
- `main.py` loses `load_config` and `configure_logging` and calls these instead (its tests move to
  `tests/test_config.py` and `tests/test_logger.py`).
- New config sections: `tracking` (D13), `api` (D9), `selection` (D14), `rl` (D18), `benchmarks` (D19), `guards`,
  `positions`; new sections `target_weights` and `rebalancing` (D17, D19); new keys `paths.positions`, `model.type`. New `.env.example` names: `WANDB_API_KEY`,
  `WANDB_ENTITY`, `WANDB_MODE`, `MODEL_SOURCE`, `WANDB_MODEL_ALIAS`, `APP_VERSION`.
- `tests/conftest.py`: sets `WANDB_MODE=disabled` and `MODEL_SOURCE=local` before any `src` import, so every test
  runs offline.

### scripts/make_sample_prices.py (Ricardo A., data for CI)

```python
def make_sample_prices(tickers: list[str], start: str, end: str, seed: int, columns: dict) -> pd.DataFrame
def main(argv: list[str] | None = None) -> None   # reads config.yaml; writes paths.raw_data and paths.inference_data
```

- Business days from `data.start` to `data.end`; one row per (date, ticker); columns from `data.columns`
  (`date_column`, `ticker_column`, `close_column`, `volume_column`). Never hard-code them.
- Random walk per ticker with a fixed seed: daily volatility roughly SPY 1.2%, EFA 1.3%, IEF 0.4%, GLD 1.0%,
  BIL 0.02%, a small positive drift, closes above 0, volumes above 0. Same seed, same file (test it).
- The inference file is the last `data.inference_days` trading days of the same frame.
- Logs that the data is **synthetic** at WARNING level, so nobody mistakes it for the real snapshot.
- Tests (`tests/test_make_sample_prices.py`): same seed gives identical frames; every ticker has every date;
  the output passes `validate.validate_dataframe` with the config's rules; `main()` writes both files under
  `tmp_path` (patch the config paths).
- `columns` is the `data.columns` mapping (column names are never written in the code).
- Run it as a module from the repo root: `python -m scripts.make_sample_prices` (running the file directly
  cannot import `src`). It **refuses to overwrite** an existing output file unless `--force` is passed, so a
  real snapshot is never replaced by accident.
- CI calls it before `python -m src.main` (issue "Infra: CI"); CI starts from an empty `data/`, so no `--force`.

**Acceptance tests: write these first, then the code until they pass.**

| Test | Given | Expect |
| --- | --- | --- |
| `test_unit_same_seed_gives_identical_prices` | two calls with the same tickers, dates and seed | the two frames are equal; a different seed gives different closes |
| `test_unit_every_ticker_has_every_business_day` | 5 tickers, 2024-01-01 to 2024-03-29 | rows = business days x 5; no missing value; no duplicated (date, ticker) |
| `test_unit_prices_and_volumes_are_positive` | any seed | every close > 0 and every volume > 0 |
| `test_unit_columns_come_from_config` | column names day, symbol, px, qty passed in | the frame has exactly those columns |
| `test_integration_output_passes_validate` | the generated frame and the config's rules | `validate.validate_dataframe(..., price_columns=cols)` returns True |
| `test_integration_main_writes_both_files` | config paths patched to `tmp_path` | raw file and inference file exist; the inference file has `data.inference_days` dates |
| `test_unit_raises_on_end_before_start` | end date before start date | `ValueError` naming both dates |

### tracking.py (Tate, W&B)

```python
def start_run(cfg: dict, env: dict[str, str], job_type: str = "train") -> Run | None
def log_metrics(run: Run | None, metrics: dict[str, float], prefix: str = "") -> None
def log_table(run: Run | None, name: str, table: pd.DataFrame) -> None
def log_dataset(run: Run | None, path: Path, name: str) -> None          # output artifact, type "dataset"
def log_model(run: Run | None, path: Path, name: str, metadata: dict) -> None   # type "model", alias "latest"
def finish_run(run: Run | None, failed: bool = False) -> None
def download_model(env: dict[str, str], project: str, name: str, alias: str) -> Path   # for api.py, no run
```

- **Contract (course): every function accepts `run=None` and then does nothing.** `start_run` returns `None` when
  `WANDB_MODE` is `disabled` or `tracking.enabled` is false. `main.py` never imports `wandb`.
- `start_run`: `wandb.init(project, entity, name, group, tags, job_type, config=cfg, mode=WANDB_MODE)`.
- `finish_run(failed=True)`: `run.finish(exit_code=1)`, so a pipeline error marks the run failed.
- `log_model` only runs after the gate passed (D8). Nobody moves `prod` in code: a person promotes in the W&B UI.
- `download_model`: `wandb.Api().artifact(f"{entity}/{project}/{name}:{alias}").download()`; returns the
  `.joblib` path. Raises `FileNotFoundError` with a clear message for every way the model cannot be fetched: the
  alias does not exist, W&B is unreachable or rejects the key (`wandb.errors.Error`), or the download holds no
  `.joblib`. One error type, so the API catches one thing and serves today's rule.
- Config section `tracking`: `enabled`, `project`, `name`, `group`, `tags`, `dataset_artifact`, `model_artifact`,
  `log_predictions_table`.
- Tests (`tests/test_tracking.py`), all offline: with `run=None` every function returns `None` and calls nothing;
  `start_run` returns `None` when disabled; with a fake run object (a `MagicMock`), `log_metrics` prefixes keys,
  `finish_run(failed=True)` passes `exit_code=1`, `log_model` builds a model artifact; `download_model` with a
  patched `wandb.Api` returns the file path and raises on a missing alias.

**Acceptance tests: write these first, then the code until they pass.**

| Test | Given | Expect |
| --- | --- | --- |
| `test_unit_every_function_does_nothing_with_run_none` | `run=None` passed to each of the log and finish functions | each returns None and no `wandb` call is made (patch `wandb` with a mock and assert it is untouched) |
| `test_unit_start_run_returns_none_when_disabled` | `WANDB_MODE=disabled`, then `tracking.enabled: false` | `start_run` returns None both times; `wandb.init` is never called |
| `test_unit_start_run_passes_project_entity_and_config` | mode `offline`, a patched `wandb.init` | it is called once with the project, entity, name, group, tags, job_type and the config |
| `test_unit_log_metrics_adds_the_prefix` | a fake run, metrics `{'accuracy': 0.5}`, prefix `val/` | `run.log` is called with `{'val/accuracy': 0.5}` |
| `test_unit_finish_run_marks_failure` | a fake run, `failed=True` | `run.finish(exit_code=1)`; with `failed=False`, `exit_code=0` |
| `test_unit_log_model_builds_a_model_artifact` | a fake run, a file in `tmp_path` | an artifact of type `model` with the file added is logged with alias `latest` |
| `test_unit_download_model_returns_the_joblib_path` | a patched `wandb.Api` whose artifact downloads to `tmp_path` | the returned path ends in `.joblib` and exists |
| `test_unit_download_model_raises_on_missing_alias` | the patched API raises for the alias | `FileNotFoundError` that names the artifact and the alias |

### portfolio.live_state (Riccardo B., the state row for one live request)

```python
def live_state(close_wide: pd.DataFrame, volume_wide: pd.DataFrame, market_features: pd.DataFrame,
               weights: pd.Series, drawdown: float, weeks_since_rebalance: int, target: pd.Series,
               cfg: dict) -> pd.DataFrame
```

- Returns **one row**, indexed by the last date in `close_wide` (name from `problem.id_column`), with exactly the
  model's input columns in the same order as `build_decision_table` (`w_<T>`, `drift_<T>`, `drawdown`,
  `weeks_since_rebalance`, `full_rebalance_cost`, the market features) plus `rule_action` (today's rule on that
  date: quarter start or a breached drift band).
- Reuses `_state_row` logic and `current_rule_action`; no new maths. "Quarter start" = the first decision date in
  a quarter month, worked out from the dates in `close_wide`.
- Raises `ValueError` (clear message) if weights do not match the tickers, do not sum to 1, `drawdown > 0`,
  `weeks_since_rebalance < 0`, or the market features are missing on the last date (too little history).
- Tests: a known-value row (weights equal to target: drifts 0, `rule_action` 0 on a mid-quarter date); a
  quarter-start date gives `rule_action` 1.0; columns equal `build_decision_table`'s feature columns on the same
  synthetic prices (integration); one test per raise.

**Acceptance tests: write these first, then the code until they pass.**

| Test | Given | Expect |
| --- | --- | --- |
| `test_unit_live_state_known_values_at_target` | synthetic prices, weights equal to target, drawdown 0, weeks 0, a mid-quarter date | one row; every `drift_<T>` is 0; `rule_action` is 0.0; `drawdown` is 0 |
| `test_unit_live_state_rule_fires_on_quarter_start` | the last date is the first decision of a quarter month | `rule_action` is 1.0 |
| `test_unit_live_state_rule_fires_on_band_breach` | one weight 6 points above target, mid-quarter | `rule_action` is 1.0; with 4 points it is 0.0 |
| `test_integration_live_state_columns_match_decision_table` | the same synthetic prices through `build_decision_table` | same feature columns in the same order (everything except the target column) |
| `test_unit_live_state_index_is_the_last_date` | any valid input | the index is the last date of `close_wide`, named `problem.id_column` |
| `test_unit_live_state_raises (one per case)` | weights for the wrong tickers; weights summing to 0.9; drawdown 0.1; weeks -1; too little price history | `ValueError` saying what is wrong and how to fix it |

### Review frequency in config (portfolio.py)

```python
def decision_dates(trading_days: pd.DatetimeIndex, start: str, end: str, review: str) -> pd.DatetimeIndex
```

- Setting `rebalancing.review`: `daily` (every trading day), `weekly` (last trading day of the week, today's
  behaviour and the default) or `monthly` (last trading day of the month). The allowed values come from
  `config.ALLOWED_VALUES`; anything else raises `ValueError`.
- Every caller passes it from the config (`simulate`, `build_decision_table`, `live_state`, `main.py`). The
  quarter-start rule for today's process stays as it is. `weeks_since_rebalance` keeps its name and counts reviews.
- Changing the review frequency changes the labels, so it means a retrain. The comparison across daily, weekly
  and monthly is done on the train years only (issue "grid search"), always net of trading costs.
- Tests: known dates for each value on a small calendar; the weekly default equals the Stage 2 result (regression
  test on the synthetic prices); the unknown-value raise; `build_decision_table` runs for each value.

**Acceptance tests: write these first, then the code until they pass.**

| Test | Given | Expect |
| --- | --- | --- |
| `test_unit_decision_dates_known_values_per_review` | business days of Jan 2024 | `weekly`: 4 or 5 last-trading-days-of-week; `daily`: every day; `monthly`: only 2024-01-31 |
| `test_unit_weekly_default_reproduces_stage_2` | the synthetic prices used by the existing tests | the decision table equals the one built before this change (regression) |
| `test_unit_decision_dates_raises_on_unknown_review` | `review='hourly'` | `ValueError` listing daily, weekly, monthly |
| `test_integration_decision_table_builds_for_each_review` | each of the three values | a non-empty table with the same columns |

### trades.py (positions in, proposed portfolio and trades out)

```python
def positions_to_weights(positions: pd.DataFrame, tickers: list[str], ticker_column: str,
                         value_column: str) -> tuple[pd.Series, float]
def proposed_portfolio(weights: pd.Series, target: pd.Series, action: float, total_value: float) -> pd.DataFrame
def trade_list(weights: pd.Series, proposed: pd.DataFrame, total_value: float, min_trade: float) -> pd.DataFrame
```

- Positions file (`paths.positions`, CSV): one row per holding with a ticker and a market value. Column names
  come from a new config section `positions` (`ticker_column`, `value_column`). Only tickers in `data.tickers`.
- `positions_to_weights`: weights in config ticker order (0 for a ticker not held) and the total value. Raises
  `ValueError` for an unknown ticker, a negative or missing value, a repeated ticker, or a total of 0.
- `proposed_portfolio`: uses `portfolio.apply_action` (no new maths); columns `ticker`, `weight`, `value`.
- `trade_list`: columns `ticker`, `side` (`buy` or `sell`), `value`, `weight_before`, `weight_after`; trades
  smaller than `portfolio.min_trade` (as a share of the portfolio) are dropped; buys and sells net to 0.
- Batch use: when `paths.positions` exists, `main.py` writes `reports/proposed_portfolio.csv` and
  `reports/trades.csv` for the latest decision date. The API uses the same three functions.
- Tests: known-value example (60/40 drifted to 65/35, action 1.0 and 0.5); weights sum to 1; net trade value 0;
  `min_trade` drops small trades; one test per raise; integration with `portfolio.apply_action`.

**Acceptance tests: write these first, then the code until they pass.**

| Test | Given | Expect |
| --- | --- | --- |
| `test_unit_positions_to_weights_known_values` | positions 650 and 350 in two tickers | weights 0.65 and 0.35 in config ticker order; total 1000 |
| `test_unit_missing_ticker_gets_zero_weight` | positions for 4 of 5 configured tickers | the fifth weight is 0.0; weights sum to 1 |
| `test_unit_proposed_portfolio_known_values` | weights 0.65/0.35, target 0.60/0.40, total 1000 | action 1.0 gives 0.60/0.40; 0.5 gives 0.625/0.375; 0.0 leaves it; values = weight x total |
| `test_unit_trade_list_nets_to_zero` | the 1.0 case above | one sell of 50 and one buy of 50; buys minus sells is 0; columns ticker, side, value, weight_before, weight_after |
| `test_unit_min_trade_drops_small_trades` | a drift of 0.1 points and `min_trade` 0.005 | the trade list is empty |
| `test_unit_positions_to_weights_raises (one per case)` | unknown ticker; negative value; missing value; repeated ticker; total 0 | `ValueError` naming the ticker or the problem |
| `test_integration_matches_apply_action` | random weights and target | `proposed_portfolio` weights equal `portfolio.apply_action` |

### selection.py (Farah, walk-forward check and grid search)

```python
def walk_forward_windows(dates: pd.DatetimeIndex, first_train_end: str, last_end: str, block: str,
                         gap_days: int) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]
def walk_forward_scores(table: pd.DataFrame, market_data: tuple, cfg: dict,
                        model_params: dict) -> pd.DataFrame
def grid_search(table: pd.DataFrame, market_data: tuple, cfg: dict, run: Run | None) -> tuple[dict, pd.DataFrame]
```

- `walk_forward_windows`: expanding train window; each score block is one year (`block: "YE"`); `gap_days`
  trading days dropped between train end and block start (5, so labels never overlap). Only dates **up to the
  train end** (2019-12-31): validation and test are never touched here.
- `walk_forward_scores`: for each window, `train.train_model` on the train rows, then the same simulator and
  `evaluate.max_drawdown_reduction` as the gate, against today's rule on that block. One row per window:
  `block_start`, `block_end`, `max_drawdown_reduction`, `agent_mdd`, `baseline_mdd`.
- `grid_search`: tries every combination in `selection.grid` (e.g. `C: [0.1, 1, 10]`,
  `class_weight: ["balanced", null]`), scores each with the **median** walk-forward `max_drawdown_reduction`, logs
  every try with `tracking.log_metrics(run, ..., prefix="grid/")`, returns the best params and the full table
  (also saved to `reports/selection.csv`). The count of tries goes in the README (multiple-testing note).
- Config section `selection`: `enabled`, `block`, `gap_days`, `grid`. `main.py` uses the best params only when
  `selection.enabled` is true.
- Tests: window boundaries and the gap on a small calendar (known values); no window reaches past the train end;
  the grid tries exactly the product of the lists; best params match the highest median on a stubbed scorer;
  every raise tested (empty grid, unknown param, window with no rows).

**Acceptance tests: write these first, then the code until they pass.**

| Test | Given | Expect |
| --- | --- | --- |
| `test_unit_windows_known_values` | business days 2010 to 2014, first train end 2011-12-31, block one year, gap 5 | 3 windows; each block is one calendar year; exactly 5 trading days dropped before each block |
| `test_unit_no_window_reaches_past_the_last_end` | last end 2013-12-31 | no block date after it |
| `test_unit_train_and_block_never_overlap` | any window | max(train) < min(block), with the gap between |
| `test_unit_walk_forward_scores_one_row_per_window` | tiny synthetic table, a stub scorer | columns block_start, block_end, max_drawdown_reduction, agent_mdd, baseline_mdd |
| `test_unit_grid_tries_exactly_the_product` | grid C: [0.1, 1], class_weight: ['balanced', None], a stub scorer counting calls | 4 tries, one per combination |
| `test_unit_grid_picks_the_highest_median` | stub scores per combination | best params are the ones with the highest median score |
| `test_unit_grid_logs_every_try` | a fake run | `tracking.log_metrics` called 4 times with prefix `grid/`; with `run=None` nothing breaks |
| `test_unit_selection_raises (one per case)` | empty grid; unknown parameter name; a window with no rows | `ValueError` with a clear message |

### rl_agent.py (the reinforcement learning challenger)

```python
def collect_transitions(market_data: tuple, target: pd.Series, cfg: dict, start: str, end: str,
                        n_episodes: int, seed: int) -> pd.DataFrame
def fit_q_agent(transitions: pd.DataFrame, feature_columns: list[str], actions: list[float], params: dict,
                random_state: int) -> QAgent
class QAgent:            # predict(X) -> action per row; predict_proba(X) -> one column per action; classes_
```

- **Method: fitted Q iteration** (batch reinforcement learning, scikit-learn only). The Stage 2 classifier looks
  one week ahead; the agent learns the value of an action over many weeks: Q(state, action) = reward now +
  `gamma` x the best Q in the next state.
- `collect_transitions`: runs `portfolio.simulate` over the **train years only** with seeded random policies
  (`n_episodes` runs), and records for every decision: state, action, reward, next state, `done`. The reward is
  the README's: next window's return minus trading cost minus the penalty for new drops below the previous high.
  It comes from `portfolio.realised_rewards`, which shares its windows and its reward function with the labels.
  `done` is False on every row: the data ends, the portfolio does not.
- `fit_q_agent`: `n_iterations` rounds; each round fits one regressor on (state, action) against
  `reward + gamma * max over actions of Q(next state)`. Small model (`HistGradientBoostingRegressor`, shallow).
- `QAgent.predict` picks the action with the highest Q. `predict_proba` is a softmax of the Q values divided by
  their usual gap on the train states (`typical_gap`, set at fit time) times `rl.temperature`, so
  `infer.run_inference` and the 60% fallback work unchanged. It is a ranking score, not a calibrated
  probability (README open gap 6): say so in the model card. `fallback_share(agent, X, min_probability)` says
  how often serving would use today's rule instead of the agent: log it next to the gate result.
- README mitigations for little data: train `rl.seeds` (5) agents and report the **median**
  `max_drawdown_reduction`; the registered agent is the median one. Log every seed to W&B.
- Same gate, same simulator, same costs as the classifier: `model.type` in config is `classifier` or `fitted_q`,
  and it decides which one is the candidate for the gate (no automatic pick on validation, which would be one more
  look at validation). With `rl.enabled: true` and `model.type: classifier` the agent is trained and reported too.
  Test 2022-25 is opened once, for a model that passed.
- Config section `rl`: `enabled`, `gamma`, `n_iterations`, `n_episodes`, `seeds`, `temperature`, `estimator`
  (with `early_stopping` set explicitly).
- Tests: on a toy problem where one action always pays more, the agent learns it (known value); same seed gives
  the same agent; transitions never reach past the train end; windows join with no gaps or overlaps;
  `predict_proba` rows sum to 1; every raise tested.

**Acceptance tests: write these first, then the code until they pass.**

| Test | Given | Expect |
| --- | --- | --- |
| `test_unit_agent_learns_the_better_action_on_a_toy_problem` | hand-made transitions where action 1.0 always earns 1 and the others 0 | `predict` returns 1.0 for every state |
| `test_unit_agent_prefers_delayed_reward_with_gamma` | a two-step toy problem where action A pays 0 now and 10 next, B pays 1 now and 0 next | with gamma 0.9 the agent picks A; with gamma 0 it picks B |
| `test_unit_same_seed_same_agent` | two fits with the same seed | identical predictions; a different seed is allowed to differ |
| `test_unit_predict_proba_rows_sum_to_one` | any fitted agent | shape (rows, number of actions); each row sums to 1; `classes_` equals the configured actions |
| `test_unit_transitions_stay_inside_the_train_years` | collect with end 2019-12-31 on synthetic prices to 2021 | no decision or reward date after the end |
| `test_unit_transition_windows_join_up` | one episode | each reward window starts after its execution date and ends where the next begins (no gap, no overlap) |
| `test_integration_agent_runs_through_infer_and_simulate` | a fitted agent | `infer.run_inference(agent, X, include_proba=True)` works and `portfolio.simulate` accepts its policy |
| `test_unit_rl_raises (one per case)` | empty transitions; an action not in the configured list; gamma outside [0, 1] | `ValueError` with a clear message |

### benchmarks.py (traditional portfolios, reported)

```python
def fit_weights(train_returns: pd.DataFrame, method: str, params: dict) -> pd.Series
def run_benchmarks(market_data: tuple, cfg: dict, start: str, end: str) -> pd.DataFrame
```

- Methods (README "Benchmarks"): `min_variance`, `max_sharpe`, `mean_variance` (risk aversion from config) and
  `risk_parity`. Long only, weights sum to 1, **fitted on the train years only** (2007-09-07 to 2019).
  Use `scipy.optimize.minimize` (SLSQP); add `scipy` to `environment.yml`.
- `run_benchmarks`: each weight vector runs through the **same** `portfolio.simulate` and costs as the agent, with
  today's rule keeping it at its own weights. One row per method: weights, `max_drawdown`, annual return,
  volatility, Sharpe, trading cost. Saved to `reports/benchmarks.csv`, added to `metrics.json` under `benchmarks`,
  logged to W&B as a table.
- Config section `benchmarks`: `enabled`, `methods`, `risk_aversion`.
- Reported, never gated. Optional later (stretch): let a user pick one as the target weights in the API
  ("allocation mode"); the default stays the approved targets.
- Tests: weights are long only and sum to 1; two uncorrelated assets with equal variance give 50/50 for
  `min_variance` and `risk_parity` (known value); only train rows are used (a test that poisons later rows);
  unknown method raises; the table has one row per method.

**Acceptance tests: write these first, then the code until they pass.**

| Test | Given | Expect |
| --- | --- | --- |
| `test_unit_two_equal_uncorrelated_assets_give_half_half` | two return series with equal variance, zero correlation | `min_variance` and `risk_parity` both give 0.5 / 0.5 (tolerance 0.01) |
| `test_unit_min_variance_prefers_the_calmer_asset` | one asset with a third of the other's volatility | its weight is above 0.5 |
| `test_unit_weights_are_long_only_and_sum_to_one` | each of the four methods on random returns | every weight >= 0; the sum is 1; index in config ticker order |
| `test_unit_only_train_rows_are_used` | returns where rows after the train end are replaced by huge numbers | the weights do not change |
| `test_unit_unknown_method_raises` | `method='equal_weight'` | `ValueError` listing the four methods |
| `test_integration_run_benchmarks_one_row_per_method` | synthetic prices through the real simulator | 4 rows with max_drawdown, annual return, volatility, Sharpe, trading cost |

### rules.py (the simple-rule challengers from the business case)

```python
RULES: tuple[str, ...]     # names accepted in config: current, band_2_5, band_1_5, persistent_drift, asymmetric, matched_exposure
def make_rule_policy(name: str, target: pd.Series, decisions: pd.DatetimeIndex, close_wide: pd.DataFrame,
                     cfg: dict) -> Callable[[pd.Series], float]
def run_challengers(market_data: tuple, target: pd.Series, cfg: dict, start: str, end: str,
                    agent_daily: pd.DataFrame | None = None) -> pd.DataFrame
def select_challenger(results: pd.DataFrame, guards: dict) -> str
```

- The README's six rules, each a policy through the **same** `portfolio.simulate` and costs: today's rule
  (quarterly plus 5 points); band 2.5 points; band 1.5 points; persistent drift (at least 2 ETFs beyond 2.5 points
  on each of the last 5 trading days); asymmetric (full rebalance when drift or drawdown passes a threshold, then
  gradual steps back with a minimum wait); matched-exposure fixed mix (constant weights holding the agent's own
  average equity share, rebalanced quarterly; needs `agent_daily`).
- Every number (bands, days, thresholds, wait, which tickers are "equity") comes from a `challengers` config
  section. They are the team's assumptions (README): keep the README's values.
- `run_challengers`: one row per rule with max drawdown, return, cost, tracking error and the guard results.
  `select_challenger`: the smallest max drawdown on validation 2020-21 **among rules that pass the guards**.
  Saved to `reports/challengers.csv`, added to `metrics.json` (`challenger`), logged to W&B.
- The release decision (guards spec) then also requires the agent's max drawdown to be smaller than the selected
  rule's on test, as the README's "Do not release if" list says.
- Tests: each rule on a tiny price path with a known answer (when it trades and when it does not); the persistent
  rule needs all 5 days; selection ignores a rule that fails a guard; unknown rule name raises.

**Acceptance tests: write these first, then the code until they pass.**

| Test | Given | Expect |
| --- | --- | --- |
| `test_unit_tolerance_band_fires_only_outside_the_band` | band 2.5 points; drift 3 points, then 2 points | action 1.0, then 0.0 |
| `test_unit_calendar_fires_only_on_the_calendar_date` | quarterly; a quarter-start decision, then a mid-quarter one | 1.0, then 0.0 |
| `test_unit_persistent_drift_needs_every_day` | 2 assets beyond the band on 5 days; then on only 4 of the 5 | 1.0, then 0.0 |
| `test_unit_asymmetric_steps_back_gradually` | a breach, then calm weeks | 1.0 at the breach; later steps of 0.5 no sooner than `min_wait` apart |
| `test_unit_today_rule_equals_stage_2` | the same prices through `make_rule_policy('current', ...)` and `make_current_rule_policy` | identical actions on every decision date |
| `test_unit_select_challenger_ignores_rules_that_fail_a_guard` | a results table where the best drawdown fails a guard | the best passing rule is selected |
| `test_unit_unknown_rule_raises` | `name='momentum'` | `ValueError` listing the rule names |
| `test_integration_run_challengers_one_row_per_rule` | synthetic prices | one row per rule with drawdown, return, cost, tracking error, guard results |

### Guards and the release decision (evaluate.py)

```python
def tracking_error(portfolio_values: pd.Series, target_values: pd.Series, periods_per_year: int) -> float
def guard_report(agent: dict, baseline: dict, limits: dict) -> dict
def release_decision(gate_met: bool, guards: dict) -> dict
```

- The README's guards, reported with every run (not part of the gate): annual net return no more than
  `guards.max_return_shortfall` (0.0025) below the baseline; annual trading cost no more than
  `guards.max_cost_ratio` (2.0) times the baseline's; `tracking_error` no more than `guards.max_tracking_error`.
- `tracking_error`: annualised standard deviation of the difference between the portfolio's and the
  target-weight portfolio's weekly returns (README open gap 2). The limit X is set once from today's rule on
  validation 2020-21 and written to `config.yaml`.
- `guard_report`: one entry per guard with `value`, `limit`, `passed`; goes into `metrics.json` under `guards`.
- `release_decision`: only called for a model that passed the gate, on test 2022-25, **once**. Returns
  `release: bool` and the list of reasons (README "Do not release if"). Logged to W&B with prefix `test/`.
- Tests: known-value tracking error (identical paths give 0); each guard passes and fails at its limit; the
  decision lists every failed reason; raises on mismatched or empty series.

**Acceptance tests: write these first, then the code until they pass.**

| Test | Given | Expect |
| --- | --- | --- |
| `test_unit_tracking_error_is_zero_for_identical_paths` | the same value series twice | 0.0 |
| `test_unit_tracking_error_known_value` | weekly return differences alternating +1% and -1% | std x sqrt(52), to 6 decimals |
| `test_unit_each_guard_passes_and_fails_at_its_limit` | return shortfall exactly at 0.0025 and just beyond; cost ratio exactly 2.0 and just beyond | passed True at the limit, False beyond |
| `test_unit_guard_report_shape` | agent and baseline metric dicts | one entry per guard with value, limit, passed |
| `test_unit_release_decision_lists_every_failed_reason` | gate met, two guards failed | `release` False and both reasons named; all passed gives `release` True |
| `test_unit_release_decision_never_releases_a_failed_gate` | gate not met, guards all fine | `release` False with the gate as the reason |
| `test_unit_guards_raise (one per case)` | series of different lengths; empty series; a missing limit | `ValueError` with a clear message |

### robustness.py (the checks the business case lists)

```python
def feasibility_ceiling(market_data: tuple, target: pd.Series, cfg: dict, start: str, end: str) -> dict
def cost_sensitivity(run_policy: Callable[[dict], pd.DataFrame], cfg: dict) -> pd.DataFrame
def without_largest_episode(daily_agent: pd.DataFrame, daily_baseline: pd.DataFrame) -> dict
def bootstrap_p_value(agent_returns: pd.Series, other_returns: pd.Series, n: int, block: int, seed: int) -> float
def holm_adjust(p_values: dict[str, float]) -> dict[str, float]
```

- `feasibility_ceiling` (README open gap 7): a timing-only policy with perfect foresight (it picks the action the
  label says was best) on 2007 to 2021. Its `max_drawdown_reduction` is the ceiling: what timing alone could
  ever achieve. Reported next to the gate target; the README says the KPI target is set from it.
- `cost_sensitivity`: rerun the agent and today's rule with the realistic cost model already in `costs.py`
  (exits cost more than entries, plus market impact). Reports the gate metric under both cost models.
- `without_largest_episode`: the test-period result again with the largest drawdown episode removed; if the pass
  depends on that one episode, the result is reported as **fragile**.
- `bootstrap_p_value` and `holm_adjust`: a seeded block bootstrap on weekly returns for "the agent is better than
  X", one p-value per challenger comparison, then the Holm correction across them. The number of agent versions
  tried (grid search, seeds, frequencies) is logged beside it.
- Output: `reports/robustness.json`, a W&B table, and a section in the model card. Reported, never gated. A
  12-month shadow period with no drawdown is "inconclusive" (model card text). The optional VIX input is future work.
- Tests: Holm on a textbook example (known values); bootstrap is reproducible per seed and near 0.5 for two
  identical series; the ceiling is at least as good as today's rule on a toy path; episode removal on a path with
  one obvious crash; every raise tested.

**Acceptance tests: write these first, then the code until they pass.**

| Test | Given | Expect |
| --- | --- | --- |
| `test_unit_holm_textbook_example` | p-values 0.01, 0.04, 0.03 for three comparisons | adjusted 0.03, 0.06, 0.06 |
| `test_unit_bootstrap_is_reproducible` | two calls with the same seed | the same p-value |
| `test_unit_bootstrap_near_half_for_identical_series` | the same return series twice | p-value between 0.4 and 0.6 |
| `test_unit_bootstrap_small_for_a_clearly_better_series` | one series 1% higher every week | p-value below 0.05 |
| `test_unit_ceiling_is_at_least_todays_rule` | a toy price path | the perfect-foresight reduction is >= 0 |
| `test_unit_largest_episode_removed` | a path with one obvious crash | the crash dates are excluded and `fragile` is reported when the pass depends on them |
| `test_unit_cost_sensitivity_two_rows` | a stub policy runner | one row for `flat` and one for `realistic` costs |
| `test_unit_robustness_raises (one per case)` | empty p-values; a p-value outside [0, 1]; series of different lengths | `ValueError` with a clear message |

### allocation.py (the two choices a user makes)

```python
def resolve_target(method: str, cfg: dict, train_returns: pd.DataFrame | None) -> pd.Series
def target_explanation(target: pd.Series, cfg: dict, train_returns: pd.DataFrame | None) -> dict
    # {"fallback": bool, "reason": str, "expected_tracking_error": float}; train_returns needed for any
    # target that is not the policy mix (ex-ante tracking error to the policy weights)
def resolve_policy(method: str, model: Pipeline | None, target: pd.Series, decisions: pd.DatetimeIndex,
                   close_wide: pd.DataFrame, cfg: dict) -> Callable[[pd.Series], float]
def explain(state_row: pd.Series, action: float, fallback: bool, cfg: dict) -> str
```

- Two choices, in the words a portfolio manager uses.
  **Target weights** (`target_weights.method`): `policy` uses the fund's own mix listed in
  `target_weights.policy`; `min_variance`, `max_sharpe`, `mean_variance` and `risk_parity` calculate one with
  `benchmarks.py`, fitted on the train years.
  **Rebalancing** (`rebalancing.method`): `smart` (our model or RL agent decides at each review: leave it, go
  halfway, go fully back); `calendar` (fully, on the schedule in `rebalancing.calendar`); `tolerance_band`
  (fully, when any holding is more than `rebalancing.tolerance_band` percentage points from target);
  `calendar_and_band` (both: today's process is this with `quarterly` and `5`).
- We never call this "tactical": tactical asset allocation means moving *away* from the targets on a market
  view. The tool only decides how to return to them. The README's two unusual rules (persistent drift,
  asymmetric) are challengers we test against (`rules.py`), not choices a user is offered.
- The API takes `target_method` and `rebalancing_method` as optional fields (review frequency is fixed when the
  model is trained, so it is not a request field), plus a free-text `portfolio_id` that is echoed back: an
  allocator usually runs one portfolio or one sleeve through the tool, not its whole book.
- `explain`: one plain sentence with every answer, for example "Rebalance fully: SPY is 6.1 points above target,
  outside the 5-point band" or "No trade: every holding is within 2 points of target".
- Honest limits, enforced in code and written in the model card: the model is trained for the target set in
  config. With a calculated target the request falls back to the calendar-and-band rule (`fallback: true`,
  reason given). Calculated-target serving uses the rule even after research retraining with that target.
  A calculated target changes the mandate the Investment Committee approved. The API reason reports
  the fixed training-period tracking error to the policy weights stored at refit time.
  The API and pipeline load reviewed weights from `target_weights.calculated`, refitted at each retrain.
- **Three places catch a wrong value, all from `config.ALLOWED_VALUES`:** `load_config` at start-up; the API's
  request model (enumerations, so a wrong value is a 422 that lists the allowed ones and `/docs` shows a
  dropdown); and `resolve_target` / `resolve_policy`, which raise `ValueError` with the allowed list.
- No new maths: `resolve_target` calls `benchmarks.fit_weights`; `resolve_policy` builds the calendar and band
  rules from `rules.make_rule_policy` or wraps the model.
- Tests: defaults (`policy`, `smart`) reproduce the Stage 2 behaviour (regression); each target method returns
  long-only weights that sum to 1 in config ticker order; each rebalancing method returns a callable policy;
  `calendar_and_band` with `quarterly` and `5` equals today's rule on the same prices; unknown names raise with
  the allowed list; `explain` on three known states.

**Acceptance tests: write these first, then the code until they pass.**

| Test | Given | Expect |
| --- | --- | --- |
| `test_unit_defaults_reproduce_stage_2` | `policy` and `smart` | target equals `target_weights.policy`; the policy is the model's |
| `test_unit_each_calculated_target_is_valid` | each of the four calculated methods on synthetic returns | long only, sums to 1, config ticker order |
| `test_unit_calendar_and_band_equals_todays_rule` | `calendar_and_band` with `quarterly` and 5 | the same actions as today's rule on the same prices |
| `test_unit_each_rebalancing_method_returns_a_policy` | each of the four methods | a callable that returns a float in [0, 1] |
| `test_unit_unknown_names_raise_with_the_allowed_list` | `min_risk`, then `smrt` | `ValueError` listing the allowed values from `config.ALLOWED_VALUES` |
| `test_unit_explain_three_known_states` | a band breach; all within band; a fallback | sentences naming the ticker and its drift, 'no trade', and the fallback reason |
| `test_unit_calculated_target_falls_back_to_the_rule` | a calculated target with a model trained on the policy mix | fallback True and a reason; tracking error to the policy weights reported |

### api.py (the service)

```python
app = FastAPI(title="ETF rebalancing advisor", version=APP_VERSION)
@app.get("/health")   -> {"status": "ok", "model_loaded": bool, "model_version": str, "policy": "model" | "rule",
                          "version": str, "commit": str}
# version: APP_VERSION (default "dev"); commit: APP_COMMIT, baked into the release image
# (default "local"). model_version is "rule" when no model is loaded, so the deploy checks commit, not it.
@app.post("/predict") -> PredictResponse
```

- Request (decision **D9**): `as_of`, `positions` (list of `{ticker, market_value}`), `drawdown`,
  `weeks_since_rebalance`, `prices` (records with the `data.columns` names, at least `api.min_history_days`
  trading days). Field checks in the pydantic models: only tickers from `config.yaml`, values not negative, total
  above 0, `drawdown <= 0`. `as_of` can be any trading day: a request on a day that is not a Friday is
  treated as that week's decision (the state is built at that day's close).
- Response: `as_of`, `prediction`, `probability`, `recommendation` (0.0, 0.5 or 1.0), `fallback`, `rule_action`,
  `model_version`, `reason` (one plain sentence), `proposed_portfolio` (ticker, weight, value) and `trades`
  (ticker, side, value).
- Flow, **no new ML logic**: `trades.positions_to_weights`, then `clean_data.clean_dataframe`,
  `validate.validate_dataframe`, `clean_data.to_wide`, `features.compute_market_features`,
  `portfolio.live_state`, `infer.run_inference`, `infer.apply_fallback`, `trades.proposed_portfolio`,
  `trades.trade_list`. A `ValueError` from any step becomes HTTP 400 with its message.
- Startup (lifespan): `config.load_config`, `config.load_env`, `logger.setup_logging`; model from `MODEL_SOURCE`
  (`local`: `utils.load_model(paths.model)`; `wandb`: `tracking.download_model`). No model: rule-only mode
  (D8, D12), `/health` still 200 with `policy: "rule"`.
- Logs one INFO line per request (as_of, recommendation, fallback, latency in ms). Never logs the whole payload.
- Tests (`tests/test_api.py`, FastAPI `TestClient`, offline): `/health` with and without a model; `/predict`
  returns the D9 fields; recommendation in {0.0, 0.5, 1.0}; trades sum to zero net value; missing field gives 422;
  unknown ticker gives 422; too little history gives 400 with the message; rule-only mode answers with
  `fallback: true`. Build the model in a fixture from tiny synthetic prices (no real data, no network).

**Acceptance tests: write these first, then the code until they pass.**

| Test | Given | Expect |
| --- | --- | --- |
| `test_integration_health_with_and_without_a_model` | the app started with a tiny fitted model, then with none | 200 both times; `model_loaded` true then false; `policy` `model` then `rule` |
| `test_integration_predict_returns_the_contract` | the sample request built from synthetic prices | 200 with every D9 field; `recommendation` in {0.0, 0.5, 1.0}; `reason` is a non-empty sentence |
| `test_integration_trades_net_to_zero` | the same request | buys minus sells is 0; proposed weights sum to 1 |
| `test_integration_missing_field_is_422` | the request without `positions` | 422 naming the field |
| `test_integration_unknown_ticker_is_422` | a position in a ticker not in the config | 422 naming the ticker |
| `test_integration_too_little_history_is_400` | 10 days of prices | 400 with the message from the pipeline |
| `test_integration_wrong_choice_is_422` | `rebalancing_method: 'smrt'` | 422 listing the allowed values |
| `test_integration_rule_only_mode_answers` | no model loaded | 200 with `fallback` true and `model_version` `rule` |
| `test_unit_no_payload_in_the_logs` | one request with `caplog` | the log line has as_of, recommendation, latency; no prices, no positions |

### Monitoring (Raja, after the API)

```python
def log_prediction_batch(run: Run | None, rows: list[dict]) -> None    # tracking.py, small addition
```

- `api.py` keeps the last `api.monitoring_batch_size` answers in memory; when the batch is full it sends them to
  W&B as one table (`tracking.log_prediction_batch`) under a `serve` run, then clears the batch. Tracking off:
  nothing happens. Monitoring must never break serving (catch and log, never raise to the client).
- Logged per row: `as_of`, `recommendation`, `probability`, `fallback`, `model_version`, latency.
- README section "Monitoring": what is logged, what would trigger a retrain (fallback rate above 50% for 4 weeks,
  or the live drawdown reduction below 0 over a quarter), and that no personal data is involved.
- Tests: a batch is sent exactly when full and then cleared; a tracking error is logged, not raised; with
  tracking disabled nothing is called.

**Acceptance tests: write these first, then the code until they pass.**

| Test | Given | Expect |
| --- | --- | --- |
| `test_unit_batch_is_sent_when_full_and_cleared` | `monitoring_batch_size` 3 and three requests | `tracking.log_prediction_batch` called once with 3 rows; the buffer is empty after |
| `test_unit_no_send_before_the_batch_is_full` | two requests | not called |
| `test_unit_tracking_error_never_reaches_the_client` | `log_prediction_batch` raising | the request still returns 200 and the error is logged |
| `test_unit_tracking_disabled_calls_nothing` | `run=None` | no call and no error |

### scripts/smoke_test.py and scripts/sample_request.json (Raja)

```text
python scripts/smoke_test.py --url http://localhost:8000 [--expect-version v1.1.0] [--expect-commit SHA] [--timeout 300]
```

- **Standard library only** (`urllib`, `json`, `argparse`, `time`): runs anywhere, no installs.
- Polls `GET /health` every 10 s until `status == "ok"` (and `version` equals `--expect-version`, and
  `commit` matches `--expect-commit` as a short or full SHA, when given) or the timeout ends. Then `POST /predict` with `scripts/sample_request.json` and checks the D9 response
  fields and that `recommendation` is 0.0, 0.5 or 1.0. Exit 0 if all good, 1 otherwise, with one clear line why.
- `scripts/make_sample_request.py` builds `sample_request.json` from the synthetic prices (last 70 trading days,
  target weights, drawdown 0, weeks 0), so the sample never contains the real snapshot.
- Tests (`tests/test_smoke_test.py`): run the checks against a tiny local `http.server` stub (healthy, unhealthy,
  wrong version, bad response) or patch `urllib.request.urlopen`; each exit path tested.

**Acceptance tests: write these first, then the code until they pass.**

| Test | Given | Expect |
| --- | --- | --- |
| `test_unit_healthy_service_exits_0` | a stub server: healthy `/health`, valid `/predict` | exit code 0 |
| `test_unit_unhealthy_until_timeout_exits_1` | `/health` always 503, timeout 1 s | exit code 1 and one line saying health never became ok |
| `test_unit_wrong_version_exits_1` | `--expect-version v9.9.9` against `/health` `version` `dev` | exit code 1 naming both versions |
| `test_unit_wrong_commit_exits_1` | `--expect-commit` of another SHA | exit code 1 naming both commits |
| `test_unit_bad_predict_response_exits_1` | `/predict` without `recommendation`, then with 0.7 | exit code 1 each time, naming the problem |
| `test_unit_only_standard_library_is_imported` | the script's imports (parse it with `ast`) | every import is in `sys.stdlib_module_names` |
| `test_unit_sample_request_is_valid` | `scripts/sample_request.json` | it validates against the API's request model and contains no real prices |

### Dockerfile, .dockerignore, conda-lock.yml (Tate)

- `environment.yml` adds: `fastapi`, `uvicorn`, `pydantic`, `httpx` (test client), `python-dotenv`, `wandb`,
  `conda-lock`. Then `conda-lock -f environment.yml -p linux-64` writes `conda-lock.yml`. Change one, regenerate
  and commit both.
- `Dockerfile`: pinned micromamba base image, install from `conda-lock.yml`, copy `config.yaml` and `src/` only,
  `EXPOSE 8000`, `HEALTHCHECK` that calls `/health` with Python's `urllib`, start with
  `uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}` (a host may set `PORT`).
- `.dockerignore`: allow-list (ignore everything, then allow `config.yaml`, `conda-lock.yml`, `src/`). `.env`,
  `data/`, `models/`, `tests/`, `notebooks/` never enter the image.
- Check: `docker build -t etf-advisor .`, `docker run -p 8000:8000 -e MODEL_SOURCE=local etf-advisor`, then
  `python scripts/smoke_test.py --url http://localhost:8000` (rule-only mode is fine without a model).
- README: build and run commands, image size.

**Done checks (run each one and paste the output in the PR):**

- `docker build -t etf-advisor .` succeeds from a clean clone
- `docker run --rm etf-advisor ls -a /app` shows only `config.yaml`, `conda-lock.yml`, `src/` (no `.env`, `data/`, `tests/`)
- `docker run -p 8000:8000 -e MODEL_SOURCE=local etf-advisor` then `curl localhost:8000/health` returns 200
- `docker inspect` shows a HEALTHCHECK; the image size is in the PR
- `conda-lock.yml` was regenerated from the current `environment.yml` (the lock's content hash matches)

### ci.yml extended (Miguel)

Steps on every PR into `dev` and `main` (and pushes to them): checkout, flake8, environment from
`conda-lock.yml` (and a check that the lock matches `environment.yml`), `python -m scripts.make_sample_prices`,
`python -m src.main` with `WANDB_MODE=disabled`, `pytest --cov=src --cov-branch --cov-fail-under=90`,
`radon cc src scripts -s -n C` must print nothing, `docker build`. Triggers also include `merge_group` (the dev
merge queue) and `workflow_call` (deploy.yml reruns this CI on the release commit).
No secrets in CI. The job name stays `lint-and-test` (it is the required check in the rulesets).

**Done checks (run each one and paste the output in the PR):**

- The job is still called `lint-and-test` and is green on the PR itself
- A deliberate flake8 error on a scratch commit turns it red, then revert
- The log shows: lint, sample prices generated, `python -m src.main` exit 0, pytest with the coverage floor, radon printing nothing, `docker build`
- No secret is referenced anywhere in `ci.yml`

### main.py wiring (Miguel)

- `config.load_config`, `config.load_env`, `logger.setup_logging` at the start; `tracking.start_run`.
- After cleaning: `tracking.log_dataset(run, processed clean.csv, "etf-prices")`.
- `selection.enabled`: `selection.grid_search` on the train years first, its best params go to `train_model`.
- Gate met: test audit once on 2022-25 (logged with prefix `test/`, report only), `utils.save_model`,
  `tracking.log_model`. Gate not met: D8 (no model saved or registered; predictions from today's rule).
- Any error: `tracking.finish_run(run, failed=True)`, log it, exit 1. Success: `finish_run(run)`, exit 0.

**Done checks (run each one and paste the output in the PR):**

- `test_integration_pipeline_*` still really runs `main()` offline
- `test_unit_gate_met_saves_and_registers`: gate patched to pass: model file written, `tracking.log_model` called once, test audit logged with prefix `test/`
- `test_unit_gate_not_met_saves_nothing`: gate patched to fail: no model file, `log_model` never called, predictions all `fallback` true, exit code 0
- `test_unit_error_marks_the_run_failed`: a step raising `ValueError`: `finish_run(run, failed=True)` and exit code 1
- `test_unit_main_never_imports_wandb`: parse `src/main.py` with `ast`: no `wandb` import

### deploy.yml (Miguel)

Runbook: `docs/DEPLOY.md` (set-up, dry run, release, rollback). This section is the contract only.

- Trigger: GitHub Release **published** (and `workflow_dispatch` from a tag, to roll back; from `dev` for the dry run).
- Flow: release -> CI on the released commit (`uses: ./.github/workflows/ci.yml`) -> one `docker build` with the
  commit and version baked in -> push `ghcr.io/<org>/etf-rebalancing-api:<sha>`, `:<tag>` and `:release` -> the host
  timer pulls `:release`, canary-tests it on a side port and swaps it in when `/health` is ok ->
  `python3 scripts/smoke_test.py --url "$SERVICE_URL" --expect-commit "$GITHUB_SHA" --timeout 900`.
  The commit, not the tag: the host keeps the old container serving until the new one is healthy, and
  `model_version` is `"rule"` while no model passes the gate.
- No deploy secret: the workflow pushes with its own `GITHUB_TOKEN`; `SERVICE_URL` is a repository variable.
- Rollback: run the workflow from the previous tag (it re-points `:release` to that tag's existing image), or move
  the W&B `prod` alias back and restart.

**Done checks (run each one and paste the output in the PR):**

- A manual run (`workflow_dispatch`) deploys and the smoke-test step passes
- A deliberately wrong `SERVICE_URL` (or `--expect-commit`) makes the workflow fail (then put it back)
- The workflow log shows no secret value (there is none to show: only `GITHUB_TOKEN` is used)
- Rollback steps in the README were followed once by someone who did not write them

### README model card and Stage 3 docs (Ricardo A.)

- **Model card** section (course outcome 6, for business readers): purpose and users; data (source, dates,
  snapshot checksum); model and features; the gate and the honest result; walk-forward results per year; known
  limits (COVID in validation, about 950 weekly decisions, flat costs); what triggers a retrain; who owns it.
- Run and serve: install (conda-lock), get the data, `python -m src.main`, `uvicorn src.api:app`, `/docs`,
  Docker, the live URL, a `curl` example with `sample_request.json`, W&B project link.
- Business case section 1: Stage 3 update (what the service changes for the desk, cost of running it).
- Keeps README, `TEAM_BRIEF.md` and `config.yaml` agreeing on the gate (graded since Stage 2).
- The business case in section 1 is **not rewritten**: it was submitted. Stage 3 adds results next to its
  commitments (RL agent, challengers, benchmarks, guards, release decision, robustness) and closes open gaps.
- **Future work** section, written so the next developer can start from it: any universe or whole book (why the
  per-ticker inputs block it, and the asset-agnostic design that removes it), continuous training, rolling
  target weights, a stronger agent, named portfolios. One paragraph each: what, why not now, first step.

**Done checks (run each one and paste the output in the PR):**

- A teammate who did not write it follows install, run, serve, Docker and the `curl` example on their own machine, and every command works
- Gate metric, threshold and split agree across README, `TEAM_BRIEF.md`, `config.yaml` (grep for the three)
- Every Stage 3 commitment in section 1 has a result or a plain 'not met' next to it
- The Future work section has what, why not now and a first step for each item

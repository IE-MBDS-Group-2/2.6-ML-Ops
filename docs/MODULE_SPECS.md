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
  -> main._split_by_date                (train 2007-09 to 2019 / validation 2020-21 / test 2022-25 sealed)
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
def clean_dataframe(df_raw: pd.DataFrame, target_column: str | None = None) -> pd.DataFrame   # contract
def to_wide(df: pd.DataFrame, value_column: str, tickers: list[str]) -> pd.DataFrame          # ours
```

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
                       tickers: list[str] | None = None, max_abs_daily_return: float | None = None) -> bool  # contract + extras
```

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
def split_data(X, y, test_size: float, val_size: float, random_state: int, stratify: bool) -> tuple  # contract, kept and tested
def _split_by_date(table: pd.DataFrame, train_end: str, val_end: str) -> tuple                      # used by the pipeline
def main() -> None                                                                                   # contract
```

- It loads `config.yaml` once and fails fast on a missing section (`KeyError`). It configures logging from the config.
- It runs the flow above, logging `Quality gate met` or `NOT met`, with `max_drawdown_reduction` on validation.
- It writes `data/processed/clean.csv`, `models/model.joblib`, `reports/predictions.csv` (decision date first) and
  `reports/metrics.json`.
- It catches `ValueError`, `OSError` and `KeyError`, logs the cause and the next action, and exits 1.

Tests:
- `split_data` sizes with no overlap.
- `_split_by_date` boundaries.
- A missing raw file exits 1.
- **Pipeline test:** synthetic prices for 5 tickers over about 3 years in `tmp_path`, with config paths and dates
  patched. Then the 4 outputs exist, the model reloads, and there's one prediction row per inference decision date.

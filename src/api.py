"""
api.py  [Stage 3: THE SERVICE]
==============================
Purpose   Answer one portfolio's question live: given its positions and recent prices, rebalance or not, and
          which trades get there
Input     POST /predict: as_of, positions, drawdown, weeks_since_rebalance, prices (D9), and the optional
          choices target_method, rebalancing_method, portfolio_id
Output    The recommendation (0.0, 0.5 or 1.0), whether today's rule decided (fallback), one plain-words reason,
          the proposed portfolio and the trade list; GET /health says whether a model is loaded
Run       uvicorn src.api:app   (then open /docs)
Calls     trades, clean_data, validate, features, portfolio.live_state, allocation, infer, tracking, utils

Why it matters
- No new ML logic: every step is a call to the module the training pipeline uses, so a live answer and the
  backtest cannot drift apart
- No model past the gate (D8, D12): the service still starts and answers with today's rule (fallback true)
- Monitoring: answers go to W&B in batches (api.monitoring_batch_size) under a serve run; a tracking error is
  logged, never raised, so monitoring can never break serving
- Wrong input is the client's to fix: 422 for a field that breaks the request model (with the allowed values),
  400 with the pipeline's own message for data the pipeline cannot use; payloads are never logged

Tests     tests/test_api.py
"""

import logging
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from enum import Enum
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, create_model, model_validator
from sklearn.pipeline import Pipeline

from src import allocation, clean_data, config, features, infer
from src import logger as log_setup
from src import portfolio, tracking, trades, utils, validate

logger = logging.getLogger(__name__)

# Read once at import: the request model needs the tickers and column names before the app starts.
CFG = config.load_config(config.CONFIG_PATH)
RULE_VERSION = "rule"          # model_version when no model is loaded
POLICY_MODEL, POLICY_RULE = "model", "rule"
SMART = "smart"                # the rebalancing method where the model decides
NO_MODEL_REASON = "No model has passed the quality gate, so today's rule decides."
MS_PER_SECOND = 1000


def _choices(name: str, values: list[str] | tuple[str, ...]) -> type[Enum]:
    """An enumeration of allowed strings, so a wrong value is a 422 listing them and /docs shows a dropdown.

    Inputs:  name, the enumeration's name in the OpenAPI schema; values, the allowed strings.
    Outputs: a str Enum whose members are the values.
    Why:     the allowed lists come from config (config.ALLOWED_VALUES, data.tickers), never from this file.
    """
    return Enum(name, {value: value for value in values}, type=str)


Ticker = _choices("Ticker", CFG["data"]["tickers"])
TargetMethod = _choices("TargetMethod", config.ALLOWED_VALUES[allocation.TARGET_KEY])
RebalancingMethod = _choices("RebalancingMethod", config.ALLOWED_VALUES[allocation.REBALANCING_KEY])

_COLUMNS = CFG["data"]["columns"]
PriceRecord = create_model(
    "PriceRecord",
    **{
        _COLUMNS["date_column"]: (date, ...),
        _COLUMNS["ticker_column"]: (Ticker, ...),
        _COLUMNS["close_column"]: (float, ...),
        _COLUMNS["volume_column"]: (float, ...),
    },
)


class Position(BaseModel):
    """One holding: an ETF from config.yaml and its market value (0 or more)."""

    ticker: Ticker
    market_value: float = Field(ge=0)


class PredictRequest(BaseModel):
    """The D9 request: one portfolio's positions and recent prices, plus the user's two optional choices."""

    as_of: date
    positions: list[Position] = Field(min_length=1)
    drawdown: float = Field(le=0, description="value / running peak - 1; 0 at the peak")
    weeks_since_rebalance: int = Field(ge=0)
    prices: list[PriceRecord] = Field(min_length=1)
    target_method: TargetMethod = TargetMethod(CFG["target_weights"]["method"])
    rebalancing_method: RebalancingMethod = RebalancingMethod(CFG["rebalancing"]["method"])
    portfolio_id: str | None = None

    @model_validator(mode="after")
    def _total_above_zero(self) -> "PredictRequest":
        """Reject positions that add up to nothing: there is no portfolio to rebalance."""
        if sum(position.market_value for position in self.positions) <= 0:
            raise ValueError("positions add up to a total market value of 0. Send at least one holding above 0.")
        return self


class Holding(BaseModel):
    """One row of the proposed portfolio."""

    ticker: str
    weight: float
    value: float


class Trade(BaseModel):
    """One order to place."""

    ticker: str
    side: str
    value: float
    weight_before: float
    weight_after: float


class PredictResponse(BaseModel):
    """The D9 answer."""

    as_of: date
    portfolio_id: str | None
    prediction: float | None
    probability: float | None
    recommendation: float
    fallback: bool
    rule_action: float
    model_version: str
    reason: str
    target_method: str
    rebalancing_method: str
    proposed_portfolio: list[Holding]
    trades: list[Trade]


class HealthResponse(BaseModel):
    """GET /health: the service is up, which model serves, and which release and commit are deployed."""

    status: str
    model_loaded: bool
    model_version: str
    policy: str
    version: str
    commit: str


def _model_source(cfg: dict, env: dict[str, str]) -> tuple[Path, str]:
    """Where the model file is, and the version name /health reports for it.

    Inputs:  cfg, the full config (paths.model, tracking.project, tracking.model_artifact); env, from
             config.load_env (MODEL_SOURCE, WANDB_MODEL_ALIAS).
    Outputs: (path of the .joblib file, version name).
    Raises:  FileNotFoundError from tracking.download_model when no W&B version has the alias.
    """
    if env["MODEL_SOURCE"] == "wandb":
        name, alias = cfg["tracking"]["model_artifact"], env["WANDB_MODEL_ALIAS"]
        return tracking.download_model(env, cfg["tracking"]["project"], name, alias), f"{name}:{alias}"
    path = config.resolve_paths(cfg)["model"]
    return path, f"local:{path.name}"


def load_serving_model(cfg: dict, env: dict[str, str]) -> tuple[Pipeline | None, str]:
    """Load the model to serve, or start in rule-only mode when there is none.

    Inputs:  cfg, the full config; env, from config.load_env.
    Outputs: (model, version name), or (None, "rule") when no model file or W&B alias exists.
    Why:     D8 and D12: a model that failed the gate is never saved, so the service must still start and answer
             with today's rule.
    """
    try:
        path, version = _model_source(cfg, env)
        model = utils.load_model(path)
    except FileNotFoundError as error:
        logger.warning("No model to serve (%s). Rule-only mode: today's rule decides.", error)
        return None, RULE_VERSION
    logger.info("Serving model %s", version)
    return model, version


class PredictionMonitor:
    """Keeps the last answers in memory and sends them to W&B as one table when the batch is full.

    Why: monitoring (README "Monitoring") without one W&B call per request; and it can never break serving,
    because every tracking error is caught and logged, never raised to the client.
    """

    def __init__(self, run: tracking.wandb.Run | None, batch_size: int) -> None:
        """Inputs: run, the W&B serve run or None (tracking off: nothing is kept or sent); batch_size, rows per
        table (api.monitoring_batch_size)."""
        self.run = run
        self.batch_size = batch_size
        self.rows: list[dict] = []
        self._lock = threading.Lock()   # requests run in a thread pool

    def record(self, row: dict) -> None:
        """Add one answer; send and clear the batch once it holds batch_size rows."""
        if self.run is None:
            return
        with self._lock:
            self.rows.append(row)
            if len(self.rows) < self.batch_size:
                return
            batch, self.rows = self.rows, []
        self._send(batch)

    def flush(self) -> None:
        """Send whatever is left (at shutdown), so the last partial batch is not lost."""
        with self._lock:
            batch, self.rows = self.rows, []
        if batch:
            self._send(batch)

    def _send(self, batch: list[dict]) -> None:
        """Log one batch to W&B; a tracking error is logged, never raised."""
        try:
            tracking.log_prediction_batch(self.run, batch)
        except tracking.TRACKING_ERRORS as error:
            logger.error("Monitoring: could not send %d predictions to W&B (%s). Serving continues.",
                         len(batch), error)


def start_monitoring(cfg: dict, env: dict[str, str]) -> PredictionMonitor:
    """Start the W&B serve run (or none when tracking is off) and the prediction buffer.

    Inputs:  cfg, the full config (tracking, api.monitoring_batch_size); env, from config.load_env.
    Outputs: the monitor; with tracking off, or when W&B cannot be reached, a monitor that does nothing.
    Why:     a W&B outage must never stop the service from starting.
    """
    try:
        run = tracking.start_run(cfg, env, job_type="serve")
    except tracking.TRACKING_ERRORS as error:
        logger.error("Monitoring off: could not start the W&B serve run (%s). Serving continues.", error)
        run = None
    return PredictionMonitor(run, cfg["api"]["monitoring_batch_size"])


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Start-up: read .env, set up logging, load the model (or go rule-only), start monitoring; shutdown: send
    the last batch and close the W&B run.

    Inputs:  application, the FastAPI app; its state gets env, model, model_version and monitor.
    Outputs: yields once the service is ready.
    Why:     the model is loaded once, not per request.
    """
    env = config.load_env()
    log_setup.setup_logging(CFG["logging"]["level"])
    application.version = env["APP_VERSION"]
    application.state.env = env
    application.state.model, application.state.model_version = load_serving_model(CFG, env)
    application.state.monitor = start_monitoring(CFG, env)
    yield
    application.state.monitor.flush()
    try:
        tracking.finish_run(application.state.monitor.run)
    except tracking.TRACKING_ERRORS as error:
        logger.error("Monitoring: could not close the W&B serve run (%s).", error)


app = FastAPI(title="ETF rebalancing advisor", version=config.ENV_DEFAULTS["APP_VERSION"], lifespan=lifespan)


def _positions_frame(body: PredictRequest, cfg: dict) -> pd.DataFrame:
    """The request's positions as a table with the positions column names from config.yaml."""
    names = cfg["positions"]
    return pd.DataFrame({
        names["ticker_column"]: [position.ticker.value for position in body.positions],
        names["value_column"]: [position.market_value for position in body.positions],
    })


def _check_history(close_wide: pd.DataFrame, as_of: date, min_days: int) -> None:
    """Raise unless the prices end on as_of and cover at least min_days trading days.

    Inputs:  close_wide (date x ticker), already cut at as_of; as_of, the request date; min_days,
             api.min_history_days.
    Outputs: None.
    Raises:  ValueError saying what to send instead.
    """
    if close_wide.empty or close_wide.index[-1] != pd.Timestamp(as_of):
        raise ValueError(f"No prices for every ticker on as_of {as_of}. Send prices up to and including as_of.")
    if len(close_wide) < min_days:
        raise ValueError(f"Only {len(close_wide)} trading days of prices up to {as_of}. Send at least {min_days} "
                         "(api.min_history_days): the longest return feature needs them.")


def market_inputs(body: PredictRequest, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The request's prices through the training pipeline's own steps: clean, validate, wide, features.

    Inputs:  body, the request; cfg, the full config (data, features, api).
    Outputs: (close_wide, volume_wide, market_features), ending on as_of.
    Raises:  ValueError from validate, to_wide, features or the history check.
    """
    cols, tickers = cfg["data"]["columns"], cfg["data"]["tickers"]
    prices = pd.DataFrame([record.model_dump(mode="json") for record in body.prices])
    clean = clean_data.clean_dataframe(prices, date_column=cols["date_column"],
                                       sort_by=[cols["date_column"], cols["ticker_column"]])
    clean = clean[clean[cols["date_column"]] <= pd.Timestamp(body.as_of)]
    validate.validate_dataframe(clean, cfg["data"]["required_columns"], tickers=tickers,
                                max_abs_daily_return=cfg["data"]["max_abs_daily_return"], price_columns=cols)
    close_wide = clean_data.to_wide(clean, cols["close_column"], tickers, cols["date_column"], cols["ticker_column"])
    volume_wide = clean_data.to_wide(clean, cols["volume_column"], tickers, cols["date_column"],
                                     cols["ticker_column"]).reindex(close_wide.index)
    _check_history(close_wide, body.as_of, cfg["api"]["min_history_days"])
    market = features.compute_market_features(close_wide, cfg["features"]["return_weeks"],
                                              cfg["features"]["vol_days"])
    return close_wide, volume_wide, market


def _model_decision(model: Pipeline, state: pd.DataFrame, cfg: dict) -> dict:
    """The model's answer for the live state, with the 60% fallback to today's rule (infer.py)."""
    columns = [name for name in state.columns if name != portfolio.RULE_ACTION_COLUMN]
    inputs = state[columns]
    predictions = infer.run_inference(model, inputs, include_proba=True)
    answer = infer.apply_fallback(predictions, state[portfolio.RULE_ACTION_COLUMN], inputs.isna().any(axis=1),
                                  cfg["fallback"]["min_probability"]).iloc[0]
    recommendation, fallback = float(answer["recommendation"]), bool(answer["fallback"])
    return {"prediction": float(answer["prediction"]), "probability": float(answer["probability"]),
            "recommendation": recommendation, "fallback": fallback,
            "reason": allocation.explain(state.iloc[0], recommendation, fallback, cfg)}


def _rule_decision(method: str, state: pd.DataFrame, close_wide: pd.DataFrame, target: pd.Series,
                   cfg: dict) -> dict:
    """The chosen rule's answer; for smart with no model loaded, today's rule as the fallback (D8)."""
    row = state.iloc[0]
    decisions = portfolio.decision_dates(close_wide.index, str(close_wide.index[0].date()), str(row.name.date()))
    policy = allocation.resolve_policy(method, None, target, decisions, close_wide, cfg)
    action = float(policy(row.drop(portfolio.RULE_ACTION_COLUMN)))
    fallback = method == SMART
    reason = allocation.explain(row, action, False, cfg)
    return {"prediction": None, "probability": None, "recommendation": action, "fallback": fallback,
            "reason": f"{NO_MODEL_REASON} {reason}" if fallback else reason}


def recommend(body: PredictRequest, model: Pipeline | None, model_version: str, cfg: dict) -> PredictResponse:
    """The whole /predict flow, calling only existing modules.

    Inputs:  body, the validated request; model, the loaded model or None; model_version, its name;
             cfg, the full config.
    Outputs: the D9 response.
    Why:     kept apart from the route so it can be tested and timed without HTTP.
    Raises:  ValueError from any step (the route turns it into a 400 with the message).
    """
    names = cfg["positions"]
    weights, total = trades.positions_to_weights(_positions_frame(body, cfg), cfg["data"]["tickers"],
                                                 names["ticker_column"], names["value_column"])
    prices = market_inputs(body, cfg)
    target = allocation.resolve_target(body.target_method.value, cfg, None)
    state = portfolio.live_state(*prices, weights, body.drawdown, body.weeks_since_rebalance, target, cfg)
    method = body.rebalancing_method.value
    if method == SMART and model is not None:
        decision = _model_decision(model, state, cfg)
    else:
        decision = _rule_decision(method, state, prices[0], target, cfg)
    proposed = trades.proposed_portfolio(weights, target, decision["recommendation"], total)
    orders = trades.trade_list(weights, proposed, total, cfg["portfolio"]["min_trade"])
    return PredictResponse(
        as_of=body.as_of, portfolio_id=body.portfolio_id, **decision,
        rule_action=float(state[portfolio.RULE_ACTION_COLUMN].iloc[0]), model_version=model_version,
        target_method=body.target_method.value, rebalancing_method=method,
        proposed_portfolio=proposed.to_dict(orient="records"), trades=orders.to_dict(orient="records"),
    )


@app.get("/health", response_model=HealthResponse)
def health(http_request: Request) -> HealthResponse:
    """Liveness and what is served: 200 whenever the app is up, with or without a model."""
    state = http_request.app.state
    loaded = state.model is not None
    return HealthResponse(status="ok", model_loaded=loaded, model_version=state.model_version,
                          policy=POLICY_MODEL if loaded else POLICY_RULE, version=state.env["APP_VERSION"],
                          commit=state.env["RENDER_GIT_COMMIT"])


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, http_request: Request) -> PredictResponse:
    """One recommendation for one portfolio. A ValueError from the pipeline becomes a 400 with its message."""
    started = time.perf_counter()
    state = http_request.app.state
    try:
        answer = recommend(body, state.model, state.model_version, CFG)
    except ValueError as error:
        logger.warning("Predict rejected for as_of %s: %s", body.as_of, error)
        raise HTTPException(status_code=400, detail=str(error)) from error
    latency_ms = (time.perf_counter() - started) * MS_PER_SECOND
    logger.info("Predict as_of %s: recommendation %.1f, fallback %s, latency %.1f ms",
                answer.as_of, answer.recommendation, answer.fallback, latency_ms)
    state.monitor.record({"as_of": str(answer.as_of), "recommendation": answer.recommendation,
                          "probability": answer.probability, "fallback": answer.fallback,
                          "model_version": answer.model_version, "latency_ms": latency_ms})
    return answer

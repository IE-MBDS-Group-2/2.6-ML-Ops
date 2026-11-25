"""Tests for src/api.py with FastAPI's TestClient: offline, tiny synthetic prices, a tiny fitted model.

Never the real data/ folder, never the network: the model is built in a fixture and handed to the app.
"""
import logging
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import allocation, api, config, features, portfolio, train, utils

CFG = api.CFG
TICKERS = CFG["data"]["tickers"]
COLS = CFG["data"]["columns"]
TARGET = pd.Series(CFG["target_weights"]["policy"], dtype=float)[TICKERS]
D9_FIELDS = {"as_of", "prediction", "probability", "recommendation", "fallback", "rule_action", "model_version",
             "reason", "proposed_portfolio", "trades"}
TOTAL = 1_000_000.0


def price_records(periods: int = 80, end: str = "2024-05-17", seed: int = 0) -> list[dict]:
    """Random-walk closes and constant volumes for the configured ETFs, as long records (one per date, ticker)."""
    days = pd.bdate_range(end=end, periods=periods)
    returns = np.random.default_rng(seed).normal(0.0003, 0.01, size=(periods, len(TICKERS)))
    close = 100 * np.cumprod(1 + returns, axis=0)
    return [{COLS["date_column"]: str(day.date()), COLS["ticker_column"]: ticker,
             COLS["close_column"]: float(close[i, j]), COLS["volume_column"]: 1_000_000.0}
            for i, day in enumerate(days) for j, ticker in enumerate(TICKERS)]


def request_body(periods: int = 80, shifts: dict | None = None, **overrides: object) -> dict:
    """A valid /predict body: positions at target (plus shifts, in weight points), drawdown 0, weeks 0."""
    weights = TARGET + pd.Series(shifts or {}, dtype=float).reindex(TICKERS, fill_value=0.0)
    records = price_records(periods)
    body = {
        "as_of": records[-1][COLS["date_column"]],
        "positions": [{"ticker": ticker, "market_value": float(weights[ticker] * TOTAL)} for ticker in TICKERS],
        "drawdown": 0.0,
        "weeks_since_rebalance": 0,
        "prices": records,
        "portfolio_id": "sleeve-a",
    }
    body.update(overrides)
    return body


@pytest.fixture(scope="module")
def fitted_model():
    """A tiny Stage 2 model fitted on random rows with the live state's columns and all three actions."""
    request = api.PredictRequest(**request_body())
    close, volume, market = api.market_inputs(request, CFG)
    state = portfolio.live_state(close, volume, market, TARGET, 0.0, 0, TARGET, CFG)
    columns = [name for name in state.columns if name != portfolio.RULE_ACTION_COLUMN]
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(0.0, 0.05, size=(60, len(columns))), columns=columns)
    y = pd.Series([0.0, 0.5, 1.0] * 20)
    model = train.train_model(X, y, features.get_feature_preprocessor(columns), "classification",
                              CFG["model"]["params"], CFG["model"]["random_state"])
    model.target_metadata_ = allocation.model_target(CFG)
    return model


@pytest.fixture
def start_app(monkeypatch):
    """Start the app with a given model (or None) instead of reading models/ or W&B."""
    def start(model, version: str = "local:model.joblib") -> TestClient:
        monkeypatch.setattr(api, "load_serving_model",
                            lambda cfg, env: (model, version if model is not None else api.RULE_VERSION))
        monkeypatch.setattr(api.log_setup, "setup_logging", lambda *args, **kwargs: None)
        return TestClient(api.app)
    return start


def signed_total(trades: list[dict]) -> float:
    """Buys minus sells."""
    return sum(row["value"] if row["side"] == "buy" else -row["value"] for row in trades)


# ---------- /health ----------

def test_integration_health_with_and_without_a_model(start_app, fitted_model):
    with start_app(fitted_model) as client:
        with_model = client.get("/health")
    with start_app(None) as client:
        without = client.get("/health")
    assert with_model.status_code == 200 and without.status_code == 200
    assert with_model.json()["model_loaded"] is True and with_model.json()["policy"] == "model"
    assert without.json()["model_loaded"] is False and without.json()["policy"] == "rule"
    assert without.json()["model_version"] == "rule"


def test_unit_health_reports_version_and_commit(start_app, monkeypatch):
    monkeypatch.setenv("APP_VERSION", "v1.1.0")
    monkeypatch.setenv("APP_COMMIT", "abc1234")
    with start_app(None) as client:
        body = client.get("/health").json()
    assert body["version"] == "v1.1.0"
    assert body["commit"] == "abc1234"
    assert body["model_version"] == "rule"


def test_unit_health_defaults_version_dev_and_commit_local(start_app, monkeypatch):
    monkeypatch.delenv("APP_VERSION", raising=False)
    monkeypatch.delenv("APP_COMMIT", raising=False)
    monkeypatch.setattr(api.config, "PROJECT_ROOT", api.config.PROJECT_ROOT / "no-such-folder")
    with start_app(None) as client:
        body = client.get("/health").json()
    assert body["version"] == "dev"
    assert body["commit"] == "local"


# ---------- /predict: the contract ----------

def test_integration_predict_returns_the_contract(start_app, fitted_model):
    with start_app(fitted_model) as client:
        response = client.post("/predict", json=request_body())
    assert response.status_code == 200, response.text
    body = response.json()
    assert D9_FIELDS <= set(body)
    assert body["recommendation"] in {0.0, 0.5, 1.0}
    assert body["prediction"] in {0.0, 0.5, 1.0}
    assert 0.0 <= body["probability"] <= 1.0
    assert body["model_version"] == "local:model.joblib"
    assert body["portfolio_id"] == "sleeve-a"
    assert body["target_method"] == "policy" and body["rebalancing_method"] == "smart"
    assert isinstance(body["reason"], str) and body["reason"].strip().endswith(".")


def test_integration_trades_net_to_zero(start_app, fitted_model):
    with start_app(fitted_model) as client:
        body = client.post("/predict", json=request_body(shifts={"SPY": 0.08, "IEF": -0.08})).json()
    weights = [row["weight"] for row in body["proposed_portfolio"]]
    assert [row["ticker"] for row in body["proposed_portfolio"]] == TICKERS
    assert sum(weights) == pytest.approx(1.0)
    assert signed_total(body["trades"]) == pytest.approx(0.0, abs=1e-6)
    assert sum(row["value"] for row in body["proposed_portfolio"]) == pytest.approx(TOTAL)


def test_integration_rule_only_mode_answers(start_app):
    with start_app(None) as client:
        response = client.post("/predict", json=request_body(shifts={"SPY": 0.08, "IEF": -0.08}))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["fallback"] is True
    assert body["model_version"] == "rule"
    assert body["prediction"] is None and body["probability"] is None
    assert body["recommendation"] == body["rule_action"] == 1.0
    assert body["reason"].startswith(api.NO_MODEL_REASON)
    assert "SPY is 8.0 points above target" in body["reason"]


def test_integration_rule_only_mode_no_trade_inside_the_band(start_app):
    # 17 May 2024 is mid-quarter, so with every holding on target today's rule does nothing.
    with start_app(None) as client:
        body = client.post("/predict", json=request_body()).json()
    assert body["recommendation"] == 0.0
    assert body["trades"] == []
    assert "No trade" in body["reason"]


def test_integration_chosen_rule_is_not_a_fallback(start_app, fitted_model):
    with start_app(fitted_model) as client:
        body = client.post("/predict", json=request_body(
            shifts={"SPY": 0.03, "IEF": -0.03}, rebalancing_method="tolerance_band")).json()
    assert body["fallback"] is False
    assert body["prediction"] is None
    assert body["recommendation"] == 0.0     # 3 points is inside the 5-point band
    assert body["rebalancing_method"] == "tolerance_band"


# ---------- /predict: wrong input ----------

def test_integration_missing_field_is_422(start_app):
    body = request_body()
    del body["positions"]
    with start_app(None) as client:
        response = client.post("/predict", json=body)
    assert response.status_code == 422
    assert "positions" in response.text


def test_integration_unknown_ticker_is_422(start_app):
    body = request_body()
    body["positions"].append({"ticker": "QQQ", "market_value": 10.0})
    with start_app(None) as client:
        response = client.post("/predict", json=body)
    assert response.status_code == 422
    assert "QQQ" in response.text


@pytest.mark.parametrize("field, value", [("drawdown", 0.1), ("weeks_since_rebalance", -1)])
def test_integration_out_of_range_field_is_422(start_app, field, value):
    with start_app(None) as client:
        response = client.post("/predict", json=request_body(**{field: value}))
    assert response.status_code == 422
    assert field in response.text


def test_integration_positions_totalling_zero_is_422(start_app):
    body = request_body(positions=[{"ticker": "SPY", "market_value": 0.0}])
    with start_app(None) as client:
        response = client.post("/predict", json=body)
    assert response.status_code == 422
    assert "total market value of 0" in response.text


def test_integration_wrong_choice_is_422(start_app):
    with start_app(None) as client:
        response = client.post("/predict", json=request_body(rebalancing_method="smrt"))
    assert response.status_code == 422
    for allowed in ("smart", "calendar", "tolerance_band", "calendar_and_band"):
        assert allowed in response.text


def test_integration_too_little_history_is_400(start_app):
    with start_app(None) as client:
        response = client.post("/predict", json=request_body(periods=10))
    assert response.status_code == 400
    assert "Only 10 trading days" in response.json()["detail"]


def test_integration_no_prices_on_as_of_is_400(start_app):
    with start_app(None) as client:
        response = client.post("/predict", json=request_body(as_of="2024-05-20"))
    assert response.status_code == 400
    assert "as_of 2024-05-20" in response.json()["detail"]


def test_integration_repeated_ticker_is_400(start_app):
    body = request_body()
    body["positions"].append({"ticker": "SPY", "market_value": 10.0})
    with start_app(None) as client:
        response = client.post("/predict", json=body)
    assert response.status_code == 400
    assert "SPY" in response.json()["detail"]


def test_integration_bad_prices_are_400(start_app):
    body = request_body()
    body["prices"][0][COLS["close_column"]] = -1.0
    with start_app(None) as client:
        response = client.post("/predict", json=body)
    assert response.status_code == 400


@pytest.mark.parametrize("target_method", config.CALCULATED_TARGETS)
def test_integration_band_rule_uses_stored_calculated_weights(start_app, target_method):
    """A full band-triggered trade goes to the reviewed target weights."""
    with start_app(None) as client:
        response = client.post("/predict", json=request_body(
            target_method=target_method, rebalancing_method="tolerance_band", shifts={"SPY": 0.25, "IEF": -0.25},
        ))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["recommendation"] == 1.0
    assert {row["ticker"]: row["weight"] for row in body["proposed_portfolio"]} == pytest.approx(
        CFG["target_weights"]["calculated"][target_method])
    assert body["fallback"] is False


@pytest.mark.parametrize("target_method", config.CALCULATED_TARGETS)
@pytest.mark.parametrize("loaded", [True, False])
def test_integration_calculated_smart_target_falls_back_with_reason(start_app, fitted_model, target_method, loaded):
    """Calculated targets work with smart and bypass a model trained for the policy mix."""
    with start_app(fitted_model if loaded else None) as client:
        response = client.post("/predict", json=request_body(target_method=target_method))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["fallback"] is True
    assert body["prediction"] is None
    assert body["probability"] is None
    assert "calendar-and-band" in body["reason"]
    assert "tracking error" in body["reason"]
    assert body["recommendation"] == body["rule_action"]


def test_integration_policy_request_does_not_use_a_model_for_another_target(start_app, fitted_model, monkeypatch):
    """A config fitted for a calculated mix cannot serve model decisions for the policy mix."""
    monkeypatch.setitem(CFG["target_weights"], "method", "min_variance")
    with start_app(fitted_model) as client:
        response = client.post("/predict", json=request_body(target_method="policy"))
    assert response.status_code == 200, response.text
    assert response.json()["fallback"] is True
    assert response.json()["prediction"] is None
    assert api.NO_MODEL_REASON not in response.json()["reason"]
    assert "requested policy weights" in response.json()["reason"]


# ---------- logging ----------

def test_unit_no_payload_in_the_logs(start_app, caplog):
    body = request_body()
    with caplog.at_level(logging.INFO, logger="src.api"), start_app(None) as client:
        client.post("/predict", json=body)
    lines = [record.getMessage() for record in caplog.records if record.name == "src.api"]
    request_lines = [line for line in lines if line.startswith("Predict as_of")]
    assert len(request_lines) == 1
    assert body["as_of"] in request_lines[0]
    assert "recommendation" in request_lines[0] and "latency" in request_lines[0]
    first_close = f"{body['prices'][0][COLS['close_column']]:.4f}"
    for line in lines:
        assert "market_value" not in line and first_close not in line and "sleeve-a" not in line


# ---------- start-up: which model is served ----------

def test_unit_load_serving_model_local_file(tmp_path, fitted_model):
    utils.save_model(fitted_model, tmp_path / "model.joblib")
    cfg = {**CFG, "paths": {**CFG["paths"], "model": str(tmp_path / "model.joblib")}}
    model, version = api.load_serving_model(cfg, {"MODEL_SOURCE": "local"})
    assert model is not None
    assert version == "local:model.joblib"


def test_unit_load_serving_model_reports_the_version_baked_into_the_image(tmp_path, fitted_model):
    utils.save_model(fitted_model, tmp_path / "model.joblib")
    cfg = {**CFG, "paths": {**CFG["paths"], "model": str(tmp_path / "model.joblib")}}
    env = {"MODEL_SOURCE": "local", "MODEL_VERSION": "rebalance-model:v7"}
    assert api.load_serving_model(cfg, env)[1] == "rebalance-model:v7"


def test_unit_load_serving_model_without_a_file_is_rule_only(tmp_path):
    cfg = {**CFG, "paths": {**CFG["paths"], "model": str(tmp_path / "absent.joblib")}}
    assert api.load_serving_model(cfg, {"MODEL_SOURCE": "local"}) == (None, "rule")


def test_unit_load_serving_model_from_wandb(tmp_path, monkeypatch, fitted_model):
    utils.save_model(fitted_model, tmp_path / "model.joblib")
    calls = []

    def download(env, project, name, alias):
        calls.append((project, name, alias))
        return tmp_path / "model.joblib"

    monkeypatch.setattr(api.tracking, "download_model", download)
    model, version = api.load_serving_model(CFG, {"MODEL_SOURCE": "wandb", "WANDB_MODEL_ALIAS": "prod"})
    assert model is not None
    assert version == f"{CFG['tracking']['model_artifact']}:prod"
    assert calls == [(CFG["tracking"]["project"], CFG["tracking"]["model_artifact"], "prod")]


def test_unit_missing_wandb_alias_is_rule_only(monkeypatch):
    def download(env, project, name, alias):
        raise FileNotFoundError(f"Model {name}:{alias} not found in W&B")

    monkeypatch.setattr(api.tracking, "download_model", download)
    assert api.load_serving_model(CFG, {"MODEL_SOURCE": "wandb", "WANDB_MODEL_ALIAS": "prod"}) == (None, "rule")


# ---------- monitoring ----------

@pytest.fixture
def monitored(monkeypatch, start_app):
    """Start the app with a fake serve run, batch size 3, and a recorder in place of log_prediction_batch."""
    sent = []
    monkeypatch.setitem(CFG["api"], "monitoring_batch_size", 3)
    monkeypatch.setattr(api.tracking, "start_run", lambda cfg, env, job_type: MagicMock(name=job_type))
    monkeypatch.setattr(api.tracking, "finish_run", lambda run, failed=False: None)
    monkeypatch.setattr(api.tracking, "log_prediction_batch", lambda run, rows: sent.append(list(rows)))
    return start_app(None), sent


def test_unit_batch_is_sent_when_full_and_cleared(monitored):
    client, sent = monitored
    with client:
        for _ in range(3):
            assert client.post("/predict", json=request_body()).status_code == 200
        assert len(sent) == 1 and len(sent[0]) == 3
        assert client.app.state.monitor.rows == []
    row = sent[0][0]
    assert set(row) == {"as_of", "recommendation", "probability", "fallback", "model_version", "latency_ms"}
    assert row["model_version"] == "rule" and row["fallback"] is True


def test_unit_no_send_before_the_batch_is_full(monitored):
    client, sent = monitored
    with client:
        for _ in range(2):
            client.post("/predict", json=request_body())
        assert sent == []
        assert len(client.app.state.monitor.rows) == 2


def test_unit_shutdown_sends_the_last_partial_batch(monitored):
    client, sent = monitored
    with client:
        client.post("/predict", json=request_body())
    assert len(sent) == 1 and len(sent[0]) == 1


def test_unit_tracking_error_never_reaches_the_client(monitored, monkeypatch, caplog):
    client, _ = monitored

    def broken(run, rows):
        raise RuntimeError("W&B is down")

    monkeypatch.setattr(api.tracking, "log_prediction_batch", broken)
    with caplog.at_level(logging.ERROR, logger="src.api"), client:
        statuses = [client.post("/predict", json=request_body()).status_code for _ in range(3)]
        assert client.app.state.monitor.rows == []
    assert statuses == [200, 200, 200]
    assert any("W&B is down" in record.getMessage() for record in caplog.records)


def test_unit_tracking_disabled_calls_nothing(monkeypatch, start_app):
    calls = []
    monkeypatch.setattr(api.tracking, "log_prediction_batch", lambda run, rows: calls.append(rows))
    with start_app(None) as client:
        for _ in range(CFG["api"]["monitoring_batch_size"] + 1):
            assert client.post("/predict", json=request_body()).status_code == 200
        assert client.app.state.monitor.run is None
        assert client.app.state.monitor.rows == []
    assert calls == []


def test_unit_wandb_outage_at_start_up_still_serves(monkeypatch, start_app, caplog):
    def unreachable(cfg, env, job_type):
        raise OSError("network is unreachable")

    monkeypatch.setattr(api.tracking, "start_run", unreachable)
    with caplog.at_level(logging.ERROR, logger="src.api"), start_app(None) as client:
        assert client.get("/health").status_code == 200
        assert client.app.state.monitor.run is None
    assert any("Monitoring off" in record.getMessage() for record in caplog.records)


def test_unit_failing_finish_run_is_logged(monitored, monkeypatch, caplog):
    client, _ = monitored

    def broken(run, failed=False):
        raise RuntimeError("cannot close")

    monkeypatch.setattr(api.tracking, "finish_run", broken)
    with caplog.at_level(logging.ERROR, logger="src.api"), client:
        pass
    assert any("cannot close" in record.getMessage() for record in caplog.records)


def test_unit_loading_rejects_model_metadata_after_policy_weights_change(tmp_path, fitted_model):
    """An artifact cannot silently serve a different target after configuration changes."""
    utils.save_model(fitted_model, tmp_path / "model.joblib")
    cfg = {**CFG, "paths": {**CFG["paths"], "model": str(tmp_path / "model.joblib")},
           "target_weights": {**CFG["target_weights"], "policy": dict(CFG["target_weights"]["policy"])}}
    cfg["target_weights"]["policy"].update(SPY=0.35, IEF=0.35)
    assert api.load_serving_model(cfg, {"MODEL_SOURCE": "local"}) == (None, "rule")


def test_unit_loading_rejects_a_legacy_artifact_without_target_metadata(tmp_path, fitted_model):
    """Unknown historical training targets require a new reviewed model."""
    import copy

    legacy = copy.deepcopy(fitted_model)
    del legacy.target_metadata_
    utils.save_model(legacy, tmp_path / "model.joblib")
    cfg = {**CFG, "paths": {**CFG["paths"], "model": str(tmp_path / "model.joblib")}}
    assert api.load_serving_model(cfg, {"MODEL_SOURCE": "local"}) == (None, "rule")


def test_integration_calculated_target_keeps_rule_reason_and_fixed_training_risk(start_app):
    """The rule explains its action and the target's risk estimate stays independent of request prices."""
    reasons = []
    with start_app(None) as client:
        for seed in (3, 17):
            answer = client.post("/predict", json=request_body(
                target_method="min_variance", prices=price_records(seed=seed))).json()
            assert "No trade:" in answer["reason"] or "Rebalance" in answer["reason"]
            reasons.append(answer["reason"].split("a year.")[0])
    assert reasons[0] == reasons[1]

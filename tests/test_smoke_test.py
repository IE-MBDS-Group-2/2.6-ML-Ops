"""Tests for scripts/smoke_test.py and scripts/make_sample_request.py.

The smoke test runs against a tiny local http.server stub (healthy, unhealthy, wrong version or commit, bad
answers), so every exit path is covered offline. The sample request is checked against the API's request model.
"""
import ast
import json
import logging
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from scripts import make_sample_request, smoke_test
from src import api

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
HEALTHY = {"status": "ok", "model_loaded": False, "model_version": "rule", "policy": "rule", "version": "dev",
           "commit": "0123456789abcdef0123456789abcdef01234567"}
ANSWER = {"as_of": "2025-12-31", "prediction": None, "probability": None, "recommendation": 0.0, "fallback": True,
          "rule_action": 0.0, "model_version": "rule", "reason": "No trade.", "proposed_portfolio": [], "trades": []}


class Stub(BaseHTTPRequestHandler):
    """Answers /health and /predict with whatever the test put in `routes`: path -> (status, body)."""

    routes: dict = {}
    agents: list = []

    def _answer(self) -> None:
        Stub.agents.append(self.headers.get("User-Agent"))
        status, body = self.routes.get(self.path, (404, {"detail": "Not Found"}))
        text = body if isinstance(body, str) else json.dumps(body)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(text.encode("utf-8"))

    def do_GET(self) -> None:
        self._answer()

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self._answer()

    def log_message(self, *args: object) -> None:
        """Keep the test output quiet."""


@pytest.fixture
def stub():
    """Start the stub on a free port; yields a function that sets the answers and returns the base URL."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def serve(health=(200, HEALTHY), predict=(200, ANSWER)) -> str:
        Stub.routes = {"/health": health, "/predict": predict}
        Stub.agents = []
        return f"http://127.0.0.1:{server.server_address[1]}"

    yield serve
    server.shutdown()
    server.server_close()


def run(url: str, *extra: str) -> int:
    """The smoke test's exit code, with fast polling."""
    return smoke_test.main(["--url", url, "--interval", "0.05", "--timeout", "1", *extra])


def failure_line(caplog) -> str:
    """The one FAIL line the smoke test logged."""
    lines = [record.getMessage() for record in caplog.records if record.getMessage().startswith("FAIL")]
    assert len(lines) == 1
    return lines[0]


# ---------- exit paths ----------

def test_unit_healthy_service_exits_0(stub, caplog):
    with caplog.at_level(logging.INFO):
        assert run(stub()) == 0
    assert any(record.getMessage().startswith("PASS") for record in caplog.records)


def test_unit_every_call_sends_the_smoke_test_user_agent(stub):
    assert run(stub()) == 0
    assert len(Stub.agents) >= 2   # the /health GET and the /predict POST
    assert set(Stub.agents) == {"etf-smoke-test/1"}


def test_unit_unhealthy_until_timeout_exits_1(stub, caplog):
    assert run(stub(health=(503, {"detail": "starting"}))) == 1
    assert "/health never became ok within 1 s" in failure_line(caplog)


def test_unit_status_not_ok_exits_1(stub, caplog):
    assert run(stub(health=(200, {**HEALTHY, "status": "starting"}))) == 1
    assert "'starting'" in failure_line(caplog)


def test_unit_wrong_version_exits_1(stub, caplog):
    assert run(stub(), "--expect-version", "v9.9.9") == 1
    line = failure_line(caplog)
    assert "'v9.9.9'" in line and "'dev'" in line


def test_unit_right_version_exits_0(stub):
    assert run(stub(health=(200, {**HEALTHY, "version": "v1.1.0"})), "--expect-version", "v1.1.0") == 0


def test_unit_wrong_commit_exits_1(stub, caplog):
    assert run(stub(), "--expect-commit", "fedcba9876543210") == 1
    line = failure_line(caplog)
    assert "fedcba9876543210" in line and HEALTHY["commit"] in line


@pytest.mark.parametrize("sha", [HEALTHY["commit"], HEALTHY["commit"][:7]])
def test_unit_short_or_full_commit_exits_0(stub, sha):
    assert run(stub(), "--expect-commit", sha) == 0


def test_unit_too_short_commit_never_matches():
    assert not smoke_test.same_commit("0123", HEALTHY["commit"])
    assert not smoke_test.same_commit(HEALTHY["commit"], "local")


@pytest.mark.parametrize("answer, problem", [
    ({key: value for key, value in ANSWER.items() if key != "recommendation"}, "missing ['recommendation']"),
    ({**ANSWER, "recommendation": 0.7}, "recommendation is 0.7"),
])
def test_unit_bad_predict_response_exits_1(stub, caplog, answer, problem):
    assert run(stub(predict=(200, answer))) == 1
    assert problem in failure_line(caplog)


def test_unit_predict_http_error_exits_1(stub, caplog):
    assert run(stub(predict=(400, {"detail": "Only 10 trading days"}))) == 1
    assert "HTTP 400" in failure_line(caplog) and "Only 10 trading days" in failure_line(caplog)


def test_unit_answer_that_is_not_json_exits_1(stub, caplog):
    assert run(stub(predict=(200, "<html>oops</html>"))) == 1
    assert "not JSON" in failure_line(caplog)


def test_unit_unreachable_service_exits_1(caplog):
    assert run("http://127.0.0.1:9") == 1
    assert "unreachable" in failure_line(caplog)


def test_unit_unreadable_sample_request_exits_1(stub, caplog, tmp_path):
    assert run(stub(), "--request", str(tmp_path / "absent.json")) == 1
    assert "cannot read the sample request" in failure_line(caplog)


# ---------- the script itself ----------

def test_unit_only_standard_library_is_imported():
    tree = ast.parse((SCRIPTS / "smoke_test.py").read_text(encoding="utf-8"))
    modules = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    modules |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert modules
    for module in modules:
        assert module.split(".")[0] in sys.stdlib_module_names, module


# ---------- the sample request ----------

def test_unit_sample_request_is_valid():
    body = json.loads((SCRIPTS / "sample_request.json").read_text(encoding="utf-8"))
    api.PredictRequest(**body)
    date_column = api.CFG["data"]["columns"]["date_column"]
    assert len({record[date_column] for record in body["prices"]}) == 70


def test_unit_sample_request_holds_only_synthetic_prices():
    # The file is exactly what the seeded random walk writes, so it cannot hold the real snapshot.
    cfg = yaml.safe_load((SCRIPTS.parent / "config.yaml").read_text(encoding="utf-8"))
    body = json.loads((SCRIPTS / "sample_request.json").read_text(encoding="utf-8"))
    assert body == json.loads(json.dumps(make_sample_request.build_request(cfg, 70, 1_000_000)))


def test_integration_sample_request_answers_200(monkeypatch):
    monkeypatch.setattr(api, "load_serving_model", lambda cfg, env: (None, api.RULE_VERSION))
    monkeypatch.setattr(api.log_setup, "setup_logging", lambda *args, **kwargs: None)
    body = json.loads((SCRIPTS / "sample_request.json").read_text(encoding="utf-8"))
    with TestClient(api.app) as client:
        answer = client.post("/predict", json=body)
    assert answer.status_code == 200, answer.text
    assert set(smoke_test.D9_FIELDS) <= set(answer.json())


def test_unit_make_sample_request_main_writes_the_file(tmp_path):
    out = tmp_path / "request.json"
    make_sample_request.main(["--out", str(out), "--days", "65"])
    body = json.loads(out.read_text(encoding="utf-8"))
    assert len(body["prices"]) == 65 * len(body["positions"])
    assert body["drawdown"] == 0.0 and body["weeks_since_rebalance"] == 0


def test_unit_sample_request_raises_below_the_api_minimum():
    cfg = yaml.safe_load((SCRIPTS.parent / "config.yaml").read_text(encoding="utf-8"))
    needed = cfg["api"]["min_history_days"]
    with pytest.raises(ValueError, match=f"at least {needed} trading days"):
        make_sample_request.build_request(cfg, needed - 1, 1_000_000)


def test_unit_sample_request_uses_the_ci_generator():
    # One synthetic generator for CI and the sample (same function, same seed): 70 dates up to as_of, all tickers.
    cfg = yaml.safe_load((SCRIPTS.parent / "config.yaml").read_text(encoding="utf-8"))
    body = make_sample_request.build_request(cfg, 70, 1_000_000)
    columns = cfg["data"]["columns"]
    dates = sorted({record[columns["date_column"]] for record in body["prices"]})
    assert len(dates) == 70 and dates[-1] == body["as_of"]
    assert {record[columns["ticker_column"]] for record in body["prices"]} == set(cfg["data"]["tickers"])

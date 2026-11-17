"""
smoke_test.py  [Stage 3: IS THE LIVE SERVICE UP AND ANSWERING?]
===============================================================
Purpose   After a deploy, check the service end to end: /health becomes ok (on the expected release or commit),
          then /predict answers the sample request with the D9 fields
Input     --url of the service; optional --expect-version (the /health version), --expect-commit (the /health
          commit, short or full SHA), --timeout in seconds, --request (default scripts/sample_request.json)
Output    Exit code 0 when every check passes, 1 otherwise, with one log line saying why
Run       python scripts/smoke_test.py --url http://localhost:8000
          python scripts/smoke_test.py --url "$SERVICE_URL" --expect-commit "$GITHUB_SHA" --timeout 900
Used by   deploy.yml after each GitHub Release; anyone checking a local or Docker run

Why it matters
- Standard library only (urllib, json, argparse, time): it runs on any machine and in CI with no installs
- It polls, because a fresh deploy takes minutes and the old release answers until the new one is live: the
  commit check stops a green result coming from the old release

Tests     tests/test_smoke_test.py
"""

import argparse
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger("smoke_test")

DEFAULT_REQUEST = Path(__file__).with_name("sample_request.json")
D9_FIELDS = ("as_of", "prediction", "probability", "recommendation", "fallback", "rule_action", "model_version",
             "reason", "proposed_portfolio", "trades")
ACTIONS = (0.0, 0.5, 1.0)
HTTP_OK = 200
MAX_CALL_SECONDS = 60      # one HTTP call never waits longer than this, whatever the overall timeout
SHORT_SHA = 7              # the shortest commit SHA accepted as a match


def call(url: str, timeout: float, body: dict | None = None) -> tuple[int, dict | None, str]:
    """GET (or POST a JSON body to) a URL and read a JSON answer.

    Inputs:  url; timeout, seconds for this one call; body, the JSON to POST, or None for a GET.
    Outputs: (HTTP status or 0 when unreachable, the JSON answer or None, a short note on what went wrong).
    Why:     every failure becomes data the caller can report in one line, never an exception.
    """
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status, text = response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        return error.code, None, f"HTTP {error.code}: {error.read().decode('utf-8', 'replace')[:200]}"
    except (urllib.error.URLError, OSError) as error:
        return 0, None, f"unreachable ({error})"
    try:
        return status, json.loads(text), ""
    except json.JSONDecodeError:
        return status, None, f"HTTP {status} but the answer is not JSON: {text[:200]!r}"


def same_commit(expected: str, actual: str) -> bool:
    """True when two commit SHAs name the same commit.

    Inputs:  expected, the SHA the deploy shipped (short or full); actual, the SHA /health reports.
    Outputs: True when one is a prefix of the other and the shorter has at least 7 characters.
    Why:     the workflow passes the full SHA, a person may type a short one; too short a prefix could match by luck.
    """
    shorter = min(len(expected), len(actual))
    return shorter >= SHORT_SHA and (actual.startswith(expected) or expected.startswith(actual))


def health_problem(status: int, body: dict | None, note: str, expect: dict) -> str:
    """Why one /health answer is not good enough yet; "" when it is.

    Inputs:  status, body and note from call; expect, {"version": str | None, "commit": str | None}.
    Outputs: a short reason, or "".
    """
    if status != HTTP_OK or body is None:
        return note or f"HTTP {status}"
    if body.get("status") != "ok":
        return f"status is {body.get('status')!r}, not 'ok'"
    if expect["version"] and body.get("version") != expect["version"]:
        return f"version is {body.get('version')!r}, expected {expect['version']!r}"
    if expect["commit"] and not same_commit(expect["commit"], str(body.get("commit", ""))):
        return f"commit is {body.get('commit')!r}, expected {expect['commit']!r}"
    return ""


def wait_for_health(base_url: str, expect: dict, timeout: float, interval: float) -> str:
    """Poll /health until it is ok (and on the expected version and commit) or the timeout ends.

    Inputs:  base_url; expect, {"version": ..., "commit": ...} (None skips that check); timeout and interval,
             in seconds.
    Outputs: "" once healthy, otherwise one line saying health never became ok and the last reason.
    """
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        problem = health_problem(*call(f"{base_url}/health", max(min(remaining, MAX_CALL_SECONDS), 1)), expect)
        if not problem:
            return ""
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return f"/health never became ok within {timeout:g} s: {problem}"
        logger.info("Waiting for /health: %s", problem)
        time.sleep(min(interval, remaining))


def predict_problem(base_url: str, request_body: dict, timeout: float) -> str:
    """POST the sample request to /predict and check the answer; "" when it has every D9 field and a valid action.

    Inputs:  base_url; request_body, the sample request; timeout, seconds for the call.
    Outputs: a one-line reason, or "".
    """
    status, body, note = call(f"{base_url}/predict", timeout, request_body)
    if status != HTTP_OK or body is None:
        return f"/predict failed: {note or f'HTTP {status}'}"
    missing = [field for field in D9_FIELDS if field not in body]
    if missing:
        return f"/predict answer is missing {missing}"
    if body["recommendation"] not in ACTIONS:
        return f"/predict recommendation is {body['recommendation']!r}, expected one of {list(ACTIONS)}"
    return ""


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """The command-line options.

    Inputs:  argv, the arguments (None reads sys.argv).
    Outputs: url, expect_version, expect_commit, timeout and request.
    Why:     the deploy workflow and a person at a terminal call the same script with the same options.
    """
    parser = argparse.ArgumentParser(description="Check that the rebalancing service is up and answering.")
    parser.add_argument("--url", required=True, help="base URL of the service, e.g. http://localhost:8000")
    parser.add_argument("--expect-version", default=None, help="the /health version that must be live")
    parser.add_argument("--expect-commit", default=None, help="the /health commit that must be live (short or full)")
    parser.add_argument("--timeout", type=float, default=300, help="seconds to wait for /health (default 300)")
    parser.add_argument("--interval", type=float, default=10, help="seconds between /health polls (default 10)")
    parser.add_argument("--request", type=Path, default=DEFAULT_REQUEST, help="the /predict body to send")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the checks and return the exit code: 0 when everything passed, 1 otherwise.

    Inputs:  argv, the command-line arguments (None reads sys.argv).
    Outputs: 0 or 1; one log line says the result.
    """
    args = parse_args(argv)
    base_url = args.url.rstrip("/")
    try:
        request_body = json.loads(args.request.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        logger.error("FAIL: cannot read the sample request %s (%s)", args.request, error)
        return 1
    expect = {"version": args.expect_version, "commit": args.expect_commit}
    problem = (wait_for_health(base_url, expect, args.timeout, args.interval)
               or predict_problem(base_url, request_body, MAX_CALL_SECONDS))
    if problem:
        logger.error("FAIL: %s", problem)
        return 1
    logger.info("PASS: %s is healthy and /predict answers the sample request", base_url)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.exit(main())

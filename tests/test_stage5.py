"""Stage 5 tests: structured logs + /metrics counters (offline, TestClient).

Covers: /metrics incrementing after /solve, a successful logistic+nesterov
solve recording iterations + converged=true in BOTH the metrics and the JSON
log stream, an inapplicable pair still returning 422 and counted as an ERROR
(not a convergence failure), and validation errors counted as errors. All
offline and docker-free; the metrics singleton is reset per test module."""

from __future__ import annotations

import json
import logging

from fastapi.testclient import TestClient

from convex_optimization.app import app
from convex_optimization.cli import APPLICABLE
from convex_optimization.observability import metrics

client = TestClient(app)


class _Capture(logging.Handler):
    """Collect the observability logger's JSON lines."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[dict] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(json.loads(record.getMessage()))


def test_metrics_counts_solve_and_latency() -> None:
    before = metrics.snapshot()
    r = client.post("/solve", json={"problem": "lasso", "method": "fista"})
    assert r.status_code == 200
    after = metrics.snapshot()
    assert after["solve"]["count"] == before["solve"]["count"] + 1
    assert after["requests_total"] == before["requests_total"] + 1
    assert after["latency_ms"]["window"] >= 1
    assert after["latency_ms"]["p50"] is not None
    # A converging solve must NOT count as a convergence failure.
    assert after["solve"]["convergence_failures"] == before["solve"]["convergence_failures"]


def test_successful_solve_records_iterations_and_converged_in_metrics_and_logs() -> None:
    capture = _Capture()
    logger = logging.getLogger("convex_optimization.observability")
    logger.addHandler(capture)
    try:
        r = client.post("/solve", json={"problem": "logistic", "method": "nesterov"})
        assert r.status_code == 200
        body = r.json()
        assert body["converged"] is True
        assert body["iterations"] >= 1
    finally:
        logger.removeHandler(capture)

    snap = metrics.snapshot()["solve"]
    assert snap["iterations_last"] == body["iterations"]
    assert snap["converged_last"] is True

    solve_logs = [rec for rec in capture.records if rec["endpoint"] == "/solve"]
    assert solve_logs, "expected at least one structured /solve log line"
    last = solve_logs[-1]
    assert last["status"] == 200
    assert last["problem"] == "logistic"
    assert last["method"] == "nesterov"
    assert last["iterations"] == body["iterations"]
    assert last["converged"] is True
    assert last["error_class"] is None
    assert isinstance(last["latency_ms"], float)
    # No iterate vectors or history rows in the log line.
    assert "x" not in last and "history" not in last


def test_inapplicable_pair_is_422_error_not_convergence_failure() -> None:
    before = metrics.snapshot()
    r = client.post("/solve", json={"problem": "lasso", "method": "nesterov"})
    assert r.status_code == 422
    assert r.json()["detail"]["applicable_methods"] == list(APPLICABLE["lasso"])
    after = metrics.snapshot()
    assert after["status_counts"]["4xx"] == before["status_counts"]["4xx"] + 1
    # 4xx are errors, NEVER convergence failures.
    assert after["solve"]["convergence_failures"] == before["solve"]["convergence_failures"]


def test_validation_error_is_422_and_counted_as_error() -> None:
    before = metrics.snapshot()
    r = client.post("/solve", json={"problem": "logistic", "method": "gd", "tail": 500})
    assert r.status_code == 422  # tail > 20 rejected by request validation
    after = metrics.snapshot()
    assert after["status_counts"]["4xx"] == before["status_counts"]["4xx"] + 1
    assert after["solve"]["convergence_failures"] == before["solve"]["convergence_failures"]


def test_metrics_endpoint_is_get_and_json() -> None:
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["requests_total"] >= 1
    assert "convergence_failure_definition" in body["solve"]
    assert body["scope"].startswith("in-process counters")

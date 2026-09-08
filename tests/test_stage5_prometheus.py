"""Phase 2 tests: the Prometheus text-exposition contract (offline, TestClient).

Covers the shared Phase 2 contract for GET /metrics/prometheus: 200 + exact
content type, required metric families (with HELP/TYPE even when empty),
bounded label values (route-template endpoints, HTTP verbs, used status codes,
the app's error-class vocabulary), request counting, latency series, error
counting after a triggering request, and the up gauge. Also asserts the JSON
GET /metrics contract is unchanged. No network, no new dependencies; the
Prometheus singleton is reset around every test."""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from convex_optimization.app import app
from convex_optimization.observability import ERROR_INAPPLICABLE, ERROR_VALIDATION
from convex_optimization.prometheus import (
    CONTENT_TYPE,
    ERRORS_TOTAL,
    LATENCY_BUCKETS,
    LATENCY_SECONDS,
    REQUESTS_TOTAL,
    UNMATCHED_ENDPOINT,
    UP,
    prometheus_metrics,
)

client = TestClient(app)

JSON_METRICS_KEYS = {
    "scope",
    "uptime_seconds",
    "requests_total",
    "requests_by_endpoint",
    "status_counts",
    "errors",
    "solve",
    "latency_ms",
}

# Bounded label vocabularies (the app's actually-used values plus the static
# FastAPI documentation routes that would appear if scraped).
ALLOWED_ENDPOINTS = {
    "/health",
    "/metrics",
    "/metrics/prometheus",
    "/solve",
    "/openapi.json",
    "/docs",
    "/redoc",
    "/docs/oauth2-redirect",
    UNMATCHED_ENDPOINT,
}
ALLOWED_METHODS = {"GET", "POST"}
ALLOWED_STATUSES = {"200", "404", "422", "500"}
ALLOWED_ERROR_CLASSES = {ERROR_VALIDATION, ERROR_INAPPLICABLE, "server_error"}

_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')
_SAMPLE_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})? (\S+)$")


def _parse(text: str) -> list[tuple[str, dict[str, str], float]]:
    """Minimal exposition parser for assertions (text -> name, labels, value)."""
    samples = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _SAMPLE_RE.match(line)
        assert m is not None, f"unparseable exposition line: {line!r}"
        labels = dict(_LABEL_RE.findall(m.group(2) or ""))
        samples.append((m.group(1), labels, float(m.group(3))))
    return samples


def _get(name: str, labels: dict[str, str], text: str) -> float | None:
    for n, ls, value in _parse(text):
        if n == name and ls == labels:
            return value
    return None


@pytest.fixture(autouse=True)
def _reset_prometheus_singleton():
    prometheus_metrics.reset()
    yield
    prometheus_metrics.reset()


def test_prometheus_endpoint_returns_200_with_exact_content_type() -> None:
    r = client.get("/metrics/prometheus")
    assert r.status_code == 200
    assert r.headers["content-type"] == CONTENT_TYPE
    assert "convex_optimization_up 1" in r.text


def test_all_required_families_and_types_present() -> None:
    text = client.get("/metrics/prometheus").text
    # Exactly one TYPE line per family (histogram child series inherit the
    # family's type; there is no separate TYPE line for _bucket/_sum/_count).
    assert f"# TYPE {REQUESTS_TOTAL} counter" in text
    assert f"# TYPE {ERRORS_TOTAL} counter" in text
    assert f"# TYPE {LATENCY_SECONDS} histogram" in text
    assert f"# TYPE {UP} gauge" in text
    # 4 generic Phase 2 families + 6 Phase 3 convex-specific families.
    assert text.count("# TYPE ") == 10
    # Families are declared even when they have no samples yet.
    for name in (REQUESTS_TOTAL, ERRORS_TOTAL, LATENCY_SECONDS, UP):
        assert f"# HELP {name} " in text, name
        assert f"# TYPE {name} " in text, name
    # The gauge has its sample immediately.
    assert _get(UP, {}, text) == 1


def test_request_increments_requests_total_with_bounded_labels() -> None:
    before = _get(
        REQUESTS_TOTAL,
        {"endpoint": "/health", "method": "GET", "status": "200"},
        client.get("/metrics/prometheus").text,
    )
    r = client.get("/health")
    assert r.status_code == 200
    text = client.get("/metrics/prometheus").text
    after = _get(
        REQUESTS_TOTAL,
        {"endpoint": "/health", "method": "GET", "status": "200"},
        text,
    )
    assert after == (before or 0) + 1


def test_latency_histogram_exposed_with_documented_buckets() -> None:
    client.get("/health")
    text = client.get("/metrics/prometheus").text
    base = {"endpoint": "/health", "method": "GET"}
    le_values = []
    cumulative = []
    for name, labels, value in _parse(text):
        if name == f"{LATENCY_SECONDS}_bucket" and labels.items() >= base.items():
            le_values.append(labels["le"])
            cumulative.append(value)
    assert le_values == [f"{b:g}" for b in LATENCY_BUCKETS] + ["+Inf"]
    assert cumulative == sorted(cumulative), "bucket counts must be cumulative"
    total = _get(f"{LATENCY_SECONDS}_count", base, text)
    assert total is not None and total >= 1
    assert _get(f"{LATENCY_SECONDS}_bucket", {**base, "le": "+Inf"}, text) == total
    total_sum = _get(f"{LATENCY_SECONDS}_sum", base, text)
    assert total_sum is not None and total_sum >= 0.0


def test_error_counter_increments_after_triggering_request() -> None:
    # Inapplicable (problem, method) pair -> 422 -> ERROR_INAPPLICABLE.
    r = client.post("/solve", json={"problem": "lasso", "method": "nesterov"})
    assert r.status_code == 422
    text = client.get("/metrics/prometheus").text
    assert (
        _get(
            ERRORS_TOTAL,
            {"endpoint": "/solve", "method": "POST", "error_class": ERROR_INAPPLICABLE},
            text,
        )
        == 1
    )
    # Validation failure (tail out of range) -> 422 -> ERROR_VALIDATION.
    r = client.post("/solve", json={"problem": "gd", "method": "gd", "tail": 500})
    assert r.status_code == 422
    text = client.get("/metrics/prometheus").text
    assert (
        _get(
            ERRORS_TOTAL,
            {"endpoint": "/solve", "method": "POST", "error_class": ERROR_VALIDATION},
            text,
        )
        == 1
    )


def test_all_label_values_bounded() -> None:
    # Exercise every status class the app actually produces.
    client.get("/health")
    client.get("/metrics")
    client.get("/metrics/prometheus")
    client.get("/no-such-route")
    client.post("/solve", json={"problem": "lasso", "method": "fista"})  # 200
    client.post("/solve", json={"problem": "lasso", "method": "nesterov"})  # 422
    text = client.get("/metrics/prometheus").text
    for name, labels, _value in _parse(text):
        if name == REQUESTS_TOTAL:
            assert set(labels) == {"endpoint", "method", "status"}
            assert labels["endpoint"] in ALLOWED_ENDPOINTS, labels
            assert labels["method"] in ALLOWED_METHODS, labels
            assert labels["status"] in ALLOWED_STATUSES, labels
        elif name == ERRORS_TOTAL:
            assert set(labels) == {"endpoint", "method", "error_class"}
            assert labels["endpoint"] in ALLOWED_ENDPOINTS, labels
            assert labels["method"] in ALLOWED_METHODS, labels
            assert labels["error_class"] in ALLOWED_ERROR_CLASSES, labels
        elif name == f"{LATENCY_SECONDS}_bucket":
            assert set(labels) == {"endpoint", "method", "le"}, labels
            assert labels["endpoint"] in ALLOWED_ENDPOINTS, labels
            assert labels["method"] in ALLOWED_METHODS, labels
        elif name in (f"{LATENCY_SECONDS}_sum", f"{LATENCY_SECONDS}_count"):
            assert set(labels) == {"endpoint", "method"}, labels
            assert labels["endpoint"] in ALLOWED_ENDPOINTS, labels
            assert labels["method"] in ALLOWED_METHODS, labels
        elif name == UP:
            assert labels == {}


def test_up_gauge_is_one() -> None:
    text = client.get("/metrics/prometheus").text
    assert _get(UP, {}, text) == 1


def test_json_metrics_contract_unchanged() -> None:
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    assert set(r.json()) == JSON_METRICS_KEYS
    # The exposition endpoint did not alter the JSON snapshot's shape.
    client.get("/metrics/prometheus")
    r2 = client.get("/metrics")
    assert set(r2.json()) == JSON_METRICS_KEYS


def test_writer_is_deterministic_for_identical_state() -> None:
    client.post("/solve", json={"problem": "lasso", "method": "fista"})
    client.post("/solve", json={"problem": "lasso", "method": "nesterov"})  # 422 error
    client.get("/metrics/prometheus")  # ensure both requests above are recorded
    first = prometheus_metrics.render()
    second = prometheus_metrics.render()
    assert first == second
    # All four families have samples in the fully recorded state.
    names = {name for name, _, _ in _parse(first)}
    assert {
        REQUESTS_TOTAL,
        ERRORS_TOTAL,
        f"{LATENCY_SECONDS}_bucket",
        f"{LATENCY_SECONDS}_sum",
        f"{LATENCY_SECONDS}_count",
        UP,
    } <= names
    # A fresh response body parses as valid exposition with the same families.
    body = client.get("/metrics/prometheus").text
    assert {name for name, _, _ in _parse(body)} >= {
        REQUESTS_TOTAL,
        ERRORS_TOTAL,
        f"{LATENCY_SECONDS}_bucket",
        UP,
    }

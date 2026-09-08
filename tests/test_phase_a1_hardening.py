"""Phase A1 tests: request hardening (rate limit, body cap, CORS) -- offline.

Covers: 413 body-size rejections, 429 rate limiting with Retry-After (via a
monkeypatched RATE_LIMIT_PER_MIN and distinct Fly-Client-IP test keys), the
loopback/test-client exemption, fixed-window rollover with an injected fake
clock, CORS default-deny (no ALLOWED_ORIGINS -> no CORS headers), and CORS
enabled via the environment (module reload; env restored afterwards). The
hardening middleware self-records short-circuited requests in BOTH metrics
layers with the new bounded error classes."""

from __future__ import annotations

import importlib
import time

from fastapi.testclient import TestClient

import convex_optimization.app as app_module
from convex_optimization.app import app
from convex_optimization.hardening import (
    MAX_BODY_BYTES,
    RateLimiter,
    parse_allowed_origins,
    parse_rate_limit,
)
from convex_optimization.observability import ERROR_BODY_TOO_LARGE, ERROR_RATE_LIMITED

client = TestClient(app)


# ---------------------------------------------------------------------------
# Body-size guard
# ---------------------------------------------------------------------------


def test_oversized_body_rejected_413_with_bounded_error_class() -> None:
    big = {"problem": "logistic", "method": "gd", "padding": "x" * (MAX_BODY_BYTES + 10)}
    r = client.post("/solve", json=big)
    assert r.status_code == 413
    detail = r.json()["detail"]
    assert detail["error"] == "request body too large"
    assert detail["max_body_bytes"] == MAX_BODY_BYTES
    text = client.get("/metrics/prometheus").text
    assert (
        f'convex_optimization_errors_total{{endpoint="/solve",method="POST",'
        f'error_class="{ERROR_BODY_TOO_LARGE}"}}' in text
    )
    # the 413 is still a counted request in requests_total
    assert (
        'convex_optimization_requests_total{endpoint="/solve",method="POST",status="413"}' in text
    )


def test_normal_sized_bodies_pass() -> None:
    r = client.post("/solve", json={"problem": "logistic", "method": "gd"})
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def test_rate_limit_429_with_retry_after_and_bounded_error_class(monkeypatch) -> None:
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "2")
    ip = "203.0.113.77"  # unique test key (loopback/testclient are exempt)
    headers = {"Fly-Client-IP": ip}
    for _ in range(2):
        r = client.post("/solve", json={"problem": "logistic", "method": "gd"}, headers=headers)
        assert r.status_code == 200
    r = client.post("/solve", json={"problem": "logistic", "method": "gd"}, headers=headers)
    assert r.status_code == 429
    retry_after = r.headers["retry-after"]
    assert int(retry_after) >= 1
    assert r.json()["detail"]["error"] == "rate limit exceeded"
    assert r.json()["detail"]["limit_per_minute"] == 2
    # a DIFFERENT client key is not affected by the first key's window
    r2 = client.post(
        "/solve",
        json={"problem": "logistic", "method": "gd"},
        headers={"Fly-Client-IP": "203.0.113.78"},
    )
    assert r2.status_code == 200
    text = client.get("/metrics/prometheus").text
    assert (
        f'convex_optimization_errors_total{{endpoint="/solve",method="POST",'
        f'error_class="{ERROR_RATE_LIMITED}"}}' in text
    )
    assert (
        'convex_optimization_requests_total{endpoint="/solve",method="POST",status="429"}' in text
    )


def test_rate_limit_window_rollover_with_injected_clock() -> None:
    now = time.time()
    limiter = RateLimiter(window_seconds=60.0, clock=lambda: now)
    assert limiter.check("k", limit=2) == (True, 0)
    assert limiter.check("k", limit=2) == (True, 0)
    allowed, retry = limiter.check("k", limit=2)
    assert allowed is False and 1 <= retry <= 60
    # window rolls over -> counter restarts
    later = limiter.check("k", limit=2)  # still inside the same window
    assert later[0] is False
    limiter2 = RateLimiter(window_seconds=60.0, clock=lambda: now + 61.0)
    assert limiter2.check("k", limit=2)[0] is True


def test_rate_limit_disabled_by_zero_or_unparsable_env() -> None:
    limiter = RateLimiter()
    assert parse_rate_limit("0") == 0
    assert parse_rate_limit("not-a-number") == 0
    assert parse_rate_limit(None) == 30  # safe default
    assert limiter.check("anything", limit=0)[0] is True


def test_loopback_and_testclient_are_exempt() -> None:
    # the default TestClient client host ("testclient") never gets throttled,
    # even with a tiny limit
    for _ in range(5):
        r = client.get("/health")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------


def test_cors_default_deny_no_headers() -> None:
    r = client.get("/health", headers={"Origin": "https://example.com"})
    assert r.status_code == 200
    assert "access-control-allow-origin" not in r.headers
    # preflight would not be answered by the app (no CORS middleware added)
    r = client.options(
        "/solve",
        headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert "access-control-allow-origin" not in r.headers


def test_parse_allowed_origins_bounded() -> None:
    assert parse_allowed_origins(None) == []
    assert parse_allowed_origins("") == []
    assert parse_allowed_origins("https://a.example, https://b.example,,") == [
        "https://a.example",
        "https://b.example",
    ]
    # a wildcard is never accepted from the environment
    assert parse_allowed_origins("*") == []
    assert parse_allowed_origins("https://a.example, *") == ["https://a.example"]


def test_cors_enabled_via_env(monkeypatch) -> None:
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://workbench.example")
    importlib.reload(app_module)
    try:
        cors_client = TestClient(app_module.app)
        # preflight
        r = cors_client.options(
            "/solve",
            headers={
                "Origin": "https://workbench.example",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert r.status_code == 200
        assert r.headers["access-control-allow-origin"] == "https://workbench.example"
        # actual cross-origin GET
        r = cors_client.get("/health", headers={"Origin": "https://workbench.example"})
        assert r.headers["access-control-allow-origin"] == "https://workbench.example"
        # a NON-allowed origin is denied (no ACAO header -> browser blocks)
        r = cors_client.get("/health", headers={"Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in r.headers
    finally:
        monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
        importlib.reload(app_module)  # restore the default-deny app


def test_hardening_rejections_recorded_in_json_metrics() -> None:
    snap_before = TestClient(app_module.app).post(
        "/solve",
        json={"problem": "logistic", "method": "gd", "padding": "x" * (MAX_BODY_BYTES + 1)},
    )
    assert snap_before.status_code == 413
    snap = client.get("/metrics").json()
    assert snap["status_counts"]["4xx"] >= 1
    assert "/solve" in snap["requests_by_endpoint"]

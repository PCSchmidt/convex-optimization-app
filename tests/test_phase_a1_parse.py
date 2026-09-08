"""Phase A1 tests: the /parse pipeline (offline, stub provider + fake LLM).

Covers: the stub provider success path (verified parse), the unverified
verdict (200 with verified=false and mismatch reasons), garbage text (422
parse_invalid), provider-not-configured (503), the OpenAI-compatible provider
with a monkeypatched urllib transport (success, invalid JSON, upstream
failure), the deterministic verifier at unit level, and the bounded
parse-outcome Prometheus family. No network, no keys."""

from __future__ import annotations

import io
import json
import urllib.error

import pytest
from fastapi.testclient import TestClient

from convex_optimization import parsing
from convex_optimization.app import app
from convex_optimization.observability import (
    ERROR_PARSE_INVALID,
    ERROR_PROVIDER_NOT_CONFIGURED,
    ERROR_PROVIDER_UNAVAILABLE,
)
from convex_optimization.prometheus import prometheus_metrics

client = TestClient(app)


@pytest.fixture(autouse=True)
def _stub_provider(monkeypatch):
    """Point the provider at the documented test double for this module."""
    monkeypatch.setenv("LLM_PROVIDER", "stub")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    yield


# ---------------------------------------------------------------------------
# Stub provider (test double) end to end
# ---------------------------------------------------------------------------


def test_parse_success_verified() -> None:
    r = client.post(
        "/parse",
        json={"text": "minimize the lasso objective with 50 variables, seed 7, lambda 0.2"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["parse_method"] == "stub"
    assert body["verified"] is True
    assert body["mismatches"] == []
    req = body["problem_request"]
    assert req["problem"] == "lasso"
    assert req["method"] == "fista"  # suggested method for the nonsmooth lasso
    assert req["params"]["seed"] == 7
    assert req["params"]["n_vars"] == 50
    assert req["params"]["lam"] == 0.2
    assert req["params"]["n_rows"] == 60  # deterministic auto-fill
    # The returned request is directly POSTable to /solve. (This particular
    # 60x50 lasso does not reach tol=1e-10 within max_iter=2000 -- that is
    # the documented convergence-failure signal, so only the 200 is asserted.)
    solve = client.post("/solve", json=req)
    assert solve.status_code == 200
    assert solve.json()["problem"] == "lasso"
    # bounded outcome family counted
    assert prometheus_metrics.parse_outcome_counts().get("verified", 0) >= 1


def test_parse_least_squares_suggested_method_is_nesterov() -> None:
    r = client.post("/parse", json={"text": "solve least squares with 20 variables, seed 2"})
    assert r.status_code == 200
    req = r.json()["problem_request"]
    assert req["problem"] == "least_squares"
    assert req["method"] == "nesterov"


def test_parse_unverified_conflicting_numbers() -> None:
    r = client.post(
        "/parse",
        json={"text": "lasso with 30 variables and 50 features"},
    )
    assert r.status_code == 200  # parsed, but the text does not support it
    body = r.json()
    assert body["verified"] is False
    assert body["mismatches"], "conflicting dimension numbers must surface"
    assert prometheus_metrics.parse_outcome_counts().get("parsed", 0) >= 1


def test_parse_garbage_is_422_parse_invalid() -> None:
    r = client.post("/parse", json={"text": "hello world, what is the weather"})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"].startswith("could not parse")
    text = client.get("/metrics/prometheus").text
    assert (
        f'convex_optimization_errors_total{{endpoint="/parse",method="POST",'
        f'error_class="{ERROR_PARSE_INVALID}"}}' in text
    )
    assert 'convex_optimization_parse_outcomes_total{outcome="failed"}' in text


def test_parse_provider_not_configured_is_clean_503(monkeypatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    r = client.post("/parse", json={"text": "minimize least squares with 10 variables"})
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert detail["error"] == "no LLM provider configured"
    assert "LLM_API_KEY" in detail["hint"]
    text = client.get("/metrics/prometheus").text
    assert (
        f'convex_optimization_errors_total{{endpoint="/parse",method="POST",'
        f'error_class="{ERROR_PROVIDER_NOT_CONFIGURED}"}}' in text
    )
    assert 'convex_optimization_parse_outcomes_total{outcome="provider_not_configured"}' in text


def test_parse_rejects_unknown_fields_and_empty_text() -> None:
    r = client.post("/parse", json={"text": "lasso with seed 3", "extra": 1})
    assert r.status_code == 422
    r = client.post("/parse", json={"text": ""})
    assert r.status_code == 422
    r = client.post("/parse", json={"text": "x" * (parsing.MAX_PARSE_TEXT_CHARS + 1)})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# OpenAI-compatible provider (stdlib urllib; offline via a fake urlopen)
# ---------------------------------------------------------------------------


def _fake_urlopen(payload: dict):
    class _Response(io.BytesIO):
        def __enter__(self):  # context manager protocol
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(payload).encode()

    def fake(request, timeout=None):  # signature-agnostic
        return _Response()

    return fake


def _canned_content(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def test_openai_provider_success_path(monkeypatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)  # exercise the real provider path
    spec_json = json.dumps({"problem": "lasso", "params": {"seed": 7, "n_vars": 50, "lam": 0.2}})
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setattr(
        parsing.urllib.request, "urlopen", _fake_urlopen(_canned_content(spec_json))
    )
    r = client.post(
        "/parse", json={"text": "minimize the lasso with 50 variables, seed 7, lambda 0.2"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["parse_method"] == "llm"
    assert body["verified"] is True
    assert body["problem_request"]["problem"] == "lasso"


def test_openai_provider_garbage_json_is_422(monkeypatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)  # exercise the real provider path
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setattr(
        parsing.urllib.request,
        "urlopen",
        _fake_urlopen(_canned_content("I think the answer is a lasso problem, maybe")),
    )
    r = client.post("/parse", json={"text": "lasso with 10 variables"})
    assert r.status_code == 422
    assert r.json()["detail"]["error"].startswith("could not parse")


def test_openai_provider_upstream_failure_is_502(monkeypatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)  # exercise the real provider path
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    def failing(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(parsing.urllib.request, "urlopen", failing)
    r = client.post("/parse", json={"text": "lasso with 10 variables"})
    assert r.status_code == 502
    text = client.get("/metrics/prometheus").text
    assert (
        f'convex_optimization_errors_total{{endpoint="/parse",method="POST",'
        f'error_class="{ERROR_PROVIDER_UNAVAILABLE}"}}' in text
    )
    assert 'convex_optimization_parse_outcomes_total{outcome="provider_unavailable"}' in text


def test_openai_provider_needs_key() -> None:
    provider = parsing.OpenAICompatibleProvider(api_key="")
    with pytest.raises(parsing.ProviderNotConfigured):
        provider.parse("lasso with 10 variables")


# ---------------------------------------------------------------------------
# Deterministic verifier (unit level)
# ---------------------------------------------------------------------------


def test_verifier_flags_invented_values() -> None:
    verified, mismatches = parsing.verify_parse("lasso with 10 variables", "lasso", {"seed": 7})
    assert verified is False
    assert any("seed" in m for m in mismatches)


def test_verifier_accepts_minimal_honest_parse() -> None:
    verified, mismatches = parsing.verify_parse("the lasso problem", "lasso", {})
    assert verified is True
    assert mismatches == []


def test_verifier_flags_wrong_kind() -> None:
    verified, mismatches = parsing.verify_parse("plain least squares", "lasso", {})
    assert verified is False
    assert any("least_squares" in m or "keywords" in m for m in mismatches)


def test_parse_outcome_vocabulary_is_bounded() -> None:
    from convex_optimization.prometheus import PARSE_OUTCOMES

    assert set(PARSE_OUTCOMES) == {
        "parsed",
        "verified",
        "failed",
        "provider_unavailable",
        "provider_not_configured",
    }
    with pytest.raises(ValueError):
        prometheus_metrics.record_parse("bogus_outcome")

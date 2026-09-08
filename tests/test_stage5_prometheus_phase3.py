"""Phase 3 tests: convex-app-specific Prometheus families (offline, TestClient).

Extends the Phase 2 exposition with the convex-specific families:
solves_total, convergence_successes_total / convergence_failures_total,
solve_latency_seconds, iterations, final_objective_gap -- all labelled ONLY
by registry problem/solver_method names (cli.PROBLEMS / cli.METHODS), recorded
only for HTTP 200 solves. Inapplicable-pair 422s must create NO
problem/solver_method series. No network, no new dependencies."""

from __future__ import annotations

import math
import re

import pytest
from fastapi.testclient import TestClient

from convex_optimization.app import app
from convex_optimization.cli import APPLICABLE, METHODS, PROBLEMS
from convex_optimization.prometheus import (
    CONVERGENCE_FAILURES_TOTAL,
    CONVERGENCE_SUCCESSES_TOTAL,
    FINAL_OBJECTIVE_GAP,
    GAP_BUCKETS,
    ITERATION_BUCKETS,
    ITERATIONS,
    SOLVE_LATENCY_SECONDS,
    SOLVES_TOTAL,
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

DOMAIN_FAMILIES = {
    SOLVES_TOTAL,
    CONVERGENCE_SUCCESSES_TOTAL,
    CONVERGENCE_FAILURES_TOTAL,
    f"{SOLVE_LATENCY_SECONDS}_bucket",
    f"{SOLVE_LATENCY_SECONDS}_sum",
    f"{SOLVE_LATENCY_SECONDS}_count",
    f"{ITERATIONS}_bucket",
    f"{ITERATIONS}_sum",
    f"{ITERATIONS}_count",
    f"{FINAL_OBJECTIVE_GAP}_bucket",
    f"{FINAL_OBJECTIVE_GAP}_sum",
    f"{FINAL_OBJECTIVE_GAP}_count",
}

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


def _le_values(name: str, base: dict[str, str], text: str) -> list[str]:
    return [
        labels["le"]
        for n, labels, _ in _parse(text)
        if n == name and labels.items() >= base.items()
    ]


@pytest.fixture(autouse=True)
def _reset_prometheus_singleton():
    prometheus_metrics.reset()
    yield
    prometheus_metrics.reset()


def test_successful_solve_increments_solves_total_and_convergence_success() -> None:
    r = client.post("/solve", json={"problem": "lasso", "method": "fista"})
    assert r.status_code == 200
    body = r.json()
    assert body["converged"] is True  # benchmark lasso+fista converges (1436 iters)
    text = client.get("/metrics/prometheus").text
    pair = {"problem": "lasso", "solver_method": "fista"}
    assert set(pair) <= {k for n, ls, _ in _parse(text) if n == SOLVES_TOTAL for k in ls}
    assert _get(SOLVES_TOTAL, pair, text) == 1
    assert _get(CONVERGENCE_SUCCESSES_TOTAL, pair, text) == 1
    assert _get(CONVERGENCE_FAILURES_TOTAL, pair, text) is None
    # Exactly one successful solve recorded in total.
    assert sum(v for n, _, v in _parse(text) if n == SOLVES_TOTAL) == 1


def test_solve_histograms_expose_bucket_sum_count_after_solve() -> None:
    r = client.post("/solve", json={"problem": "lasso", "method": "fista"})
    assert r.status_code == 200
    body = r.json()
    text = client.get("/metrics/prometheus").text
    pair = {"problem": "lasso", "solver_method": "fista"}
    for family in (SOLVE_LATENCY_SECONDS, ITERATIONS, FINAL_OBJECTIVE_GAP):
        for suffix in ("_sum", "_count"):
            assert _get(f"{family}{suffix}", pair, text) is not None, family + suffix
        # _bucket lines carry the extra `le` label, so match on a label
        # superset (same convention as _le_values below), not exact equality.
        for n, ls, _ in _parse(text):
            if n == f"{family}_bucket":
                assert ls.keys() >= pair.keys(), f"{family}_bucket labels {ls}"
                break
        else:
            pytest.fail(f"{family}_bucket series missing for {pair}")
    # Iterations: single observation, so _sum equals the reported iteration
    # count and exactly one bucket is non-zero (cumulative step at its bound).
    assert _get(f"{ITERATIONS}_count", pair, text) == 1
    assert _get(f"{ITERATIONS}_sum", pair, text) == body["iterations"]
    cumulative = [
        v for n, ls, v in _parse(text) if n == f"{ITERATIONS}_bucket" and ls.items() >= pair.items()
    ]
    assert cumulative == sorted(cumulative)
    assert cumulative[-2:] == [1, 1]  # one observation at/below the last bound
    # Latency and gap: one observation each.
    assert _get(f"{SOLVE_LATENCY_SECONDS}_count", pair, text) == 1
    assert _get(f"{FINAL_OBJECTIVE_GAP}_count", pair, text) == 1
    gap_sum = _get(f"{FINAL_OBJECTIVE_GAP}_sum", pair, text)
    assert gap_sum is not None
    assert math.isclose(gap_sum, body["final_objective_gap"], rel_tol=1e-6, abs_tol=1e-15)


def test_documented_bucket_layouts_exposed() -> None:
    client.post("/solve", json={"problem": "logistic", "method": "gd"})
    text = client.get("/metrics/prometheus").text
    pair = {"problem": "logistic", "solver_method": "gd"}
    assert _le_values(f"{ITERATIONS}_bucket", pair, text) == [
        f"{b:g}" for b in ITERATION_BUCKETS
    ] + ["+Inf"]
    assert _le_values(f"{FINAL_OBJECTIVE_GAP}_bucket", pair, text) == [
        f"{b:g}" for b in GAP_BUCKETS
    ] + ["+Inf"]
    # Solve latency reuses the documented Phase 2 latency layout.
    from convex_optimization.prometheus import LATENCY_BUCKETS

    assert _le_values(f"{SOLVE_LATENCY_SECONDS}_bucket", pair, text) == [
        f"{b:g}" for b in LATENCY_BUCKETS
    ] + ["+Inf"]


def test_inapplicable_pair_422_creates_no_problem_series() -> None:
    r = client.post("/solve", json={"problem": "lasso", "method": "nesterov"})
    assert r.status_code == 422
    text = client.get("/metrics/prometheus").text
    domain_samples = [(n, ls) for n, ls, _ in _parse(text) if n in DOMAIN_FAMILIES]
    assert domain_samples == [], domain_samples
    # The request IS still counted as a bounded error, never as a solve.
    assert (
        _get(
            "convex_optimization_errors_total",
            {"endpoint": "/solve", "method": "POST", "error_class": "inapplicable_pair"},
            text,
        )
        == 1
    )


def test_validation_error_422_creates_no_problem_series() -> None:
    r = client.post("/solve", json={"problem": "lasso", "method": "fista", "tail": 500})
    assert r.status_code == 422
    text = client.get("/metrics/prometheus").text
    assert [(n, ls) for n, ls, _ in _parse(text) if n in DOMAIN_FAMILIES] == []


def test_label_values_stay_within_registry_sets_after_varied_requests() -> None:
    seen_solves = 0
    for problem in APPLICABLE:
        for method in METHODS:
            r = client.post("/solve", json={"problem": problem, "method": method})
            if r.status_code == 200:
                seen_solves += 1
            else:
                assert r.status_code == 422
    client.get("/no-such-route")  # 404 must not create domain series either
    text = client.get("/metrics/prometheus").text
    total_solves = 0
    for name, labels, _value in _parse(text):
        if name in DOMAIN_FAMILIES:
            if name.endswith(("_bucket", "_sum", "_count")):
                assert set(labels) == {"problem", "solver_method"} | (
                    {"le"} if name.endswith("_bucket") else set()
                ), (name, labels)
            else:
                assert set(labels) == {"problem", "solver_method"}, (name, labels)
            assert labels["problem"] in set(PROBLEMS), labels
            assert labels["solver_method"] in set(METHODS), labels
            if name == SOLVES_TOTAL:
                total_solves += int(_value)
    assert total_solves == seen_solves == len(APPLICABLE) * 0 + 6  # 6 applicable pairs
    # Registry sanity: APPLICABLE is a subset of the name registries.
    assert set(APPLICABLE) <= set(PROBLEMS)
    assert {m for ms in APPLICABLE.values() for m in ms} <= set(METHODS)


def test_convergence_failure_counter_recorded_directly_on_collector() -> None:
    # No API-reachable non-converging case exists: every Stage 2 benchmark
    # (problem, method) run converged=true (see experiments/run_log.json), so
    # the failure counter is exercised directly on the collector (no network).
    prometheus_metrics.record_solve(
        "lasso",
        "ista",
        latency_seconds=0.001,
        iterations=2000,
        final_objective_gap=1e-4,
        converged=False,
    )
    text = prometheus_metrics.render()
    pair = {"problem": "lasso", "solver_method": "ista"}
    assert _get(SOLVES_TOTAL, pair, text) == 1
    assert _get(CONVERGENCE_FAILURES_TOTAL, pair, text) == 1
    assert _get(CONVERGENCE_SUCCESSES_TOTAL, pair, text) is None
    # The non-converged solve is still observed in the histograms (HTTP 200).
    assert _get(f"{ITERATIONS}_count", pair, text) == 1
    assert _get(f"{FINAL_OBJECTIVE_GAP}_count", pair, text) == 1


def test_json_metrics_contract_still_unchanged_after_solves() -> None:
    client.post("/solve", json={"problem": "logistic", "method": "nesterov"})
    r = client.get("/metrics")
    assert r.status_code == 200
    assert set(r.json()) == JSON_METRICS_KEYS
    assert r.json()["solve"]["count"] >= 1


def test_exposition_remains_deterministic_with_domain_families() -> None:
    client.post("/solve", json={"problem": "logistic", "method": "gd"})
    client.get("/metrics/prometheus")  # ensure the solve above is recorded
    assert prometheus_metrics.render() == prometheus_metrics.render()

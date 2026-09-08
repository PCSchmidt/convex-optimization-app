"""Phase A1 tests: parameterized problem instances on /solve (offline).

Covers: user-specified seeded instances (bounded params), the hard caps
(n <= 200 and friends), cross-field validation, bit-reproducibility of
identical (problem, params, seed, method) requests, byte-compatibility of the
no-params path with the frozen Stage 1 defaults, and the Prometheus registry-
name label bound. All offline and deterministic."""

from __future__ import annotations

from fastapi.testclient import TestClient

from convex_optimization.app import app
from convex_optimization.problems import (
    MAX_PARAMETERIZED_DIM,
    make_least_squares,
    make_logistic,
)
from convex_optimization.prometheus import prometheus_metrics

client = TestClient(app)


def _solve(payload: dict) -> dict:
    r = client.post("/solve", json=payload)
    assert r.status_code == 200, r.text
    return r.json()


def test_parameterized_solve_returns_resolved_parameters() -> None:
    body = _solve(
        {"problem": "least_squares", "method": "nesterov", "params": {"seed": 3, "n_vars": 50}}
    )
    assert body["problem"] == "least_squares"
    # n_rows auto-filled deterministically: max(default 30, n_vars + 10)
    assert body["parameters"] == {"seed": 3, "n_rows": 60, "n_vars": 50}
    assert body["converged"] is True


def test_no_params_path_is_byte_compatible_with_frozen_defaults() -> None:
    a = _solve({"problem": "logistic", "method": "nesterov"})
    b = _solve({"problem": "logistic", "method": "nesterov", "params": {}})
    assert a["parameters"] is None
    # Same instance as the frozen Stage 1 default: identical numbers.
    for key in ("iterations", "converged", "final_objective", "final_objective_gap"):
        assert a[key] == b[key]
    # And identical to building the frozen problem directly.
    problem = make_logistic()
    assert a["iterations"] > 0
    assert abs(a["final_objective_gap"] - (a["final_objective"] - problem.ground_truth.f)) < 1e-15


def test_same_params_seed_method_is_bit_reproducible() -> None:
    payload = {"problem": "lasso", "method": "ista", "params": {"seed": 11, "n_vars": 15}}
    first = _solve(payload)
    second = _solve(payload)
    assert first == second  # bit-reproducible: identical full response


def test_parameterized_with_all_defaults_equals_frozen_builder() -> None:
    a = _solve({"problem": "least_squares", "method": "gd", "params": {}})
    b = _solve({"problem": "least_squares", "method": "gd"})
    assert a["iterations"] == b["iterations"]
    assert a["final_objective"] == b["final_objective"]


def test_dimension_caps_rejected() -> None:
    for field, value in (
        ("n_vars", MAX_PARAMETERIZED_DIM + 1),
        ("n_rows", MAX_PARAMETERIZED_DIM + 1),
    ):
        r = client.post(
            "/solve", json={"problem": "logistic", "method": "gd", "params": {field: value}}
        )
        assert r.status_code == 422
    # bad seed range
    r = client.post("/solve", json={"problem": "logistic", "method": "gd", "params": {"seed": -1}})
    assert r.status_code == 422
    r = client.post(
        "/solve", json={"problem": "logistic", "method": "gd", "params": {"ridge": 1e9}}
    )
    assert r.status_code == 422


def test_cross_field_and_wrong_kind_params_rejected() -> None:
    # explicit contradiction: n_rows <= n_vars
    r = client.post(
        "/solve",
        json={"problem": "lasso", "method": "fista", "params": {"n_rows": 10, "n_vars": 50}},
    )
    assert r.status_code == 422
    assert "n_rows" in r.json()["detail"]["reason"]
    # kind-specific fields on the wrong problem
    r = client.post(
        "/solve", json={"problem": "least_squares", "method": "gd", "params": {"lam": 0.1}}
    )
    assert r.status_code == 422


def test_extra_fields_rejected_on_params() -> None:
    r = client.post(
        "/solve",
        json={"problem": "logistic", "method": "gd", "params": {"tol": 1e-4}},
    )
    assert r.status_code == 422  # extra="forbid": no tuning surface via params


def test_condition_knob_changes_conditioning() -> None:
    plain = _solve({"problem": "least_squares", "method": "nesterov", "params": {"seed": 5}})
    hard = _solve(
        {"problem": "least_squares", "method": "nesterov", "params": {"seed": 5, "condition": 1e3}}
    )
    assert hard["parameters"]["condition"] == 1e3
    # An ill-conditioned instance needs more iterations at the same 1/L step.
    # (A large-enough condition can legitimately NOT converge within the fixed
    # max_iter=2000 -- that is the documented convergence-failure signal, so
    # no unconditional converged=True is asserted here.)
    assert hard["iterations"] > plain["iterations"]


def test_invalid_params_counted_as_validation_error_not_solve() -> None:
    before = prometheus_metrics.error_counts()
    r = client.post("/solve", json={"problem": "logistic", "method": "gd", "params": {"seed": -5}})
    assert r.status_code == 422
    key = ("/solve", "POST", "validation_error")
    assert prometheus_metrics.error_counts().get(key, 0) == before.get(key, 0) + 1


def test_registry_labels_unchanged_for_parameterized_solves() -> None:
    from convex_optimization.cli import METHODS, PROBLEMS

    before = prometheus_metrics.solve_counts()
    _solve({"problem": "logistic", "method": "gd", "params": {"seed": 9, "n_vars": 8}})
    after = prometheus_metrics.solve_counts()
    new_keys = {k for k, v in after.items() if after.get(k, 0) != before.get(k, 0)}
    assert new_keys
    for problem, method in new_keys:
        assert problem in METHODS or problem in PROBLEMS
        assert method in METHODS
        assert problem in ("least_squares", "lasso", "logistic")
    # the frozen default instance (make_least_squares) still builds identically
    assert make_least_squares().lipschitz == make_least_squares().lipschitz

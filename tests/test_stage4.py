"""Stage 4 tests: the FastAPI serving layer (offline, via TestClient).

Covers GET /health and POST /solve for a valid pair, an inapplicable pair
(422 with the applicable-methods list, mirroring the CLI), and request
validation errors. All offline: seeded problems, no network, no keys, and
Stage 1-2 default settings only (no tuning surface)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from convex_optimization.app import app
from convex_optimization.cli import APPLICABLE

client = TestClient(app)


def test_health() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_solve_valid_smooth_pair() -> None:
    r = client.post("/solve", json={"problem": "logistic", "method": "nesterov"})
    assert r.status_code == 200
    body = r.json()
    assert body["problem"] == "logistic"
    assert body["method"] == "nesterov"
    assert body["converged"] is True
    assert abs(body["final_objective_gap"]) <= 1e-8
    assert body["final_residual"] <= 1e-6
    assert 1 <= len(body["history_tail"]) <= 20
    assert body["history_tail"][-1]["iteration"] == body["iterations"]


def test_solve_valid_proximal_pair() -> None:
    r = client.post("/solve", json={"problem": "lasso", "method": "fista", "tail": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["converged"] is True
    assert abs(body["final_objective_gap"]) <= 1e-8
    assert body["final_residual"] <= 1e-6
    assert len(body["history_tail"]) == 3
    # The lasso truth is the documented eps-smoothed SciPy approximate reference.
    assert "smoothed" in body["ground_truth_source"]


def test_solve_deterministic() -> None:
    a = client.post("/solve", json={"problem": "least_squares", "method": "gd"}).json()
    b = client.post("/solve", json={"problem": "least_squares", "method": "gd"}).json()
    assert a["iterations"] == b["iterations"]
    assert a["final_objective_gap"] == b["final_objective_gap"]


def test_solve_inapplicable_pair_returns_422_with_applicable_list() -> None:
    r = client.post("/solve", json={"problem": "lasso", "method": "nesterov"})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["applicable_methods"] == list(APPLICABLE["lasso"])
    assert "not applicable" in detail["error"]


def test_solve_unknown_method_is_4xx() -> None:
    r = client.post("/solve", json={"problem": "logistic", "method": "admm"})
    assert 400 <= r.status_code < 500


def test_solve_unknown_problem_is_4xx() -> None:
    r = client.post("/solve", json={"problem": "portfolio", "method": "gd"})
    assert 400 <= r.status_code < 500


def test_solve_tail_bounded() -> None:
    r = client.post("/solve", json={"problem": "logistic", "method": "gd", "tail": 500})
    assert r.status_code == 422  # tail > 20 rejected by request validation

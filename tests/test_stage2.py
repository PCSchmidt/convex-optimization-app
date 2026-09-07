"""Stage 2 evaluation tests: the added ISTA method and the applicable-pairs
matrix. Fast and offline; no wall-time assertions (too noisy for CI).

ISTA is FISTA minus the momentum sequence, with the IDENTICAL Stage 1 step
(1/L, prox at step 1/L) and identical tolerance/stopping rule. It exists so
the Stage 2 benchmark compares accelerated vs non-accelerated proximal
gradient without changing any constant.
"""

from __future__ import annotations

import numpy as np

from convex_optimization import cli
from convex_optimization.methods import ista
from convex_optimization.problems import make_lasso, make_least_squares

GAP_TOL = 1e-8  # same criterion as Stage 1 (absolute gap vs the approximate lasso reference)
KKT_TOL = 1e-6  # prox-gradient-map residual, same criterion as Stage 1


def test_applicable_pairs_matrix_has_two_lasso_prox_methods():
    assert cli.APPLICABLE == {
        "least_squares": ("gd", "nesterov"),
        "lasso": ("fista", "ista"),
        "logistic": ("gd", "nesterov"),
    }
    # inapplicable pairs stay rejected (smooth methods on the nonsmooth lasso)
    import pytest

    with pytest.raises(ValueError):
        cli.solve("lasso", "gd", max_iter=10, tol=1e-8)


def test_ista_matches_lasso_reference_with_stage1_constants():
    problem = make_lasso()
    result = ista(
        problem.objective,
        problem.smooth_gradient,
        problem.prox,
        problem.x0,
        smooth_lipschitz=problem.lipschitz,
        max_iter=2000,
        tol=1e-10,
    )
    assert result.converged
    assert abs(result.final_objective_gap(problem.ground_truth.f)) <= GAP_TOL
    assert result.final_residual() <= KKT_TOL


def test_ista_proximal_gradient_monotone_decrease():
    """With the 1/L step the proximal-gradient map is monotone for the full
    objective, so ISTA (unlike FISTA) must decrease the objective every step."""
    problem = make_lasso()
    result = ista(
        problem.objective,
        problem.smooth_gradient,
        problem.prox,
        problem.x0,
        smooth_lipschitz=problem.lipschitz,
        max_iter=200,  # do not need convergence to check monotonicity
        tol=1e-300,
    )
    objs = result.history.objectives
    assert all(objs[i + 1] <= objs[i] + 1e-15 for i in range(len(objs) - 1))


def test_ista_reduces_to_slow_but_correct_solver_on_smooth_problem():
    """ISTA with the identity prox is plain fixed-step gradient descent."""
    problem = make_least_squares()
    result = ista(
        problem.objective,
        problem.gradient,
        lambda x: x,  # identity prox: composite problem reduces to the smooth one
        problem.x0,
        smooth_lipschitz=problem.lipschitz,
        max_iter=2000,
        tol=1e-10,
    )
    assert result.converged
    assert float(np.linalg.norm(result.x - problem.ground_truth.x)) <= 1e-6


def test_benchmark_runner_writes_schema_compliant_rows(tmp_path, monkeypatch):
    """Smoke-check the runner on one cell: schema, types, determinism of n_iter."""
    import csv as _csv
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    import run_benchmark

    row = run_benchmark.run_cell("logistic", 1, "nesterov")
    required = {
        "problem",
        "seed",
        "method",
        "n_iter",
        "wall_time_s",
        "final_gap",
        "final_residual",
        "converged",
        "notes",
    }
    assert required <= set(row)
    assert row["problem"] == "logistic" and row["seed"] == 1 and row["method"] == "nesterov"
    assert row["converged"] == "True"
    assert int(row["n_iter"]) > 0
    assert float(row["wall_time_s"]) >= 0.0
    # the CSV metadata columns round-trip
    fieldnames = ["date_utc", "git_commit", "numpy_version", "scipy_version", *required]
    out = tmp_path / "row.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerow(row)
    with open(out, encoding="utf-8") as f:
        back = next(_csv.DictReader(f))
    assert int(back["n_iter"]) == row["n_iter"]

"""Stage 1 correctness tests: self-implemented methods vs ground truth.

Tolerance criteria are stated per problem and chosen honestly:

- least_squares (smooth, strongly convex, cond(L) ~ 67, closed-form truth):
  iterate distance ``|x_final - x*| <= 1e-6`` plus objective gap <= 1e-8.
- lasso (nonsmooth; the minimizer is unique here but the ground truth is an
  eps-smoothed SciPy reference, so x-distance is not meaningful against it):
  absolute objective gap to the reference <= 1e-8, and the reference itself is
  certified by a prox-gradient-map residual <= 1e-6.
- logistic (smooth, strongly convex, cond(L) ~ 5.5, SciPy truth):
  objective gap <= 1e-8 plus gradient norm <= 1e-6.

These are final-accuracy criteria; no wall-time or method-vs-method
benchmarking is claimed (that is Stage 2 work).
"""

from __future__ import annotations

import numpy as np
import pytest

from convex_optimization import cli
from convex_optimization.history import History, Result
from convex_optimization.methods import fista, gradient_descent
from convex_optimization.problems import make_problem

X_TOL = 1e-6  # iterate distance (only used vs exact closed-form truth)
GAP_TOL = 1e-8  # objective gap vs ground truth
GRAD_TOL = 1e-6  # gradient norm at final iterate (smooth problems)
KKT_TOL = 1e-6  # prox-gradient-map residual certifying the SciPy lasso point


def _final_xdist(result, truth_x: np.ndarray) -> float:
    return float(np.linalg.norm(result.x - truth_x))


def test_least_squares_closed_form_matches_scipy():
    """The closed-form truth really is the optimum: SciPy L-BFGS-B agrees."""
    problem = make_problem("least_squares")
    from scipy.optimize import minimize

    ref = minimize(
        problem.objective,
        problem.x0,
        jac=problem.gradient,
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-18, "gtol": 1e-14},
    )
    assert abs(ref.fun - problem.ground_truth.f) <= 1e-10


def test_least_squares_gd_and_nesterov_reach_truth():
    problem = make_problem("least_squares")
    for method in ("gd", "nesterov"):
        _, result = cli.solve("least_squares", method, max_iter=2000, tol=1e-10)
        assert result.converged
        assert _final_xdist(result, problem.ground_truth.x) <= X_TOL
        assert result.final_objective_gap(problem.ground_truth.f) <= GAP_TOL


def test_lasso_fista_matches_scipy_reference():
    problem = make_problem("lasso")
    # certify the SciPy reference point satisfies the Lasso KKT condition
    assert problem.residual(problem.ground_truth.x) <= KKT_TOL
    _, result = cli.solve("lasso", "fista", max_iter=5000, tol=1e-10)
    assert result.converged
    assert abs(result.final_objective_gap(problem.ground_truth.f)) <= GAP_TOL
    assert result.final_residual() <= KKT_TOL


def test_logistic_gd_and_nesterov_match_scipy_truth():
    problem = make_problem("logistic")
    for method in ("gd", "nesterov"):
        _, result = cli.solve("logistic", method, max_iter=2000, tol=1e-10)
        assert result.converged
        assert abs(result.final_objective_gap(problem.ground_truth.f)) <= GAP_TOL
        assert result.final_residual() <= GRAD_TOL


def test_every_method_returns_full_history():
    _, result = cli.solve("least_squares", "gd", max_iter=50, tol=1e-300)
    assert result.n_iter == 50
    assert len(result.history) == result.n_iter + 1  # initial point + updates
    assert all(np.isfinite(result.history.objectives))
    assert all(np.isfinite(result.history.residuals))
    # fixed-step gradient descent on a smooth objective decreases monotonically
    objs = result.history.objectives
    assert all(objs[i + 1] <= objs[i] + 1e-15 for i in range(len(objs) - 1))


def test_history_tail_rows():
    history = History()
    for k in range(10):
        history.append(float(k), float(k) / 10)
    rows = history.tail(3)
    assert rows == [(7, 7.0, 0.7), (8, 8.0, 0.8), (9, 9.0, 0.9)]
    assert len(history) == 10


def test_result_gap_is_objective_difference():
    x = np.zeros(2)
    history = History()
    history.append(1.0, 0.5)
    result = Result(x=x, history=history, n_iter=0, converged=False, tol=1e-8)
    assert result.final_objective_gap(0.25) == pytest.approx(0.75)


def test_backtracking_gd_decreases_monotonically_without_known_lipschitz():
    """Backtracking GD reaches the optimum, with a documented float64 floor.

    Near the minimizer the required Armijo trial step becomes so small that
    ``x - t * g`` rounds back to ``x`` in float64; the gradient norm then
    stalls around 1e-7 instead of reaching 1e-10. We therefore assert
    convergence at tol=1e-7 plus the objective-gap criterion (the gap is far
    below GAP_TOL even at that gradient-norm floor).
    """
    problem = make_problem("least_squares")
    result = gradient_descent(
        problem.objective,
        problem.gradient,
        problem.x0,
        step_size="backtracking",
        max_iter=2000,
        tol=1e-7,
    )
    assert result.converged
    assert _final_xdist(result, problem.ground_truth.x) <= X_TOL
    assert result.final_objective_gap(problem.ground_truth.f) <= GAP_TOL
    objs = result.history.objectives
    assert all(objs[i + 1] <= objs[i] + 1e-15 for i in range(len(objs) - 1))


def test_fista_on_smooth_problem_with_identity_prox_also_converges():
    """FISTA reduces to accelerated gradient when prox is the identity."""
    problem = make_problem("least_squares")
    identity = lambda x: x
    result = fista(
        problem.objective,
        problem.gradient,
        identity,
        problem.x0,
        smooth_lipschitz=problem.lipschitz,
        max_iter=2000,
        tol=1e-10,
    )
    assert result.converged
    assert _final_xdist(result, problem.ground_truth.x) <= X_TOL


def test_invalid_step_size_and_bad_pairs_raise():
    problem = make_problem("least_squares")
    with pytest.raises(ValueError):
        gradient_descent(problem.objective, problem.gradient, problem.x0, step_size=0.0)
    with pytest.raises(ValueError):
        cli.solve("lasso", "gd", max_iter=10, tol=1e-8)  # gd is smooth-only here
    with pytest.raises(KeyError):
        cli.solve("nope", "gd", max_iter=10, tol=1e-8)


def test_cli_prints_gap_and_history_rows(capsys):
    code = cli.main(["--problem", "logistic", "--method", "gd"])
    assert code == 0
    out = capsys.readouterr().out
    assert "problem=logistic" in out
    assert "final_objective_gap=" in out
    assert "ground_truth_source" in out
    # the tail is printed as integer-iteration rows
    assert "  " in out and any(line.strip()[0].isdigit() for line in out.splitlines())

"""CLI: run ONE problem with ONE method and report convergence vs ground truth.

Usage (from the repo root, with the project venv active):

    python -m convex_optimization.cli --problem lasso --method fista

Prints the final objective gap against the SciPy (or closed-form) ground
truth, the stopping criterion, and the last few history rows
(iteration, objective, residual). Everything runs offline and deterministically.
"""

from __future__ import annotations

import argparse

from .methods import fista, gradient_descent, nesterov_ag
from .problems import make_problem

METHODS = ("gd", "nesterov", "fista")
PROBLEMS = ("least_squares", "lasso", "logistic")

# Which (problem, method) pairs are mathematically meaningful. The methods are
# first-order: gradient descent and Nesterov need a smooth objective; FISTA
# handles the nonsmooth L1 term via the proximal map (and is the designated
# method for the Lasso).
APPLICABLE = {
    "least_squares": ("gd", "nesterov"),
    "lasso": ("fista",),
    "logistic": ("gd", "nesterov"),
}


def solve(problem_name: str, method: str, max_iter: int, tol: float):
    """Run one (problem, method) pair and return (problem, result)."""
    problem = make_problem(problem_name)
    if method not in APPLICABLE[problem_name]:
        raise ValueError(
            f"method {method!r} is not applicable to problem {problem_name!r}; "
            f"applicable: {APPLICABLE[problem_name]}"
        )

    if method == "gd":
        # Fixed step 1/L: largest step with a guaranteed monotone decrease for
        # an L-smooth convex objective; deterministic, no line search needed.
        result = gradient_descent(
            problem.objective,
            problem.gradient,
            problem.x0,
            step_size=1.0 / problem.lipschitz,
            max_iter=max_iter,
            tol=tol,
        )
    elif method == "nesterov":
        result = nesterov_ag(
            problem.objective,
            problem.gradient,
            problem.x0,
            lipschitz=problem.lipschitz,
            strong_convexity=problem.strong_convexity,
            max_iter=max_iter,
            tol=tol,
        )
    else:  # fista
        result = fista(
            problem.objective,
            problem.smooth_gradient,
            problem.prox,
            problem.x0,
            smooth_lipschitz=problem.lipschitz,
            max_iter=max_iter,
            tol=tol,
        )
    return problem, result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=PROBLEMS, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--tol", type=float, default=1e-10)
    args = parser.parse_args(argv)

    problem, result = solve(args.problem, args.method, args.max_iter, args.tol)
    truth = problem.ground_truth
    gap = result.final_objective_gap(truth.f)

    print(f"problem={problem.name} method={args.method}")
    print(f"ground_truth_source: {truth.source}")
    print(f"iterations={result.n_iter} converged={result.converged} tol={result.tol}")
    print(f"f(x_final)={result.final_objective():.12e} f*={truth.f:.12e}")
    print(f"final_objective_gap={gap:.3e}")
    print(f"final_residual={result.final_residual():.3e}")
    print("last history rows (iteration, objective, residual):")
    for it, obj, res in result.history.tail(5):
        print(f"  {it:5d}  {obj:.12e}  {res:.3e}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

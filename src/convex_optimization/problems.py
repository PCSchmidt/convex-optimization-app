"""Deterministic benchmark problems with SciPy (or closed-form) ground truth.

All problems are small (10-40 dimensions), CPU-friendly, and fully
deterministic: every random draw comes from a fixed NumPy seed. SciPy appears
ONLY inside ``Problem.ground_truth`` as a reference solver; the methods in
``methods.py`` never call SciPy.

Residual conventions (what "per-iteration residual" means per problem):

- smooth problems (least squares, logistic): the gradient norm at the iterate.
- Lasso (nonsmooth): the norm of the scaled prox-gradient map
  ``|| x - prox(x - grad g(x) / L) || * L`` with ``prox`` at step ``1 / L``.
  This is zero exactly when 0 belongs to the subdifferential of the full
  objective, so it is the correct stationarity measure for a nonsmooth
  composite objective.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class GroundTruth:
    """A reference optimum plus an honest description of how it was computed."""

    x: np.ndarray
    f: float
    source: str  # e.g. "closed form (normal equations)" or "scipy L-BFGS-B"


@dataclass
class Problem:
    name: str
    objective: Callable[[np.ndarray], float]
    gradient: Callable[[np.ndarray], np.ndarray] | None  # None for lasso
    residual: Callable[[np.ndarray], float]
    x0: np.ndarray
    lipschitz: float  # L of the objective (smooth) or smooth part (lasso)
    strong_convexity: float | None  # None for lasso
    prox: Callable[[np.ndarray], np.ndarray] | None  # set for lasso
    smooth_objective: Callable[[np.ndarray], float] | None  # smooth part (lasso)
    smooth_gradient: Callable[[np.ndarray], np.ndarray] | None  # smooth part (lasso)
    ground_truth: GroundTruth
    smooth_methods: tuple[str, ...]  # methods applicable beyond fista

    def __post_init__(self) -> None:
        self.name = str(self.name)


def _least_squares_data(seed: int = 0, n: int = 30, d: int = 20) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, d))
    x_true = rng.standard_normal(d)
    b = A @ x_true + 0.05 * rng.standard_normal(n)
    return A, b


def make_least_squares(seed: int = 0, n: int = 30, d: int = 20) -> Problem:
    """Least squares: min 0.5 * ||A x - b||^2 on seeded synthetic data.

    Smooth and strongly convex (A has full column rank by construction with
    n > d and a continuous distribution). Ground truth is the documented
    closed form: the unique minimizer of the normal equations, computed with
    ``numpy.linalg.lstsq`` (a linear-algebra factorization, not an iterative
    first-order solver).
    """
    A, b = _least_squares_data(seed, n, d)
    AtA = A.T @ A
    L = float(np.linalg.eigvalsh(AtA).max())
    mu = float(np.linalg.eigvalsh(AtA).min())

    def objective(x: np.ndarray) -> float:
        r = A @ x - b
        return 0.5 * float(r @ r)

    def gradient(x: np.ndarray) -> np.ndarray:
        return A.T @ (A @ x - b)

    x_star = np.linalg.lstsq(A, b, rcond=None)[0]
    f_star = objective(x_star)
    truth = GroundTruth(x=x_star, f=f_star, source="closed form (normal equations via lstsq)")

    return Problem(
        name="least_squares",
        objective=objective,
        gradient=gradient,
        residual=lambda x: float(np.linalg.norm(gradient(x))),
        x0=np.zeros(d),
        lipschitz=L,
        strong_convexity=mu,
        prox=None,
        smooth_objective=None,
        smooth_gradient=None,
        ground_truth=truth,
        smooth_methods=("gd", "nesterov"),
    )


def make_lasso(seed: int = 0, n: int = 30, d: int = 20, lam: float = 0.1) -> Problem:
    """L1-regularized least squares (Lasso): min 0.5 ||A x - b||^2 + lam ||x||_1.

    Convex but nonsmooth. The applicable self-implemented methods are the
    proximal ones (ISTA/FISTA). Ground truth is a SciPy reference: L-BFGS-B on
    a smoothed objective ``0.5 ||A x - b||^2 + lam * sum(sqrt(x_i^2 + eps^2))``
    with eps = 1e-10 and the analytic smoothed gradient supplied. This is an
    approximate reference: its objective exceeds the true optimum by at most
    roughly ``eps * sqrt(d)``, which is far below every tolerance used in the
    tests. The tests additionally verify that the FISTA iterate does not beat
    the reference by more than the same slack.
    """
    A, b = _least_squares_data(seed, n, d)
    AtA = A.T @ A
    L = float(np.linalg.eigvalsh(AtA).max())
    eps = 1e-10

    def smooth_objective(x: np.ndarray) -> float:
        r = A @ x - b
        return 0.5 * float(r @ r)

    def smooth_gradient(x: np.ndarray) -> np.ndarray:
        return A.T @ (A @ x - b)

    def objective(x: np.ndarray) -> float:
        return smooth_objective(x) + lam * float(np.abs(x).sum())

    def smoothed_objective(x: np.ndarray) -> float:
        return smooth_objective(x) + lam * float(np.sqrt(x * x + eps * eps).sum())

    def gradient_of_smoothed(x: np.ndarray) -> np.ndarray:
        return smooth_gradient(x) + lam * x / np.sqrt(x * x + eps * eps)

    gradient_of_smoothed_objective = smoothed_objective

    def prox(x: np.ndarray) -> np.ndarray:
        # soft-thresholding at step 1/L: prox_{(lam/L) ||.||_1}
        t = lam / L
        return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)

    def residual(x: np.ndarray) -> float:
        # scaled prox-gradient map norm: zero iff x is a Lasso minimizer
        g = smooth_gradient(x)
        return float(np.linalg.norm(x - prox(x - g / L)) * L)

    ref = minimize(
        gradient_of_smoothed_objective,
        np.zeros(d),
        jac=gradient_of_smoothed,
        method="L-BFGS-B",
        options={"maxiter": 5000, "ftol": 1e-18, "gtol": 1e-14},
    )
    x_star = np.asarray(ref.x, dtype=float)
    truth = GroundTruth(
        x=x_star,
        f=objective(x_star),
        source=(
            "scipy L-BFGS-B on the eps=1e-10 smoothed Lasso objective "
            "(analytic gradient); approximate reference, bias < ~1e-9"
        ),
    )

    return Problem(
        name="lasso",
        objective=objective,
        gradient=None,
        residual=residual,
        x0=np.zeros(d),
        lipschitz=L,
        strong_convexity=None,
        prox=prox,
        smooth_objective=smooth_objective,
        smooth_gradient=smooth_gradient,
        ground_truth=truth,
        smooth_methods=(),
    )


def make_logistic(seed: int = 1, n: int = 40, d: int = 10, ridge: float = 0.1) -> Problem:
    """L2-regularized logistic regression on a small seeded synthetic set.

    min (1/n) sum_i log(1 + exp(-y_i * a_i^T x)) + (ridge/2) ||x||^2 with
    y_i in {-1, +1}. Smooth and strongly convex (the ridge term guarantees
    strong convexity regardless of the data). Ground truth is the SciPy
    optimum: L-BFGS-B with the exact analytic gradient, which is reliable for
    this smooth objective.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, d))
    x_true = rng.standard_normal(d)
    logits = A @ x_true
    y = np.where(rng.random(n) < 1.0 / (1.0 + np.exp(-logits)), 1.0, -1.0)

    AtA = A.T @ A
    # grad (mean logistic loss) has Lipschitz constant (1/(4n)) * lambda_max(A^T A)
    L = float(np.linalg.eigvalsh(AtA).max() / (4.0 * n) + ridge)

    def objective(x: np.ndarray) -> float:
        z = -y * (A @ x)
        return float(np.logaddexp(0.0, z).mean() + 0.5 * ridge * (x @ x))

    def gradient(x: np.ndarray) -> np.ndarray:
        s = -y / (1.0 + np.exp(y * (A @ x)))  # d/dx of log(1+exp(-y a^T x)) terms
        return s @ A / n + ridge * x

    truth_ref = minimize(
        objective,
        np.zeros(d),
        jac=gradient,
        method="L-BFGS-B",
        options={"maxiter": 5000, "ftol": 1e-18, "gtol": 1e-14},
    )
    x_star = np.asarray(truth_ref.x, dtype=float)
    truth = GroundTruth(
        x=x_star, f=objective(x_star), source="scipy L-BFGS-B with analytic gradient"
    )

    return Problem(
        name="logistic",
        objective=objective,
        gradient=gradient,
        residual=lambda x: float(np.linalg.norm(gradient(x))),
        x0=np.zeros(d),
        lipschitz=L,
        strong_convexity=ridge,
        prox=None,
        smooth_objective=None,
        smooth_gradient=None,
        ground_truth=truth,
        smooth_methods=("gd", "nesterov"),
    )


def make_problem(name: str) -> Problem:
    builders = {
        "least_squares": make_least_squares,
        "lasso": make_lasso,
        "logistic": make_logistic,
    }
    if name not in builders:
        raise KeyError(f"unknown problem {name!r}; choose from {sorted(builders)}")
    return builders[name]()

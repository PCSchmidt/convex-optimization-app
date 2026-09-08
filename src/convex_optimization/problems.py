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
from functools import lru_cache

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


def _least_squares_from_data(A: np.ndarray, b: np.ndarray) -> Problem:
    """Build the least-squares Problem from a fixed (A, b) pair.

    Shared by ``make_least_squares`` and the conditioned parameterized
    variant; the construction (eigenvalues, lstsq truth, closures) is
    byte-identical for identical (A, b).
    """
    d = A.shape[1]
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


def make_least_squares(seed: int = 0, n: int = 30, d: int = 20) -> Problem:
    """Least squares: min 0.5 * ||A x - b||^2 on seeded synthetic data.

    Smooth and strongly convex (A has full column rank by construction with
    n > d and a continuous distribution). Ground truth is the documented
    closed form: the unique minimizer of the normal equations, computed with
    ``numpy.linalg.lstsq`` (a linear-algebra factorization, not an iterative
    first-order solver).
    """
    A, b = _least_squares_data(seed, n, d)
    return _least_squares_from_data(A, b)


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


# ---------------------------------------------------------------------------
# Phase A1: seeded, deterministic PARAMETERIZED problem instances.
#
# The frozen Stage 2 benchmark fixtures above are unchanged (same functions,
# same defaults, byte-identical outputs). This section adds user-specified
# instances of the SAME three registry problems for the public API: same
# builders, same conventions, bounded dimensions/weights, fully reproducible
# from (kind, seed, dimensions, weight). Nothing here tunes the solvers.
# ---------------------------------------------------------------------------

# Hard dimension caps (rows AND variables): a public request cannot ask for
# an arbitrarily large instance. 200 is chosen so one request stays well
# under a second of CPU for the solver AND the SciPy ground-truth pass.
MAX_PARAMETERIZED_DIM = 200
# Bounds for regularization weights and the condition-number knob.
MIN_PARAMETERIZED_WEIGHT = 1e-6
MAX_PARAMETERIZED_WEIGHT = 100.0
MAX_PARAMETERIZED_CONDITION = 1e6
MAX_PARAMETERIZED_SEED = 2**31 - 1

# Kind-specific defaults: a parameterized instance whose fields are all left
# at the defaults builds the EXACT frozen Stage 1 benchmark instance.
PARAMETERIZED_DEFAULTS: dict[str, dict[str, float | int]] = {
    "least_squares": {"seed": 0, "n_rows": 30, "n_vars": 20},
    "lasso": {"seed": 0, "n_rows": 30, "n_vars": 20, "lam": 0.1},
    "logistic": {"seed": 1, "n_rows": 40, "n_vars": 10, "ridge": 0.1},
}


@dataclass(frozen=True)
class ParameterSpec:
    """Fully resolved parameters of one parameterized instance.

    ``kind`` is a registry problem name (``least_squares`` / ``lasso`` /
    ``logistic``); the instance's ``Problem.name`` stays the registry name so
    the APPLICABLE matrix and the metrics labels are unchanged.
    """

    kind: str
    seed: int
    n_rows: int
    n_vars: int
    lam: float | None = None
    ridge: float | None = None
    condition: float | None = None

    def to_dict(self) -> dict[str, float | int]:
        """JSON-serializable view (only the fields meaningful for the kind)."""
        out: dict[str, float | int] = {
            "seed": self.seed,
            "n_rows": self.n_rows,
            "n_vars": self.n_vars,
        }
        if self.lam is not None:
            out["lam"] = self.lam
        if self.ridge is not None:
            out["ridge"] = self.ridge
        if self.condition is not None:
            out["condition"] = self.condition
        return out


def resolve_parameter_spec(
    kind: str,
    *,
    seed: int | None = None,
    n_rows: int | None = None,
    n_vars: int | None = None,
    lam: float | None = None,
    ridge: float | None = None,
    condition: float | None = None,
) -> ParameterSpec:
    """Validate user parameters against the kind and fill kind defaults.

    Raises ``ValueError`` with a bounded, user-facing message on any out-of-
    range value, wrong kind/field combination, or dimensional inconsistency.
    The returned spec is fully resolved (no ``None`` where a default applies).
    """
    if kind not in PARAMETERIZED_DEFAULTS:
        raise ValueError(f"unknown problem {kind!r}; choose from {sorted(PARAMETERIZED_DEFAULTS)}")
    defaults = PARAMETERIZED_DEFAULTS[kind]
    n_rows_resolved = defaults["n_rows"] if n_rows is None else n_rows
    if n_rows is None and n_vars is not None:
        # Deterministic auto-fill: a user-specified variable count can exceed
        # the kind's default row count (which would violate n_rows > n_vars).
        # Keep the default when it already fits, else take n_vars + 10 rows.
        n_rows_resolved = max(int(defaults["n_rows"]), n_vars + 10)
    spec = ParameterSpec(
        kind=kind,
        seed=int(defaults["seed"] if seed is None else seed),
        n_rows=int(n_rows_resolved),
        n_vars=int(defaults["n_vars"] if n_vars is None else n_vars),
        lam=lam,
        ridge=ridge,
        condition=condition,
    )
    if not 0 <= spec.seed <= MAX_PARAMETERIZED_SEED:
        raise ValueError(f"seed must be in [0, {MAX_PARAMETERIZED_SEED}]")
    if not 4 <= spec.n_rows <= MAX_PARAMETERIZED_DIM:
        raise ValueError(f"n_rows must be in [4, {MAX_PARAMETERIZED_DIM}]")
    if not 2 <= spec.n_vars <= MAX_PARAMETERIZED_DIM:
        raise ValueError(f"n_vars must be in [2, {MAX_PARAMETERIZED_DIM}]")
    if spec.lam is not None and kind != "lasso":
        raise ValueError("lam is only valid for the lasso problem")
    if spec.ridge is not None and kind != "logistic":
        raise ValueError("ridge is only valid for the logistic problem")
    if spec.condition is not None and kind != "least_squares":
        raise ValueError("condition is only valid for the least_squares problem")
    if kind in ("least_squares", "lasso") and spec.n_rows <= spec.n_vars:
        # The seeded data needs more rows than variables for a full-rank A
        # (strong convexity of the least-squares objective).
        raise ValueError("n_rows must be greater than n_vars for least_squares/lasso")
    for name, value in (("lam", spec.lam), ("ridge", spec.ridge)):
        if value is not None and not MIN_PARAMETERIZED_WEIGHT <= value <= MAX_PARAMETERIZED_WEIGHT:
            raise ValueError(
                f"{name} must be in [{MIN_PARAMETERIZED_WEIGHT}, {MAX_PARAMETERIZED_WEIGHT}]"
            )
    if spec.condition is not None and not 1.0 <= spec.condition <= MAX_PARAMETERIZED_CONDITION:
        raise ValueError(f"condition must be in [1.0, {MAX_PARAMETERIZED_CONDITION}]")
    return spec


def _conditioned_least_squares(seed: int, n_rows: int, n_vars: int, condition: float) -> Problem:
    """Least squares with a geometric column-scaling condition knob.

    Columns of A are scaled by ``geomspace(1, sqrt(condition), n_vars)``, so
    the sample covariance A^T A becomes (approximately) condition-number
    ``condition`` -- approximately because the random data adds its own
    mild factor. The ACTUAL Lipschitz constant and strong convexity are
    computed from the scaled matrix (as in every builder), so all step sizes
    and momentum constants remain exact for the built instance.
    """
    A, b = _least_squares_data(seed, n_rows, n_vars)
    A = A * np.geomspace(1.0, float(condition) ** 0.5, n_vars)
    return _least_squares_from_data(A, b)


@lru_cache(maxsize=64)
def _build_parameterized(
    kind: str,
    seed: int,
    n_rows: int,
    n_vars: int,
    lam: float | None,
    ridge: float | None,
    condition: float | None,
) -> Problem:
    """Cached builder: identical parameters return the identical Problem.

    The cache is safe because a Problem is immutable by convention (closures
    over fixed seeded data) and rebuilt deterministically; identical
    parameters produce bit-identical instances even on a cache miss.
    """
    if kind == "least_squares":
        if condition is not None:
            return _conditioned_least_squares(seed, n_rows, n_vars, condition)
        return make_least_squares(seed, n_rows, n_vars)
    if kind == "lasso":
        return make_lasso(seed, n_rows, n_vars, float(lam) if lam is not None else 0.1)
    if kind == "logistic":
        return make_logistic(seed, n_rows, n_vars, float(ridge) if ridge is not None else 0.1)
    raise KeyError(f"unknown problem {kind!r}")  # unreachable via resolve_parameter_spec


def make_parameterized(
    kind: str,
    *,
    seed: int | None = None,
    n_rows: int | None = None,
    n_vars: int | None = None,
    lam: float | None = None,
    ridge: float | None = None,
    condition: float | None = None,
) -> tuple[Problem, ParameterSpec]:
    """Build one user-specified instance of a registry problem.

    Returns ``(problem, resolved_spec)``. Raises ``ValueError`` on any
    invalid parameter (dimension caps, weight ranges, wrong field for the
    kind, n_rows <= n_vars for least_squares/lasso). Same arguments always
    produce a bit-reproducible instance (fixed NumPy seed; cached builder).
    """
    spec = resolve_parameter_spec(
        kind, seed=seed, n_rows=n_rows, n_vars=n_vars, lam=lam, ridge=ridge, condition=condition
    )
    return _build_parameterized(
        spec.kind, spec.seed, spec.n_rows, spec.n_vars, spec.lam, spec.ridge, spec.condition
    ), spec


def make_problem(name: str) -> Problem:
    builders = {
        "least_squares": make_least_squares,
        "lasso": make_lasso,
        "logistic": make_logistic,
    }
    if name not in builders:
        raise KeyError(f"unknown problem {name!r}; choose from {sorted(builders)}")
    return builders[name]()

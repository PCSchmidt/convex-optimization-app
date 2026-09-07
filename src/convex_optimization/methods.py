"""First-order optimization methods implemented from scratch in NumPy.

SciPy is used only for ground-truth references in ``problems.py``; it is never
part of the solver path here.

Methods
-------
- ``gradient_descent``: fixed step size or Armijo backtracking line search.
- ``nesterov_ag``: Nesterov accelerated gradient with the constant momentum
  tuned for L-smooth, mu-strongly-convex objectives.
- ``ista``: non-accelerated proximal gradient for composite objectives
  ``smooth + nonsmooth`` (FISTA minus the momentum sequence; used for Lasso).
- ``fista``: accelerated proximal gradient (Beck-Teboulle) for composite
  objectives ``smooth + nonsmooth`` (used for Lasso / ISTA-FISTA).

Step-size rules are documented on each function and in the README.
"""

from __future__ import annotations

import numpy as np

from .history import History, Result

_RESIDUAL_EPS = 1e-300  # guard against divide-by-zero in backtracking


def _finish(x: np.ndarray, history: History, updates: int, tol: float, converged: bool) -> Result:
    return Result(x=x, history=history, n_iter=updates, converged=converged, tol=tol)


def _record(
    history: History,
    x: np.ndarray,
    objective,
    residual_fn,
) -> float:
    residual = float(residual_fn(x))
    history.append(objective(x), residual)
    return residual


def gradient_descent(
    objective,
    gradient,
    x0: np.ndarray,
    step_size: float | str = "backtracking",
    max_iter: int = 2000,
    tol: float = 1e-10,
    residual_fn=None,
) -> Result:
    """Gradient descent with a fixed step size or Armijo backtracking.

    Parameters
    ----------
    step_size:
        A positive float uses the fixed rule ``x_{k+1} = x_k - step_size * grad``.
        The string ``"backtracking"`` uses Armijo backtracking: start from a
        trial step of 1.0 and shrink by ``rho=0.5`` until
        ``f(x - t*g) <= f(x) - c1 * t * ||g||^2`` with ``c1 = 1e-4``.

    Fixed ``1/L`` is the textbook choice for L-smooth convex objectives: it is
    the largest step guaranteed to decrease the objective monotonically and it
    keeps every benchmark deterministic. Backtracking is provided for callers
    who do not know ``L``.

    ``residual_fn`` defaults to the gradient norm.
    """
    if not (isinstance(step_size, str) and step_size == "backtracking"):
        step_size = float(step_size)
        if step_size <= 0:
            raise ValueError("step_size must be positive or 'backtracking'")

    if residual_fn is None:
        residual_fn = lambda x: float(np.linalg.norm(gradient(x)))

    c1, rho, t0 = 1e-4, 0.5, 1.0
    x = np.asarray(x0, dtype=float).copy()
    history = History()
    residual = _record(history, x, objective, residual_fn)
    converged = residual <= tol

    for _ in range(max_iter):
        if converged:
            break
        g = np.asarray(gradient(x), dtype=float)
        f_x = float(objective(x))
        if isinstance(step_size, float):
            t = step_size
        else:
            t = t0
            while float(objective(x - t * g)) > f_x - c1 * t * float(np.dot(g, g)) + _RESIDUAL_EPS:
                t *= rho
                if t < 1e-20:
                    break
        x = x - t * g
        residual = _record(history, x, objective, residual_fn)
        converged = residual <= tol

    return _finish(x, history, len(history) - 1, tol, converged)


def nesterov_ag(
    objective,
    gradient,
    x0: np.ndarray,
    lipschitz: float,
    strong_convexity: float,
    max_iter: int = 2000,
    tol: float = 1e-10,
    residual_fn=None,
) -> Result:
    """Nesterov accelerated gradient with constant momentum.

    For an L-smooth, mu-strongly-convex objective with kappa = L / mu, the
    scheme uses step size ``1/L`` and momentum
    ``gamma = (sqrt(kappa) - 1) / (sqrt(kappa) + 1)``, which achieves the
    O((1 - 1/sqrt(kappa))^k) convergence rate instead of gradient descent's
    O((1 - 1/kappa)^k) (Nesterov 1983; constant-step variant for the strongly
    convex case). This is why the benchmark problems declare their strong
    convexity: the momentum is a function of the known conditioning.

    ``residual_fn`` defaults to the gradient norm evaluated at the iterate.
    """
    if lipschitz <= 0 or strong_convexity <= 0:
        raise ValueError("lipschitz and strong_convexity must be positive")
    kappa = lipschitz / strong_convexity
    gamma = (np.sqrt(kappa) - 1.0) / (np.sqrt(kappa) + 1.0)

    if residual_fn is None:
        residual_fn = lambda x: float(np.linalg.norm(gradient(x)))

    x = np.asarray(x0, dtype=float).copy()
    x_prev = x.copy()
    history = History()
    residual = _record(history, x, objective, residual_fn)
    converged = residual <= tol

    for _ in range(max_iter):
        if converged:
            break
        y = x + gamma * (x - x_prev)
        x_new = y - gradient(y) / lipschitz
        x_prev, x = x, x_new
        residual = _record(history, x, objective, residual_fn)
        converged = residual <= tol

    return _finish(x, history, len(history) - 1, tol, converged)


def fista(
    objective,
    smooth_gradient,
    prox,
    x0: np.ndarray,
    smooth_lipschitz: float,
    max_iter: int = 5000,
    tol: float = 1e-10,
    residual_fn=None,
) -> Result:
    """Accelerated proximal gradient (FISTA, Beck & Teboulle 2009).

    Solves ``min f(x) = g(x) + h(x)`` where ``g`` is ``L``-smooth and ``h`` is
    convex but possibly nonsmooth, via

        x_{k+1} = prox_{h / L}( y_k - grad g(y_k) / L )

    with the FISTA momentum sequence ``t_{k+1} = (1 + sqrt(1 + 4 t_k^2)) / 2``.
    The step ``1/L`` is the largest step for which the proximal-gradient map
    is guaranteed contractive for the smooth part; ``prox`` must therefore
    correspond to step ``1 / smooth_lipschitz``.

    ``residual_fn`` defaults to the norm of the scaled prox-gradient map,
    ``||x - prox(x - grad g(x) / L)|| / L``, a stationarity measure that is
    zero exactly when a subgradient of the full objective vanishes.
    """
    if smooth_lipschitz <= 0:
        raise ValueError("smooth_lipschitz must be positive")

    def prox_grad_norm(x: np.ndarray) -> float:
        g = np.asarray(smooth_gradient(x), dtype=float)
        return float(np.linalg.norm(x - prox(x - g / smooth_lipschitz)) * smooth_lipschitz)

    if residual_fn is None:
        residual_fn = prox_grad_norm

    x = np.asarray(x0, dtype=float).copy()
    y = x.copy()
    t = 1.0
    history = History()
    residual = _record(history, x, objective, residual_fn)
    converged = residual <= tol

    for _ in range(max_iter):
        if converged:
            break
        x_new = np.asarray(prox(y - smooth_gradient(y) / smooth_lipschitz), dtype=float)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x, t = x_new, t_new
        residual = _record(history, x, objective, residual_fn)
        converged = residual <= tol

    return _finish(x, history, len(history) - 1, tol, converged)


def ista(
    objective,
    smooth_gradient,
    prox,
    x0: np.ndarray,
    smooth_lipschitz: float,
    max_iter: int = 5000,
    tol: float = 1e-10,
    residual_fn=None,
) -> Result:
    """Non-accelerated proximal gradient (ISTA) for ``min g(x) + h(x)``.

    ISTA is exactly FISTA without the momentum sequence: iterate

        x_{k+1} = prox_{h / L}( x_k - grad g(x_k) / L )

    with the same fixed step ``1 / smooth_lipschitz`` (and therefore the same
    ``prox`` at step ``1 / L``) as FISTA. Without acceleration the guaranteed
    rate is O(1/k) instead of FISTA's O(1/k^2), but the proximal-gradient map
    with step ``1 / L`` is monotone for the full objective, so the recorded
    objective values decrease at every step.

    ``ista`` exists so the benchmark can compare accelerated vs
    non-accelerated proximal gradient with everything else held identical:
    same step size, same tolerance, same stopping rule, same residual.
    No step size or tolerance was changed for the benchmark.

    ``residual_fn`` defaults to the norm of the scaled prox-gradient map,
    ``||x - prox(x - grad g(x) / L)|| * L``, identical to FISTA's.
    """
    if smooth_lipschitz <= 0:
        raise ValueError("smooth_lipschitz must be positive")

    def prox_grad_norm(x: np.ndarray) -> float:
        g = np.asarray(smooth_gradient(x), dtype=float)
        return float(np.linalg.norm(x - prox(x - g / smooth_lipschitz)) * smooth_lipschitz)

    if residual_fn is None:
        residual_fn = prox_grad_norm

    x = np.asarray(x0, dtype=float).copy()
    history = History()
    residual = _record(history, x, objective, residual_fn)
    converged = residual <= tol

    for _ in range(max_iter):
        if converged:
            break
        x = np.asarray(prox(x - smooth_gradient(x) / smooth_lipschitz), dtype=float)
        residual = _record(history, x, objective, residual_fn)
        converged = residual <= tol

    return _finish(x, history, len(history) - 1, tol, converged)

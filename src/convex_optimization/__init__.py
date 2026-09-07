"""Convex optimization algorithms with reproducible benchmarking.

Stage 1 core: first-order methods implemented from scratch in NumPy
(gradient descent, Nesterov accelerated gradient, FISTA for the Lasso),
deterministic benchmark problems, per-iteration convergence history, and
SciPy / closed-form ground truths. SciPy is used only as a reference
solver, never as the method.
"""

from .history import History, Result
from .methods import fista, gradient_descent, nesterov_ag
from .problems import (
    GroundTruth,
    Problem,
    make_lasso,
    make_least_squares,
    make_logistic,
    make_problem,
)

__version__ = "0.2.0"

__all__ = [
    "GroundTruth",
    "History",
    "Problem",
    "Result",
    "fista",
    "gradient_descent",
    "make_lasso",
    "make_least_squares",
    "make_logistic",
    "make_problem",
    "nesterov_ag",
]

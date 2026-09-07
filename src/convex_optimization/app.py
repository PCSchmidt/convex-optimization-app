"""Stage 4 serving layer: a small FastAPI app over the Stage 1-2 solvers.

NOT a production deployment and not tuned: the endpoints run the exact Stage 1
methods (via ``cli.solve``) with the Stage 1-2 defaults (``tol=1e-10``,
``max_iter=2000``, zero start, default-seeded problems). There are no
step-size, tolerance, or iteration changes here; this file adds no numerical
code. Applicable (problem, method) pairs mirror the CLI's ``APPLICABLE``
matrix (single-sourced; nothing is duplicated):

    least_squares: gd, nesterov
    lasso:         fista, ista
    logistic:      gd, nesterov

The lasso ground truth is the documented eps-smoothed SciPy approximate
reference (computed CPU-only at request time); the smooth problems use their
documented closed-form / SciPy truths. Everything is offline and
deterministic: no secrets, no keys, no network calls at request time.

Run locally (from the repo root):

    PYTHONPATH=src .venv/Scripts/python.exe -m uvicorn convex_optimization.app:app --port 8000

Interactive docs: http://localhost:8000/docs
"""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .cli import APPLICABLE, METHODS, PROBLEMS, solve

app = FastAPI(
    title="convex-optimization-app",
    description=(
        "Stage 4 serving layer over the from-scratch Stage 1 first-order "
        "methods. Local/offline demo only; not a production service."
    ),
    version="0.4.0",
)

# Upper bound on the returned history tail so responses stay small and
# bounded. The full history stays available through the CLI / Python API.
MAX_TAIL = 20
DEFAULT_TAIL = 5


class SolveRequest(BaseModel):
    """One (problem, method) solve with the Stage 1-2 default settings."""

    problem: Literal["least_squares", "lasso", "logistic"]
    method: Literal["gd", "nesterov", "fista", "ista"]
    # Number of trailing history rows to return (bounded; full history is NOT
    # returned by the API).
    tail: int = Field(default=DEFAULT_TAIL, ge=1, le=MAX_TAIL)


class HistoryRow(BaseModel):
    iteration: int
    objective: float
    residual: float


class SolveResponse(BaseModel):
    problem: str
    method: str
    ground_truth_source: str
    iterations: int
    converged: bool
    tol: float
    final_objective: float
    ground_truth_objective: float
    final_objective_gap: float
    final_residual: float
    history_tail: list[HistoryRow]


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe: the service is up (no solver run)."""
    return {"status": "ok"}


@app.post(
    "/solve",
    response_model=SolveResponse,
    responses={422: {"description": "Invalid or inapplicable (problem, method) pair"}},
)
def solve_endpoint(request: SolveRequest) -> SolveResponse:
    """Run one Stage 1 method on one documented problem, as-is.

    Returns the iteration count, convergence flag, final objective gap against
    the documented ground truth, final residual, and the last ``tail`` history
    rows. Invalid (problem, method) pairs return HTTP 422 with the list of
    applicable methods, mirroring the CLI.
    """
    if request.problem not in APPLICABLE or request.method not in METHODS:
        # Literal validation already rejects these; kept defensive.
        raise HTTPException(
            status_code=422,
            detail={
                "error": "unknown problem or method",
                "problems": list(PROBLEMS),
                "methods": list(METHODS),
            },
        )
    if request.method not in APPLICABLE[request.problem]:
        raise HTTPException(
            status_code=422,
            detail={
                "error": (
                    f"method {request.method!r} is not applicable to problem {request.problem!r}"
                ),
                "applicable_methods": list(APPLICABLE[request.problem]),
            },
        )

    problem, result = solve(request.problem, request.method)
    tail_rows = result.history.tail(min(request.tail, MAX_TAIL))
    return SolveResponse(
        problem=problem.name,
        method=request.method,
        ground_truth_source=problem.ground_truth.source,
        iterations=result.n_iter,
        converged=result.converged,
        tol=result.tol,
        final_objective=result.final_objective(),
        ground_truth_objective=float(problem.ground_truth.f),
        final_objective_gap=result.final_objective_gap(problem.ground_truth.f),
        final_residual=result.final_residual(),
        history_tail=[
            HistoryRow(iteration=i, objective=obj, residual=res) for i, obj, res in tail_rows
        ],
    )

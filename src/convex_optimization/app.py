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

Stage 5 observability (LOCAL ONLY): every request emits one structured JSON
log line to stdout, and ``GET /metrics`` exposes in-process counters (request
count, latency percentiles, 4xx/5xx error rates, and the convex-specific
convergence-failure rate). Counters reset on restart; there is no Prometheus,
Grafana, alerting, or persistence. See ``observability.py``.
"""

from __future__ import annotations

import time
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .cli import APPLICABLE, METHODS, PROBLEMS, solve
from .observability import ERROR_INAPPLICABLE, ERROR_SERVER, ERROR_VALIDATION, metrics

app = FastAPI(
    title="convex-optimization-app",
    description=(
        "Stage 4 serving layer over the from-scratch Stage 1 first-order "
        "methods. Local/offline demo only; not a production service."
    ),
    version="0.5.0",
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Count pydantic validation failures (4xx) and keep the default 422 body.

    Validation errors are request ERRORS for the metrics; they are never
    convergence failures (only a completed 200 solve can fail to converge).
    """
    metrics.record(request.url.path, 422, 0.0, error_class=ERROR_VALIDATION)
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(exc.errors())})


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
    t0 = time.perf_counter()
    response = {"status": "ok"}
    metrics.record("/health", 200, (time.perf_counter() - t0) * 1000.0)
    return response


@app.get("/metrics")
def metrics_endpoint() -> dict:
    """Stage 5 metrics (LOCAL ONLY): in-process counters, reset on restart.

    JSON snapshot: request counts, latency summary and percentiles, 4xx/5xx
    error rates, and the convergence-failure rate (200 solves that returned
    ``converged=false``, i.e. hit ``max_iter=2000`` without meeting
    ``tol=1e-10``). 4xx validation / inapplicable-pair responses count as
    errors, NOT convergence failures. There is no persistence, no scrape
    target, and no alerting; counters reset when the process restarts.
    """
    snapshot = metrics.snapshot()  # snapshot first: do not count this call
    metrics.record("/metrics", 200, 0.0)
    return snapshot


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

    Every request is counted and logged (one structured JSON line): 4xx are
    request errors; a 200 with ``converged=false`` is the convergence-failure
    signal. Iterate vectors and history arrays are never logged.
    """
    t0 = time.perf_counter()

    def elapsed_ms() -> float:
        return (time.perf_counter() - t0) * 1000.0

    if request.problem not in APPLICABLE or request.method not in METHODS:
        # Literal validation already rejects these; kept defensive.
        metrics.record("/solve", 422, elapsed_ms(), error_class=ERROR_VALIDATION)
        raise HTTPException(
            status_code=422,
            detail={
                "error": "unknown problem or method",
                "problems": list(PROBLEMS),
                "methods": list(METHODS),
            },
        )
    if request.method not in APPLICABLE[request.problem]:
        metrics.record(
            "/solve",
            422,
            elapsed_ms(),
            problem=request.problem,
            method=request.method,
            error_class=ERROR_INAPPLICABLE,
        )
        raise HTTPException(
            status_code=422,
            detail={
                "error": (
                    f"method {request.method!r} is not applicable to problem {request.problem!r}"
                ),
                "applicable_methods": list(APPLICABLE[request.problem]),
            },
        )

    try:
        problem, result = solve(request.problem, request.method)
    except Exception:
        # Solver crash: counted and logged as a 5xx server error (not a
        # convergence failure), then re-raised for the default 500 handling.
        metrics.record(
            "/solve",
            500,
            elapsed_ms(),
            problem=request.problem,
            method=request.method,
            error_class=ERROR_SERVER,
        )
        raise
    tail_rows = result.history.tail(min(request.tail, MAX_TAIL))
    response = SolveResponse(
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
    metrics.record(
        "/solve",
        200,
        elapsed_ms(),
        problem=problem.name,
        method=request.method,
        iterations=result.n_iter,
        converged=result.converged,
    )
    return response

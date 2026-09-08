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
convergence-failure rate). Counters reset on restart; there is no Grafana,
alerting, or persistence. See ``observability.py``.

Phase 2 shared contract: ``GET /metrics/prometheus`` additionally exposes the
generic Prometheus text-exposition families (requests_total, errors_total,
request_latency_seconds, up) written by ``prometheus.py`` -- stdlib only, no
prometheus_client. The JSON ``GET /metrics`` snapshot is unchanged.

Phase 3 convex-specific families on the same endpoint: solves_total,
convergence_successes_total / convergence_failures_total,
solve_latency_seconds, iterations, and final_objective_gap -- all labelled by
registry problem/solver_method NAMES only, recorded ONLY for HTTP 200 solves
(failed requests touch errors_total exclusively). The convergence-failure
rate is derived in dashboards as failures_total / solves_total.
"""

from __future__ import annotations

import time
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from starlette.routing import Match

from .cli import APPLICABLE, METHODS, PROBLEMS, solve
from .observability import ERROR_INAPPLICABLE, ERROR_SERVER, ERROR_VALIDATION, metrics
from .prometheus import (
    CONTENT_TYPE,
    UNKNOWN_LABEL,
    UNMATCHED_ENDPOINT,
    prometheus_metrics,
)

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
    endpoint = _route_template(request)
    metrics.record(endpoint, 422, 0.0, error_class=ERROR_VALIDATION)
    prometheus_metrics.record_error(endpoint, request.method, ERROR_VALIDATION)
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(exc.errors())})


def _bounded_registry_label(name: str, allowed: tuple[str, ...] | frozenset[str]) -> str:
    """Registry-name label bound (defense in depth).

    ``problem``/``solver_method`` labels use ONLY ``cli.PROBLEMS`` /
    ``cli.METHODS`` names. By construction they are already validated
    (pydantic Literal + the APPLICABLE matrix reject everything else before a
    solve runs); a hypothetically unregistered name would fall back to the
    fixed token ``unknown`` rather than create an unbounded series.
    """
    return name if name in allowed else UNKNOWN_LABEL


def _route_template(request: Request) -> str:
    """Route template for the request (bounded label), or ``unmatched``.

    Starlette 1.6 does not put the matched route in the ASGI scope, so we
    re-match against the declared routes. The result is always a route
    template (e.g. ``/solve``) or the fixed token ``unmatched`` -- never a
    concrete URL path -- keeping the Prometheus ``endpoint`` label bounded.
    """
    for route in app.routes:
        match, _ = route.matches(request.scope)
        if match == Match.FULL:
            return route.path
    return UNMATCHED_ENDPOINT


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next) -> Response:
    """Feed the Phase 2 Prometheus families (requests_total + latency).

    Errors are NOT counted here: ``errors_total`` is recorded only at the
    explicit error-class sites (validation handler, inapplicable pair, solver
    server error) so each error is counted exactly once with its bounded
    ``error_class``. An unhandled exception is counted as one 500 request.
    """
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        prometheus_metrics.record_request(
            _route_template(request), request.method, 500, time.perf_counter() - start
        )
        raise
    prometheus_metrics.record_request(
        _route_template(request),
        request.method,
        response.status_code,
        time.perf_counter() - start,
    )
    return response


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


@app.get("/metrics/prometheus")
def prometheus_metrics_endpoint() -> Response:
    """Phase 2 shared contract: Prometheus text exposition (generic families).

    Same four families as every sibling portfolio app: ``requests_total``,
    ``errors_total``, ``request_latency_seconds`` (histogram), and ``up``,
    with the ``convex_optimization_`` prefix and low-cardinality labels
    (route-template endpoint, HTTP verb, status, bounded error_class).
    Rendered by the stdlib-only writer in ``prometheus.py``; the JSON
    ``GET /metrics`` snapshot above is unchanged.
    """
    body = prometheus_metrics.render()  # render first: do not count this call
    metrics.record("/metrics/prometheus", 200, 0.0)
    return Response(content=body, media_type=CONTENT_TYPE)


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
        prometheus_metrics.record_error("/solve", "POST", ERROR_INAPPLICABLE)
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
        prometheus_metrics.record_error("/solve", "POST", ERROR_SERVER)
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
    # Phase 3 Prometheus domain families: HTTP 200 solves only (the JSON
    # snapshot's success_count semantics); registry-name labels; a 4xx/5xx
    # request never creates a problem/solver_method series.
    prometheus_metrics.record_solve(
        _bounded_registry_label(problem.name, PROBLEMS),
        _bounded_registry_label(request.method, METHODS),
        latency_seconds=elapsed_ms() / 1000.0,
        iterations=response.iterations,
        final_objective_gap=response.final_objective_gap,
        converged=response.converged,
    )
    return response

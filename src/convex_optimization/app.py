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
convergence-failure rate). Counters reset on restart. Phase 6 adds the local
compose Grafana + Prometheus stack (prometheus.yml + provisioning/ + the
grafana service in docker-compose.yml, host ports 9092/3002) which scrapes
``/metrics/prometheus`` into a committed 14-panel dashboard; there is still
no alerting and no external monitoring. See ``observability.py``.

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

Phase A1 (public readiness, single-instance demo grade -- see README):

- POST /solve additionally accepts bounded ``params`` (seed, n_rows, n_vars,
  plus lam/ridge/condition) to solve USER-SPECIFIED seeded instances of the
  same three registry problems; without ``params`` the frozen Stage 1
  benchmark instance is used, unchanged.
- POST /parse turns a natural-language problem description into the
  structured /solve request, with a deterministic mechanical verifier
  (``verified`` + ``mismatches``) and never trusted blindly. LLM-backed via
  ``LLM_API_KEY``; without a key it returns the documented 503
  ``provider_not_configured`` error.
- Hardening: per-IP fixed-window rate limiting (429 + Retry-After,
  ``RATE_LIMIT_PER_MIN``), a 64 KiB request-body cap (413), and
  env-configurable CORS (``ALLOWED_ORIGINS``; empty default = same-origin
  only). New bounded error classes: body_too_large, rate_limited,
  provider_not_configured, provider_unavailable, parse_invalid; plus the
  ``convex_optimization_parse_outcomes_total`` family. See
  ``hardening.py`` / ``parsing.py``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.routing import Match

from . import parsing
from .cli import APPLICABLE, METHODS, PROBLEMS, solve
from .hardening import (
    HardeningMiddleware,
    parse_allowed_origins,
)
from .observability import (
    ERROR_INAPPLICABLE,
    ERROR_PARSE_INVALID,
    ERROR_PROVIDER_NOT_CONFIGURED,
    ERROR_PROVIDER_UNAVAILABLE,
    ERROR_SERVER,
    ERROR_VALIDATION,
    metrics,
)
from .problems import make_parameterized, resolve_parameter_spec
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
    version="0.7.0",
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


# --- Phase A1 hardening -----------------------------------------------------
# Middleware order (Starlette: the LAST added runs FIRST): the Prometheus
# middleware below is the innermost custom layer; hardening sits outside it
# and records its own short-circuited 429/413 responses; CORS (when any
# ALLOWED_ORIGINS are configured) is outermost and answers preflights.
app.add_middleware(HardeningMiddleware, fastapi_app=app)

_allowed_origins = parse_allowed_origins(os.environ.get("ALLOWED_ORIGINS"))
if _allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins,  # explicit list only; never "*"
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
        max_age=600,
    )
# Default (no ALLOWED_ORIGINS): no CORS middleware at all -> NO
# Access-Control-Allow-Origin headers -> browsers deny every cross-origin
# request (same-origin only). The safe default requires no configuration.

# Upper bound on the returned history tail so responses stay small and
# bounded. The full history stays available through the CLI / Python API.
MAX_TAIL = 20
DEFAULT_TAIL = 5


class ProblemParams(BaseModel):
    """Bounded parameters of a user-specified problem instance (Phase A1).

    All fields are optional: omitted fields fall back to the kind's frozen
    Stage 1 defaults (see ``problems.PARAMETERIZED_DEFAULTS``), so the frozen
    benchmark instances are reproducible through this model as well. Hard
    caps: dimensions <= 200, weights in [1e-6, 100], condition in [1, 1e6],
    seed in [0, 2^31-1]. Extra fields are rejected.
    """

    model_config = ConfigDict(extra="forbid")

    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    n_rows: int | None = Field(default=None, ge=4, le=200)
    n_vars: int | None = Field(default=None, ge=2, le=200)
    lam: float | None = Field(default=None, gt=0, le=100.0)
    ridge: float | None = Field(default=None, gt=0, le=100.0)
    condition: float | None = Field(default=None, ge=1.0, le=1e6)


class SolveRequest(BaseModel):
    """One (problem, method) solve with the Stage 1-2 default settings.

    Without ``params`` the FROZEN Stage 1 benchmark instance is solved,
    exactly as before Phase A1 (byte-identical behavior). With ``params`` a
    user-specified seeded instance of the SAME registry problem is built
    (bounded dimensions/weights; bit-reproducible for identical inputs).
    The solvers themselves are never retuned: same 1/L steps, tol=1e-10,
    max_iter=2000, zero start.
    """

    model_config = ConfigDict(extra="forbid")

    problem: Literal["least_squares", "lasso", "logistic"]
    method: Literal["gd", "nesterov", "fista", "ista"]
    # Number of trailing history rows to return (bounded; full history is NOT
    # returned by the API).
    tail: int = Field(default=DEFAULT_TAIL, ge=1, le=MAX_TAIL)
    params: ProblemParams | None = None


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
    # Resolved instance parameters when the request carried ``params``
    # (user-specified instance); null for the frozen benchmark instance.
    parameters: dict[str, float | int] | None = None


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
        errors, NOT convergence failures. This JSON contract is unchanged by the
        Phase 6 stack; there is no alerting, and counters reset when the process
        restarts.
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

    # Phase A1: user-specified instance (bounded, validated, cached by
    # problems.make_parameterized). Cross-field checks (e.g. n_rows > n_vars,
    # kind-specific fields) run here even though pydantic already bounded the
    # individual ranges -- defense in depth with the same error class.
    parameters: dict[str, float | int] | None = None
    instance = None
    if request.params is not None:
        try:
            spec = resolve_parameter_spec(
                request.problem,
                seed=request.params.seed,
                n_rows=request.params.n_rows,
                n_vars=request.params.n_vars,
                lam=request.params.lam,
                ridge=request.params.ridge,
                condition=request.params.condition,
            )
        except ValueError as exc:
            metrics.record("/solve", 422, elapsed_ms(), error_class=ERROR_VALIDATION)
            prometheus_metrics.record_error("/solve", "POST", ERROR_VALIDATION)
            raise HTTPException(
                status_code=422,
                detail={"error": "invalid problem parameters", "reason": str(exc)},
            ) from exc
        problem, spec = make_parameterized(
            spec.kind,
            seed=spec.seed,
            n_rows=spec.n_rows,
            n_vars=spec.n_vars,
            lam=spec.lam,
            ridge=spec.ridge,
            condition=spec.condition,
        )
        parameters = spec.to_dict()
        instance = problem

    try:
        problem, result = solve(request.problem, request.method, problem=instance)
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
        parameters=parameters,
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


# ---------------------------------------------------------------------------
# Phase A1: POST /parse -- natural-language -> structured /solve request.
# Provider in parsing.py (LLM via env, or the test-only stub); every parse is
# mechanically verified against the text and NEVER trusted blindly.
# ---------------------------------------------------------------------------


class ParseRequest(BaseModel):
    """One natural-language problem description to parse."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=parsing.MAX_PARSE_TEXT_CHARS)


class ParsedParams(BaseModel):
    """The parsed instance parameters (same shape/bounds as ProblemParams)."""

    model_config = ConfigDict(extra="forbid")

    seed: int | None = None
    n_rows: int | None = None
    n_vars: int | None = None
    lam: float | None = None
    ridge: float | None = None
    condition: float | None = None


class ParsedProblemRequest(BaseModel):
    """A complete, directly POSTable /solve request body."""

    problem: Literal["least_squares", "lasso", "logistic"]
    method: Literal["gd", "nesterov", "fista", "ista"]
    params: ParsedParams
    tail: int = Field(default=DEFAULT_TAIL, ge=1, le=MAX_TAIL)


class ParseResponse(BaseModel):
    """One parse outcome: the /solve request plus its verification verdict.

    ``verified=false`` (HTTP 200) means the spec was produced but the text
    does not fully support it -- ``mismatches`` says exactly what failed.
    The UI should surface that verdict, not silently accept the parse.
    """

    problem_request: ParsedProblemRequest
    parse_method: str  # "llm" (production) or "stub" (test double)
    verified: bool
    mismatches: list[str]


def _fail_parse(
    status: int,
    error_class: str,
    outcome: str,
    detail: dict,
    elapsed_ms: float,
) -> None:
    """Record one /parse failure in all three metrics layers, then raise."""
    metrics.record("/parse", status, elapsed_ms, error_class=error_class)
    prometheus_metrics.record_error("/parse", "POST", error_class)
    prometheus_metrics.record_parse(outcome)
    raise HTTPException(status_code=status, detail=detail)


# Suggested method per problem kind (the benchmark's fastest converging pair):
# smooth kinds -> nesterov; the nonsmooth lasso -> fista.
SUGGESTED_METHOD = {"least_squares": "nesterov", "lasso": "fista", "logistic": "nesterov"}


@app.post(
    "/parse",
    response_model=ParseResponse,
    responses={
        422: {"description": "Text could not be parsed into a valid problem spec"},
        502: {"description": "Configured upstream LLM failed or timed out"},
        503: {"description": "No LLM provider configured (set LLM_API_KEY)"},
    },
)
def parse_endpoint(request: ParseRequest) -> ParseResponse:
    """Parse one natural-language problem description into a /solve request.

    The parse is mechanical-verified against the original text (problem-type
    keywords, dimension numbers, seed integer, weight numbers); the response
    carries ``verified`` and ``mismatches``. Outcomes are counted in the
    bounded ``convex_optimization_parse_outcomes_total`` family. The text is
    never logged and never persisted.
    """
    t0 = time.perf_counter()

    def elapsed_ms() -> float:
        return (time.perf_counter() - t0) * 1000.0

    try:
        problem, params, parse_method, verified, mismatches = parsing.parse_text(request.text)
        spec = resolve_parameter_spec(problem, **params)
    except parsing.ProviderNotConfigured as exc:
        _fail_parse(
            503,
            ERROR_PROVIDER_NOT_CONFIGURED,
            "provider_not_configured",
            {
                "error": "no LLM provider configured",
                "hint": "set LLM_API_KEY (and optionally LLM_BASE_URL / LLM_MODEL) to enable parsing",
                "reason": str(exc)[:200],
            },
            elapsed_ms(),
        )
    except parsing.ProviderUnavailable as exc:
        _fail_parse(
            502,
            ERROR_PROVIDER_UNAVAILABLE,
            "provider_unavailable",
            {"error": "upstream LLM call failed", "reason": str(exc)[:200]},
            elapsed_ms(),
        )
    except (parsing.ParseFailure, ValueError) as exc:
        _fail_parse(
            422,
            ERROR_PARSE_INVALID,
            "failed",
            {
                "error": "could not parse the text into a valid problem spec",
                "reason": str(exc)[:200],
            },
            elapsed_ms(),
        )

    method = SUGGESTED_METHOD[problem]
    response = ParseResponse(
        problem_request=ParsedProblemRequest(
            problem=problem,
            method=method,
            params=ParsedParams(**spec.to_dict()),
            tail=DEFAULT_TAIL,
        ),
        parse_method=parse_method,
        verified=verified,
        mismatches=mismatches,
    )
    prometheus_metrics.record_parse("verified" if verified else "parsed")
    metrics.record("/parse", 200, elapsed_ms())
    return response


def _mount_ui(a: FastAPI) -> None:
    """Serve the built workbench UI from the API process (Option A, same origin).

    Opt-in via ``SERVE_UI=1`` so local dev, tests, and the CLI smoke check are
    unchanged by default. ``UI_DIST_DIR`` overrides the dist directory
    (default: ``<repo root>/ui/dist``, i.e. ``ui/dist`` next to ``src/``). The
    mount is added LAST so every API route (including /docs and /health) keeps
    precedence; ``html=True`` serves ``index.html`` at ``/``. No SPA fallback:
    unknown paths 404 honestly. This is same-origin serving -- no CORS headers
    are involved.
    """
    if os.environ.get("SERVE_UI") != "1":
        return
    dist = Path(
        os.environ.get(
            "UI_DIST_DIR",
            Path(__file__).resolve().parents[2] / "ui" / "dist",
        )
    )
    if not (dist / "index.html").is_file():
        return
    a.mount("/", StaticFiles(directory=str(dist), html=True), name="ui")


_mount_ui(app)

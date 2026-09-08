"""Prometheus text exposition writer + collector (STDLIB ONLY).

Phase 2 shared observability contract (generic HTTP families) plus the Phase 3
convex-app-specific families. Alongside the existing in-process JSON
``GET /metrics`` snapshot (unchanged), the app exposes ``GET
/metrics/prometheus`` returning valid Prometheus text exposition. This module
hand-rolls the exposition writer: no ``prometheus_client``, no new
dependencies, no lockfile changes. The writer is deterministic (fixed family
order, sorted label tuples) and unit-testable offline.

Phase 2 generic families (prefix ``convex_optimization_``):

- ``convex_optimization_requests_total`` (counter): endpoint, method, status.
- ``convex_optimization_errors_total`` (counter): endpoint, method,
  error_class -- the SAME bounded error-class vocabulary as the JSON logs
  (``validation_error`` / ``inapplicable_pair`` / ``server_error``).
- ``convex_optimization_request_latency_seconds`` (histogram): endpoint,
  method; exports ``_bucket`` / ``_sum`` / ``_count``.
- ``convex_optimization_up`` (gauge): 1 while the app is serving.

Phase 3 convex-app-specific families (labels: ``problem``, ``solver_method``
-- registry NAMES only, never numeric parameters; ``iterations`` and
``final_objective_gap`` are quantities, not labels):

- ``convex_optimization_solves_total`` (counter): every HTTP 200 /solve
  response (matching the JSON snapshot's ``solve.success_count`` semantics).
- ``convex_optimization_convergence_successes_total`` /
  ``convex_optimization_convergence_failures_total`` (counters): HTTP 200
  solves split on the Stage 1 ``converged`` flag. Convergence failure keeps
  the JSON definition: a 200 solve that hit ``max_iter=2000`` without meeting
  ``tol=1e-10``. The rate is deliberately NOT exported as a gauge -- derive
  ``failures_total / solves_total`` in dashboards from the counters.
- ``convex_optimization_solve_latency_seconds`` (histogram): per successful
  (HTTP 200) solve, documented LATENCY_BUCKETS layout.
- ``convex_optimization_iterations`` (histogram with integer buckets): the
  Stage 1 iteration count per successful solve; top bucket = ``max_iter``.
- ``convex_optimization_final_objective_gap`` (histogram): final objective
  gap vs ground truth per successful solve, on log-scale buckets (see
  GAP_BUCKETS for the rationale; gaps can be marginally negative against the
  eps-smoothed approximate lasso reference and fall into the first bucket).

Cardinality is bounded by construction:

- ``endpoint`` is a route template or the literal ``unmatched`` token.
- ``method`` is the HTTP verb. ``status`` is an actually-used HTTP code.
- ``error_class`` comes only from the bounded taxonomy in
  ``observability.py``.
- ``problem`` / ``solver_method`` come from the ``cli`` registries
  (``PROBLEMS`` / ``METHODS``); the app bounds them before recording and
  would map a hypothetically unregistered name to the fixed token
  ``unknown`` (unreachable by construction: pydantic Literal validation +
  the APPLICABLE matrix reject everything else before any solve runs).

Request ids, user problem/method choices beyond the registry names, exception
messages, and full URLs are NEVER label values.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence

METRIC_PREFIX = "convex_optimization"

# Exact Content-Type required by the Prometheus text exposition format.
CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

# Phase 2 generic family names (counters end in _total; latency in seconds).
REQUESTS_TOTAL = f"{METRIC_PREFIX}_requests_total"
ERRORS_TOTAL = f"{METRIC_PREFIX}_errors_total"
LATENCY_SECONDS = f"{METRIC_PREFIX}_request_latency_seconds"
UP = f"{METRIC_PREFIX}_up"

# Phase 3 convex-app-specific family names.
SOLVES_TOTAL = f"{METRIC_PREFIX}_solves_total"
CONVERGENCE_SUCCESSES_TOTAL = f"{METRIC_PREFIX}_convergence_successes_total"
CONVERGENCE_FAILURES_TOTAL = f"{METRIC_PREFIX}_convergence_failures_total"
SOLVE_LATENCY_SECONDS = f"{METRIC_PREFIX}_solve_latency_seconds"
ITERATIONS = f"{METRIC_PREFIX}_iterations"
FINAL_OBJECTIVE_GAP = f"{METRIC_PREFIX}_final_objective_gap"

# Latency buckets in SECONDS: the prometheus_client default layout, kept so
# dashboards built against other portfolio apps align. The /solve path runs
# small in-process NumPy solves (typically well under 1 s), so the low end is
# dense and the top bucket covers 10 s of worst-case headroom.
LATENCY_BUCKETS: tuple[float, ...] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    2.5,
    7.5,
    10.0,
)

# Iteration buckets: integers, spread to the Stage 1 max_iter=2000 (the
# observable range on the Stage 2 benchmark is ~33-1451 iterations, so the
# decades give headroom on both sides).
ITERATION_BUCKETS: tuple[float, ...] = (1, 10, 25, 50, 100, 250, 500, 1000, 2000)

# Final-objective-gap buckets: log-scale decades. With tol=1e-10, converged
# solves land at |gap| <~ 1e-10 (benchmark log: 0 to ~1e-9, including one
# marginally negative value vs the eps-smoothed approximate lasso reference,
# which falls into the first bucket). Decades below 1e-10 are dense for
# converged solves; decades above give headroom for non-converged or larger
# problems, with 1.0 as the top bound (unit-scale objective gaps).
GAP_BUCKETS: tuple[float, ...] = (
    1e-12,
    1e-11,
    1e-10,
    1e-09,
    1e-08,
    1e-06,
    1e-04,
    1e-02,
    1.0,
)

# Literal endpoint label for requests that matched no declared route (e.g.
# unknown paths -> 404). Bounded: it is a fixed token, never the raw path.
UNMATCHED_ENDPOINT = "unmatched"

# Fixed fallback label for a hypothetically unregistered problem/method name
# (unreachable by construction; see module docstring).
UNKNOWN_LABEL = "unknown"

_HELP = {
    UP: "1 when the app is serving.",
    REQUESTS_TOTAL: "Total HTTP requests handled, by endpoint, method, and status code.",
    ERRORS_TOTAL: (
        "Total HTTP requests counted as errors, with the app's bounded "
        "error_class vocabulary (validation_error, inapplicable_pair, server_error)."
    ),
    LATENCY_SECONDS: "End-to-end HTTP request latency in seconds.",
    SOLVES_TOTAL: "Total HTTP 200 /solve responses, by problem and solver method.",
    CONVERGENCE_SUCCESSES_TOTAL: (
        "HTTP 200 solves that met the stopping criterion (converged=true)."
    ),
    CONVERGENCE_FAILURES_TOTAL: (
        "HTTP 200 solves that hit max_iter=2000 without meeting tol=1e-10 "
        "(converged=false). Derive the rate as failures_total / solves_total."
    ),
    SOLVE_LATENCY_SECONDS: "Latency of successful (HTTP 200) solves, in seconds.",
    ITERATIONS: "Stage 1 iteration count per successful (HTTP 200) solve.",
    FINAL_OBJECTIVE_GAP: ("Final objective gap vs ground truth per successful (HTTP 200) solve."),
}

_TYPE = {
    UP: "gauge",
    REQUESTS_TOTAL: "counter",
    ERRORS_TOTAL: "counter",
    LATENCY_SECONDS: "histogram",
    SOLVES_TOTAL: "counter",
    CONVERGENCE_SUCCESSES_TOTAL: "counter",
    CONVERGENCE_FAILURES_TOTAL: "counter",
    SOLVE_LATENCY_SECONDS: "histogram",
    ITERATIONS: "histogram",
    FINAL_OBJECTIVE_GAP: "histogram",
}


def escape_label_value(value: str) -> str:
    """Escape a label value per the Prometheus text exposition format."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def format_labels(pairs: Sequence[tuple[str, str]]) -> str:
    """Render an ordered label set: ``{k="v",...}`` (empty pairs -> ``""``)."""
    if not pairs:
        return ""
    inner = ",".join(f'{key}="{escape_label_value(value)}"' for key, value in pairs)
    return "{" + inner + "}"


def _fmt_bucket(bound: float) -> str:
    """Compact, deterministic bucket bound (``0.005``, ``1``, ``1e-12``)."""
    return f"{bound:g}"


def _fmt_value(value: float) -> str:
    """Sample value formatting (deterministic across runs).

    ``%g`` keeps small sums meaningful: objective gaps are ~1e-10 or below,
    which six fixed decimals would round to ``0.000000``.
    """
    return f"{value:.6g}"


class _Series:
    """Per-label-tuple histogram state (non-cumulative bucket counts)."""

    __slots__ = ("bucket_counts", "total_count", "total_sum")

    def __init__(self, n_buckets: int) -> None:
        self.bucket_counts = [0] * n_buckets
        self.total_sum = 0.0
        self.total_count = 0


class _Histogram:
    """One histogram family: per-key series over a fixed bucket layout."""

    def __init__(self, buckets: tuple[float, ...]) -> None:
        self.buckets = tuple(sorted(buckets))
        self.series: dict[tuple, _Series] = {}

    def observe(self, key: tuple, value: float) -> None:
        series = self.series.get(key)
        if series is None:
            series = _Series(len(self.buckets))
            self.series[key] = series
        for i, bound in enumerate(self.buckets):
            if value <= bound:
                series.bucket_counts[i] += 1
                break
        series.total_sum += value
        series.total_count += 1

    def snapshot(self) -> list[tuple[tuple, tuple[int, ...], float, int]]:
        """Sorted (key, per-bucket counts, sum, count) tuples."""
        return sorted(
            (key, tuple(s.bucket_counts), s.total_sum, s.total_count)
            for key, s in self.series.items()
        )

    def clear(self) -> None:
        self.series.clear()


class PrometheusCollector:
    """Thread-safe collector + deterministic Prometheus text writer."""

    def __init__(
        self,
        buckets: tuple[float, ...] = LATENCY_BUCKETS,
        iteration_buckets: tuple[float, ...] = ITERATION_BUCKETS,
        gap_buckets: tuple[float, ...] = GAP_BUCKETS,
    ) -> None:
        self._lock = threading.Lock()
        self._buckets = tuple(sorted(buckets))
        self._requests: dict[tuple[str, str, str], int] = {}
        self._errors: dict[tuple[str, str, str], int] = {}
        self._latencies = _Histogram(self._buckets)
        self._solves: dict[tuple[str, str], int] = {}
        self._convergence_success: dict[tuple[str, str], int] = {}
        self._convergence_failure: dict[tuple[str, str], int] = {}
        self._solve_latencies = _Histogram(self._buckets)
        self._iterations = _Histogram(tuple(sorted(iteration_buckets)))
        self._gaps = _Histogram(tuple(sorted(gap_buckets)))

    def record_request(
        self, endpoint: str, method: str, status: int, latency_seconds: float
    ) -> None:
        """Count one handled request and add its latency to the histogram."""
        req_key = (endpoint, method, str(status))
        with self._lock:
            self._requests[req_key] = self._requests.get(req_key, 0) + 1
            self._latencies.observe((endpoint, method), latency_seconds)

    def record_error(self, endpoint: str, method: str, error_class: str) -> None:
        """Count one error-classified request (bounded vocabulary only)."""
        key = (endpoint, method, error_class)
        with self._lock:
            self._errors[key] = self._errors.get(key, 0) + 1

    def record_solve(
        self,
        problem: str,
        solver_method: str,
        *,
        latency_seconds: float,
        iterations: int,
        final_objective_gap: float,
        converged: bool,
    ) -> None:
        """Count one HTTP 200 /solve response (the JSON 'success' semantics).

        Every HTTP 200 solve is counted in ``solves_total`` and observed in
        the latency / iterations / gap histograms; the convergence counters
        split on the Stage 1 ``converged`` flag. Failed (4xx/5xx) requests
        NEVER create problem/solver_method series -- they only touch
        ``errors_total``.
        """
        key = (problem, solver_method)
        with self._lock:
            self._solves[key] = self._solves.get(key, 0) + 1
            self._solve_latencies.observe(key, latency_seconds)
            self._iterations.observe(key, float(iterations))
            self._gaps.observe(key, final_objective_gap)
            if converged:
                self._convergence_success[key] = self._convergence_success.get(key, 0) + 1
            else:
                self._convergence_failure[key] = self._convergence_failure.get(key, 0) + 1

    def counts(self) -> dict[tuple[str, str, str], int]:
        """Copy of the requests_total series (for tests/introspection)."""
        with self._lock:
            return dict(self._requests)

    def error_counts(self) -> dict[tuple[str, str, str], int]:
        """Copy of the errors_total series (for tests/introspection)."""
        with self._lock:
            return dict(self._errors)

    def solve_counts(self) -> dict[tuple[str, str], int]:
        """Copy of the solves_total series (for tests/introspection)."""
        with self._lock:
            return dict(self._solves)

    def convergence_counts(self) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str], int]]:
        """Copies of (successes, failures) series (for tests/introspection)."""
        with self._lock:
            return dict(self._convergence_success), dict(self._convergence_failure)

    def render(self) -> str:
        """Full exposition document, deterministic for a given state."""
        with self._lock:
            requests = sorted(self._requests.items())
            errors = sorted(self._errors.items())
            latencies = self._latencies.snapshot()
            solves = sorted(self._solves.items())
            successes = sorted(self._convergence_success.items())
            failures = sorted(self._convergence_failure.items())
            solve_latencies = self._solve_latencies.snapshot()
            iterations = self._iterations.snapshot()
            gaps = self._gaps.snapshot()

        lines: list[str] = []

        def family(name: str) -> None:
            lines.append(f"# HELP {name} {_HELP[name]}")
            lines.append(f"# TYPE {name} {_TYPE[name]}")

        family(UP)
        lines.append(f"{UP} 1")

        family(REQUESTS_TOTAL)
        for (endpoint, method, status), count in requests:
            labels = format_labels((("endpoint", endpoint), ("method", method), ("status", status)))
            lines.append(f"{REQUESTS_TOTAL}{labels} {count}")

        family(ERRORS_TOTAL)
        for (endpoint, method, error_class), count in errors:
            labels = format_labels(
                (("endpoint", endpoint), ("method", method), ("error_class", error_class))
            )
            lines.append(f"{ERRORS_TOTAL}{labels} {count}")

        family(LATENCY_SECONDS)
        _render_histogram(lines, LATENCY_SECONDS, latencies, self._buckets, ("endpoint", "method"))

        family(SOLVES_TOTAL)
        for (problem, solver_method), count in solves:
            labels = format_labels((("problem", problem), ("solver_method", solver_method)))
            lines.append(f"{SOLVES_TOTAL}{labels} {count}")

        family(CONVERGENCE_SUCCESSES_TOTAL)
        for (problem, solver_method), count in successes:
            labels = format_labels((("problem", problem), ("solver_method", solver_method)))
            lines.append(f"{CONVERGENCE_SUCCESSES_TOTAL}{labels} {count}")

        family(CONVERGENCE_FAILURES_TOTAL)
        for (problem, solver_method), count in failures:
            labels = format_labels((("problem", problem), ("solver_method", solver_method)))
            lines.append(f"{CONVERGENCE_FAILURES_TOTAL}{labels} {count}")

        family(SOLVE_LATENCY_SECONDS)
        _render_histogram(
            lines,
            SOLVE_LATENCY_SECONDS,
            solve_latencies,
            self._buckets,
            ("problem", "solver_method"),
        )

        family(ITERATIONS)
        _render_histogram(
            lines, ITERATIONS, iterations, self._iterations.buckets, ("problem", "solver_method")
        )

        family(FINAL_OBJECTIVE_GAP)
        _render_histogram(
            lines, FINAL_OBJECTIVE_GAP, gaps, self._gaps.buckets, ("problem", "solver_method")
        )

        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        """Clear all series (test isolation only; counters reset on restart)."""
        with self._lock:
            self._requests.clear()
            self._errors.clear()
            self._latencies.clear()
            self._solves.clear()
            self._convergence_success.clear()
            self._convergence_failure.clear()
            self._solve_latencies.clear()
            self._iterations.clear()
            self._gaps.clear()


def _render_histogram(
    lines: list[str],
    name: str,
    snapshot: list[tuple[tuple, tuple[int, ...], float, int]],
    buckets: tuple[float, ...],
    label_names: tuple[str, ...],
) -> None:
    """Append one histogram family's bucket/sum/count lines."""
    for key, counts, total_sum, total_count in snapshot:
        base_pairs = list(zip(label_names, key))
        cumulative = 0
        for bound, n in zip(buckets, counts, strict=True):
            cumulative += n
            le_labels = format_labels([*base_pairs, ("le", _fmt_bucket(bound))])
            lines.append(f"{name}_bucket{le_labels} {cumulative}")
        le_labels = format_labels([*base_pairs, ("le", "+Inf")])
        lines.append(f"{name}_bucket{le_labels} {total_count}")
        lines.append(f"{name}_sum{format_labels(base_pairs)} {_fmt_value(total_sum)}")
        lines.append(f"{name}_count{format_labels(base_pairs)} {total_count}")


# Process-wide singleton (imported by app.py), mirroring ``observability.metrics``.
prometheus_metrics = PrometheusCollector()

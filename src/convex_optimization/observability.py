"""Stage 5 observability: structured logs + in-process counters (LOCAL ONLY).

This is convex-app-specific serving observability for the Stage 4 API. It is
NOT a production monitoring stack: no Grafana, no alerting, no persistence.
All state lives in this process and RESETS ON RESTART. (The Phase 2
Prometheus text-exposition endpoint lives in ``prometheus.py``; it scrapes
the same in-process counters plus the generic HTTP families, still with no
persistence and no third-party client library.)

Two artifacts per request:

- A single-line JSON log record on stdout (stdlib ``logging``, no extra deps):
  request id, endpoint, HTTP status, latency, problem, method, iteration
  count, converged flag, error class. Never the history arrays, never any
  iterate vector ``x`` -- counts, flags, and latencies only. No secrets exist
  in this app and none are ever logged.
- In-process counters exposed at ``GET /metrics``: request counts, latency
  summary and percentiles, 4xx/5xx error rates, and the CONVERGENCE-FAILURE
  rate, defined precisely as:

      a /solve request that completed with HTTP 200 but ``converged == false``,
      i.e. the Stage 1 method exhausted its fixed ``max_iter = 2000``
      iterations without meeting the stopping criterion (residual <=
      ``tol = 1e-10``). HTTP 4xx responses (request validation, inapplicable
      (problem, method) pairs) count as ERRORS, never as convergence
      failures; HTTP 5xx counts as a server error, also not a convergence
      failure.
"""

from __future__ import annotations

import json
import logging
import math
import sys
import threading
import time
import uuid
from collections import deque

_LOGGER_NAME = "convex_optimization.observability"

# Error-class taxonomy (error_class field in logs; 4xx/5xx only).
ERROR_INAPPLICABLE = "inapplicable_pair"
ERROR_VALIDATION = "validation_error"
ERROR_SERVER = "server_error"

# Bounded latency window for percentiles: the most recent requests only, so
# the metrics endpoint cannot grow without bound in a long-lived process.
LATENCY_WINDOW = 10_000


def _configure_logger() -> logging.Logger:
    """One stdout handler emitting raw single-line JSON records."""
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def _percentile(sorted_vals: list[float], pct: float) -> float | None:
    """Linear-interpolated percentile of an already-sorted list."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (len(sorted_vals) - 1) * pct / 100.0
    lo, hi = math.floor(rank), math.ceil(rank)
    if lo == hi:
        return sorted_vals[int(rank)]
    return sorted_vals[lo] * (hi - rank) + sorted_vals[hi] * (rank - lo)


class Metrics:
    """Thread-safe in-process counters (reset on process restart)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start = time.time()
        self._logger = _configure_logger()
        self._requests_total = 0
        self._by_endpoint: dict[str, int] = {}
        self._status = {"2xx": 0, "4xx": 0, "5xx": 0}
        # /solve-only solver counters (the convergence-failure signal).
        self._solve_count = 0
        self._solve_success = 0
        self._convergence_failures = 0
        self._iterations_last: int | None = None
        self._converged_last: bool | None = None
        self._latencies_ms: deque[float] = deque(maxlen=LATENCY_WINDOW)

    def record(
        self,
        endpoint: str,
        status: int,
        latency_ms: float,
        *,
        problem: str | None = None,
        method: str | None = None,
        iterations: int | None = None,
        converged: bool | None = None,
        error_class: str | None = None,
    ) -> str:
        """Update counters and emit one structured JSON log line.

        Returns the request id (also present in the log record). ``problem``,
        ``method``, ``iterations`` and ``converged`` are meaningful only for
        completed solves; ``error_class`` is set for 4xx/5xx only. Iterate
        vectors and history arrays are NEVER passed here or logged.
        """
        request_id = uuid.uuid4().hex[:12]
        bucket = "2xx" if status < 300 else ("4xx" if status < 500 else "5xx")
        with self._lock:
            self._requests_total += 1
            self._by_endpoint[endpoint] = self._by_endpoint.get(endpoint, 0) + 1
            self._status[bucket] += 1
            self._latencies_ms.append(latency_ms)
            if endpoint == "/solve":
                self._solve_count += 1
                if status == 200:
                    self._solve_success += 1
                    self._iterations_last = iterations
                    self._converged_last = converged
                    if converged is False:
                        # The Stage 5 convergence-failure definition: HTTP 200
                        # with converged=false (max_iter=2000 hit, tol=1e-10
                        # not met). 4xx/5xx are counted as errors instead.
                        self._convergence_failures += 1
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "level": "info" if status < 500 else "error",
            "request_id": request_id,
            "endpoint": endpoint,
            "status": status,
            "latency_ms": round(latency_ms, 3),
            "problem": problem,
            "method": method,
            "iterations": iterations,
            "converged": converged,
            "error_class": error_class,
        }
        self._logger.info(json.dumps(record, ensure_ascii=False))
        return request_id

    def snapshot(self) -> dict:
        """Current counters as a JSON-serializable dict (for GET /metrics)."""
        with self._lock:
            total = self._requests_total
            status = dict(self._status)
            lat = sorted(self._latencies_ms)
            solve_count = self._solve_count
            success = self._solve_success
            failures = self._convergence_failures
            by_endpoint = dict(self._by_endpoint)
            iterations_last = self._iterations_last
            converged_last = self._converged_last
            start = self._start

        def rate(numer: int, denom: int) -> float | None:
            return round(numer / denom, 6) if denom else None

        return {
            "scope": (
                "in-process counters on seeded synthetic problems; "
                "reset on process restart; NOT production monitoring/alerting"
            ),
            "uptime_seconds": round(time.time() - start, 3),
            "requests_total": total,
            "requests_by_endpoint": by_endpoint,
            "status_counts": status,
            "errors": {
                "total_4xx": status["4xx"],
                "total_5xx": status["5xx"],
                "rate_4xx": rate(status["4xx"], total),
                "rate_5xx": rate(status["5xx"], total),
            },
            "solve": {
                "count": solve_count,
                "success_count": success,
                "convergence_failures": failures,
                "convergence_failure_rate": rate(failures, success),
                "convergence_failure_definition": (
                    "HTTP 200 /solve response with converged=false: the method "
                    "hit max_iter=2000 without meeting tol=1e-10. 4xx "
                    "(validation, inapplicable pair) are errors, NOT "
                    "convergence failures; 5xx are server errors."
                ),
                "iterations_last": iterations_last,
                "converged_last": converged_last,
            },
            "latency_ms": {
                "window": min(len(lat), LATENCY_WINDOW),
                "mean": round(sum(lat) / len(lat), 3) if lat else None,
                "p50": round(_percentile(lat, 50), 3) if lat else None,
                "p90": round(_percentile(lat, 90), 3) if lat else None,
                "p99": round(_percentile(lat, 99), 3) if lat else None,
                "max": round(lat[-1], 3) if lat else None,
            },
        }


# Process-wide singleton; FastAPI dependency-free (imported by app.py).
metrics = Metrics()

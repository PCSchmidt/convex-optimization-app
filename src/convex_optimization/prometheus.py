"""Phase 2 shared observability contract: Prometheus text exposition (STDLIB ONLY).

Portfolio-wide Phase 2 contract: alongside the existing in-process JSON
``GET /metrics`` snapshot (unchanged), the app must expose ``GET
/metrics/prometheus`` returning valid Prometheus text exposition for four
GENERIC metric families. This module hand-rolls the exposition writer: no
``prometheus_client``, no new dependencies, no lockfile changes. The writer is
deterministic (fixed family order, sorted label tuples) and unit-testable
offline.

Families (prefix ``convex_optimization_``):

- ``convex_optimization_requests_total`` (counter): endpoint, method, status.
- ``convex_optimization_errors_total`` (counter): endpoint, method,
  error_class -- the SAME bounded error-class vocabulary as the JSON logs
  (``validation_error`` / ``inapplicable_pair`` / ``server_error``).
- ``convex_optimization_request_latency_seconds`` (histogram): endpoint,
  method; exports ``_bucket`` / ``_sum`` / ``_count``.
- ``convex_optimization_up`` (gauge): 1 while the app is serving.

Cardinality is bounded by construction:

- ``endpoint`` is a route template (``/health``, ``/metrics``,
  ``/metrics/prometheus``, ``/solve``, ...) or the literal ``unmatched`` for
  requests that matched no declared route. Never a full URL.
- ``method`` is the HTTP verb. ``status`` is an actually-used HTTP code.
- ``error_class`` comes only from the bounded taxonomy in
  ``observability.py``.

Request ids, user problem/method choices, exception messages, and full URLs
are NEVER label values (the ``method`` label is the HTTP verb only). Domain
metrics (convergence failures, iterations) remain in the JSON endpoint and
are deliberately NOT exported here in Phase 2.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence

METRIC_PREFIX = "convex_optimization"

# Exact Content-Type required by the Prometheus text exposition format.
CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

# Metric family names (counters end in _total; latency is in seconds).
REQUESTS_TOTAL = f"{METRIC_PREFIX}_requests_total"
ERRORS_TOTAL = f"{METRIC_PREFIX}_errors_total"
LATENCY_SECONDS = f"{METRIC_PREFIX}_request_latency_seconds"
UP = f"{METRIC_PREFIX}_up"

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

# Literal endpoint label for requests that matched no declared route (e.g.
# unknown paths -> 404). Bounded: it is a fixed token, never the raw path.
UNMATCHED_ENDPOINT = "unmatched"

_HELP = {
    UP: "1 when the app is serving.",
    REQUESTS_TOTAL: "Total HTTP requests handled, by endpoint, method, and status code.",
    ERRORS_TOTAL: (
        "Total HTTP requests counted as errors, with the app's bounded "
        "error_class vocabulary (validation_error, inapplicable_pair, server_error)."
    ),
    LATENCY_SECONDS: "End-to-end HTTP request latency in seconds.",
}

_TYPE = {
    UP: "gauge",
    REQUESTS_TOTAL: "counter",
    ERRORS_TOTAL: "counter",
    LATENCY_SECONDS: "histogram",
}


def escape_label_value(value: str) -> str:
    """Escape a label value per the Prometheus text exposition format."""
    return value.replace("\\", "\\\\").replace('"', '"').replace("\n", "\\n")


def format_labels(pairs: Sequence[tuple[str, str]]) -> str:
    """Render an ordered label set: ``{k="v",...}`` (empty pairs -> ``""``)."""
    if not pairs:
        return ""
    inner = ",".join(f'{key}="{escape_label_value(value)}"' for key, value in pairs)
    return "{" + inner + "}"


def _fmt_bucket(bound: float) -> str:
    """Compact, deterministic bucket bound (``0.005``, ``1``, ``+Inf``)."""
    return f"{bound:g}"


def _fmt_value(value: float) -> str:
    """Fixed-precision sample value (deterministic across runs)."""
    return f"{value:.6f}"


class _LatencySeries:
    """Per-(endpoint, method) histogram state."""

    __slots__ = ("bucket_counts", "total_count", "total_sum")

    def __init__(self, n_buckets: int) -> None:
        # NON-cumulative per-bucket counts; cumulative form is built at render.
        self.bucket_counts = [0] * n_buckets
        self.total_sum = 0.0
        self.total_count = 0


class PrometheusCollector:
    """Thread-safe collector + deterministic Prometheus text writer."""

    def __init__(self, buckets: tuple[float, ...] = LATENCY_BUCKETS) -> None:
        self._lock = threading.Lock()
        self._buckets = tuple(sorted(buckets))
        self._requests: dict[tuple[str, str, str], int] = {}
        self._errors: dict[tuple[str, str, str], int] = {}
        self._latencies: dict[tuple[str, str], _LatencySeries] = {}

    def record_request(
        self, endpoint: str, method: str, status: int, latency_seconds: float
    ) -> None:
        """Count one handled request and add its latency to the histogram."""
        req_key = (endpoint, method, str(status))
        lat_key = (endpoint, method)
        with self._lock:
            self._requests[req_key] = self._requests.get(req_key, 0) + 1
            series = self._latencies.get(lat_key)
            if series is None:
                series = _LatencySeries(len(self._buckets))
                self._latencies[lat_key] = series
            for i, bound in enumerate(self._buckets):
                if latency_seconds <= bound:
                    series.bucket_counts[i] += 1
                    break
            series.total_sum += latency_seconds
            series.total_count += 1

    def record_error(self, endpoint: str, method: str, error_class: str) -> None:
        """Count one error-classified request (bounded vocabulary only)."""
        key = (endpoint, method, error_class)
        with self._lock:
            self._errors[key] = self._errors.get(key, 0) + 1

    def counts(self) -> dict[tuple[str, str, str], int]:
        """Copy of the requests_total series (for tests/introspection)."""
        with self._lock:
            return dict(self._requests)

    def error_counts(self) -> dict[tuple[str, str, str], int]:
        """Copy of the errors_total series (for tests/introspection)."""
        with self._lock:
            return dict(self._errors)

    def render(self) -> str:
        """Full exposition document, deterministic for a given state."""
        with self._lock:
            requests = sorted(self._requests.items())
            errors = sorted(self._errors.items())
            latencies = sorted(
                (
                    key,
                    tuple(series.bucket_counts),
                    series.total_sum,
                    series.total_count,
                )
                for key, series in self._latencies.items()
            )

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
        for (endpoint, method), counts, total_sum, total_count in latencies:
            base = format_labels((("endpoint", endpoint), ("method", method)))
            cumulative = 0
            for bound, n in zip(self._buckets, counts, strict=True):
                cumulative += n
                lines.append(
                    f'{LATENCY_SECONDS}_bucket{base[:-1]},le="{_fmt_bucket(bound)}"}} {cumulative}'
                )
            lines.append(f'{LATENCY_SECONDS}_bucket{base[:-1]},le="+Inf"}} {total_count}')
            lines.append(f"{LATENCY_SECONDS}_sum{base} {_fmt_value(total_sum)}")
            lines.append(f"{LATENCY_SECONDS}_count{base} {total_count}")

        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        """Clear all series (test isolation only; counters reset on restart)."""
        with self._lock:
            self._requests.clear()
            self._errors.clear()
            self._latencies.clear()


# Process-wide singleton (imported by app.py), mirroring ``observability.metrics``.
prometheus_metrics = PrometheusCollector()

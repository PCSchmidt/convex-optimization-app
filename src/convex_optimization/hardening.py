"""Phase A1 request hardening: per-IP rate limiting + body-size guard.

In-process, single-instance, demo-grade (see README Phase A section): there
is no auth, no distributed state, and no persistence -- every rate-limit
window resets when the process restarts. The goal is to bound accidental or
casual abuse of a publicly deployed demo, not to stop a determined attacker.

Components:

- :class:`RateLimiter` -- fixed-window per-key counter (window: 60 s by
  default), injectable clock and window length for deterministic offline
  tests, thread-safe, with bounded key memory (stale windows are evicted).
  The limit is read from the ``RATE_LIMIT_PER_MIN`` environment variable at
  CHECK time (default 30; values <= 0 disable limiting) so tests can
  monkeypatch the environment without rebuilding the app.
- :class:`HardeningMiddleware` -- pure-ASGI middleware (no BaseHTTPMiddleware
  buffering): returns 429 with ``Retry-After`` when a client exceeds the
  window, and 413 when a request body exceeds ``MAX_BODY_BYTES`` (64 KiB).
  Short-circuited requests are recorded in BOTH metrics layers here (the
  inner Prometheus/observability middleware never sees them), with the same
  bounded error classes used everywhere else.
- :func:`resolve_client_key` -- the rate-limit key. Behind fly.io the proxy
  sets ``Fly-Client-IP`` (client-supplied values are overwritten by the
  proxy); otherwise the direct socket address is used. Loopback requests and
  the ASGI test client are EXEMPT: local health checks, the local compose
  Prometheus scraper, and the offline test suite must never be throttled.
- :func:`parse_allowed_origins` -- ``ALLOWED_ORIGINS`` parsing (comma-
  separated; empty/absent = same-origin only, i.e. no CORS headers at all).
  A wildcard is never accepted from the environment.

Client keys, raw URLs, and body contents are never logged.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections.abc import Awaitable, Callable

from starlette.datastructures import Headers

from .observability import ERROR_BODY_TOO_LARGE, ERROR_RATE_LIMITED, metrics
from .prometheus import prometheus_metrics

# Reject any request body larger than this (Content-Length header based, with
# a capped streamed read for chunked bodies). ~64 KB is far above every
# legitimate /solve or /parse request (bounded fields, few hundred bytes).
MAX_BODY_BYTES = 64 * 1024
RATE_LIMIT_PER_MIN_DEFAULT = 30
RATE_LIMIT_WINDOW_SECONDS = 60.0
# Hard cap on tracked client keys (bounded memory); stale entries evicted.
MAX_RATE_LIMIT_KEYS = 10_000

ASGIReceive = Callable[[], Awaitable[dict]]
ASGISend = Callable[[dict], Awaitable[None]]
ASGIApp = Callable[[dict, ASGIReceive, ASGISend], Awaitable[None]]

# Address values that are always exempt from rate limiting: the ASGI test
# client (offline test suite) and loopback (local health checks, the local
# compose Prometheus scraper on the same host).
EXEMPT_CLIENT_HOSTS = frozenset({"testclient", "127.0.0.1", "::1", "localhost"})


def parse_rate_limit(raw: str | None) -> int:
    """Parse ``RATE_LIMIT_PER_MIN``; <= 0 or unparsable means disabled.

    An unparsable value disables limiting rather than guessing (documented
    fail-open choice: a typo must not lock out the operator's own monitoring).
    """
    if raw is None:
        return RATE_LIMIT_PER_MIN_DEFAULT
    try:
        return int(raw)
    except ValueError:
        return 0


def parse_allowed_origins(raw: str | None) -> list[str]:
    """Parse ``ALLOWED_ORIGINS``: comma-separated scheme+host origins.

    Empty/absent -> ``[]`` (same-origin only; NO CORS headers are emitted, so
    browsers deny every cross-origin request). A literal ``*`` is rejected
    from the environment: the app never configures a wildcard allow-all.
    """
    if not raw:
        return []
    origins = [item.strip() for item in raw.split(",") if item.strip()]
    return [item for item in origins if item != "*"]


def resolve_client_key(scope: dict) -> str | None:
    """Rate-limit key for a request scope, or ``None`` if exempt.

    ``Fly-Client-IP`` (set by the fly.io proxy) wins over the direct socket
    address; both are only used as an opaque bounded string key and are never
    logged. Returns ``None`` for loopback / test-client requests.
    """
    headers = Headers(scope=scope)
    fly_ip = headers.get("fly-client-ip")
    if fly_ip:
        return fly_ip
    client = scope.get("client")
    if not client:
        return None
    host = str(client[0])
    if host in EXEMPT_CLIENT_HOSTS:
        return None
    return host


class RateLimiter:
    """Fixed-window per-key limiter (thread-safe; injectable clock/window).

    One counter per client key per window. When a key's window rolls over,
    its count restarts. Keys are evicted once they are stale AND the store
    exceeds ``MAX_RATE_LIMIT_KEYS``, so memory stays bounded on a public
    endpoint.
    """

    def __init__(
        self,
        window_seconds: float = RATE_LIMIT_WINDOW_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._window = float(window_seconds)
        self._clock = clock
        self._lock = threading.Lock()
        self._buckets: dict[str, tuple[float, int]] = {}

    def check(self, key: str, limit: int | None = None) -> tuple[bool, int]:
        """Count one request for ``key``; return (allowed, retry_after_s).

        ``limit`` defaults to the ``RATE_LIMIT_PER_MIN`` environment value
        (parsed at check time, so tests can reconfigure without rebuilds).
        ``retry_after_s`` is 0 when allowed, else the ceil-seconds until the
        current window closes. Disabled limits (<= 0) always allow.
        """
        if limit is None:
            limit = parse_rate_limit(os.environ.get("RATE_LIMIT_PER_MIN"))
        if limit <= 0:
            return True, 0
        now = self._clock()
        window_start = now - (now % self._window)
        with self._lock:
            if len(self._buckets) > MAX_RATE_LIMIT_KEYS:
                # Bounded memory: drop every bucket from an older window.
                self._buckets = {k: v for k, v in self._buckets.items() if v[0] == window_start}
            start, count = self._buckets.get(key, (window_start, 0))
            if start != window_start:
                start, count = window_start, 0
            if count >= limit:
                retry_after = max(1, math.ceil(start + self._window - now))
                return False, retry_after
            self._buckets[key] = (window_start, count + 1)
            return True, 0

    def reset(self) -> None:
        """Clear all state (test isolation only; resets on restart anyway)."""
        with self._lock:
            self._buckets.clear()


# Process-wide singleton (state lives in-process; resets on restart).
rate_limiter = RateLimiter()


def _record_short_circuit(
    route_template: str, method: str, status: int, latency_seconds: float, error_class: str
) -> None:
    """Record a 429/413 in BOTH metrics layers (same convention as in-app).

    The hardening middleware runs OUTSIDE the Prometheus/observability
    middleware, so a short-circuited request never reaches those layers --
    record it here to keep ``requests_total``, ``latency`` and the JSON
    snapshot complete. Error classes are the bounded taxonomy only.
    """
    prometheus_metrics.record_request(route_template, method, status, latency_seconds)
    prometheus_metrics.record_error(route_template, method, error_class)
    metrics.record(route_template, status, latency_seconds * 1000.0, error_class=error_class)


def _error_detail(error: str, **fields: object) -> bytes:
    """JSON body for a hardening rejection (bounded, no user input echoed)."""
    return json.dumps({"detail": {"error": error, **fields}}).encode()


class HardeningMiddleware:
    """Pure-ASGI 429 rate limiting + 413 body-size guard (Phase A1)."""

    def __init__(self, app: ASGIApp, fastapi_app: object) -> None:
        self.app = app
        self._fastapi_app = fastapi_app

    def _route_template(self, scope: dict) -> str:
        """Route template for the scope (bounded label), or ``unmatched``.

        Same contract as ``app._route_template``; kept here because the
        hardening middleware runs before the app's own helper is reachable.
        """
        from starlette.routing import Match  # local import: avoids cycles

        for route in getattr(self._fastapi_app, "routes", []):
            match, _ = route.matches(scope)
            if match == Match.FULL:
                return route.path
        return "unmatched"

    async def __call__(self, scope: dict, receive: ASGIReceive, send: ASGISend) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        t0 = time.perf_counter()
        method = scope.get("method", "")

        # --- rate limiting (per client key; loopback/test-client exempt) ---
        key = resolve_client_key(scope)
        if key is not None:
            allowed, retry_after = rate_limiter.check(key)
            if not allowed:
                template = self._route_template(scope)
                body = _error_detail(
                    "rate limit exceeded",
                    retry_after_seconds=retry_after,
                    limit_per_minute=parse_rate_limit(os.environ.get("RATE_LIMIT_PER_MIN")),
                )
                headers = [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"retry-after", str(retry_after).encode()),
                ]
                await send({"type": "http.response.start", "status": 429, "headers": headers})
                await send({"type": "http.response.body", "body": body})
                _record_short_circuit(
                    template, method, 429, time.perf_counter() - t0, ERROR_RATE_LIMITED
                )
                return

        # --- body-size guard on request bodies ---
        if method in ("POST", "PUT", "PATCH"):
            headers = Headers(scope=scope)
            content_length = headers.get("content-length")
            if content_length is not None:
                try:
                    declared = int(content_length)
                except ValueError:
                    declared = 0
                if declared > MAX_BODY_BYTES:
                    await self._reject_body(scope, send, method, t0, declared)
                    return
            else:
                # No Content-Length (chunked body): read with a hard cap and
                # replay the consumed bytes downstream.
                ok, body = await self._capped_read(receive)
                if not ok:
                    await self._reject_body(scope, send, method, t0, None)
                    return

                async def replay_receive() -> dict:
                    if not replay_receive.sent:  # type: ignore[attr-defined]
                        replay_receive.sent = True  # type: ignore[attr-defined]
                        return {"type": "http.request", "body": body, "more_body": False}
                    message = await receive()
                    return message

                replay_receive.sent = False  # type: ignore[attr-defined]
                receive = replay_receive

        await self.app(scope, receive, send)

    async def _capped_read(self, receive: ASGIReceive) -> tuple[bool, bytes]:
        body = b""
        while True:
            message = await receive()
            if message["type"] != "http.request":
                return True, body
            body += message.get("body", b"")
            if len(body) > MAX_BODY_BYTES:
                return False, body
            if not message.get("more_body"):
                return True, body

    async def _reject_body(
        self, scope: dict, send: ASGISend, method: str, t0: float, declared: int | None
    ) -> None:
        template = self._route_template(scope)
        fields: dict[str, object] = {"max_body_bytes": MAX_BODY_BYTES}
        if declared is not None:
            fields["declared_bytes"] = declared
        body = _error_detail("request body too large", **fields)
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ]
        await send({"type": "http.response.start", "status": 413, "headers": headers})
        await send({"type": "http.response.body", "body": body})
        _record_short_circuit(template, method, 413, time.perf_counter() - t0, ERROR_BODY_TOO_LARGE)

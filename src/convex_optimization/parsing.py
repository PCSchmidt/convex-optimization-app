"""Phase A1 natural-language -> problem-spec parsing pipeline.

POST /parse turns a natural-language description (e.g. ``"minimize ||Ax-b||^2
with 50 variables, seed 7"``) into the structured request that POST /solve
accepts. Two providers behind one interface (mirroring the youtube app's
generation.py design, with stdlib urllib instead of httpx -- no new
dependencies):

- :class:`OpenAICompatibleProvider` -- the PRODUCTION path: any
  OpenAI-compatible /chat/completions endpoint (OpenAI, OpenRouter, vLLM,
  Ollama, ...). Configured entirely from the environment: ``LLM_API_KEY``
  (required to activate), ``LLM_BASE_URL`` (default
  ``https://openrouter.ai/api/v1``), ``LLM_MODEL``. With no key configured
  the endpoint returns a documented ``provider_not_configured`` error -- it
  never crashes, and the UI hides the feature.
- :class:`StubProvider` -- a deterministic rule-based (regex/heuristic)
  parser. THIS IS A TEST DOUBLE ONLY, clearly not the production parser: it
  recognizes a handful of phrasings so the offline test suite can exercise
  the endpoint. Selected explicitly via ``LLM_PROVIDER=stub`` (test hook);
  never the default.

CREDIBILITY FEATURE -- never trust the parse blindly: every parsed spec is
passed through :func:`verify_parse`, a DETERMINISTIC mechanical verifier that
checks the structured spec against the original text (problem-type keywords,
dimension numbers, seed integer, weight numbers). The /parse response carries
``verified`` plus human-readable ``mismatches``; a 200 with ``verified=false``
means "parsed, but the text does not fully support it".

Outcome families: provider/parse errors use the bounded error classes
``provider_not_configured`` (503), ``provider_unavailable`` (502) and
``parse_invalid`` (422); every outcome is counted in the bounded
``convex_optimization_parse_outcomes_total`` Prometheus family.

The provider only ever sees the user's text; the API key is sent only to the
configured base URL and is never logged or echoed.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Protocol

# Bounded timeout for the upstream LLM call (a public endpoint must not hang).
LLM_TIMEOUT_SECONDS = 30.0
LLM_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
LLM_DEFAULT_MODEL = "openai/gpt-4o-mini"

# Cap on user text (defense in depth under the 64 KiB body cap).
MAX_PARSE_TEXT_CHARS = 2000


class ParseFailure(Exception):
    """The text could not be turned into a valid problem spec (422 class)."""


class ProviderNotConfigured(Exception):
    """No LLM provider is configured (503 class; UI hides the feature)."""


class ProviderUnavailable(Exception):
    """The configured upstream LLM failed or timed out (502 class)."""


class ParseProvider(Protocol):
    """Anything that maps one text to a raw spec dict (or raises)."""

    def parse(self, text: str) -> dict: ...


# ---------------------------------------------------------------------------
# Raw spec shape shared by both providers:
#   {"problem": str|None, "params": {field: value|None}, ...}
# Unknown/extra keys are dropped by complete_spec and flagged by the verifier.
# ---------------------------------------------------------------------------

_NUM = r"[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"

# Kind keyword patterns, most specific first (lasso before least_squares).
_KIND_PATTERNS: tuple[tuple[str, str], ...] = (
    ("lasso", r"\blasso\b|\bl\s*1\b|l_1|l1[- ]regulariz|\|\|x\|\|_1"),
    (
        "least_squares",
        r"least[- ]?squares|\|\|\s*a\s*x\s*-\s*b\s*\|\||ordinary least|normal equations",
    ),
    ("logistic", r"\blogistic\b|classification|sigmoid|cross[- ]entropy"),
)

_VAR_UNITS = r"(?:variables?|features?|dimension(?:s)?|dims?|parameters?)"
_VAR_PATTERNS = (
    rf"(?P<n>({_NUM}))\s*{_VAR_UNITS}\b",
    rf"{_VAR_UNITS}\s*(?:of|=|:)?\s*(?P<n>{_NUM})\b",
)
_SEED_PATTERN = r"seed\s*(?:=|:)?\s*(?P<n>[0-9]+)\b"
_LAM_PATTERN = (
    rf"(?:\blam\b|\blambda\b|l1[- ]weight|regularization(?:\s*weight)?)\s*(?:=|:|of)?\s*"
    rf"(?P<n>{_NUM})"
)
_RIDGE_PATTERN = rf"\bridge\b\s*(?:=|:|of)?\s*(?P<n>{_NUM})"


def _find_numbers(text: str, pattern: str) -> list[float]:
    """All numbers captured by ``pattern``'s ``n`` group (duplicates kept)."""
    return [float(m.group("n")) for m in re.finditer(pattern, text, re.IGNORECASE)]


class StubProvider:
    """Deterministic rule-based parser -- A TEST DOUBLE, not production.

    Recognizes a small set of phrasings with fixed regexes so the offline
    test suite can exercise /parse end to end without keys or network. It is
    deliberately naive: the production path is OpenAICompatibleProvider.
    Selected via ``LLM_PROVIDER=stub`` only.
    """

    def parse(self, text: str) -> dict:
        lowered = text.lower()
        kind = None
        for candidate, pattern in _KIND_PATTERNS:
            if re.search(pattern, lowered):
                kind = candidate
                break
        if kind is None:
            raise ParseFailure("could not identify a supported problem type in the text")
        params: dict[str, float | int | None] = {}
        for pattern in _VAR_PATTERNS:
            found = _find_numbers(lowered, pattern)
            if found:
                # First match wins; conflicts are flagged by the verifier.
                params["n_vars"] = int(found[0])
                break
        seeds = _find_numbers(lowered, _SEED_PATTERN)
        if seeds:
            params["seed"] = int(seeds[0])
        lams = _find_numbers(lowered, _LAM_PATTERN)
        if lams:
            params["lam"] = lams[0]
        ridges = _find_numbers(lowered, _RIDGE_PATTERN)
        if ridges:
            params["ridge"] = ridges[0]
        return {"problem": kind, "params": params}


SYSTEM_PROMPT = (
    "You convert one natural-language description of a convex optimization "
    "problem into strict JSON. Output ONLY a JSON object, no prose, no code "
    "fences, exactly this schema:\n"
    '{"problem": "least_squares" | "lasso" | "logistic",\n'
    ' "params": {"seed": int|null, "n_vars": int|null, "n_rows": int|null,\n'
    '            "lam": number|null, "ridge": number|null, "condition": number|null}}\n'
    "Use null for anything the text does not state. least_squares = minimize "
    "||Ax-b||^2; lasso = ||Ax-b||^2 + lambda*||x||_1; logistic = regularized "
    "logistic regression. Never invent numbers that are not in the text."
)


class OpenAICompatibleProvider:
    """Chat-completions client for any OpenAI-compatible endpoint (stdlib).

    Uses ``urllib.request`` (no new dependencies). One POST per parse; the
    response must contain a strict-JSON spec or the parse fails.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = LLM_TIMEOUT_SECONDS,
    ) -> None:
        self.model = model or os.environ.get("LLM_MODEL") or LLM_DEFAULT_MODEL
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL") or LLM_DEFAULT_BASE_URL).rstrip(
            "/"
        )
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or ""
        self.timeout_seconds = timeout_seconds

    def parse(self, text: str) -> dict:
        if not self.api_key:
            raise ProviderNotConfigured("LLM_API_KEY is not set")
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
        except ProviderNotConfigured:
            raise
        except (
            urllib.error.URLError,
            TimeoutError,
            KeyError,
            IndexError,
            ValueError,
            OSError,
        ) as exc:
            raise ProviderUnavailable(f"upstream LLM call failed: {exc}") from exc
        return self._parse_content(content)

    def _parse_content(self, content: str) -> dict:
        """Extract the strict-JSON spec from the model's message content."""
        stripped = content.strip()
        if stripped.startswith("```"):  # tolerate a single fenced block
            stripped = stripped.strip("`")
            stripped = stripped.removeprefix("json")
            stripped = stripped.strip()
        try:
            spec = json.loads(stripped)
        except ValueError as exc:
            raise ParseFailure(f"model did not return valid JSON: {exc}") from exc
        if not isinstance(spec, dict) or "problem" not in spec:
            raise ParseFailure("model JSON did not contain a 'problem' field")
        params = spec.get("params")
        if params is not None and not isinstance(params, dict):
            raise ParseFailure("model JSON 'params' is not an object")
        return {"problem": spec.get("problem"), "params": params or {}}


def provider_configured() -> bool:
    """True when the production LLM path is activated (LLM_API_KEY set)."""
    return bool(os.environ.get("LLM_API_KEY"))


def get_provider() -> ParseProvider:
    """Resolve the provider from the environment (per request).

    - ``LLM_PROVIDER=stub`` -> :class:`StubProvider` (documented TEST hook;
      the offline suite uses it; never the production default).
    - ``LLM_API_KEY`` set -> :class:`OpenAICompatibleProvider`.
    - otherwise -> ``ProviderNotConfigured`` (the endpoint maps this to the
      documented 503 ``provider_not_configured`` error class).
    """
    if os.environ.get("LLM_PROVIDER") == "stub":
        return StubProvider()
    if provider_configured():
        return OpenAICompatibleProvider()
    raise ProviderNotConfigured("no LLM provider configured; set LLM_API_KEY to enable /parse")


# ---------------------------------------------------------------------------
# Spec completion + deterministic verification
# ---------------------------------------------------------------------------

# Fields a spec may carry per kind (everything else is dropped + flagged).
_ALLOWED_PARAM_FIELDS = frozenset({"seed", "n_rows", "n_vars", "lam", "ridge", "condition"})


def complete_spec(spec: dict) -> tuple[str, dict]:
    """Normalize a raw provider spec into (kind, params) for /solve.

    Drops unknown fields (returned in ``dropped`` for the verifier), rejects
    unknown problem names and non-numeric field values, and leaves the
    per-field validation/range checks and defaults to
    ``problems.resolve_parameter_spec`` (the single source of truth). Raises
    :class:`ParseFailure` when the spec cannot describe a valid request.
    """
    problem = spec.get("problem")
    if problem not in ("least_squares", "lasso", "logistic"):
        raise ParseFailure(f"unknown problem type {problem!r} in parsed spec")
    raw_params = spec.get("params") or {}
    params: dict[str, float | int] = {}
    dropped: list[str] = []
    for key, value in raw_params.items():
        if key not in _ALLOWED_PARAM_FIELDS:
            dropped.append(str(key))
            continue
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ParseFailure(f"parsed field {key!r} is not a number")
        params[key] = value
    return str(problem), params


def verify_parse(text: str, problem: str, params: dict[str, float | int]) -> tuple[bool, list[str]]:
    """Deterministically check a parsed spec against the original text.

    Returns ``(verified, mismatches)``. Checks are mechanical ONLY (regex
    lookups, integer/float equality); nothing is trusted because an LLM
    produced it:

    - the text must contain a keyword of the parsed problem type;
    - every number the text states for a parameter family must match the
      parsed value (conflicting numbers in the text -> mismatch);
    - a parsed parameter the text never states -> mismatch (invented value).

    Checks that do not apply (e.g. no dimension phrasing in the text) are
    skipped, so a minimal honest parse ("lasso with seed 7") verifies.
    """
    mismatches: list[str] = []
    lowered = text.lower()

    matched_kinds = [k for k, pattern in _KIND_PATTERNS if re.search(pattern, lowered)]
    if problem not in matched_kinds:
        mismatches.append(
            f"no keywords for problem {problem!r} in the text"
            + (f" (text matches: {matched_kinds})" if matched_kinds else "")
        )

    checks = (
        ("n_vars", _VAR_PATTERNS),
        ("seed", (_SEED_PATTERN,)),
        ("lam", (_LAM_PATTERN,)),
        ("ridge", (_RIDGE_PATTERN,)),
    )
    for field, patterns in checks:
        if field not in params:
            continue
        stated: list[float] = []
        for pattern in patterns:
            stated.extend(_find_numbers(lowered, pattern))
        if not stated:
            mismatches.append(f"parsed {field}={params[field]} but the text never states it")
        elif not all(_num_equal(float(params[field]), n) for n in stated):
            # Conservative rule: if the text states CONFLICTING numbers for the
            # same family ("30 variables and 50 features"), the parse is NOT
            # verified unless it matches every stated number. Credibility
            # first: ambiguity surfaces, it is never silently resolved.
            mismatches.append(f"text states {field}={stated} but parsed {field}={params[field]}")

    # condition / n_rows cannot be stated textually in a checkable way; only
    # flag them if the text ALSO does not support the kind at all (handled
    # above). They are accepted as structural completion, not text claims.
    verified = not mismatches and problem in ("least_squares", "lasso", "logistic")
    return verified, mismatches


def _num_equal(a: float, b: float) -> bool:
    """Float equality with a relative tolerance (text '0.1' vs parsed 0.1)."""
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def parse_text(text: str) -> tuple[str, dict, str, bool, list[str]]:
    """Full pipeline: provider parse -> complete -> verify.

    Returns ``(problem, params, parse_method, verified, mismatches)``.
    Raises ``ParseFailure`` / ``ProviderUnavailable`` / ``ProviderNotConfigured``
    exactly like the provider call (the endpoint maps them to error classes).
    """
    if len(text) > MAX_PARSE_TEXT_CHARS:
        raise ParseFailure(f"text exceeds {MAX_PARSE_TEXT_CHARS} characters")
    provider = get_provider()
    parse_method = "stub" if isinstance(provider, StubProvider) else "llm"
    spec = provider.parse(text)
    problem, params = complete_spec(spec)
    # Fold dropped unknown keys into the mismatch list (credibility signal).
    dropped = [
        str(key)
        for key in (spec.get("params") or {})
        if str(key) not in _ALLOWED_PARAM_FIELDS and (spec.get("params") or {})[key] is not None
    ]
    verified, mismatches = verify_parse(text, problem, params)
    mismatches = [*[f"ignored unsupported field {key!r}" for key in dropped], *mismatches]
    verified = verified and not dropped
    return problem, params, parse_method, verified, mismatches

# ROADMAP.md - convex-optimization-app

Optimization application with a strong **rigor / reproducibility / benchmarking** story. This roadmap drives the build, ship, deploy, monitor, and maintain stages, with the emphasis on reproducible experiments and honest benchmarking.

## North star

A reviewer can clone this repo, run one command, and see a set of optimization methods implemented correctly, benchmarked against ground-truth solvers, with reproducible run logs and a documented lifecycle. The README tells the full lifecycle story, with rigor as the standout.

## Stage 0 - Foundation (prereq)

- [x] Inspect existing code and inventory what works vs. what is broken. (See `INVENTORY.md`.)
- [x] Decide stack (see AGENTS.md defaults) and document the choice. (Python 3.11+, pytest, ruff; numerical core pins deferred to Stage 1.)
- [x] Set up Python 3.11+ environment with pinned dependencies and a lockfile. (`requirements-lock.txt`, `make setup`.)
- [x] Add `.gitignore` for secrets, artifacts, and caches.
- [x] Establish a test harness and CI (GitHub Actions) that runs tests + lint.
- [x] Write the README skeleton with the Motivation / Method / Results / Limitations / Operational notes structure.

**Acceptance:** `git clone && make setup && make test` succeeds on a clean machine.

## Stage 1 - Build (optimization core)

- [x] Implement classic methods: gradient descent, accelerated (e.g., Nesterov), and at least one proximal/ADMM method.
- [x] Define a set of standard convex problems (e.g., least squares, logistic regression, L1-regularized) as benchmarks.
- [x] Ground truth: compare against SciPy solvers on the same problems.
- [x] Convergence tracking: record objective value and gradient norm per iteration.

**Acceptance:** Each method converges on the benchmark problems and matches ground-truth solutions within tolerance.

## Stage 2 - Evaluate (before optimizing)

- [x] Metrics: iterations to convergence, wall time, final objective gap, and stability across seeds.
- [x] Benchmark all methods across all problems; record results in an `experiments/` run log.
- [x] Fixed seeds and pinned versions for full reproducibility.

**Acceptance:** Benchmark results are recorded and reproducible. No tuning happens before this.

## Stage 3 - Ship (versioned, reproducible)

> Note: the Python floor was raised to 3.12 in this stage. numpy 2.5.3 from
> `requirements-lock.txt` requires Python >= 3.12, and pip install of the lock on Python 3.11
> fails (reproduced via `docker build --build-arg PYTHON_TAG=3.11`). The lockfile is unchanged.

- [x] Code versioning: tag releases; pin numerical library versions.
- [x] Artifact bundle: documented way to store and load experiment configs and results.
- [x] `requirements.lock` and a reproducible build path.
- [x] Containerize the app (Dockerfile) and provide `docker-compose.yml`.
- [x] CI/CD pipeline that builds, tests, and produces a tagged artifact.

**Acceptance:** A tagged release can be rebuilt and run reproducibly.

## Stage 4 - Deploy

- [x] FastAPI serving layer exposing the optimization methods as a documented API.
- [x] Deploy target decision: local Docker Compose (minimum) or a public endpoint (optional).
- [x] Environment-based configuration (no hardcoded secrets).
- [x] Document the deployment runbook.

**Acceptance:** The app runs from the container and responds to health + solve endpoints.

## Stage 5 - Monitor

- [x] Structured logging of solve requests, iterations, and latency.
- [x] Metrics endpoint exposing: request count, latency percentiles, error rate, convergence-failure rate.
- [x] Optional: Prometheus/Grafana dashboard. (Phase 6: compose `prometheus`
  + `grafana` services on env-overridable host ports 9092 / 3002; committed
  `prometheus.yml` scrape config; declarative Grafana provisioning
  (`provisioning/datasources`, `provisioning/dashboards` with dashboard JSON
  uid `convex-optimization-app`, 14 panels incl. app-specific
  convergence/iteration/gap families). Verified live: Prometheus target UP,
  instant queries return real values from generated traffic
  (`sum(convex_optimization_solves_total)` = 112), Grafana datasource health
  200, dashboard uid 200. Only intentional persistence: `grafana-data` named
  volume.)
- [x] Phase 2 shared observability contract: `GET /metrics/prometheus` Prometheus text
  exposition (stdlib-only writer in `src/convex_optimization/prometheus.py`; generic
  families requests_total / errors_total / request_latency_seconds / up with the
  `convex_optimization_` prefix; JSON `GET /metrics` unchanged).
- [x] App-specific Prometheus families (same endpoint): solves_total, solve_latency_seconds,
  convergence_successes_total / convergence_failures_total, iterations, final_objective_gap —
  labeled ONLY by registry problem/solver_method names; inapplicable-pair 422s create no
  problem/method series; JSON `GET /metrics` unchanged.
- [x] Document "what could degrade" (ill-conditioned inputs, numerical instability, solver failures).

**Acceptance:** A reviewer can see how the service is observed and what signals would indicate a problem.

## Stage 6 - Maintain

- [x] Refresh path: how to re-run the benchmark suite and regenerate results.
- [x] Rollback path: revert to a previous code/artifact version.
- [x] Runbook for common incidents (non-convergence, numerical instability, API failures).
- [x] One documented incident write-up (real or realistic) showing the maintain loop.

**Acceptance:** The maintain loop is documented and executable, not just described.

## Phase A - Public readiness (deployment project; backend A1 complete)

The app is being deployed publicly on fly.io behind a purchased domain with a React
workbench UI. Scope is explicitly DEMO-GRADE and documented as such (no production-
readiness claims): single instance, in-process rate limiting, no auth.

- [x] **A1 backend hardening** (this phase):
  - Parameterized problem instances: user-specified seeded variants of the SAME three
    registry problems on POST /solve (seed, n_rows, n_vars <= 200 hard cap, lam/ridge,
    least_squares condition knob). Frozen Stage 2 benchmark suite and the no-params path
    stay byte-identical; identical (problem, params, seed, method) is bit-reproducible.
    Iteration cap stays the fixed max_iter=2000 (larger instances may honestly return
    converged=false).
  - Request hardening: per-IP fixed-window rate limiting (429 + Retry-After,
    RATE_LIMIT_PER_MIN, default 30/min, in-process only), 64 KiB body cap (413), strict
    pydantic validation with extra="forbid", env-configurable CORS (ALLOWED_ORIGINS;
    empty default = same-origin only; wildcard never accepted).
  - POST /parse: natural language -> structured /solve request via an OpenAI-compatible
    provider (stdlib urllib, LLM_API_KEY/LLM_BASE_URL/LLM_MODEL; clean 503
    provider_not_configured without a key) plus a deterministic mechanical verifier
    (verified flag + mismatches; parses are never trusted blindly). StubProvider is a
    documented test double. New bounded error classes + the
    convex_optimization_parse_outcomes_total family on the existing writer.
  - 94 tests pass (60 pre-existing, unchanged except the brittle Prometheus family-count
    assertion 10 -> 11), ruff check + format clean.
- [x] **A2 workbench UI (local dev server)**: React + Vite + TypeScript app in `ui/`
  (recharts + vitest, offline tests, `make ui` / `ui-build` / `ui-test`). Panels:
  problem/method/params form with client-side caps mirroring the server, 429
  Retry-After countdown, convergence chart from the history tail (log-scale gap),
  converged vs 'hit the 2000-iteration cap' badge states, per-method explanations,
  and the /parse NL box (hidden with an honest note on 503; verified=false
  mismatches surfaced before running the spec). `/parse` stays
  provider-dependent: without `LLM_API_KEY` the panel hides itself. NOT a
  production deployment; fly.io wiring and TLS remain open below.
- [ ] A2 deployment: fly.io app, purchased domain, TLS at the proxy, workbench
  wiring to the deployed backend, deploy runbook evidence.
- [ ] Honest scope note to carry forward: single instance, in-process rate limiting
  (resets on restart), no auth, no per-user quotas, no distributed state. Demo-grade.

## Portfolio presentation

- [ ] README tells the full lifecycle story with real benchmark numbers, emphasizing rigor.
- [ ] Link the repo from `pcschmidt.github.io`.
- [ ] Prepare a 3-sentence interview arc per lifecycle stage.

## Definition of done

All stages complete, tests green, benchmarks reproducible, deployment reproducible, monitoring documented, and the README honestly reflects what is implemented.
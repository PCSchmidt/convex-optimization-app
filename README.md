# Convex Optimization App

## Motivation

This app is the optimization + rigor pillar of a personal portfolio. Its goal
is to demonstrate correct, well-documented convex optimization algorithms and
reproducible benchmarking against SciPy solvers. The emphasis is on honest,
reviewable engineering: fixed seeds, pinned dependencies, recorded run logs,
and convergence behavior shown rather than claimed.

The pre-refactor app (archived in this repository's git history before removal) delegated all solving to PuLP/CVXPY
wrappers with no self-implemented algorithm and no convergence record. Stage 1
replaces that approach with first-order methods written from scratch in NumPy.

## Method

Stage 1 core, all implemented from scratch in NumPy (`src/convex_optimization/`).
SciPy is used only as a ground-truth reference, never as the method.

### Algorithms and step-size rules

- **Gradient descent** (`methods.gradient_descent`). Fixed step
  `x_{k+1} = x_k - (1/L) * grad f(x_k)` where `L` is the (analytically known)
  Lipschitz constant of the gradient. `1/L` is the largest step with a
  guaranteed monotone decrease for an L-smooth convex objective, so it fits
  both smooth benchmarks and keeps every run deterministic. An Armijo
  backtracking line search (`step_size="backtracking"`, `c1 = 1e-4`,
  `rho = 0.5`) is provided for callers who do not know `L`; it is monotone by
  construction but stalls at a gradient norm of about 1e-7 in float64 near the
  optimum (the required trial step rounds the update to zero), so the
  benchmarks use the fixed `1/L` rule.
- **Nesterov accelerated gradient** (`methods.nesterov_ag`), for L-smooth,
  mu-strongly-convex objectives. Step `1/L`, constant momentum
  `gamma = (sqrt(kappa) - 1) / (sqrt(kappa) + 1)` with `kappa = L / mu`, which
  achieves the O((1 - 1/sqrt(kappa))^k) rate instead of gradient descent's
  O((1 - 1/kappa)^k). Both benchmarks declare their exact `L` and `mu`, so the
  momentum is derived from known problem conditioning, not tuned.
- **FISTA** (accelerated proximal gradient, Beck & Teboulle 2009;
  `methods.fista`), for composite `smooth + nonsmooth` problems such as the
  Lasso: `x_{k+1} = prox_{h/L}(y_k - grad g(y_k) / L)` with the standard
  momentum sequence. For the Lasso, `prox` is soft-thresholding at
  `lambda / L`, the exact proximal operator of `lambda * ||.||_1` at step
  `1/L`. The step `1/L` (L = Lipschitz constant of the smooth part) is the
  largest step for which the proximal-gradient map is guaranteed contractive.
- **ISTA** (non-accelerated proximal gradient, `methods.ista`; added in
  Stage 2 for benchmarking): exactly FISTA minus the momentum sequence, with
  the IDENTICAL `1/L` step, identical `prox`, identical tolerance and
  stopping rule. It exists so the benchmark can compare accelerated vs
  non-accelerated proximal gradient with nothing else changed. No constant
  was tuned for the benchmark.

Every method returns a `Result` whose `History` records the objective value
and a residual at every iterate, including the initial point. Residuals:
gradient norm for the smooth problems; for the nonsmooth Lasso, the norm of
the scaled prox-gradient map `|| x - prox(x - grad g(x) / L) || * L`, which is
zero exactly when 0 is in the subdifferential of the full objective.

### Benchmark problems (deterministic, small, CPU-friendly)

- **least_squares** (`problems.make_least_squares`): `min 0.5 ||Ax - b||^2`,
  A in R^(30x20) and b from `numpy.random.default_rng(0)`, noise 0.05.
  Smooth, strongly convex, kappa ~ 67.
- **lasso** (`problems.make_lasso`): `min 0.5 ||Ax - b||^2 + 0.1 ||x||_1`,
  same seeded data (seed 0, 30x20). Convex, nonsmooth.
- **logistic** (`problems.make_logistic`): mean logistic loss on a seeded
  synthetic set (seed 1, n = 40, d = 10) plus `(0.1/2)||x||^2` ridge.
  Smooth, strongly convex, kappa ~ 5.5.

### Ground truth

- least_squares: documented closed form — the unique minimizer of the normal
  equations via `numpy.linalg.lstsq` (a factorization, not an iterative
  first-order solver). The test suite cross-checks it against a SciPy
  L-BFGS-B optimum (agreement within 1e-10 in objective).
- lasso: SciPy L-BFGS-B on an eps-smoothed objective
  (`|x| -> sqrt(x^2 + eps^2)`, eps = 1e-10) with the analytic smoothed
  gradient. This is an approximate reference (bias < ~1e-9); the tests
  certify the reference point itself by its prox-gradient-map residual
  (<= 1e-6) before comparing against it.
- logistic: SciPy L-BFGS-B with the exact analytic gradient on the exact
  smooth objective.

### Tolerance honesty

- least_squares (exact closed-form truth): iterate distance
  `|x_final - x*| <= 1e-6` is asserted for gradient descent and Nesterov, plus
  objective gap <= 1e-8.
- lasso (approximate smoothed SciPy reference): the iterate distance is NOT
  asserted (the reference itself is approximate); instead the absolute
  objective gap to the reference must be <= 1e-8, and the final
  prox-gradient-map residual must be <= 1e-6.
- logistic (exact SciPy truth on a smooth problem): objective gap <= 1e-8 and
  final gradient norm <= 1e-6.

The CLI prints the final objective gap and the last five history rows for one
(problem, method) run; the Python API (`convex_optimization.solve` via
`cli.solve`) returns the full `Result` with the complete history.

## Results

Stage 2 baseline benchmark of the Stage 1 methods **as-is** (no tuning):
fixed `1/L` steps, Nesterov momentum from the known `L/mu`, `tol = 1e-10`,
`max_iter = 2000`, zero start. Applicable (problem, method) pairs only —
the smooth methods are not run on the nonsmooth Lasso. Three problem seeds
per problem (Stage 1 default seed included; dimensions unchanged). Full
per-run log: `experiments/run_log.csv` (+ `.json` metadata), regenerate with
`make eval`. All numbers below come from that log (numpy 2.5.3,
scipy 1.18.1, Git Bash on Windows).

Baseline results on small synthetic problems (NOT production benchmarks).
`n_iter` is fully deterministic (two consecutive runs reproduced it exactly);
wall time in milliseconds is the per-cell median of 3 runs and is **indicative
only** — it is noisy on this host and is not used to rank methods.

| problem | method | n_iter (seed A / B / C) | max abs gap | wall ms (range) | all converged |
| --- | --- | --- | --- | --- | --- |
| least_squares | gd | 1445 / 1475 / 668 | 1.1e-16 | 6.8-15.2 | yes |
| least_squares | nesterov | 192 / 200 / 129 | 2.3e-16 | 1.1-1.8 | yes |
| lasso | fista | 1436 / 950 / 775 | 6.6e-10 | 14.2-27.2 | yes |
| lasso | ista | 1451 / 1187 / 662 | 6.6e-10 | 13.4-25.1 | yes |
| logistic | gd | 62 / 71 / 62 | 5.6e-17 | 2.1-2.3 | yes |
| logistic | nesterov | 33 / 35 / 35 | 5.6e-17 | 0.8-1.3 | yes |

Seeds: least_squares and lasso use 0/1/2, logistic uses 1/2/3 (each problem's
Stage 1 default seed is included). "Gap" is `f(x_final) - f(x*)` against the
documented ground truth; for the lasso this is the eps=1e-10 smoothed SciPy
L-BFGS-B reference, which is approximate (bias < ~1e-9). Two lasso runs
(seed 1) land slightly BELOW that reference — expected, because the
eps-smoothing biases the reference optimum upward.

What the recorded numbers actually show:

- **Nesterov vs gradient descent: fewer iterations, consistently.** Across
  all seeds, Nesterov needed 129-200 vs 668-1475 iterations on least_squares
  (~3.5-7.7x fewer) and 33-35 vs 62-71 on logistic (~1.8-2.1x fewer). Since
  `n_iter` is deterministic, this is the robust comparison signal.
- **FISTA vs ISTA on the lasso: no consistent winner at this tolerance.**
  FISTA used fewer iterations than ISTA on seeds 0 and 1 (1436 vs 1451,
  950 vs 1187) but MORE on seed 2 (775 vs 662). The theoretical O(1/k^2) vs
  O(1/k) advantage did not translate into a clear iteration win on this small
  problem at `tol = 1e-10`. No acceleration claim is made for the lasso.
- **Stability across seeds:** all 18 runs converged, and every final gap is
  within the Stage 1 tolerance criteria (|gap| <= 1e-8; residuals <= 1e-6).

Wall time roughly tracks the iteration counts, but the cells take
milliseconds, so single timings are dominated by noise; treat them as
indicative only.

## Limitations

- First-order methods only (gradient descent, Nesterov accelerated gradient,
  FISTA). No second-order, ADMM, or constrained/SDP solvers.
- Small synthetic problems only (10-40 dimensions, seeded); no real datasets.
- Ground truth for the Lasso is an eps-smoothed SciPy reference, not an exact
  nonsmooth optimum; tolerances account for this, but it is an approximation.
- Armijo backtracking gradient descent stalls near the optimum in float64
  (documented above); the benchmarks use the fixed `1/L` step instead.
- Stage 2 baselines are measured on small synthetic problems (10-40
  dimensions, 3 seeds each) and are NOT production benchmarks. Only
  first-order methods are compared; no second-order, ADMM, or
  constrained-solver baselines.
- Wall-clock timing on the development host (Windows, Git Bash) is noisy;
  timings are reported per cell but methods are compared on iteration
  counts, which are deterministic.
- The lasso comparison (FISTA vs ISTA) shows no consistent iteration
  advantage for acceleration at `tol = 1e-10` on this problem size; the
  lasso truth is the approximate eps-smoothed SciPy reference, so lasso gaps
  inherit its bias.
- Requires Python 3.12+. The pinned lock (`requirements-lock.txt`, numpy 2.5.3) does not install
  on Python 3.11 because numpy 2.5.3 requires Python >= 3.12; the failure is reproducible with
  `docker build --build-arg PYTHON_TAG=3.11` (documented in the Dockerfile header).
- Serving (Stage 4) is a local, single-user demo API behind Docker Compose: no TLS and no
  auth at the app level. Phase A1 (public readiness) adds per-IP in-process rate limiting,
  a 64 KiB body cap, bounded parameterized instances, env-configurable CORS, and the
  verified /parse pipeline -- see the Phase A section. It stays single-instance demo-grade:
  rate-limit state is in-process (resets on restart), there is no auth, and no
  production-readiness claim is made. Observability (Stage 5) is local compose
  observability on seeded synthetic problems: structured JSON logs on stdout plus an
  in-process `/metrics` JSON endpoint whose counters RESET ON RESTART. There is no
  persistence, no Prometheus/Grafana scrape stack, no dashboards, and NO alerting — this is
  not production monitoring. The Stage 3 offline smoke solve (network_mode none) is unchanged.
- What could degrade, and the log/metric signal each failure leaves (Stage 5):
  - **Ill-conditioned inputs.** The API fixes the problems, but a future problem with a
    large condition number `kappa = L/mu` needs many more iterations at fixed step `1/L`;
    at `max_iter = 2000` it can simply run out. Signal: `/solve` returns 200 with
    `converged=false` -> `solve.convergence_failures` and
    `solve.convergence_failure_rate` rise in `/metrics`, and the JSON log line shows
    `"converged": false` with `iterations` pinned at 2000.
  - **Numerical instability.** In float64 a monotone first-order method can stall
    (the documented Armijo stall) or overflow (e.g. `exp` in the logistic loss far from
    the optimum). A stall shows up as above (`converged=false`, huge iteration count);
    an overflow or NaN raises and surfaces as HTTP 500 -> `status_counts["5xx"]` and
    `errors.rate_5xx` in `/metrics`, log line `"error_class": "server_error"`.
  - **Solver non-convergence (max_iter hit).** Same signal as ill-conditioning: a 200
    solve with `converged=false`. By definition this is the ONLY thing counted as a
    convergence failure; it means the fixed Stage 1 settings (`tol=1e-10`,
    `max_iter=2000`, `1/L` step) did not reach the criterion — no retry, no retune.
  - **Inapplicable (problem, method) pairs** (e.g. lasso + nesterov). Rejected with 422
    before any solve. Signal: `status_counts["4xx"]`, `errors.rate_4xx`, and the log line
    `"error_class": "inapplicable_pair"`. Malformed bodies (bad `tail`, unknown names)
    are 4xx with `"error_class": "validation_error"`. These are request errors, NEVER
    convergence failures.

## Operational notes

Requirements: Python 3.12+, `git`, GNU `make`, and optionally Docker (local
container runs only).

```bash
git clone <repo-url>
cd convex_optimization_app
make setup   # creates .venv, installs pinned dependencies from requirements-lock.txt
make test    # runs pytest (smoke + Stage 1 correctness) and ruff check
```

Run one offline solve (one problem, one method; prints the final objective
gap vs ground truth and the last few history rows):

```bash
make solve ARGS="--problem lasso --method fista"     # lasso | least_squares | logistic
make solve ARGS="--problem logistic --method nesterov"
```

Run the Stage 2 benchmark (all applicable problem/method pairs x 3 seeds,
writes `experiments/run_log.csv` and `experiments/run_log.json`; NOT part of
`make test`, and offline):

```bash
make eval
```

### Stage 4 serving API (LOCAL Docker Compose only)

Deploy-target decision: **local Docker Compose is the accepted minimum deploy
target. Public cloud endpoints (ngrok, Azure, AWS, or any paid hosting) were
DECLINED** — cost, and this is a portfolio demonstration of a solver API, not
a production service. Accordingly there are **no TLS, no auth, and no
multi-user serving claims**: the API below binds to localhost by default via
Docker's published port and is meant for a reviewer to run on their own
machine.

The API (`src/convex_optimization/app.py`, FastAPI) runs the EXISTING Stage 1
methods as-is via `cli.solve` — same step sizes, tolerances (`tol=1e-10`),
`max_iter=2000`, and default-seeded problems. No numerical code is duplicated.
Applicable pairs mirror the CLI's `APPLICABLE` matrix:

| problem | applicable methods |
| --- | --- |
| least_squares | gd, nesterov |
| lasso | fista, ista |
| logistic | gd, nesterov |

Endpoints (interactive docs at `/docs`):

- `GET /health` -> `{"status":"ok"}` (200).
- `POST /solve`, JSON body `{"problem": ..., "method": ..., "tail": 5}`:
  - `problem`/`method` must be valid names (422 otherwise); an INAPPLICABLE
    pair (e.g. lasso + nesterov) also returns **422** with the list of
    applicable methods, mirroring the CLI.
  - `tail` (optional, 1-20, default 5) bounds the returned history rows; the
    full history is never returned by the API (use the CLI / Python API).
  - 200 response: `iterations`, `converged`, `tol`, `final_objective`,
    `ground_truth_objective`, `final_objective_gap` (vs the documented ground
    truth — for lasso the eps-smoothed SciPy approximate reference, computed
    CPU-only at request time), `final_residual`, `ground_truth_source`,
    `history_tail` (last rows as `{iteration, objective, residual}`).

### Stage 5 observability (LOCAL compose only — not production monitoring)

Stage 5 adds two artifacts over the UNCHANGED Stage 4 API (additive only: same
endpoints, same 422 behavior, still no tol/max_iter/seed knobs):

- **Structured JSON logging.** Every request (including `/health` and
  `/metrics`) emits ONE single-line JSON record on stdout with `request_id`,
  `endpoint`, `status`, `latency_ms`, `problem`, `method`, `iterations`,
  `converged`, and `error_class`. Stdlib `logging` only — no new dependency.
  Iterate vectors and history arrays are never logged (bounded counts, flags,
  and latencies only). No secrets exist in this app and none are logged. The
  compose `api` command runs uvicorn with `--no-access-log` so these JSON
  lines are the sole stdout log stream.
- **`GET /metrics` (JSON).** In-process counters (`observability.py`):
  `requests_total`, per-endpoint counts, `status_counts` (2xx/4xx/5xx),
  `errors` (4xx/5xx totals and rates), `latency_ms` (mean, p50/p90/p99, max
  over the most recent 10,000 requests), and the solver block `solve` with
  the convex-specific counters: `convergence_failures` and
  `convergence_failure_rate`.

  **Convergence-failure definition (precise):** a `/solve` request that
  completed with HTTP 200 but `converged=false` — the method exhausted its
  fixed `max_iter = 2000` iterations without meeting the Stage 1 stopping
  criterion (residual <= `tol = 1e-10`). HTTP 4xx (request validation,
  inapplicable pairs) count as ERRORS, never as convergence failures; HTTP
  5xx counts as a server error, also not a convergence failure. The rate is
  `convergence_failures / solve.success_count`.
- **`GET /metrics/prometheus` (text).** Phase 2 shared observability
  contract (`prometheus.py`, STDLIB ONLY — no `prometheus_client`, no new
  dependencies): Prometheus text exposition with `Content-Type: text/plain;
  version=0.0.4; charset=utf-8` for the four GENERIC families
  `convex_optimization_requests_total` (labels: endpoint, method, status),
  `convex_optimization_errors_total` (labels: endpoint, method, error_class
  — the same bounded taxonomy as the JSON logs),
  `convex_optimization_request_latency_seconds` (histogram over endpoint and
  method, prometheus_client default buckets 0.005s–10s), and
  `convex_optimization_up` (1 while serving). Labels are low-cardinality by
  construction: `endpoint` is a route template (or the fixed token
  `unmatched`), never a URL. The JSON `GET /metrics` snapshot above is
  UNCHANGED. Domain counters (convergence failures, iterations) are
  deliberately NOT exported here yet.
- **Scope honesty.** Counters are in-process and RESET ON RESTART; there is
  no alerting, no retention beyond the 1-day local Prometheus TSDB, and no
  external monitoring. This is local compose observability on seeded
  synthetic problems so a reviewer can SEE the signals — not production
  monitoring. Phase 6 adds the LOCAL Grafana + Prometheus stack (next
  section); the JSON contract is unchanged.
- **Phase 6 note.** The domain counters listed as "NOT exported here yet"
  above were added in Phase 3 (`convex_optimization_solves_total`,
  `convex_optimization_convergence_successes_total` /
  `_convergence_failures_total`, `convex_optimization_solve_latency_seconds`,
  `convex_optimization_iterations`, `convex_optimization_final_objective_gap`)
  and are scraped by the Phase 6 Prometheus job into the Phase 6 Grafana
  dashboard.

### Phase 6 observability stack (LOCAL Grafana + Prometheus, compose only)

Declarative, reproducible, and offline after image pulls: scrape config,
datasource, and dashboard are committed files, not hand-clicked UI state.

- **`prometheus.yml`** — one scrape job, `convex-optimization-api`, scraping
  `api:8000/metrics/prometheus` every 5 s inside the compose network
  (`--storage.tsdb.retention.time=1d`). No federation, no cloud, no external
  monitoring.
- **`provisioning/datasources/prometheus.yml`** — auto-configures the default
  Prometheus datasource (fixed uid `convex-prom`) pointing at the compose
  `prometheus` service.
- **`provisioning/dashboards/`** — `dashboards.yml` (file provider) +
  `convex_optimization.json` (uid `convex-optimization-app`), auto-loaded
  into the "Convex Optimization" folder. 14 panels: app health (up stat),
  HTTP request rate, HTTP error rate, p50/p95/p99 latency, request rate by
  endpoint, recent-errors table, solver error classes, solve request rate by
  problem/method, solve latency p50/p95, convergence success rate, convergence
  failures, iteration count by method/problem, final objective gap. Units are
  set per panel (seconds, req/s, %, short); empty windows show the panel's
  "No data yet" note rather than a blank chart.
- **`docker-compose.yml`** — `prometheus` (pinned `prom/prometheus:v3.14.0`)
  on host `${PROMETHEUS_PORT:-9092}`, `grafana` (pinned
  `grafana/grafana-oss:13.0.2`) on host `${GRAFANA_PORT:-3002}`, plus the
  unchanged `api` (host `${PORT:-8000}`) and offline `smoke` services. The
  `prometheus` service waits for the api healthcheck.
- **Volumes (intentional persistence only).** `grafana-data` (named volume)
  keeps Grafana users/orgs/settings across `compose down`; dashboards are NOT
  kept there — they are provisioned from the committed JSON. Prometheus TSDB
  data is deliberately EPHEMERAL (no volume): the api counters it scrapes are
  in-process and reset on api restart anyway. `docker compose down` preserves
  `grafana-data`; `docker compose down -v` removes it.
- **Login:** Grafana OSS defaults `admin` / `admin` (set explicitly in
  compose; no secrets exist in this app). Anonymous access is disabled.
- **Honest scope:** counters are in-process in the api container and reset
  when it restarts; Prometheus history is capped at 1 day; nothing is exposed
  beyond localhost published ports. This is a demo stack, not production
  monitoring.

Environment variables: the app has no secrets and needs no keys. Three host
ports are configurable: `PORT` (api, default 8000), `PROMETHEUS_PORT`
(default 9092), `GRAFANA_PORT` (default 3002). The api container always
listens on 8000 internally. Default compose path is fully offline: the solve
endpoints make no network calls.

### Phase 7 validation evidence (executed 2026-09-08, Git Bash on Windows)

Executed against the running stack (api on host port 8010 because a parallel
process occupied 8000 — the env override exists for exactly this): `make
test` (60 passed, 2 warnings; ruff check passed) and `make lint` (ruff check
+ format check passed) stayed green after the Phase 6 changes (only additive
docstring/compose changes to backend code). Runtime evidence, all commands
and key outputs:

- `curl http://localhost:8010/health` → `200 {"status":"ok"}`.
- `curl http://localhost:9092/api/v1/targets` → target `convex-optimization-api`
  at `http://api:8000/metrics/prometheus`, `health: "up"`.
- `curl 'http://localhost:9092/api/v1/query?query=sum(convex_optimization_solves_total)'`
  → `"value": [.., "112"]` (112 HTTP 200 solves from generated traffic);
  `sum(rate(convex_optimization_solves_total[5m]))` ≈ `0.088` req/s at capture.
- `curl -u admin:admin http://localhost:3002/api/datasources/uid/convex-prom/health`
  → `200 {"status":"OK", "message":"Successfully queried the Prometheus API."}`.
- `curl -u admin:admin http://localhost:3002/api/dashboards/uid/convex-optimization-app`
  → `200`, 14 panels, folder "Convex Optimization".
- Every dashboard panel query was validated against the live Prometheus API
  (with `$__rate_interval` substituted by `5m`): all parse and return data
  after traffic; "Convergence failures" is intentionally empty (zero failures
  occurred — 100% convergence) and shows its no-data note.

Human screenshot pointers: Grafana dashboard at
`http://localhost:3002/d/convex-optimization-app` (login `admin`/`admin`),
Prometheus at `http://localhost:9092`, API on `${PORT:-8000}` (currently 8010).

### Phase A: public readiness (A1 backend hardening)

The service is being deployed publicly (fly.io, behind a purchased domain,
with a React workbench UI). Phase A1 is the backend half of that work. Scope
is HONESTLY DEMO-GRADE: single instance, IN-PROCESS rate limiting (resets on
restart, no shared state), NO auth, NO TLS termination at the app (fly's
proxy terminates TLS), no persistence. This is abuse-mitigation for a public
portfolio demo, not production hardening.

#### New/changed endpoints

- **POST /solve** (extended, backwards compatible): the body now optionally
  carries bounded `params` to solve a USER-SPECIFIED seeded instance of the
  same three registry problems. Without `params` the frozen Stage 1
  benchmark instance is solved, byte-identical to before. Solvers are never
  retuned: same `1/L` steps, `tol=1e-10`, `max_iter=2000`, zero start.
  - `params` fields (all optional; extra fields rejected): `seed`
    (0..2^31-1), `n_rows` (4..200), `n_vars` (2..200), `lam` (lasso only,
    1e-6..100), `ridge` (logistic only, 1e-6..100), `condition`
    (least_squares only, 1..1e6).
  - Deterministic auto-fill: `n_rows` defaults to
    `max(kind_default, n_vars + 10)` when a large `n_vars` is requested
    without rows (keeps `n_rows > n_vars` for least_squares/lasso).
  - Hard caps: dimensions <= 200; an out-of-cap or inconsistent request is a
    422 (`validation_error`). Iteration cap is the fixed `max_iter=2000`.
  - Same problem+params+seed+method is BIT-REPRODUCIBLE (identical response).
  - The 200 response gains `parameters` (the resolved instance parameters,
    or null on the frozen path). Larger ill-conditioned instances may
    legitimately return 200 with `converged=false` (the documented
    convergence-failure signal) -- the iteration cap is NOT raised.
- **POST /parse** (new): body `{"text": "<natural-language problem
  description>"}` (1..2000 chars). Response: `problem_request` (a complete,
  directly POSTable /solve body: `problem`, `method` (suggested fastest
  converging pair), `params`, `tail`), `parse_method` (`"llm"` or
  `"stub"`), `verified` (boolean), `mismatches` (list of human-readable
  reasons). A 200 with `verified=false` means "parsed, but the text does
  not fully support it" -- the UI should surface that, never silently
  accept. Errors: 422 `parse_invalid` (garbage / unparseable text), 502
  `provider_unavailable` (configured upstream LLM failed/timed out), 503
  `provider_not_configured` (no `LLM_API_KEY` set -- the UI hides the
  feature). The parse is verified MECHANICALLY against the original text
  (problem-type keywords, dimension numbers, seed integer, weight numbers;
  conflicting numbers in the text are NOT silently resolved) -- the
  product's credibility feature. The text is never logged or persisted.
  The production provider is any OpenAI-compatible /chat/completions
  endpoint via stdlib urllib (no new dependencies); `StubProvider` (select
  with `LLM_PROVIDER=stub`) is a documented TEST DOUBLE only, never the
  production parser.
- **GET /health**, **GET /metrics**, **GET /metrics/prometheus**: unchanged
  JSON contract keys; the Prometheus exposition adds ONE family,
  `convex_optimization_parse_outcomes_total{outcome=...}` with the bounded
  outcomes `parsed|verified|failed|provider_unavailable|provider_not_configured`,
  and `errors_total` gains the new bounded error classes below.

#### Error classes (bounded taxonomy, JSON logs + `errors_total`)

`validation_error`, `inapplicable_pair`, `server_error` (existing) plus
`body_too_large` (413), `rate_limited` (429), `provider_not_configured`
(503), `provider_unavailable` (502), `parse_invalid` (422).

#### Request hardening

- Per-IP rate limiting: fixed 60 s window, default 30 req/min, env
  `RATE_LIMIT_PER_MIN` (<= 0 or unparsable disables). Exceeded -> 429 with
  `Retry-After` (seconds) and the `rate_limited` error class. Client key:
  the fly proxy's `Fly-Client-IP`, else the direct socket address.
  Loopback (127.0.0.1, ::1) and the ASGI test client are EXEMPT so local
  health checks, the compose Prometheus scraper, and the test suite are
  never throttled. In-process only: windows reset on restart, keys are
  capped in memory, and there is no cross-instance coordination.
- Body-size guard: request bodies above 64 KiB (`MAX_BODY_BYTES`) are
  rejected 413 (`body_too_large`) via the Content-Length header, with a
  capped streamed read for chunked bodies.
- Strict pydantic validation: bounded enums and integer/float ranges on
  every request model; unknown fields are rejected (`extra="forbid"`) on
  /solve and /parse bodies -- there is still no tol/max_iter tuning surface.
- CORS: `ALLOWED_ORIGINS` (comma-separated origins, e.g.
  `https://workbench.example.com`). Default EMPTY = same-origin only: no
  CORS middleware is installed and no `Access-Control-Allow-*` headers are
  emitted, so browsers deny every cross-origin request. A wildcard is never
  accepted from the environment. Allowed methods: GET, POST.

#### New environment variables (all optional)

| var | default | meaning |
| --- | --- | --- |
| `ALLOWED_ORIGINS` | empty | comma-separated CORS origins; empty = same-origin only |
| `RATE_LIMIT_PER_MIN` | `30` | per-IP fixed-window request limit; <= 0 disables |
| `LLM_API_KEY` | unset | activates the LLM-backed /parse provider (never logged) |
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` | OpenAI-compatible base URL |
| `LLM_MODEL` | `openai/gpt-4o-mini` | model id for the parse provider |
| `LLM_PROVIDER` | unset | `stub` selects the test-double parser (tests only) |

#### Honest scope of Phase A1

Single-instance, in-process rate limiting (no Redis, no shared state, no
auth, no per-user quotas); the verifier is heuristic-mechanical (it checks
what the text literally states, not semantic correctness); /parse runs
SciPy ground-truth work only via /solve, never in /parse itself; the
parameterized instances are capped at 200 dimensions to bound CPU per
request. Nothing here claims production readiness.

### Workbench UI (Phase A2 — local dev server only)

`ui/` holds a React + Vite + TypeScript workbench (no marketing hero; a
technical instrument panel). It is a DEV-SERVER artifact, not a deployment:
`make ui` runs the Vite dev server, which proxies `/health`, `/metrics`,
`/metrics/prometheus`, `/solve` and `/parse` to the FastAPI backend
(`API_PORT` overrides 8000; no CORS setup needed at default same-origin).
`make ui-build` type-checks and builds a static bundle; `make ui-test` runs
the offline vitest suite (mocked fetch; 11 tests).

A plain-language guide (`ExplainerPanel`, full width below the workbench) explains
convex optimization for non-specialist readers: the bowl-shaped landscape idea, the
three problem types with real-world analogies (trend fitting, sensor selection,
yes/no questions), the four methods in everyday words, and a worked rent-prediction
example with concrete parameters (lasso + ista, seed 42, 80x50, lam 0.5) the reader
can reproduce on the page.

Panels: problem/method selection with client-side validation mirroring the
server caps (out-of-cap input blocks Solve with inline errors; `lam`/`ridge`/
`condition` are disabled outside their problem), seed with a randomize button,
solve status with a 429 Retry-After countdown, a convergence chart of the
returned history tail (objective gap on a log scale, with an honest linear
fallback when the gap is not plottable), result readout (iterations, final
objective, ground truth and source, gap, residual, client round-trip), a
per-method explanation panel written from the Stage 1 docstrings, and an NL
prompt panel. Without parameters the UI states it is solving the FROZEN Stage 1
benchmark instance; with `parameters` in the response it lists the resolved
instance values.

Honest-state handling, by design: a 200 with `converged=false` is rendered as
a distinct "hit the 2000-iteration cap (not converged)" badge (the documented
real outcome, NOT an error); a 422 inapplicable pair shows the server's
applicable-methods list; `verified=false` parses keep HTTP 200 but surface the
mismatch list prominently before "Run this spec"; and on 503
`provider_not_configured` the NL panel hides itself behind the one-line
explanation that no `LLM_API_KEY` is configured. Parsing uses an LLM plus the
mechanical verifier; the solver never does. This UI makes no
production-readiness claim and is not part of `make test` (backend tests and
ruff stay Python-only).

### Deployment runbook (fly.io + convexoptimizer.stream, Phase B)

Live at **https://convexoptimizer.stream** (also `www.`). One fly.io app,
`convex-optimizer`, runs the FastAPI API AND the built workbench UI from the
same process, same origin (Option A): the Dockerfile `COPY`s `ui/dist` and sets
`SERVE_UI=1`, `app.py` mounts it last so API routes keep precedence. TLS
terminates at fly's proxy (Let's Encrypt; the app itself never sees TLS).

Deploy from the repo root (builds the UI bundle first; Docker Desktop must be
running):

    make deploy          # npm build of ui/ + fly deploy

Secrets and environment (set via fly, never in the repo/image):

    fly secrets import -a convex-optimizer < .env   # LLM_API_KEY (OpenRouter)
    # RATE_LIMIT_PER_MIN=30 set in fly.toml [env]; optional knobs per README table above

Domain + DNS (Cloudflare, all records DNS-ONLY -- proxied/orange-cloud records
hide fly's IPs and break certificate issuance):

    A     @     -> 66.241.124.7
    AAAA  @     -> 2a09:8280:1::188:4b16:0
    A     www   -> 66.241.124.7
    AAAA  www   -> 2a09:8280:1::188:4b16:0
    CNAME _acme-challenge     -> convexoptimizer.stream.02w6zrr.flydns.net
    CNAME _acme-challenge.www -> www.convexoptimizer.stream.02w6zrr.flydns.net

Certificates: `fly certs add convexoptimizer.stream` and
`fly certs add www.convexoptimizer.stream`, then `fly certs check <host>`.
Issued 2026-09-09 (Let's Encrypt, rsa+ecdsa, ~2 month expiry, auto-renewed).

Acceptance evidence (2026-09-09, live): `/health` 200 on both hosts;
frozen `/solve` byte-reproducible across repeat calls; parameterized solve
(lasso, seed 7, n_vars 30) echoes resolved parameters; `n_vars: 500` -> 422;
70 KiB body -> 413 (`declared_bytes: 70045`); 40-request burst -> 11x 429 with
`Retry-After: 6`; `/parse` (LLM provider) returns `verified: true` with zero
mismatches on a fully-specified NL problem; UI root 200 text/html, unknown
paths 404 (no SPA fallback), `/docs` 200; Prometheus exposition carries the
full family set.

Honest scope at deploy: two shared-cpu-1x machines (fly HA default) with
auto-stop suspend when idle; in-process rate limiting per machine (windows are
per-instance, so the effective public burst ceiling is roughly 2x the
configured 30/min); no auth, no persistence, demo grade.

### Deployment runbook (local Docker Compose)

From a clean clone (Windows Git Bash; for POSIX shells the same commands work
with forward slashes as written):

```bash
git clone <repo-url>
cd convex_optimization_app
# optional pre-check (uses the venv, offline): make setup && make test
docker compose build                # or: make docker-build
docker compose up -d                # api + prometheus + grafana (Phase 6 stack)
curl --noproxy '*' http://localhost:${PORT:-8010}/health   # PORT unset -> use 8000
curl --noproxy '*' -X POST http://localhost:8010/solve   -H 'Content-Type: application/json'   -d '{"problem":"logistic","method":"nesterov"}'
curl --noproxy '*' http://localhost:8010/metrics
# observability endpoints (Phase 6):
curl --noproxy '*' http://localhost:9092/api/v1/targets         # scrape target health
curl --noproxy '*' 'http://localhost:9092/api/v1/query?query=sum(convex_optimization_solves_total)'
curl --noproxy '*' -u admin:admin http://localhost:3002/api/health
# dashboard: http://localhost:3002/d/convex-optimization-app  (login admin/admin)
docker compose down                # stops containers; keeps the grafana-data volume
```

Phase 6 verification loop (already executed once during this phase; the
traffic pattern below is exactly what was replayed against the running
stack — health checks, all six applicable (problem, method) pairs, and
inapplicable/invalid 422s for the error panels):

```bash
# generate representative traffic (repeat for a few minutes):
curl --noproxy '*' http://localhost:8010/health
curl --noproxy '*' -X POST http://localhost:8010/solve -H 'Content-Type: application/json' -d '{"problem":"least_squares","method":"nesterov"}'
curl --noproxy '*' -X POST http://localhost:8010/solve -H 'Content-Type: application/json' -d '{"problem":"lasso","method":"fista"}'
curl --noproxy '*' -X POST http://localhost:8010/solve -H 'Content-Type: application/json' -d '{"problem":"lasso","method":"nesterov"}'   # 422 inapplicable_pair
# then check population:
curl --noproxy '*' 'http://localhost:9092/api/v1/query?query=sum(convex_optimization_solves_total)'
```

Windows Git Bash notes:

- Quote the JSON body with single quotes; if your shell mangles them, write
  the body to a file and use `curl -d @body.json`.
- Use `--noproxy '*'` (or unset `HTTP_PROXY`/`HTTPS_PROXY`) if a corporate
  proxy intercepts localhost requests.
- Git Bash on Windows accepts forward slashes in paths and URLs; do not
  backslash-escape inside single-quoted strings.
- `docker compose run --rm smoke` (network_mode none) still proves the CLI
  solve path is fully offline; the `api` service uses normal networking
  because a published port is incompatible with `network_mode: none`.

### Version identity and tag convention

- Numerical stack: numpy 2.5.3 / scipy 1.18.1, from `requirements-lock.txt`
  (the only lock; unchanged since Stage 2). The versions used for the recorded
  results are also stamped into `experiments/run_log.json`.
- Problem/method matrix identity: the applicable-pairs matrix in
  `src/convex_optimization/cli.py` (`APPLICABLE`: least_squares and logistic
  with gd/nesterov; lasso with fista/ista) at the Stage 1 settings
  `tol=1e-10`, `max_iter=2000`, recorded in each bundle's `settings`.
- Git tag convention: `stageN-vX.Y.Z` (for example `stage3-v0.2.0`), placed on
  the commit whose code produced the recorded results. Tags are local release
  markers; CI does not push tags or images to any registry.

### Experiment artifact bundle (save / verify)

`make eval` is the save step: it writes `experiments/run_log.csv` plus
`experiments/run_log.json`, the bundle manifest. The manifest carries the
identity needed to regenerate every result: per row the problem, seed, method,
`n_iter`, final gap, residual and converged flag; plus the settings
(`tol=1e-10`, `max_iter=2000`), numpy/scipy versions, and the git commit.

Verify the bundle offline (`make verify`, or
`PYTHONPATH=src .venv/Scripts/python.exe experiments/run_benchmark.py --verify experiments/run_log.json`):
it rebuilds every logged cell deterministically and compares `n_iter` exactly
and the final gap within the Stage 1 criterion (`|diff| <= 1e-8`). Wall time is
NOT verified (noisy on this host, indicative only); numpy/scipy version
mismatches are reported as warnings because determinism across versions is not
guaranteed — the recomputation itself decides pass/fail. The shipped Stage 2
bundle verifies 18/18 cells on the host and inside the container.

### Stage 6 maintain: versioned refresh, rollback pointer, incident runbook

The committed `experiments/run_log.json` (+ `.csv`) is the FROZEN Stage 2 (v1)
baseline. Stage 6 maintenance is CLI-side and on-demand (offline; NO scheduler,
no cron, no production MLOps) and does NOT retune anything: the Stage 1-2
constants (1/L steps, Nesterov momentum, `tol=1e-10`, `max_iter=2000`) and the
API (still no tol/max_iter/seed knobs) are unchanged, and no Stage 2 finding is
revised.

`make eval` decision, documented honestly: `make eval` still writes the DEFAULT
paths `experiments/run_log.csv` + `experiments/run_log.json`, i.e. it overwrites
the working-tree copies of the frozen bundle. They are git-tracked, so the
committed v1 stays recoverable with
`git checkout -- experiments/run_log.json experiments/run_log.csv`. The Stage 6
refresh instead writes a VERSIONED bundle and never touches those files:

```bash
make refresh    # Stage 2 suite -> experiments/runs/<UTC-ts>/run_log.{csv,json}
```

`make refresh` (via `experiments/maintain.py`):

1. re-runs the SAME Stage 2 suite (same cells, same settings; `n_iter` is
   deterministic);
2. verifies the fresh bundle with the EXISTING Stage 3 verifier (recompute
   every cell: `n_iter` exact, gap within 1e-8; wall time excluded);
3. cross-compares it against the frozen v1 baseline (`n_iter` exact, converged
   equal, gap within 1e-8; wall time deliberately EXCLUDED -- it is noisy on
   this host and a wall-time difference is NOT a regression or an improvement);
4. only if BOTH pass, atomically moves the `current` pointer
   (`experiments/current.json`) to the new bundle. ANY failure leaves the
   pointer untouched.

Rollback and status:

```bash
make current    # show the pointer target + the pointed-to bundle's identity
make rollback   # point `current` back at the frozen Stage 2 (v1) baseline
```

The pointer stores a path relative to `experiments/` (`run_log.json` for v1;
`runs/<ts>/run_log.json` after a refresh). Identity (git commit, numpy/scipy
versions, settings) lives in the bundle the pointer targets, so after a
rollback `make current` reports the v1 identity again.

Serving does NOT read the pointer: the Stage 4 API (`app.py`) always runs the
Stage 1 methods in-process via `cli.solve`, exactly as before. Refresh and
rollback are therefore CLI-side only and require NO image rebuild (the minimal
honest option; if serving ever starts reading the pointer, the image must be
rebuilt and compose re-verified with real curls first).

#### Incident runbook (offline commands; local compose only)

1. **Solver non-convergence (`converged=false` / `max_iter` hit).** Signal: a
   200 response with `"converged": false`, `solve.convergence_failures` /
   `solve.convergence_failure_rate` rising in `/metrics`, and a log line with
   `"converged": false` and `iterations` pinned at 2000 (the Stage 5
   convergence-failure definition, unchanged).

   ```bash
   curl --noproxy '*' -X POST http://localhost:8000/solve \
     -H 'Content-Type: application/json' -d '{"problem":"logistic","method":"nesterov"}'
   curl --noproxy '*' http://localhost:8000/metrics | grep convergence
   docker compose logs api | grep '"converged": false'
   ```

   Response: do NOT retune (the Stage 1-2 constants are fixed by design; the
   API has no tol/max_iter knobs). All 18 cells of the shipped suite converge
   (18/18 in the v1 log), so a failure points at a NEW problem or pair, not a
   solver change. Check the benchmark still reproduces with `make verify`
   (18/18 expected); retuning would be a deliberate, separately documented
   decision -- never an ops action.

2. **Numerical instability (NaN/Inf).** HYPOTHETICAL for the shipped problems:
   the seeded problems are small and well-conditioned and all 18 v1 cells
   converge with finite gaps, and the API accepts only those fixed problems, so
   no request can inject diverging data. The exact check that would catch it:
   a non-finite final objective or residual (`not np.isfinite(...)` on the
   solver result before a response is built) would raise and surface as HTTP
   500 -> `status_counts["5xx"]` and `errors.rate_5xx` in `/metrics`, with the
   log line `"error_class": "server_error"` (the same overflow path documented
   under Stage 5). What a reviewer would run:

   ```bash
   curl --noproxy '*' http://localhost:8000/metrics | grep -o '"rate_5xx":[^,]*'
   docker compose logs api | grep error_class
   ```

3. **API failures (inapplicable pair 422, health down).**

   ```bash
   # inapplicable pair -> 422 (counts as an ERROR, never a convergence failure)
   curl --noproxy '*' -X POST http://localhost:8000/solve \
     -H 'Content-Type: application/json' -d '{"problem":"lasso","method":"nesterov"}'
   # health down -> check and restart locally
   curl --noproxy '*' http://localhost:8000/health || docker compose ps
   docker compose up -d api     # or: make api
   ```

4. **Benchmark/results incident (suspected drift or regression).** Run
   `make refresh`. If verification or the v1-equivalence comparison FAILS, the
   pointer does NOT move: the v1 baseline stays current, and the service was
   never affected (it does not read the pointer). Investigate the mismatch
   before any refresh. `experiments/incident.md` records one fully executed
   walkthrough (refresh -> 18/18 verify -> rollback -> identity restored).

### Docker (local only, no deployment)

```bash
docker build -t convex-optimization:local .          # or: make docker-build
docker run --rm --network none convex-optimization:local   # offline smoke: one CLI solve
docker compose run --rm smoke                        # same smoke via compose (network_mode: none)
docker compose down
```

The image installs only from `requirements-lock.txt` (~525 MB) and defaults to
one offline solve (logistic + nesterov). The Python 3.11 install failure is
reproducible with `docker build --build-arg PYTHON_TAG=3.11`.

Other targets: `make lint` (ruff check + format check), `make format`,
`make verify` (bundle check above), `make refresh` / `make rollback` /
`make current` (Stage 6 maintain, above), `make docker-build`, `make smoke`,
`make api` (Stage 4 serving via compose, host port `${PORT:-8000}`), `make clean` (removes caches and `.venv`). CI runs lint + tests on every push
and pull request, then builds the image and runs the offline smoke solve in it
as build proof (no registry push). Do not commit secrets, API keys, or large
artifacts.

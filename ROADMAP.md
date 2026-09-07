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

- [ ] Structured logging of solve requests, iterations, and latency.
- [ ] Metrics endpoint exposing: request count, latency percentiles, error rate, convergence-failure rate.
- [ ] Optional: Prometheus/Grafana dashboard.
- [ ] Document "what could degrade" (ill-conditioned inputs, numerical instability, solver failures).

**Acceptance:** A reviewer can see how the service is observed and what signals would indicate a problem.

## Stage 6 - Maintain

- [ ] Refresh path: how to re-run the benchmark suite and regenerate results.
- [ ] Rollback path: revert to a previous code/artifact version.
- [ ] Runbook for common incidents (non-convergence, numerical instability, API failures).
- [ ] One documented incident write-up (real or realistic) showing the maintain loop.

**Acceptance:** The maintain loop is documented and executable, not just described.

## Portfolio presentation

- [ ] README tells the full lifecycle story with real benchmark numbers, emphasizing rigor.
- [ ] Link the repo from `pcschmidt.github.io`.
- [ ] Prepare a 3-sentence interview arc per lifecycle stage.

## Definition of done

All stages complete, tests green, benchmarks reproducible, deployment reproducible, monitoring documented, and the README honestly reflects what is implemented.
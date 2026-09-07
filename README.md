# Convex Optimization App

## Motivation

This app is the optimization + rigor pillar of a personal portfolio. Its goal
is to demonstrate correct, well-documented convex optimization algorithms and
reproducible benchmarking against SciPy solvers. The emphasis is on honest,
reviewable engineering: fixed seeds, pinned dependencies, recorded run logs,
and convergence behavior shown rather than claimed.

## Method

Planned (NOT yet implemented — this is a Stage 0 scaffold):

- Implement classic first-order methods from scratch: gradient descent,
  accelerated variants (e.g., Nesterov), and at least one proximal/ADMM
  method.
- Define standard convex benchmark problems (least squares, logistic
  regression, L1-regularized) and compare each method against SciPy solvers
  as ground truth.
- Track objective value and gradient norm per iteration; record iterations to
  convergence, wall time, and final objective gap in an `experiments/` run log.

Currently implemented: a package skeleton (`convex_optimization`), a smoke
test suite, pinned lockfile, linting, CI, and a reproducible setup path.

## Results

None yet. No benchmarks have been run. No optimization claims should be read
into this repository until the benchmark harness exists and results are
recorded here with seeds and pinned versions.

## Limitations

- No optimization algorithms are implemented yet; the package exposes only
  its version.
- No benchmark harness, no experiment log, no SciPy ground-truth comparison.
- No serving layer, containerization, or monitoring yet (planned for later
  stages of the roadmap).
- Legacy reference code in `legacy/` is read-only and is not part of the
  installed package.

## Operational notes

Requirements: Python 3.11+, `git`, GNU `make`.

```bash
git clone <repo-url>
cd convex_optimization_app
make setup   # creates .venv, installs pinned dependencies from requirements-lock.txt
make test    # runs pytest (smoke tests) and ruff check
```

Other targets: `make lint` (ruff check + format check), `make format`,
`make clean` (removes caches and `.venv`). CI runs lint + tests on every push
and pull request via GitHub Actions. Do not commit secrets, API keys, or
large artifacts.

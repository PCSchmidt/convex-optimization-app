# Convex Optimization App

## Motivation

This app is the optimization + rigor pillar of a personal portfolio. Its goal
is to demonstrate correct, well-documented convex optimization algorithms and
reproducible benchmarking against SciPy solvers. The emphasis is on honest,
reviewable engineering: fixed seeds, pinned dependencies, recorded run logs,
and convergence behavior shown rather than claimed.

The legacy app (`legacy/`, read-only) delegated all solving to PuLP/CVXPY
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

No benchmark tables yet. Stage 1 establishes correctness, not comparisons:
`make test` verifies that every method reaches its stated tolerance on every
applicable problem. Per-run numbers (iterations, final objective gap,
residual) can be produced with `make solve ARGS="..."` (see Operational
notes). Iterations-to-convergence comparisons, wall-clock timing, multi-seed
stability, and an `experiments/` run log are Stage 2 work and are
deliberately not claimed here.

## Limitations

- First-order methods only (gradient descent, Nesterov accelerated gradient,
  FISTA). No second-order, ADMM, or constrained/SDP solvers.
- Small synthetic problems only (10-40 dimensions, seeded); no real datasets.
- Ground truth for the Lasso is an eps-smoothed SciPy reference, not an exact
  nonsmooth optimum; tolerances account for this, but it is an approximation.
- Armijo backtracking gradient descent stalls near the optimum in float64
  (documented above); the benchmarks use the fixed `1/L` step instead.
- No Stage 2 evaluation yet: no wall-clock comparisons, no multi-seed
  stability, no `experiments/` run log, no method-vs-method ranking claims.
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
make test    # runs pytest (smoke + Stage 1 correctness) and ruff check
```

Run one offline solve (one problem, one method; prints the final objective
gap vs ground truth and the last few history rows):

```bash
make solve ARGS="--problem lasso --method fista"     # lasso | least_squares | logistic
make solve ARGS="--problem logistic --method nesterov"
```

Other targets: `make lint` (ruff check + format check), `make format`,
`make clean` (removes caches and `.venv`). CI runs lint + tests on every push
and pull request via GitHub Actions. Do not commit secrets, API keys, or
large artifacts.

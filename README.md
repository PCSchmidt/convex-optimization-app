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
- No serving layer or monitoring yet (planned for Stage 4-5). Containerization is local-only
  (Stage 3 Dockerfile/compose for an offline smoke solve); no public deployment.
- Legacy reference code in `legacy/` is read-only and is not part of the
  installed package.

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
`make verify` (bundle check above), `make docker-build`, `make smoke`,
`make clean` (removes caches and `.venv`). CI runs lint + tests on every push
and pull request, then builds the image and runs the offline smoke solve in it
as build proof (no registry push). Do not commit secrets, API keys, or large
artifacts.

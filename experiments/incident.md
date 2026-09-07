# Incident write-up: Stage 6 maintain loop, fully executed (2026-09-07)

This is NOT a fake outage blog. It is a verbatim record of one real, executed
walkthrough of the Stage 6 maintain loop on this machine (Windows, Git Bash,
numpy 2.5.3 / scipy 1.18.1, Python 3.12 venv):

    refresh (writes a VERSIONED v2 bundle) -> verify v2 (18/18 n_iter match vs
    the recomputed cells AND vs the frozen v1 baseline; wall time excluded) ->
    rollback the `current` pointer to v1 -> pointer identity restored.

Scenario (realistic, not dramatized): during a periodic results audit we want
to confirm the committed Stage 2 benchmark still reproduces byte-for-byte on
`n_iter`, publish a fresh versioned bundle as evidence, and then return the
`current` pointer to the frozen v1 baseline so the repository state is
unchanged for readers. No solver code was changed; the Stage 1-2 settings
(tol=1e-10, max_iter=2000, 1/L steps) are untouched, and nothing was retuned.

## Step 1 - refresh: re-run the Stage 2 suite into a versioned bundle

Command (offline, on-demand; no scheduler):

```bash
make refresh
```

Actual output (verbatim, paths abbreviated to `...` for width):

```
PYTHONPATH=src .venv/Scripts/python.exe experiments/maintain.py refresh
refresh: running 18 cells (Stage 2 suite, settings unchanged)
refresh: wrote .../experiments/runs/20260907T214140Z/run_log.csv and .../experiments/runs/20260907T214140Z/run_log.json
verify: .../experiments/runs/20260907T214140Z/run_log.json
  recorded identity: date_utc=2026-09-07T21:41:40Z git_commit=f326d5b
  numpy: recorded=2.5.3 current=2.5.3 -> match
  scipy: recorded=1.18.1 current=1.18.1 -> match
  PASS  least_squares seed=0       gd n_iter=1445
  PASS  least_squares seed=1       gd n_iter=1475
  PASS  least_squares seed=2       gd n_iter=668
  PASS  least_squares seed=0 nesterov n_iter=192
  PASS  least_squares seed=1 nesterov n_iter=200
  PASS  least_squares seed=2 nesterov n_iter=129
  PASS          lasso seed=0    fista n_iter=1436
  PASS          lasso seed=1    fista n_iter=950
  PASS          lasso seed=2    fista n_iter=775
  PASS          lasso seed=0     ista n_iter=1451
  PASS          lasso seed=1    ista n_iter=1187
  PASS          lasso seed=2     ista n_iter=662
  PASS       logistic seed=1       gd n_iter=62
  PASS       logistic seed=2       gd n_iter=71
  PASS       logistic seed=3       gd n_iter=62
  PASS       logistic seed=1 nesterov n_iter=33
  PASS       logistic seed=2 nesterov n_iter=35
  PASS       logistic seed=3 nesterov n_iter=35
18/18 cells verified (n_iter exact, |gap diff| <= 1e-08); wall time not verified (noisy on this host, indicative only)
refresh: verification and equivalence PASSED; current -> runs/20260907T214140Z/run_log.json
```

Both gates passed: the Stage 3 verifier rebuilt every cell of the fresh bundle
(18/18, `n_iter` exact, gap within 1e-8), and the cross-bundle comparison
against the frozen v1 baseline found zero mismatches. Only then did the
pointer move.

Wall time is deliberately NOT part of any gate. It DID differ between v1 and
v2 (e.g. least_squares seed 0 gd: 0.014563 s in v1 -> 0.013595 s in v2). That
is host noise, not a regression or an improvement; `n_iter` is identical for
all 18 cells and is the only comparison signal.

## Step 2 - current: confirm the pointer now targets v2

```bash
make current
```

```
PYTHONPATH=src .venv/Scripts/python.exe experiments/maintain.py current
current: runs/20260907T214140Z/run_log.json
  date_utc=2026-09-07T21:41:40Z git_commit=f326d5b
  numpy=2.5.3 scipy=1.18.1 rows=18
  settings={"tol": 1e-10, "max_iter": 2000, "wall_time_reps": 3, "step_sizes": "Stage 1 as-is: 1/L everywhere; Nesterov momentum (sqrt(kappa)-1)/(sqrt(kappa)+1); no tuning"}
```

## Step 3 - rollback: point `current` back at the frozen v1 baseline

```bash
make rollback
```

```
PYTHONPATH=src .venv/Scripts/python.exe experiments/maintain.py rollback
rollback: current -> run_log.json (frozen Stage 2 baseline)
```

## Step 4 - current: pointer identity restored to v1

```bash
make current
```

```
PYTHONPATH=src .venv/Scripts/python.exe experiments/maintain.py current
current: run_log.json
  date_utc=2026-09-07T17:05:17Z git_commit=fe14e0d+dirty
  numpy=2.5.3 scipy=1.18.1 rows=18
  settings={"tol": 1e-10, "max_iter": 2000, "wall_time_reps": 3, "step_sizes": "Stage 1 as-is: 1/L everywhere; Nesterov momentum (sqrt(kappa)-1)/(sqrt(kappa)+1); no tuning"}
```

The pointer targets the committed Stage 2 bundle again; the identity reported
(v1's `date_utc`, its `git_commit=fe14e0d+dirty`, numpy/scipy versions,
settings) is exactly the pre-refresh identity. The frozen v1 files were never
written: `git status` before the closing commit shows only the NEW artifacts
(`experiments/current.json`, `experiments/runs/20260907T214140Z/`) plus this
file, and `git diff experiments/run_log.json` is empty.

## Failure behavior (why the pointer is safe)

A FAILED verification does NOT move the pointer, by construction
(`experiments/maintain.py`, tested in `tests/test_stage6.py::test_failed_refresh_does_not_move_pointer`):
if the fresh bundle fails the Stage 3 verifier or the v1-equivalence
comparison, `make refresh` exits 1 and prints `pointer NOT moved`. The Stage 4
service never reads the pointer (it solves in-process via `cli.solve`), so no
incumbent behavior can change without an explicit, verified refresh -- and no
image rebuild is needed for any of this.

## Outcome

- v2 bundle committed as evidence: `experiments/runs/20260907T214140Z/`.
- `current` pointer committed at the v1 baseline: `experiments/current.json`.
- No retuning, no wall-time claims, no Stage 2 finding revised.

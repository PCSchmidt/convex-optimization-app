"""Stage 2 benchmark runner: measure the Stage 1 methods AS-IS. No tuning.

Runs every applicable (problem, method) pair across >= 3 problem seeds at the
Stage 1 settings exactly as shipped (fixed 1/L steps, Nesterov momentum from
the known L/mu, tol=1e-10, max_iter=2000 via ``cli.solve`` defaults) and
records one row per run:

    date_utc, git_commit, numpy/scipy versions, problem, seed, method,
    n_iter, wall_time_s, final_gap, final_residual, converged, notes

Inapplicable pairs are NOT run and NOT faked: least_squares and logistic are
smooth, so only gd and nesterov apply; the lasso is nonsmooth, so only the
proximal methods fista and ista apply.

Honesty rules baked in:
- The lasso ground truth is the eps=1e-10 smoothed SciPy L-BFGS-B reference,
  which is approximate (bias < ~1e-9). Gaps vs it are reported against that
  reference, not a true nonsmooth optimum.
- Wall time on Windows is noisy. Each cell is run 3 times and the MEDIAN wall
  time is recorded; treat it as indicative only. The robust comparison signal
  is n_iter, which is fully deterministic.

Usage (from the repo root):
    make eval        # or: PYTHONPATH=src .venv/Scripts/python.exe experiments/run_benchmark.py

Stage 6 maintain: ``make refresh`` runs the SAME suite into a VERSIONED
bundle (experiments/runs/<ts>/), verifies it, and only then moves the
``current`` pointer; ``make rollback`` points back at the frozen Stage 2
baseline. See ``experiments/maintain.py``. ``make eval`` still writes the
default (destructive, git-tracked) experiments/run_log.* paths.

Stage 3 artifact bundle: the written experiments/run_log.json IS the bundle
manifest. It carries the identity needed to regenerate the results: per-row
(problem, seed, method, n_iter, final gap, converged), the settings
(tol=1e-10, max_iter=2000, wall_time_reps), the numpy/scipy versions, and the
git commit. Verify it offline with:

    make verify      # or: PYTHONPATH=src .venv/Scripts/python.exe experiments/run_benchmark.py --verify experiments/run_log.json

The verifier rebuilds every logged cell deterministically and compares n_iter
exactly and the final gap within the Stage 1 criterion (|diff| <= 1e-8). Wall
time is NOT verified (noisy on this host, indicative only). Environment
version/commit identity is reported; a numpy/scipy version mismatch is a
warning (determinism across versions is not guaranteed), not an error.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from convex_optimization import cli
from convex_optimization.problems import make_lasso, make_least_squares, make_logistic

# Seed plan: >= 3 seeds per problem, dimensions identical to Stage 1 (the
# builders' defaults n/d are untouched). Each problem's Stage 1 default seed
# is included so the benchmark contains the Stage 1 configuration as-is.
SEEDS = {
    "least_squares": (0, 1, 2),  # Stage 1 default seed 0
    "lasso": (0, 1, 2),  # Stage 1 default seed 0
    "logistic": (1, 2, 3),  # Stage 1 default seed 1
}
BUILDERS = {
    "least_squares": make_least_squares,
    "lasso": make_lasso,
    "logistic": make_logistic,
}
WALL_TIME_REPS = 3  # median reported; single timings on Windows are noisy
LASSO_TRUTH_NOTE = (
    "ground truth = eps=1e-10 smoothed SciPy L-BFGS-B reference (approximate, bias < ~1e-9)"
)


def git_commit() -> str:
    """Short HEAD sha, suffixed with '+dirty' if the working tree has changes.

    The benchmark is usually run before the results are committed, so the
    recorded sha is the commit the uncommitted code sits on top of; '+dirty'
    makes that explicit instead of implying the sha contains this exact code.
    """
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
        return sha + ("+dirty" if status.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_cell(problem_name: str, seed: int, method: str) -> dict:
    problem = BUILDERS[problem_name](seed=seed)
    n_iters, wall_times = [], []
    result = None
    for rep in range(WALL_TIME_REPS):
        t0 = time.perf_counter()
        _, result = cli.solve(problem_name, method, problem=problem)
        wall_times.append(time.perf_counter() - t0)
        n_iters.append(result.n_iter)
    assert len(set(n_iters)) == 1, f"non-deterministic n_iter for {problem_name}/{method}"
    assert result is not None
    gap = result.final_objective_gap(problem.ground_truth.f)

    notes = [f"wall_time = median of {WALL_TIME_REPS} runs, indicative only (noisy on Windows)"]
    if problem_name == "lasso":
        notes.append(LASSO_TRUTH_NOTE)
        if gap < 0:
            notes.append(
                "final objective slightly BELOW the approximate smoothed reference; "
                "expected because the eps-smoothing biases the reference f* upward"
            )

    return {
        "problem": problem_name,
        "seed": seed,
        "method": method,
        "n_iter": result.n_iter,
        "wall_time_s": f"{float(np.median(wall_times)):.6f}",
        "final_gap": f"{gap:.6e}",
        "final_residual": f"{result.final_residual():.6e}",
        "converged": str(result.converged),
        "notes": "; ".join(notes),
    }


def write_bundle(out_dir: Path, rows: list[dict]) -> tuple[Path, Path]:
    """Write the bundle (run_log.csv + run_log.json) into ``out_dir``.

    The schema is byte-identical to the Stage 2/3 bundle. ``out_dir`` is a
    parameter so the Stage 6 refresh can write a VERSIONED bundle
    (``experiments/runs/<ts>/``) instead of clobbering the frozen Stage 2
    baseline in ``experiments/``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "run_log.csv"
    json_path = out_dir / "run_log.json"
    fieldnames = [
        "date_utc",
        "git_commit",
        "numpy_version",
        "scipy_version",
        "problem",
        "seed",
        "method",
        "n_iter",
        "wall_time_s",
        "final_gap",
        "final_residual",
        "converged",
        "notes",
    ]
    stamp = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    commit = git_commit()
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "date_utc": stamp,
                    "git_commit": commit,
                    "numpy_version": np.__version__,
                    "scipy_version": scipy.__version__,
                    **row,
                }
            )
    meta = {
        "date_utc": stamp,
        "git_commit": commit,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "settings": {
            "tol": 1e-10,
            "max_iter": 2000,
            "wall_time_reps": WALL_TIME_REPS,
            "step_sizes": "Stage 1 as-is: 1/L everywhere; Nesterov momentum (sqrt(kappa)-1)/(sqrt(kappa)+1); no tuning",
        },
        "seeds": {k: list(v) for k, v in SEEDS.items()},
        "lasso_truth": LASSO_TRUTH_NOTE,
        "rows": rows,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return csv_path, json_path


def run_benchmark(out_dir: Path | None = None) -> int:
    """Run the full Stage 2 suite and write the bundle.

    ``out_dir`` defaults to ``experiments/`` (the historical ``make eval``
    behavior, which OVERWRITES experiments/run_log.* in the working tree;
    they are git-tracked, so the committed v1 baseline stays recoverable with
    ``git checkout -- experiments/run_log.json experiments/run_log.csv``).
    The Stage 6 refresh (``experiments/maintain.py``) instead passes a
    versioned directory and never touches the frozen baseline files.
    """
    rows = []
    for problem_name, seeds in SEEDS.items():
        for method in cli.APPLICABLE[problem_name]:
            for seed in seeds:
                row = run_cell(problem_name, seed, method)
                rows.append(row)
                print(
                    f"{row['problem']:>14s} seed={row['seed']} {row['method']:>8s} "
                    f"n_iter={row['n_iter']:>5d} wall={row['wall_time_s']}s "
                    f"gap={row['final_gap']} res={row['final_residual']} "
                    f"conv={row['converged']}"
                )
    target = Path(__file__).resolve().parent if out_dir is None else Path(out_dir)
    csv_path, json_path = write_bundle(target, rows)
    print(f"wrote {csv_path} and {json_path} ({len(rows)} rows)")
    return 0


def verify_log(path: Path, gap_tol: float = 1e-8) -> int:
    """Recompute every cell in a saved run-log bundle and compare.

    n_iter is the deterministic signal and must match EXACTLY. The final gap
    is compared to the recorded value within ``gap_tol`` (the Stage 1
    criterion; recorded gaps are rounded to 6 significant digits). Wall time
    is deliberately not compared. For the lasso rows the gap is still measured
    against the eps-smoothed approximate SciPy reference recorded in the
    bundle, unchanged from Stage 2.
    """
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)

    print(f"verify: {path}")
    print(
        f"  recorded identity: date_utc={manifest.get('date_utc')} git_commit={manifest.get('git_commit')}"
    )
    for lib, current in (("numpy", np.__version__), ("scipy", scipy.__version__)):
        recorded = manifest.get(f"{lib}_version")
        status = (
            "match"
            if recorded == current
            else ("MISMATCH (determinism across versions is not guaranteed; treating as a warning)")
        )
        print(f"  {lib}: recorded={recorded} current={current} -> {status}")

    settings = manifest.get("settings", {})
    tol = float(settings.get("tol", 1e-10))
    max_iter = int(settings.get("max_iter", 2000))
    rows = manifest.get("rows", [])
    failures = 0
    for row in rows:
        problem = BUILDERS[row["problem"]](seed=int(row["seed"]))
        _, result = cli.solve(
            row["problem"], row["method"], max_iter=max_iter, tol=tol, problem=problem
        )
        gap = result.final_objective_gap(problem.ground_truth.f)
        ok_iter = result.n_iter == int(row["n_iter"])
        ok_gap = abs(gap - float(row["final_gap"])) <= gap_tol
        ok_conv = str(result.converged) == str(row["converged"])
        ok = ok_iter and ok_gap and ok_conv
        failures += 0 if ok else 1
        status = "PASS" if ok else "FAIL"
        detail = ""
        if not ok:
            bits = []
            if not ok_iter:
                bits.append(f"n_iter recomputed={result.n_iter} recorded={row['n_iter']}")
            if not ok_gap:
                bits.append(f"gap recomputed={gap:.6e} recorded={row['final_gap']}")
            if not ok_conv:
                bits.append(f"converged recomputed={result.converged} recorded={row['converged']}")
            detail = " | " + "; ".join(bits)
        print(
            f"  {status} {row['problem']:>14s} seed={row['seed']} {row['method']:>8s} "
            f"n_iter={result.n_iter}{detail}"
        )
    print(
        f"{len(rows) - failures}/{len(rows)} cells verified "
        f"(n_iter exact, |gap diff| <= {gap_tol:g}); wall time not verified "
        f"(noisy on this host, indicative only)"
    )
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 2 benchmark runner (default) / Stage 3 bundle verifier (--verify)."
    )
    parser.add_argument(
        "--verify",
        metavar="RUN_LOG_JSON",
        default=None,
        help="verify a saved run-log bundle (e.g. experiments/run_log.json) instead of running the benchmark",
    )
    args = parser.parse_args(argv)
    if args.verify is not None:
        return verify_log(Path(args.verify))
    return run_benchmark()


if __name__ == "__main__":
    raise SystemExit(main())

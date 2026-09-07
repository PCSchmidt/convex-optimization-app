"""Stage 6 maintain: versioned benchmark refresh + 'current' pointer rollback.

The committed ``experiments/run_log.json`` (+ ``.csv``) is the FROZEN Stage 2
(v1) baseline. The Stage 6 refresh NEVER writes those files. Instead:

    refresh   re-runs the SAME Stage 2 benchmark suite (Stage 1 settings
              as-is, deterministic n_iter, wall time still indicative-only)
              into a NEW versioned directory ``experiments/runs/<UTC-ts>/``,
              verifies the fresh bundle with the existing Stage 3 verifier
              (recompute every cell: n_iter exact, gap within the Stage 1
              criterion, wall time excluded), cross-compares it against the
              frozen baseline (n_iter exact, converged equal, gap within
              1e-8; wall time deliberately EXCLUDED -- it is noisy on this
              host and a wall-time difference is NOT a regression), and only
              if BOTH pass atomically moves the ``current`` pointer
              (``experiments/current.json``) to the new bundle. Any failure
              leaves the pointer untouched.
    rollback  points ``current`` back at the frozen v1 baseline
              (``run_log.json``, relative to the pointer's directory). The
              pointer identity is whatever the TARGET bundle records
              (git commit, numpy/scipy versions, settings), so after a
              rollback ``current`` reports the v1 identity again.
    current   prints the pointer target and the pointed-to bundle's identity.

Scope honesty: this is CLI-side maintenance only. The Stage 4 serving layer
(``app.py``) does NOT read the pointer -- it always runs the Stage 1 methods
in-process via ``cli.solve`` -- so refresh/rollback require NO image rebuild.
No scheduler, no cron, no production MLOps: refresh is an on-demand offline
command.

Usage (from the repo root):
    make refresh
    make rollback
    make current
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

import run_benchmark

from convex_optimization import cli

EXPERIMENTS_DIR = Path(__file__).resolve().parent
POINTER_PATH = EXPERIMENTS_DIR / "current.json"
# Pointer targets are stored RELATIVE TO THE POINTER'S DIRECTORY (the
# experiments/ dir in the real layout), so the frozen baseline is "run_log.json".
BASELINE_TARGET = "run_log.json"
GAP_TOL = 1e-8  # the Stage 1 criterion, same as the verifier


def _utc_stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def full_cells() -> list[tuple[str, int, str]]:
    """Every applicable (problem, seed, method) cell -- the Stage 2 suite."""
    return [
        (problem, seed, method)
        for problem, seeds in run_benchmark.SEEDS.items()
        for method in cli.APPLICABLE[problem]
        for seed in seeds
    ]


def compare_bundles(baseline: dict, candidate: dict, gap_tol: float = GAP_TOL) -> list[str]:
    """Cross-bundle equivalence check; returns a list of failure strings.

    n_iter must match EXACTLY (deterministic signal), converged must match,
    and the final gap within ``gap_tol``. Wall time is deliberately EXCLUDED:
    it is noisy on this host and a wall-time difference is not a regression.
    """
    base = {(r["problem"], int(r["seed"]), r["method"]): r for r in baseline.get("rows", [])}
    cand = {(r["problem"], int(r["seed"]), r["method"]): r for r in candidate.get("rows", [])}
    failures: list[str] = []
    for key in sorted(base.keys() - cand.keys()):
        failures.append(f"cell {key}: missing from the refreshed bundle")
    for key in sorted(cand.keys() - base.keys()):
        failures.append(f"cell {key}: not present in the baseline (suite changed?)")
    for key in sorted(base.keys() & cand.keys()):
        b, c = base[key], cand[key]
        if int(b["n_iter"]) != int(c["n_iter"]):
            failures.append(f"cell {key}: n_iter baseline={b['n_iter']} refreshed={c['n_iter']}")
        if str(b["converged"]) != str(c["converged"]):
            failures.append(
                f"cell {key}: converged baseline={b['converged']} refreshed={c['converged']}"
            )
        if abs(float(b["final_gap"]) - float(c["final_gap"])) > gap_tol:
            failures.append(
                f"cell {key}: gap baseline={b['final_gap']} refreshed={c['final_gap']} "
                f"(tol {gap_tol:g})"
            )
    return failures


def read_pointer(pointer_path: Path = POINTER_PATH) -> str | None:
    """Return the pointer target (relative path) or None if unset."""
    if not pointer_path.exists():
        return None
    data = json.loads(pointer_path.read_text(encoding="utf-8"))
    return data.get("current")


def write_pointer(target: str, pointer_path: Path = POINTER_PATH, note: str | None = None) -> None:
    """Atomically point ``current`` at ``target`` (tmp file + replace)."""
    record: dict = {"current": target, "updated_utc": _utc_stamp()}
    if note:
        record["note"] = note
    tmp = pointer_path.with_name(pointer_path.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    tmp.replace(pointer_path)


def _resolve(pointer_path: Path, target: str) -> Path:
    return (pointer_path.parent / target).resolve()


def refresh(
    baseline_path: Path = EXPERIMENTS_DIR / BASELINE_TARGET,
    pointer_path: Path = POINTER_PATH,
    out_dir: Path | None = None,
    cells: list[tuple[str, int, str]] | None = None,
) -> int:
    """Run the suite into a versioned bundle; move the pointer ONLY on success.

    ``cells=None`` runs the full Stage 2 suite (the CLI default). Tests pass a
    single fast cell to exercise the machinery without a full benchmark run.
    """
    baseline_path = Path(baseline_path)
    pointer_path = Path(pointer_path)
    cells = full_cells() if cells is None else list(cells)
    if out_dir is None:
        out_dir = EXPERIMENTS_DIR / "runs" / dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(out_dir)

    if not baseline_path.exists():
        print(f"refresh: baseline {baseline_path} not found; aborting")
        return 1
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    print(f"refresh: running {len(cells)} cells (Stage 2 suite, settings unchanged)")
    rows = [run_benchmark.run_cell(problem, seed, method) for problem, seed, method in cells]
    csv_path, json_path = run_benchmark.write_bundle(out_dir, rows)
    print(f"refresh: wrote {csv_path} and {json_path}")

    # 1) self-verification with the EXISTING Stage 3 verifier (wall time excluded)
    if run_benchmark.verify_log(json_path) != 0:
        print("refresh: FAILED: refreshed bundle does not verify; pointer NOT moved")
        return 1
    # 2) cross-bundle equivalence vs the frozen baseline (wall time excluded)
    candidate = json.loads(json_path.read_text(encoding="utf-8"))
    failures = compare_bundles(baseline, candidate)
    if failures:
        for failure in failures:
            print(f"refresh: MISMATCH {failure}")
        print(
            "refresh: FAILED: refreshed bundle is not equivalent to the baseline "
            f"(n_iter exact, gap <= {GAP_TOL:g}; wall time excluded); pointer NOT moved"
        )
        return 1

    target = Path(os.path.relpath(json_path.resolve(), pointer_path.parent.resolve())).as_posix()
    write_pointer(
        target, pointer_path, note="Stage 6 refresh (verified equivalent to the v1 baseline)"
    )
    print(f"refresh: verification and equivalence PASSED; current -> {target}")
    return 0


def rollback(pointer_path: Path = POINTER_PATH) -> int:
    """Point ``current`` back at the frozen Stage 2 (v1) baseline."""
    pointer_path = Path(pointer_path)
    baseline = _resolve(pointer_path, BASELINE_TARGET)
    if not baseline.exists():
        print(f"rollback: baseline bundle {baseline} not found; pointer NOT moved")
        return 1
    write_pointer(
        BASELINE_TARGET, pointer_path, note="rolled back to the frozen Stage 2 (v1) baseline"
    )
    print(f"rollback: current -> {BASELINE_TARGET} (frozen Stage 2 baseline)")
    return 0


def current(pointer_path: Path = POINTER_PATH) -> int:
    """Print the pointer target and the pointed-to bundle's identity."""
    pointer_path = Path(pointer_path)
    target = read_pointer(pointer_path)
    if target is None:
        print(f"current: no pointer at {pointer_path}")
        return 1
    print(f"current: {target}")
    bundle_path = _resolve(pointer_path, target)
    if not bundle_path.exists():
        print(f"current: MISSING bundle {bundle_path}")
        return 1
    manifest = json.loads(bundle_path.read_text(encoding="utf-8"))
    print(f"  date_utc={manifest.get('date_utc')} git_commit={manifest.get('git_commit')}")
    print(
        f"  numpy={manifest.get('numpy_version')} scipy={manifest.get('scipy_version')} "
        f"rows={len(manifest.get('rows', []))}"
    )
    print(f"  settings={json.dumps(manifest.get('settings', {}))}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    refresh_p = sub.add_parser(
        "refresh", help="run the suite into a versioned bundle; move pointer on success"
    )
    refresh_p.add_argument(
        "--out", default=None, help="output directory (default: experiments/runs/<UTC-ts>)"
    )
    sub.add_parser("rollback", help="point 'current' back at the frozen v1 baseline")
    sub.add_parser("current", help="show the pointer target and bundle identity")
    args = parser.parse_args(argv)
    if args.command == "refresh":
        return refresh(out_dir=Path(args.out) if args.out else None)
    if args.command == "rollback":
        return rollback()
    return current()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

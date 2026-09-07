"""Stage 6 tests: versioned refresh + 'current' pointer rollback (offline).

The frozen v1 baseline (experiments/run_log.json) is NEVER written by these
tests: refresh always targets a tmp directory. Wall time is NEVER part of any
assertion (it is noisy on this host and indicative only). The full benchmark
suite is NOT run here; each test exercises the machinery on ONE fast,
deterministic cell (logistic + nesterov, seed 1: 33 iterations)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import maintain
import run_benchmark

# One fast deterministic cell (the shipped v1 log records n_iter=33 for it).
FAST_CELL = ("logistic", 1, "nesterov")


def _fresh_row() -> dict:
    """Recompute the fast cell in-process (run_cell runs 3 reps; n_iter is
    deterministic, wall time is not and is never asserted)."""
    return run_benchmark.run_cell(*FAST_CELL)


def _bundle(rows: list[dict], tmp_path: Path) -> Path:
    manifest = {
        "date_utc": "stage6-test",
        "git_commit": "stage6-test",
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "settings": {"tol": 1e-10, "max_iter": 2000},
        "rows": rows,
    }
    path = tmp_path / "run_log.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_refresh_writes_versioned_bundle_without_clobbering_baseline(tmp_path):
    row = _fresh_row()
    baseline = _bundle([row], tmp_path)
    before = baseline.read_bytes()
    pointer = tmp_path / "current.json"
    out = tmp_path / "runs" / "v2"

    rc = maintain.refresh(
        baseline_path=baseline, pointer_path=pointer, out_dir=out, cells=[FAST_CELL]
    )

    assert rc == 0
    assert (out / "run_log.json").exists()  # NEW versioned bundle written
    assert (out / "run_log.csv").exists()
    assert baseline.read_bytes() == before  # frozen baseline byte-identical
    target = json.loads(pointer.read_text(encoding="utf-8"))["current"]
    assert target == "runs/v2/run_log.json"  # pointer moved to the new bundle


def test_failed_refresh_does_not_move_pointer(tmp_path):
    row = _fresh_row()
    row["n_iter"] = int(row["n_iter"]) + 1  # tampered baseline cell
    baseline = _bundle([row], tmp_path)
    pointer = tmp_path / "current.json"
    pointer.write_text(json.dumps({"current": "run_log.json"}), encoding="utf-8")
    out = tmp_path / "runs" / "v2"

    rc = maintain.refresh(
        baseline_path=baseline, pointer_path=pointer, out_dir=out, cells=[FAST_CELL]
    )

    assert rc == 1
    # The refreshed bundle was still written (evidence), but the pointer
    # stayed on the frozen baseline: a FAILED verify never moves `current`.
    assert (out / "run_log.json").exists()
    assert json.loads(pointer.read_text(encoding="utf-8"))["current"] == "run_log.json"


def test_compare_excludes_wall_time():
    row = _fresh_row()
    same_n_iter_diff_wall = dict(row, wall_time_s="9.999999")
    assert maintain.compare_bundles({"rows": [row]}, {"rows": [same_n_iter_diff_wall]}) == []
    tampered_iter = dict(row, n_iter=int(row["n_iter"]) + 1)
    assert maintain.compare_bundles({"rows": [row]}, {"rows": [tampered_iter]})


def test_rollback_restores_v1_pointer(tmp_path):
    pointer = tmp_path / "current.json"
    (tmp_path / "runs" / "v2").mkdir(parents=True)
    (tmp_path / "runs" / "v2" / "run_log.json").write_text("{}", encoding="utf-8")
    baseline_row = _fresh_row()
    baseline = _bundle([baseline_row], tmp_path)
    pointer.write_text(json.dumps({"current": "runs/v2/run_log.json"}), encoding="utf-8")

    rc = maintain.rollback(pointer_path=pointer)

    assert rc == 0
    record = json.loads(pointer.read_text(encoding="utf-8"))
    assert record["current"] == "run_log.json"  # pointer back on the v1 baseline
    assert maintain.read_pointer(pointer) == "run_log.json"
    assert baseline.exists()


def test_refreshed_cell_matches_frozen_v1_n_iter():
    """The recomputed cell matches the shipped v1 log EXACTLY on n_iter
    (deterministic); wall time is deliberately not compared."""
    shipped = Path(__file__).resolve().parents[1] / "experiments" / "run_log.json"
    manifest = json.loads(shipped.read_text(encoding="utf-8"))
    (v1_row,) = [
        r for r in manifest["rows"] if (r["problem"], int(r["seed"]), r["method"]) == FAST_CELL
    ]
    fresh = _fresh_row()
    assert fresh["n_iter"] == int(v1_row["n_iter"])
    assert str(fresh["converged"]) == str(v1_row["converged"])
    assert abs(float(fresh["final_gap"]) - float(v1_row["final_gap"])) <= 1e-8

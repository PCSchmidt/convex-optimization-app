"""Stage 3 tests: the experiment artifact bundle verify path.

The bundle is experiments/run_log.json (+ run_log.csv): per-row problem, seed,
method, n_iter, final gap, converged, plus settings, numpy/scipy versions and
the git commit. The verifier recomputes every logged cell deterministically
(n_iter must match exactly; gap within the Stage 1 criterion) and reports the
environment identity. Wall time is intentionally not verified: it is noisy on
this host and indicative only. All tests here are offline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import run_benchmark


def _manifest(rows: list[dict]) -> dict:
    return {
        "date_utc": "bundle-test",
        "git_commit": "bundle-test",
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "settings": {"tol": 1e-10, "max_iter": 2000},
        "rows": rows,
    }


def _write(tmp_path, manifest: dict) -> Path:
    path = tmp_path / "run_log.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _row(problem: str, seed: int, method: str) -> dict:
    # run_cell returns exactly the schema written to the bundle (n_iter int,
    # final_gap/converged strings). Wall time is part of the schema but is
    # never compared by the verifier.
    return run_benchmark.run_cell(problem, seed, method)


def test_verify_passes_on_fresh_bundle(tmp_path):
    rows = [_row("logistic", 1, "nesterov"), _row("lasso", 0, "fista")]
    assert run_benchmark.verify_log(_write(tmp_path, _manifest(rows))) == 0


def test_verify_fails_on_tampered_n_iter(tmp_path):
    rows = [_row("logistic", 1, "nesterov")]
    rows[0]["n_iter"] = int(rows[0]["n_iter"]) + 1
    assert run_benchmark.verify_log(_write(tmp_path, _manifest(rows))) == 1


def test_verify_fails_on_tampered_gap(tmp_path):
    rows = [_row("logistic", 1, "nesterov")]
    rows[0]["final_gap"] = "5.0e-3"  # far outside the Stage 1 criterion (1e-8)
    assert run_benchmark.verify_log(_write(tmp_path, _manifest(rows))) == 1


def test_verify_fails_on_tampered_converged(tmp_path):
    rows = [_row("logistic", 1, "nesterov")]
    rows[0]["converged"] = "False"
    assert run_benchmark.verify_log(_write(tmp_path, _manifest(rows))) == 1


def test_verify_accepts_version_mismatch_with_warning(tmp_path, monkeypatch):
    # A numpy/scipy version mismatch is reported (determinism across versions
    # is not guaranteed) but is a warning, not a failure: the recomputation
    # itself decides pass/fail.
    monkeypatch.setattr(np, "__version__", "0.0.0-test")
    rows = [_row("logistic", 1, "nesterov")]
    assert run_benchmark.verify_log(_write(tmp_path, _manifest(rows))) == 0


def test_verify_passes_on_the_shipped_stage2_bundle():
    """The full shipped Stage 2 bundle (18 cells) recomputes exactly.

    This certifies that a reviewer can rebuild every recorded n_iter from the
    bundle identity alone (problem, seed, method, tolerances). Deterministic,
    offline, and fast (single solve per cell, no wall-time reps).
    """
    shipped = Path(__file__).resolve().parents[1] / "experiments" / "run_log.json"
    assert run_benchmark.verify_log(shipped) == 0

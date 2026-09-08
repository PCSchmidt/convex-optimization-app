> **Historical document (2026-09-08):** the `legacy/` snapshot this inventory describes was removed from the working tree after the redesign was published. Its contents remain recoverable from this repository's git history. The audit below is preserved as the Stage 0 record.

# INVENTORY.md - legacy/ reference review

Read-only snapshot of the previous GitHub app, imported into `legacy/` for
reference. Nothing here is wired into the new package yet; this file records
what exists, what works, and where the gaps are against `AGENTS.md` and
`ROADMAP.md`.

## Contents

| Path | What it is | Notes |
| --- | --- | --- |
| `app.py` | FastAPI app setup (`create_app`), Jinja2 templates, imports solvers/routes/visualize | Serves the educational UI; no packaged layout (flat module imports). |
| `routes.py` | All HTTP routes (LP/QP/SDP/conic/geometric solvers, benchmark, visualize, tutorial), loads `problems/` YAML/JSON | Monolithic (23k chars); mixes parsing, solving, and rendering. |
| `solvers.py` | Wrappers around PuLP (LP) and CVXPY (QP/SDP/conic/geometric) | Solver-backed, not self-implemented algorithms; no convergence tracking. |
| `parser.py` | Sympy-based parsing of polynomials, matrices, posynomials from text input | Decent utility code; hardcoded assumptions (`^` -> `**`). |
| `benchmark.py` | CLI benchmarking LP/QP solvers on sample problems; writes CSV | Closest ancestor of the planned `experiments/` run log; no seeds, no convergence metrics. |
| `visualize.py` | Plotly figures: LP feasible region, 3D surfaces, gradient-descent / feasible-region / interior-point / simplex animations | Gradient-descent animation exists but is illustrative, not a rigorous method. |
| `templates/` | 17 Jinja2 HTML pages (solvers, tutorial, benchmark, visualization) | Frontend for the old app; not reused in Stage 0. |
| `problems/examples.yaml`, `problems/case_studies/` | Example LP/QP problems, portfolio + transport case studies | Small, reusable problem seeds for the future benchmark suite. |
| `tutorial/quiz.yaml` | Quiz data for the tutorial pages | Educational content only. |
| `tests/test_api.py`, `test_parser.py`, `test_solvers.py` | Small pytest suites | Importorskip on cvxpy/fastapi/httpx; no lint config; no CI lint step. |
| `.github/workflows/test.yml` | CI: pip install requirements, run pytest | Tests only, no lint; installs ad-hoc (no lockfile). |
| `requirements.txt` | Fully pinned pins: cvxpy, PuLP, scipy, sympy, FastAPI, uvicorn, plotly, matplotlib | Note: mixed operators (`>=` on anyio/starlette despite "pinned"); heavy tree. |
| `Dockerfile` | python:3.11 base, uvicorn entrypoint | No compose file; no artifact/versioning story. |
| `README.md`, `CONTRIBUTING.md`, `LICENSE`, `instructions.txt` | Docs and original prompt | `instructions.txt` is the original build brief (FastHTML/visuals focus). |

## Mapping to AGENTS.md requirements

| AGENTS.md requirement | Legacy status | Gap |
| --- | --- | --- |
| Mathematical rigor (self-implemented algorithms, convergence behavior) | Solvers delegate to PuLP/CVXPY backends | No GD/Nesterov/proximal-ADMM implementations; no per-iteration convergence record. |
| Reproducible experiments (pinned deps, seeds, run log) | Requirements mostly pinned; benchmark.py writes CSV | No seeds, no structured run log with config/seed/date; CI installs without a lockfile. |
| Benchmarking vs ground truth | `benchmark.py` times PuLP/CVXPY on sample problems | Benchmarks two external solvers against each other, not custom methods vs SciPy ground truth; no metrics like iterations-to-convergence or objective gap. |
| Honest documentation | README describes features truthfully | Missing the Motivation/Method/Results/Limitations/Operational-notes skeleton. |
| Tests passing | Three small suites exist | Narrow coverage; skips silently when heavy deps missing; no lint in CI. |

## Mapping to ROADMAP.md

- **Stage 0**: legacy has a rough app but no packaging (`pyproject.toml`), no lockfile, no `make setup` path, no lint. This scaffold supplies those.
- **Stage 1** (optimization core): nothing in legacy implements classic methods from scratch. `solvers.py`/`parser.py` patterns and `problems/` examples are the useful seeds.
- **Stage 2** (evaluate): `benchmark.py` is a starting sketch; it lacks seeds, stability-across-seeds, and a structured `experiments/` log.
- **Stage 3-5** (ship/deploy/monitor): a bare Dockerfile exists; no compose, no artifact bundle, no FastAPI-solve API design, no metrics/logging endpoints.

## Explicit gaps called out by AGENTS.md

1. No self-implemented optimization algorithms (GD variants, proximal/ADMM).
2. No convergence tracking (objective value / gradient norm per iteration).
3. No reproducible experiment log with seeds and pinned lockfile.
4. No benchmark-vs-SciPy ground truth with rigorous metrics.
5. No test-correctness-against-known-solutions suite (legacy tests are smoke-level).
6. No Makefile / reproducible `make setup && make test` acceptance path.
7. No honest README lifecycle structure.

Legacy code stays in `legacy/` untouched. It informs design; it is not copied
over the new specs.

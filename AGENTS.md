# AGENTS.md - convex-optimization-app

Project instructions for AI coding agents working in this repository.

## Purpose

Turn `convex-optimization-app` into a portfolio-grade application that demonstrates rigorous, reproducible ML/optimization engineering. This repo owns the **optimization + reproducibility / benchmarking (rigor)** pillar of the portfolio and leans into the user's applied-mathematics strength from their JHU AI Engineering program.

This is a personal portfolio project, not production work. It must be honest, rigorous, and reproducible.

## Current state (as of Sep 2026)

- Dormant since Jun 2025.
- Existing functionality: an educational app showcasing convex optimization concepts.
- Python-based. The user is agnostic to stack, so a redesign is allowed if it serves the goal better.

## Non-negotiable requirements

1. **Mathematical rigor.** Correct, well-documented optimization algorithms (e.g., gradient descent variants, proximal methods, or a solver-backed approach). Show convergence behavior, not just results.
2. **Reproducible experiments.** Every experiment must be reproducible with pinned dependencies, fixed seeds, and a recorded run log.
3. **Benchmarking.** Compare methods on standard problems with clear metrics (e.g., iterations to convergence, wall time, final objective gap).
4. **Honest documentation.** Motivation, Method, Results, Limitations, Operational notes. Never claim a capability that is not implemented.
5. **Tests.** A test suite must exist and pass, including correctness checks against known solutions.

## Tech stack guidance

The user is agnostic to stack. Prefer boring, well-supported tools. Python is the default. Suggested defaults (change only with justification):

- Language: Python 3.11+
- Numerical core: NumPy/SciPy; optionally JAX for autodiff if it adds clear value
- Optimization: implement classic methods (gradient descent, accelerated methods, proximal/ADMM) and compare against SciPy solvers as ground truth
- Serving: FastAPI (to expose the app as a service)
- Packaging: Docker + docker-compose
- CI/CD: GitHub Actions
- Experiment tracking: a lightweight `experiments/` run log (JSON/CSV) with date, config, seeds, and results

## Working conventions

- Keep the model-facing tool surface small. Prefer a persistent Python REPL as the control environment.
- Run project commands from the repo root.
- After any code change, run the test suite and the linter.
- Update ROADMAP.md as work progresses. Mark completed items.
- Use git branches for each lifecycle stage. Commit with clear messages.
- Do not commit secrets, API keys, or large artifacts.

## Lifecycle stage ownership

This repo owns the **optimization + reproducibility / benchmarking (rigor)** pillar of the portfolio. Do not duplicate the RAG or forecasting work that belongs to the sibling repos.

## Context files

- `ROADMAP.md` in this directory is the source of truth for planned and completed work.
- Global instructions may also be loaded from `~/.prime/agent/AGENTS.md`; project instructions here take precedence for this repo.
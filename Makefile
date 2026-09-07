# Portable Makefile for convex_optimization_app (Linux + Windows/Git-Bash).
# Detect the venv python path: Windows uses Scripts/, POSIX uses bin/.

ifeq ($(OS),Windows_NT)
    VENV_PY := .venv/Scripts/python.exe
else
    VENV_PY := .venv/bin/python
endif

.PHONY: setup test lint format clean solve eval verify refresh rollback current docker-build smoke api

setup:
	python -m venv .venv
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install -r requirements-lock.txt

test:
	$(VENV_PY) -m pytest -q
	$(VENV_PY) -m ruff check .

# Run ONE problem with ONE method, offline:
#   make solve ARGS="--problem lasso --method fista"
solve:
	PYTHONPATH=src $(VENV_PY) -m convex_optimization.cli $(ARGS)

# Stage 2 benchmark: measures the Stage 1 methods AS-IS across seeds and
# writes experiments/run_log.csv + experiments/run_log.json. NOT part of
# `make test`: wall-time measurement is too noisy for CI assertions.
# Stage 6 note: this writes the DEFAULT paths, i.e. it overwrites the
# working-tree copies of the frozen v1 bundle (git-tracked; recover with
# `git checkout -- experiments/run_log.json experiments/run_log.csv`).
# The Stage 6 refresh (`make refresh`) writes a VERSIONED bundle instead
# and never touches experiments/run_log.*.
eval:
	PYTHONPATH=src $(VENV_PY) experiments/run_benchmark.py

# Stage 3 artifact bundle: verify the saved run log (experiments/run_log.json).
# Recomputes every logged cell deterministically (n_iter exact, gap within the
# Stage 1 criterion) and reports numpy/scipy/git identity. Offline; NOT part
# of `make test`.
verify:
	PYTHONPATH=src $(VENV_PY) experiments/run_benchmark.py --verify experiments/run_log.json

# Stage 6 maintain (offline, on-demand; NO scheduler): re-run the Stage 2
# suite into a VERSIONED bundle experiments/runs/<ts>/, verify it (n_iter
# exact vs the recomputed cells and vs the frozen v1 baseline; wall time
# excluded), and move experiments/current.json ONLY if everything passes.
refresh:
	PYTHONPATH=src $(VENV_PY) experiments/maintain.py refresh

# Point experiments/current.json back at the frozen Stage 2 (v1) baseline.
rollback:
	PYTHONPATH=src $(VENV_PY) experiments/maintain.py rollback

# Show the 'current' pointer target and the pointed-to bundle's identity.
current:
	PYTHONPATH=src $(VENV_PY) experiments/maintain.py current

lint:
	$(VENV_PY) -m ruff check .
	$(VENV_PY) -m ruff format --check .

format:
	$(VENV_PY) -m ruff format .

# Stage 3 container (LOCAL ONLY): build the image and run the offline smoke
# solve (one CLI solve inside the container, no network). NOT part of `make test`.
docker-build:
	docker build -t convex-optimization:local .

smoke:
	docker compose run --rm smoke

# Stage 4 serving (LOCAL ONLY): build and run the FastAPI app in the container
# on the host port ${PORT:-8000} (default 8000). NOT part of `make test`.
api:
	docker compose up --build api

clean:
	rm -rf .venv .pytest_cache .ruff_cache build dist
	find . -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.egg-info" -type d -prune -exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage

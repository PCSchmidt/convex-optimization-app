# Portable Makefile for convex_optimization_app (Linux + Windows/Git-Bash).
# Detect the venv python path: Windows uses Scripts/, POSIX uses bin/.

ifeq ($(OS),Windows_NT)
    VENV_PY := .venv/Scripts/python.exe
else
    VENV_PY := .venv/bin/python
endif

.PHONY: setup test lint format clean solve eval verify docker-build smoke

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
eval:
	PYTHONPATH=src $(VENV_PY) experiments/run_benchmark.py

# Stage 3 artifact bundle: verify the saved run log (experiments/run_log.json).
# Recomputes every logged cell deterministically (n_iter exact, gap within the
# Stage 1 criterion) and reports numpy/scipy/git identity. Offline; NOT part
# of `make test`.
verify:
	PYTHONPATH=src $(VENV_PY) experiments/run_benchmark.py --verify experiments/run_log.json

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

clean:
	rm -rf .venv .pytest_cache .ruff_cache build dist
	find . -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.egg-info" -type d -prune -exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage

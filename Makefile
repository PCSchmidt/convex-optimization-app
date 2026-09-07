# Portable Makefile for convex_optimization_app (Linux + Windows/Git-Bash).
# Detect the venv python path: Windows uses Scripts/, POSIX uses bin/.

ifeq ($(OS),Windows_NT)
    VENV_PY := .venv/Scripts/python.exe
else
    VENV_PY := .venv/bin/python
endif

.PHONY: setup test lint format clean solve

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

lint:
	$(VENV_PY) -m ruff check .
	$(VENV_PY) -m ruff format --check .

format:
	$(VENV_PY) -m ruff format .

clean:
	rm -rf .venv .pytest_cache .ruff_cache build dist
	find . -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.egg-info" -type d -prune -exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage

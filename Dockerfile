# Local, offline container for the convex optimization app (Stage 3).
# Installs ONLY from requirements-lock.txt (the single lock) and runs one CLI
# solve as the smoke check. No API keys, no network access needed at runtime.
#
# PYTHON FLOOR: the default base is python:3.12-slim. python:3.11 DOES NOT
# WORK: numpy 2.5.3 pinned in requirements-lock.txt requires Python >= 3.12,
# so pip install of the lock fails on 3.11. Reproduce that failure with:
#     docker build --build-arg PYTHON_TAG=3.11 -t convex-optimization:py311-check .
ARG PYTHON_TAG=3.12
FROM python:${PYTHON_TAG}-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# Install the pinned stack first so this layer caches independently of code.
COPY requirements-lock.txt ./
RUN pip install --no-cache-dir -r requirements-lock.txt

# Package (run via PYTHONPATH=src, same convention as the host Makefile),
# plus the benchmark runner and its saved Stage 2 bundle.
COPY src/ ./src/
COPY experiments/ ./experiments/
COPY pyproject.toml README.md ./

# Phase B (Option A): serve the built workbench UI from the API process,
# same origin. The dist bundle is built on the host (make ui-build) BEFORE
# docker build; ui/src and ui/node_modules are NOT needed in the image.
ENV SERVE_UI=1
COPY ui/dist/ ./ui/dist/

# Stage 4 serving (compose `api` service) overrides the command with uvicorn;
# port 8000 is the fixed container-internal API port.
EXPOSE 8000

# Default command: one fully offline solve (logistic + nesterov), the Stage 3
# smoke check. Override per run, e.g.:
#   docker run --rm <image> python -m convex_optimization.cli --problem lasso --method fista
CMD ["python", "-m", "convex_optimization.cli", "--problem", "logistic", "--method", "nesterov"]

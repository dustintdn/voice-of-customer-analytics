# Voice-of-Customer intelligence pipeline
# See docs/SPEC.md for the full build spec and CLAUDE.md for build discipline.

PY ?= python
VENV ?= .venv
BIN := $(VENV)/bin

.DEFAULT_GOAL := help
.PHONY: help setup setup-core fetch-model sample run run-offline report test lint format clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup-core: ## Light install: core + dev only (offline backends; no heavy/compiled deps)
	@test -d $(VENV) || $(PY) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -e ".[dev]"

setup: setup-core ## Full install: core + ml + llm + dev, then pre-fetch the embedding model
	@# LightGBM on macOS requires OpenMP; install if missing (no-op if present)
	@if [ "$$(uname)" = "Darwin" ] && ! brew list libomp &>/dev/null; then \
		echo "[setup] Installing libomp (required by LightGBM on macOS)..."; \
		brew install libomp; \
	fi
	$(BIN)/python -m pip install -e ".[ml,llm]"
	$(MAKE) fetch-model

fetch-model: ## Pre-download the embedding model (one-time network access; offline thereafter)
	$(BIN)/python scripts/fetch_model.py --config config/config.yaml

sample: ## Build the small committed sample dataset (slices raw CSV if present, else synthesizes)
	$(BIN)/python scripts/make_sample.py

run: ## Run the full pipeline with the configured (real) backends; LLM in --dry-run
	$(BIN)/python -m voc.pipeline run --config config/config.yaml --dry-run

run-offline: ## Run the full pipeline fully offline (core-only deps, no network)
	$(BIN)/python -m voc.pipeline run --config config/config.offline.yaml --dry-run

report: ## (Re)generate the insight report from existing stage outputs
	$(BIN)/python -m voc.pipeline report --config config/config.yaml

test: ## Run the test suite (offline, no network)
	$(BIN)/python -m pytest

lint: ## Lint with ruff
	$(BIN)/ruff check src tests scripts

format: ## Auto-format with ruff
	$(BIN)/ruff format src tests scripts
	$(BIN)/ruff check --fix src tests scripts

clean: ## Remove caches and generated (non-sample) artifacts
	rm -rf .pytest_cache .ruff_cache **/__pycache__ src/**/__pycache__
	rm -rf data/processed/* reports/*
	@touch data/processed/.gitkeep reports/.gitkeep

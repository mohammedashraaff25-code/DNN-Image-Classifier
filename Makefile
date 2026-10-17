.PHONY: help install install-dev train evaluate predict test lint format check clean

PYTHON ?= python

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-12s %s\n", $$1, $$2}'

install: ## Install runtime dependencies
	$(PYTHON) -m pip install -r requirements.txt

install-dev: ## Install runtime + development dependencies
	$(PYTHON) -m pip install -r requirements-dev.txt

train: ## Train the model (saves models/best_model.keras)
	$(PYTHON) -m src.train

evaluate: ## Evaluate the saved model on the test set
	$(PYTHON) -m src.evaluate

predict: ## Classify 10 random test images
	$(PYTHON) -m src.predict --test-samples 10

test: ## Run the test suite
	$(PYTHON) -m pytest

lint: ## Check code style with Ruff
	$(PYTHON) -m ruff check src tests

format: ## Auto-format the code with Black and Ruff
	$(PYTHON) -m black src tests
	$(PYTHON) -m ruff check src tests --fix

check: ## Run everything CI runs (lint, formatting check, tests)
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m black --check src tests
	$(PYTHON) -m pytest

clean: ## Remove caches and build artifacts
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache -o -name .ipynb_checkpoints \) -not -path "./venv/*" -prune -exec rm -rf {} +
	rm -rf .coverage htmlcov

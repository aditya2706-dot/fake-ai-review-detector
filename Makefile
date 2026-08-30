# Fake review detector -- common tasks.
PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: help setup dataset train train-fast score serve test lint clean docker

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## Create the venv and install the package (editable) + dev deps
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	$(PIP) install lightgbm || echo "lightgbm skipped; sklearn fallback will be used"
	@echo "macOS + lightgbm also needs OpenMP:  brew install libomp"

dataset:  ## Synthesise a 40k-review marketplace into data/processed/
	$(PY) -m frd.cli.make_dataset --source synthetic --n-reviews 40000

train:  ## Full study: 5 ablations x 2 split regimes, saves best model
	$(PY) -m frd.cli.train

train-fast:  ## Just the fused model on the temporal split (quick iteration)
	$(PY) -m frd.cli.train --only full_fusion --splits temporal

score:  ## Score a single review inline
	$(PY) -m frd.cli.score --text "Amazing product! Highly recommend! Best ever!"

score-file:  ## Score a whole file: make score-file FILE=path.csv
	$(PY) -m frd.cli.score --data $(FILE) --out artifacts/reports/scored.csv

serve:  ## Run the scoring API on :8000 (docs at /docs)
	.venv/bin/uvicorn frd.api.main:app --reload --port 8000

test:  ## Run the test suite
	$(PY) -m pytest -q

test-v:  ## Run the test suite, verbose
	$(PY) -m pytest -v

lint:  ## Ruff, if installed
	.venv/bin/ruff check src tests || true

clean:  ## Remove artefacts and caches
	rm -rf artifacts/models/* artifacts/reports/* artifacts/figures/*
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache

docker:  ## Build the serving image
	docker build -t fake-review-detector:latest .

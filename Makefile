# Fake review detector -- common tasks.
PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: help up stop status setup dataset train train-fast score serve test \
        lint clean docker web-setup web console web-build \
        need-venv need-data need-model need-web

help:  ## Show this list
	@echo "Run the whole thing:  make up"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

# ---- the short path --------------------------------------------------------
#
# `make up` is the whole project from a clean checkout: it does only the steps
# that have not been done yet, then runs the app. Each need-* guard is a cheap
# file test, so re-running `make up` on a warm tree just starts the servers.
# The slow step is `train` (a few minutes); everything else is seconds.

up:  ## Everything: set up whatever is missing, then run the app
	@$(MAKE) --no-print-directory need-venv
	@$(MAKE) --no-print-directory need-data
	@$(MAKE) --no-print-directory need-model
	@$(MAKE) --no-print-directory need-web
	@$(MAKE) --no-print-directory console

need-venv:
	@test -x $(PY) || { echo "==> no venv yet"; $(MAKE) --no-print-directory setup; }

need-data:
	@test -f data/processed/reviews.parquet -o -f data/processed/reviews.csv \
	  || { echo "==> no dataset yet"; $(MAKE) --no-print-directory dataset; }

need-model:
	@test -f artifacts/models/detector.joblib \
	  || { echo "==> no trained model yet (this one takes a few minutes)"; \
	       $(MAKE) --no-print-directory train; }

need-web:
	@test -d web/node_modules \
	  || { echo "==> console dependencies not installed"; \
	       $(MAKE) --no-print-directory web-setup; }

stop:  ## Stop anything left running on :8000 or :5173
	@pkill -f "uvicorn frd.api.main:app" 2>/dev/null && echo "stopped API" || echo "no API running"
	@pkill -f "$(CURDIR)/web/node_modules/.bin/vite" 2>/dev/null && echo "stopped console" || echo "no console running"

status:  ## What is set up and what is running
	@printf "venv       %s\n" "$$(test -x $(PY) && echo yes || echo 'no  -- make setup')"
	@printf "dataset    %s\n" "$$(ls data/processed/reviews.parquet data/processed/reviews.csv 2>/dev/null | head -1 || echo 'no  -- make dataset')"
	@printf "model      %s\n" "$$(test -f artifacts/models/detector.joblib && echo yes || echo 'no  -- make train')"
	@printf "web deps   %s\n" "$$(test -d web/node_modules && echo yes || echo 'no  -- make web-setup')"
	@printf "API :8000  %s\n" "$$(curl -s -m 1 -o /dev/null localhost:8000/health && echo running || echo stopped)"
	@printf "app :5173  %s\n" "$$(curl -s -m 1 -o /dev/null localhost:5173 && echo running || echo stopped)"

setup:  ## Create the venv and install the package (editable) + dev deps
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	$(PIP) install lightgbm || echo "lightgbm skipped; sklearn fallback will be used"
	@echo "macOS + lightgbm also needs OpenMP:  brew install libomp"

dataset:  ## Synthesise a 40k-review marketplace into data/processed/
	$(PY) -m frd.cli.make_dataset --source synthetic --n-reviews 40000

train:  ## Full study: 8 ablations x 2 split regimes, saves best model
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

# ---- moderation console (web/) ---------------------------------------------

web-setup:  ## Install the console's npm dependencies
	cd web && npm install

web:  ## Run the console dev server on :5173 (needs `make serve` alongside)
	cd web && npm run dev

console:  ## Run API and console together; open http://localhost:5173
	@echo "API on :8000, console on :5173. Ctrl-C stops both."
	@trap 'kill 0' INT TERM EXIT; \
	 .venv/bin/uvicorn frd.api.main:app --port 8000 & \
	 (cd web && npm run dev) & \
	 wait

web-build:  ## Production build of the console into web/dist
	cd web && npm run build

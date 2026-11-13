# Fake review detector -- common tasks.
PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: help up stop status setup dataset train train-fast score serve test \
        lint clean docker deploy deploy-logs deploy-vm deploy-vm-logs web-setup web app console \
        web-build store-reset need-venv need-data need-model need-web

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
	@$(MAKE) --no-print-directory app

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
	  || { echo "==> web dependencies not installed"; \
	       $(MAKE) --no-print-directory web-setup; }

stop:  ## Stop anything left running on :8000 or :5173
	@pkill -f "uvicorn frd.api.main:app" 2>/dev/null && echo "stopped API" || echo "no API running"
	@pkill -f "$(CURDIR)/web/node_modules/.bin/vite" 2>/dev/null && echo "stopped web app" || echo "no web app running"

status:  ## What is set up and what is running
	@printf "venv       %s\n" "$$(test -x $(PY) && echo yes || echo 'no  -- make setup')"
	@printf "dataset    %s\n" "$$(ls data/processed/reviews.parquet data/processed/reviews.csv 2>/dev/null | head -1 || echo 'no  -- make dataset')"
	@printf "model      %s\n" "$$(test -f artifacts/models/detector.joblib && echo yes || echo 'no  -- make train')"
	@printf "web deps   %s\n" "$$(test -d web/node_modules && echo yes || echo 'no  -- make web-setup')"
	@printf "shop db    %s\n" "$$(test -f data/store.db && echo yes || echo 'none yet (created on first sign-up)')"
	@printf "API :8000  %s\n" "$$(curl -s -m 1 -o /dev/null localhost:8000/health && echo running || echo stopped)"
	@printf "shop :5173 %s\n" "$$(curl -s -m 1 -o /dev/null localhost:5173 && echo running || echo stopped)"

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

docker:  ## Build the serving image (API + web app, one origin)
	docker build -t verity-market:latest .

# ---- deployment -------------------------------------------------------------
#
# The image copies the trained model and the corpus out of this working tree
# rather than training them itself -- see the Dockerfile. So the one way to
# deploy something broken is to deploy before `make up` has produced them, and
# that is what this checks.

deploy:  ## Deploy to Fly.io (needs flyctl, and `make up` to have run once)
	@test -f artifacts/models/detector.joblib \
	  || { echo "no trained model -- run 'make train' before deploying"; exit 1; }
	@test -f data/processed/reviews.parquet -o -f data/processed/reviews.csv \
	  || { echo "no corpus -- run 'make dataset' before deploying"; exit 1; }
	@command -v flyctl >/dev/null \
	  || { echo "flyctl not installed -- https://fly.io/docs/flyctl/install/"; exit 1; }
	@grep -q 'app = "verity-market"' fly.toml \
	  && echo "NOTE: fly.toml still uses the default app name; Fly app names are global." || true
	flyctl deploy

deploy-logs:  ## Tail the deployed app's logs
	flyctl logs

# Self-hosted, $0: a free Oracle Cloud or Google Cloud VM (see deploy/vm/README.md).
# The image is built ON the VM, so this machine needs no Docker -- it only
# ships the tree, including the gitignored model and corpus.
#   make deploy-vm VM=ubuntu@1.2.3.4            # domain defaults to 1-2-3-4.sslip.io
#   make deploy-vm VM=ubuntu@1.2.3.4 DOMAIN=shop.example.com
VM_HOST = $(lastword $(subst @, ,$(VM)))
DOMAIN ?= $(subst .,-,$(VM_HOST)).sslip.io

deploy-vm:  ## Deploy to your own VM: make deploy-vm VM=ubuntu@<ip>
	@test -n "$(VM)" || { echo "usage: make deploy-vm VM=ubuntu@<public-ip>"; exit 1; }
	@test -f artifacts/models/detector.joblib \
	  || { echo "no trained model -- run 'make train' before deploying"; exit 1; }
	@test -f data/processed/reviews.csv \
	  || { echo "no data/processed/reviews.csv -- the image reads the CSV (no pyarrow in it); run 'make dataset'"; exit 1; }
	rsync -az --delete \
	  --exclude .git --exclude .venv --exclude 'web/node_modules' --exclude 'web/dist' \
	  --exclude '__pycache__' --exclude 'data/raw' --exclude 'data/store.db*' \
	  ./ $(VM):app/
	ssh $(VM) 'cd app && DOMAIN=$(DOMAIN) docker compose -f deploy/vm/compose.yml up -d --build'
	@echo "==> live at https://$(DOMAIN) (first boot: allow ~30s for the certificate and the model)"

deploy-vm-logs:  ## Tail the VM deployment's logs: make deploy-vm-logs VM=ubuntu@<ip>
	ssh $(VM) 'cd app && DOMAIN=$(DOMAIN) docker compose -f deploy/vm/compose.yml logs -f --tail=100'

# ---- the web app (web/) -----------------------------------------------------
#
# One bundle, two surfaces: the shop at #/ and the moderation console at
# #/console. They are the same scores over the same corpus cache; the
# difference is what each audience is allowed to act on.

web-setup:  ## Install the web app's npm dependencies
	cd web && npm install

web:  ## Run the web dev server on :5173 (needs `make serve` alongside)
	cd web && npm run dev

app:  ## Run API and web app together; open http://localhost:5173
	@echo "API on :8000, shop on :5173 (console at /#/console). Ctrl-C stops both."
	@trap 'kill 0' INT TERM EXIT; \
	 .venv/bin/uvicorn frd.api.main:app --port 8000 & \
	 (cd web && npm run dev) & \
	 wait

console: app  ## Alias for `make app`, kept for muscle memory

web-build:  ## Production build of the web app into web/dist
	cd web && npm run build

store-reset:  ## Delete the shop's accounts, baskets and orders
	rm -f data/store.db data/store.db-wal data/store.db-shm
	@echo "shop database removed; it is recreated on the next sign-up"

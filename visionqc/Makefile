# VisionQC — one command per pipeline stage.
#
# WHY A MAKEFILE: a reviewer (or interviewer) should be able to reproduce your
# entire project with `make all`. If your README says "run these nine commands
# in order", most people run zero of them.

CONFIG ?= configs/synthetic.yaml
RUN_DIR ?= artifacts/synthetic
PY = PYTHONPATH=src python3

.PHONY: help setup data splits padim autoencoder classifier evaluate all serve test docker docker-run clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:        ## Install dependencies
	pip install -r requirements.txt

data:         ## Generate the synthetic dataset
	$(PY) -m visionqc.data.synthetic --root data/synthetic

splits:       ## Build the leak-checked split manifest
	$(PY) -m visionqc.data.splits --root data/synthetic --out $(RUN_DIR)/splits.json

padim:        ## Fit PaDiM on normal images only
	$(PY) -m visionqc.fit_padim --config $(CONFIG)

autoencoder:  ## Train the autoencoder baseline
	$(PY) -m visionqc.train_autoencoder --config $(CONFIG)

classifier:   ## Train the supervised classifier
	$(PY) -m visionqc.train_classifier --config $(CONFIG)

evaluate:     ## Calibrate thresholds and produce the full report
	$(PY) -m visionqc.evaluate --config $(CONFIG)

all: data splits padim autoencoder classifier evaluate  ## Full pipeline from scratch
	@echo "Done. Report: $(RUN_DIR)/results.md"

serve:        ## Run the API locally at http://localhost:8000/docs
	VISIONQC_RUN_DIR=$(RUN_DIR) PYTHONPATH=src \
	  uvicorn visionqc.api.main:app --reload --port 8000

test:         ## Run the test suite
	python3 -m pytest -q

test-fast:    ## Run tests, skipping the slow ones
	python3 -m pytest -q -m "not slow"

docker:       ## Build the serving image
	docker build -t visionqc:latest .

docker-run:   ## Run the container
	docker run --rm -p 8000:8000 -e VISIONQC_RUN_DIR=/app/$(RUN_DIR) visionqc:latest

clean:        ## Remove generated artifacts (keeps datasets)
	rm -rf artifacts/ .pytest_cache/ && find . -name __pycache__ -type d -exec rm -rf {} +

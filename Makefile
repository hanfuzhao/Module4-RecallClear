# Convenience targets. Run `make help` for the list.

PYTHON ?= python3
PORT ?= 7860
export PYTHONPATH := $(CURDIR)
export KMP_DUPLICATE_LIB_OK := TRUE
export TOKENIZERS_PARALLELISM := false

.PHONY: help install data features train evaluate demo app test clean all

help:  ## Show the available targets
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

install:  ## Install development dependencies
	$(PYTHON) -m pip install -r requirements.txt

data:  ## Download NHTSA recall notices
	$(PYTHON) -m scripts.make_dataset

features:  ## Build the training splits from the raw notices
	$(PYTHON) -m scripts.build_features

train:  ## Fine-tune the LoRA adapter
	$(PYTHON) -m scripts.model --train

evaluate:  ## Score the adapter against the untuned baselines
	$(PYTHON) -m scripts.evaluate_model --sample 150

demo:  ## Print one before/after comparison to the terminal
	$(PYTHON) -m scripts.demo

app:  ## Run the web app locally
	$(PYTHON) main.py --port $(PORT)

test:  ## Run the unit tests
	$(PYTHON) -m unittest discover -s tests -v

all: data features train evaluate  ## Full pipeline from scratch

clean:  ## Remove generated artefacts (keeps raw downloads)
	rm -rf models/checkpoints models/merged data/processed/*.jsonl data/outputs/*.json

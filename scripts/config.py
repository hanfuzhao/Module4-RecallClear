"""Central configuration for RecallClear.

Every path, model id, and hyperparameter used by the pipeline lives here so that
the scripts stay free of magic constants and a single edit changes the whole run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = DATA_DIR / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

ADAPTER_DIR = MODELS_DIR / "adapter"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
MERGED_DIR = MODELS_DIR / "merged"

RAW_RECALLS_PATH = RAW_DIR / "nhtsa_recalls.jsonl"
CARD_DATASET_PATH = PROCESSED_DIR / "recall_cards.jsonl"
SPLIT_PATHS = {
    "train": PROCESSED_DIR / "train.jsonl",
    "validation": PROCESSED_DIR / "validation.jsonl",
    "test": PROCESSED_DIR / "test.jsonl",
}

MANAGED_DIRECTORIES = (RAW_DIR, PROCESSED_DIR, OUTPUTS_DIR, MODELS_DIR)


def ensure_directories() -> None:
    """Create the project's data and model directories if they are missing.

    Called from each entry point rather than on import: a module that makes
    directories as a side effect of being imported surprises anyone who imports
    it for a constant, and breaks on a read-only filesystem.
    """
    for directory in MANAGED_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Data collection (U.S. DOT open-data portal)
# --------------------------------------------------------------------------- #

# NHTSA's full recalls table, published on the U.S. DOT open-data portal.
RECALLS_RESOURCE_URL = "https://data.transportation.gov/resource/6axg-epim.json"
RECALLS_PAGE_SIZE = 2000

# NHTSA stored notice text in ALL CAPS until 2013 and in normal sentence case
# from 2013 onward; starting at 2013 buys a large corpus of one text style.
RECALL_START_YEAR = 2013

REQUEST_TIMEOUT_SECONDS = 25
REQUEST_RETRIES = 5
REQUEST_BACKOFF_SECONDS = 2.0  # doubled on each retry


# --------------------------------------------------------------------------- #
# Dataset construction
# --------------------------------------------------------------------------- #

# The test split is grouped by manufacturer: every notice from these brands is
# held out, so test scores measure generalisation to unseen manufacturers
# rather than memorisation. The list mixes mass-market cars, a luxury brand, an
# EV-only maker, and a recreational-vehicle maker so the held-out set spans the
# same variety of writing styles as the training set.
HELD_OUT_BRANDS = [
    "SUBARU", "MAZDA", "TESLA", "MITSUBISHI", "VOLVO", "PORSCHE", "WINNEBAGO",
]

VALIDATION_FRACTION = 0.05
RANDOM_SEED = 20260725

# Upper bound on training rows actually read during fine-tuning. The dataset
# itself is ~11.6k cards; training reads a 4k subset because CPU training costs
# 1.6 s/sample and format learning saturates well before the full corpus.
MAX_TRAIN_EXAMPLES: int | None = 4000

# The urgency labels are 97/3 imbalanced, and the first training run proved the
# cost: the model simply never produced the rare, high-stakes levels (0% recall
# on STOP DRIVING against a 73% textual-evidence ceiling). Each rare-class
# example is therefore repeated this many times in the training subset.
RARE_CLASS_OVERSAMPLE_FACTOR = 8


# --------------------------------------------------------------------------- #
# Model + training
# --------------------------------------------------------------------------- #

# SmolLM2-135M-Instruct, chosen after benchmarking three candidates on the
# actual constraint set (a 16 GB M1 Pro to train on, a free CPU host to serve
# on). Qwen2.5-0.5B's 152k vocabulary put an epoch at ~26 h; SmolLM2-360M
# trained acceptably but needed >30 s per card on CPU at inference, which is
# not a usable app. The 135M model trains at 1.6 s/sample on CPU and generates
# a card in ~4 s (33 tok/s) -- and the task is a fixed-format rewrite, which is
# exactly what a small model can learn well from a few thousand examples.
BASE_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
HUB_ADAPTER_REPO = "HanfuZhao781/recallclear-smollm2-135m-lora"
HUB_MERGED_REPO = "HanfuZhao781/recallclear-smollm2-135m"
HUB_SPACE_ID = "HanfuZhao781/recallclear"

MAX_SEQUENCE_LENGTH = 640  # covers the 95th-percentile example; 5% are truncated
MAX_NEW_TOKENS = 320


@dataclass
class LoraSettings:
    """Hyperparameters for the LoRA adapter."""

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )


@dataclass
class TrainingSettings:
    """Hyperparameters for the supervised fine-tuning run."""

    # One pass over 10k examples beats several passes over a subset here, and
    # it is what fits in a laptop training budget (~2.5 h on an M1 Pro).
    epochs: float = 1.0
    learning_rate: float = 2e-4
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    logging_steps: int = 20
    eval_steps: int = 150
    save_steps: int = 150
    max_eval_examples: int = 160  # keep mid-training evaluation cheap
    lr_scheduler_type: str = "cosine"


LORA = LoraSettings()
TRAINING = TrainingSettings()


def resolve_device() -> str:
    """Return the torch device used for training and local inference.

    CUDA when present, otherwise CPU -- including on Apple silicon. That is a
    measured decision, not an oversight: on the development machine the MPS
    backend degraded to 4x slower than CPU under memory pressure (163 s vs
    40 s per optimizer step), because Metal caches buffers per tensor shape
    and a 16 GB unified-memory machine ends up swapping. Set
    RECALLCLEAR_DEVICE=mps to override where MPS is healthy.
    """
    import torch

    override = os.environ.get("RECALLCLEAR_DEVICE")
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# --------------------------------------------------------------------------- #
# Web application
# --------------------------------------------------------------------------- #

# Number of curated held-out notices offered as one-click demos in the UI.
DEMO_EXAMPLE_COUNT = 6

# Live campaign lookup calls the NHTSA API from the server; it can be switched
# off for offline demos or if the upstream API is unavailable.
ENABLE_LIVE_LOOKUP = os.environ.get("RECALLCLEAR_LIVE_LOOKUP", "1") == "1"

# Generation budget for the deployed app. A card is about 130 tokens, so this
# leaves headroom while stopping the untuned model -- which does not emit a stop
# token and would otherwise ramble -- from stalling a CPU host.
APP_MAX_NEW_TOKENS = int(os.environ.get("RECALLCLEAR_MAX_NEW_TOKENS", "220"))

# Where the app loads the model from. Set RECALLCLEAR_ADAPTER_REPO to the Hub
# id when running on a host that has no local training artefacts.
APP_ADAPTER_SOURCE = os.environ.get("RECALLCLEAR_ADAPTER_REPO", str(ADAPTER_DIR))

# Int8 dynamic quantisation roughly halves CPU generation latency. It is a
# no-op where the quantisation engine is missing (PyTorch on Apple silicon),
# so it is safe to leave on by default.
QUANTISE_ON_CPU = os.environ.get("RECALLCLEAR_QUANTISE", "1") == "1"

# Public project links, referenced by the model card and the README.
PROJECT_REPO_URL = os.environ.get(
    "RECALLCLEAR_REPO_URL", "https://github.com/hanfuzhao/Module4-RecallClear"
)

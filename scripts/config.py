"""Central configuration for RecallClear."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


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
    """Create the project's data and model directories if they are missing."""
    for directory in MANAGED_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)


RECALLS_RESOURCE_URL = "https://data.transportation.gov/resource/6axg-epim.json"
RECALLS_PAGE_SIZE = 2000


RECALL_START_YEAR = 2013

REQUEST_TIMEOUT_SECONDS = 25
REQUEST_RETRIES = 5
REQUEST_BACKOFF_SECONDS = 2.0


HELD_OUT_BRANDS = [
    "SUBARU", "MAZDA", "TESLA", "MITSUBISHI", "VOLVO", "PORSCHE", "WINNEBAGO",
]

VALIDATION_FRACTION = 0.05
RANDOM_SEED = 20260725


MAX_TRAIN_EXAMPLES: int | None = 4000


RARE_CLASS_OVERSAMPLE_FACTOR = 8


BASE_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
HUB_ADAPTER_REPO = "HanfuZhao781/recallclear-smollm2-135m-lora"
HUB_MERGED_REPO = "HanfuZhao781/recallclear-smollm2-135m"
HUB_SPACE_ID = "HanfuZhao781/recallclear"

MAX_SEQUENCE_LENGTH = 640
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


    epochs: float = 1.0
    learning_rate: float = 2e-4
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    logging_steps: int = 20
    eval_steps: int = 150
    save_steps: int = 150
    max_eval_examples: int = 160
    lr_scheduler_type: str = "cosine"


LORA = LoraSettings()
TRAINING = TrainingSettings()


def resolve_device() -> str:
    """Return the torch device used for training and local inference."""
    import torch

    override = os.environ.get("RECALLCLEAR_DEVICE")
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEMO_EXAMPLE_COUNT = 6


ENABLE_LIVE_LOOKUP = os.environ.get("RECALLCLEAR_LIVE_LOOKUP", "1") == "1"


APP_MAX_NEW_TOKENS = int(os.environ.get("RECALLCLEAR_MAX_NEW_TOKENS", "220"))


APP_ADAPTER_SOURCE = os.environ.get("RECALLCLEAR_ADAPTER_REPO", str(ADAPTER_DIR))


QUANTISE_ON_CPU = os.environ.get("RECALLCLEAR_QUANTISE", "1") == "1"


PROJECT_REPO_URL = os.environ.get(
    "RECALLCLEAR_REPO_URL", "https://github.com/hanfuzhao/Module4-RecallClear"
)

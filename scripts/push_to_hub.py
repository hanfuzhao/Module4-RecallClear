"""Publish the trained adapter to the Hugging Face Hub.

Pushing the adapter means the deployed app can download ~35 MB of weights at
start-up instead of carrying them in the container image, and it gives the
project a citable model page.

Requires an authenticated Hub session (``hf auth login``).

    python -m scripts.push_to_hub                # adapter only
    python -m scripts.push_to_hub --merged       # also push merged weights
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

from scripts import config

MODEL_CARD_TEMPLATE = """---
base_model: {base_model}
library_name: peft
license: apache-2.0
language:
- en
tags:
- lora
- peft
- text2text-generation
- plain-language
- public-safety
---

# RecallClear — plain-language vehicle recall explainer

A LoRA adapter for `{base_model}` that rewrites official U.S. vehicle safety
recall notices into a five-part summary an owner can act on:

```
WHAT'S WRONG / WHAT COULD HAPPEN / HOW URGENT / WHAT TO DO / WHAT IT COSTS
```

## Training data

{train_examples} recall notices published by NHTSA between 2013 and 2025
([Recalls Data](https://data.transportation.gov/d/6axg-epim), public domain).
Reference cards were derived from the notices' own structured fields using a
curated plain-language lexicon and a four-level urgency ladder anchored on
NHTSA's official `do_not_drive` and `fire_risk_when_parked` flags.

Seven manufacturers ({held_out}) were held out entirely, so evaluation measures
generalisation to unseen brands.

## Intended use and limits

This is a **reading aid**, not safety or legal advice. The official notice
remains authoritative and should always be shown alongside the rewrite.
Do-not-drive and park-outside warnings should be taken from NHTSA's flags, not
from this model: evaluation found the notice text alone often lacks the
evidence needed to recover them.

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{base_model}")
model = PeftModel.from_pretrained(base, "{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{base_model}")
```

Full pipeline, evaluation, and web app: {project_url}
"""


def build_model_card(repo_id: str, train_examples: int) -> str:
    """Render the model card shown on the Hub page."""
    return MODEL_CARD_TEMPLATE.format(
        base_model=config.BASE_MODEL_ID,
        repo_id=repo_id,
        train_examples=f"{train_examples:,}",
        held_out=", ".join(brand.title() for brand in config.HELD_OUT_BRANDS),
        project_url=config.PROJECT_REPO_URL,
    )


def count_training_examples() -> int:
    """Return how many examples the committed adapter was trained on."""
    history_path = config.ADAPTER_DIR / "training_history.json"
    if history_path.exists():
        try:
            return int(json.loads(history_path.read_text(encoding="utf-8"))["train_examples"])
        except (KeyError, ValueError, json.JSONDecodeError):
            pass
    return 0


def push_adapter(repo_id: str = config.HUB_ADAPTER_REPO, private: bool = False) -> str:
    """Upload the LoRA adapter directory and its model card."""
    if not config.ADAPTER_DIR.exists():
        raise SystemExit(f"No adapter at {config.ADAPTER_DIR}. Train one first.")

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=private)

    card_path = config.ADAPTER_DIR / "README.md"
    card_path.write_text(build_model_card(repo_id, count_training_examples()), encoding="utf-8")

    api.upload_folder(
        folder_path=str(config.ADAPTER_DIR),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add RecallClear LoRA adapter",
    )
    url = f"https://huggingface.co/{repo_id}"
    print(f"Adapter pushed to {url}")
    return url


def push_merged(repo_id: str = config.HUB_MERGED_REPO, private: bool = False) -> str:
    """Upload the merged full-weight model, building it first if needed."""
    from scripts.model import merge_adapter

    if not config.MERGED_DIR.exists() or not any(config.MERGED_DIR.iterdir()):
        merge_adapter()

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=private)
    api.upload_folder(
        folder_path=str(config.MERGED_DIR),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add merged RecallClear model",
    )
    url = f"https://huggingface.co/{repo_id}"
    print(f"Merged model pushed to {url}")
    return url


def main() -> None:
    """Command-line entry point for Hub publication."""
    parser = argparse.ArgumentParser(description="Push RecallClear weights to the Hub.")
    parser.add_argument("--adapter-repo", default=config.HUB_ADAPTER_REPO)
    parser.add_argument("--merged-repo", default=config.HUB_MERGED_REPO)
    parser.add_argument("--merged", action="store_true", help="Also push merged full weights.")
    parser.add_argument("--private", action="store_true")
    arguments = parser.parse_args()

    push_adapter(arguments.adapter_repo, private=arguments.private)
    if arguments.merged:
        push_merged(arguments.merged_repo, private=arguments.private)


if __name__ == "__main__":
    main()

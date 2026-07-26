"""Deploy the RecallClear app to a Hugging Face Space (Docker SDK).

Uploads only what the container needs: the app, the adapter, the held-out demo
notices, and the evaluation summary. Raw data, checkpoints, and the training
corpus stay out of the image.

Requires an authenticated Hub session (``hf auth login``).

    python -m scripts.deploy_space
    python -m scripts.deploy_space --space-id someone/recallclear
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

from scripts import config

# Space front-matter. `sdk: docker` tells Spaces to build our Dockerfile, and
# `app_port` must match the port gunicorn binds inside the container.
SPACE_CARD = """---
title: RecallClear
emoji: 🚗
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
short_description: Vehicle safety recall notices, rewritten in plain language
---

# RecallClear

Rewrites official U.S. vehicle safety recall notices into a five-part summary a
car owner can act on, using a LoRA fine-tune of
[`{base_model}`]({base_model_url}) trained on {train_examples} real NHTSA
notices.

Paste a notice, or enter the campaign number from your recall letter (for
example `23V123000`) to fetch the live text from NHTSA.

**This is a reading aid, not safety or legal advice.** The official notice is
authoritative and is always shown next to the rewrite. Do-not-drive and
park-outside warnings come from NHTSA's own flags, not from the model.

- Code and evaluation: [{repo_url}]({repo_url})
- Adapter: [`{adapter_repo}`](https://huggingface.co/{adapter_repo})
- Data: [NHTSA Recalls Data](https://data.transportation.gov/d/6axg-epim) (public domain)
"""

# Files copied into the Space repository, relative to the project root.
OPTIONAL_DEPLOY_FILES = (
    "data/outputs/evaluation.json",
)

DEPLOY_FILES = (
    "Dockerfile",
    "requirements-deploy.txt",
    "main.py",
    "scripts/__init__.py",
    "scripts/config.py",
    "scripts/app_service.py",
    "scripts/model.py",
    "scripts/plain_language.py",
    "scripts/prompts.py",
    "scripts/recall_lookup.py",
    "scripts/make_dataset.py",
    "templates/index.html",
    "static/css/styles.css",
    "static/js/app.js",
    "data/processed/test.jsonl",
)

DEPLOY_DIRECTORIES = ("models/adapter",)


def build_space_card(train_examples: int, adapter_repo: str) -> str:
    """Render the Space's README, which doubles as its configuration."""
    return SPACE_CARD.format(
        base_model=config.BASE_MODEL_ID,
        base_model_url=f"https://huggingface.co/{config.BASE_MODEL_ID}",
        train_examples=f"{train_examples:,}",
        repo_url=config.PROJECT_REPO_URL,
        adapter_repo=adapter_repo,
    )


def _stage_files(staging: Path) -> list[str]:
    """Copy the deployable files into a staging directory; return what was skipped."""
    import shutil

    missing: list[str] = []
    for relative in DEPLOY_FILES + OPTIONAL_DEPLOY_FILES:
        source = config.PROJECT_ROOT / relative
        if not source.exists():
            if relative not in OPTIONAL_DEPLOY_FILES:
                missing.append(relative)
            continue
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    for relative in DEPLOY_DIRECTORIES:
        source = config.PROJECT_ROOT / relative
        if not source.exists():
            missing.append(relative)
            continue
        shutil.copytree(source, staging / relative, dirs_exist_ok=True)

    return missing


def deploy(space_id: str, private: bool = False) -> str:
    """Create or update the Space and upload the application files."""
    from scripts.push_to_hub import count_training_examples

    api = HfApi()
    api.create_repo(space_id, repo_type="space", space_sdk="docker", exist_ok=True, private=private)

    with tempfile.TemporaryDirectory() as directory:
        staging = Path(directory)
        missing = _stage_files(staging)
        if missing:
            raise SystemExit(
                "Cannot deploy, these files are missing:\n  " + "\n  ".join(missing)
            )

        (staging / "README.md").write_text(
            build_space_card(count_training_examples(), config.HUB_ADAPTER_REPO), encoding="utf-8"
        )

        api.upload_folder(
            folder_path=str(staging),
            repo_id=space_id,
            repo_type="space",
            commit_message="Deploy RecallClear",
        )

    url = f"https://huggingface.co/spaces/{space_id}"
    print(f"Space deployed: {url}")
    print(f"Direct app URL: https://{space_id.replace('/', '-').lower()}.hf.space")
    return url


def main() -> None:
    """Command-line entry point for deployment."""
    parser = argparse.ArgumentParser(description="Deploy RecallClear to a Hugging Face Space.")
    parser.add_argument("--space-id", default=config.HUB_SPACE_ID)
    parser.add_argument("--private", action="store_true")
    arguments = parser.parse_args()
    deploy(arguments.space_id, private=arguments.private)


if __name__ == "__main__":
    main()

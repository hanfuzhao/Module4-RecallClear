"""One-command project setup: download data, build features, train, evaluate.

Each stage is skipped when its output already exists, so the script can be
re-run safely and used to resume a partially completed setup.

    python setup.py                 # run whatever is still missing
    python setup.py --force         # rebuild everything from scratch
    python setup.py --stages data features
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from scripts import config

STAGE_ORDER = ("data", "features", "train", "evaluate")


def _stage_heading(name: str, index: int, total: int) -> None:
    """Print a consistent banner for each pipeline stage."""
    print(f"\n{'=' * 72}\n[{index}/{total}] {name}\n{'=' * 72}", flush=True)


def _is_complete(path: Path, minimum_bytes: int = 1) -> bool:
    """Whether a stage output exists and is not empty."""
    return path.exists() and path.stat().st_size >= minimum_bytes


def run_data_stage(force: bool) -> None:
    """Download the NHTSA recall corpus."""
    from scripts.make_dataset import collect_recalls, write_jsonl

    if _is_complete(config.RAW_RECALLS_PATH, 1000) and not force:
        print(f"Raw notices already present at {config.RAW_RECALLS_PATH}; skipping.")
        return
    records = collect_recalls()
    write_jsonl(records, config.RAW_RECALLS_PATH)
    print(f"Saved {len(records)} notices.")


def run_features_stage(force: bool) -> None:
    """Build the train / validation / test splits."""
    from scripts.build_features import build_examples, split_examples, summarise
    from scripts.make_dataset import read_jsonl, write_jsonl
    import json

    if all(_is_complete(path, 1000) for path in config.SPLIT_PATHS.values()) and not force:
        print("Splits already built; skipping.")
        return

    records = read_jsonl(config.RAW_RECALLS_PATH)
    examples = build_examples(records)
    splits = split_examples(examples)
    for name, rows in splits.items():
        write_jsonl(rows, config.SPLIT_PATHS[name])
        print(f"  {name:11s} {len(rows):6d}")
    write_jsonl(examples, config.CARD_DATASET_PATH)

    stats = summarise(splits)
    stats["held_out_brands"] = config.HELD_OUT_BRANDS
    (config.OUTPUTS_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


def run_train_stage(force: bool) -> None:
    """Fine-tune the LoRA adapter."""
    from scripts.make_dataset import read_jsonl
    from scripts.model import train_lora

    adapter_weights = config.ADAPTER_DIR / "adapter_model.safetensors"
    if _is_complete(adapter_weights) and not force:
        print(f"Adapter already trained at {config.ADAPTER_DIR}; skipping.")
        return

    train_rows = read_jsonl(config.SPLIT_PATHS["train"])
    validation_rows = read_jsonl(config.SPLIT_PATHS["validation"])
    train_lora(train_rows, validation_rows)


def run_evaluate_stage(force: bool) -> None:
    """Score the adapter against the untuned baselines."""
    import json

    from scripts.evaluate_model import evaluate

    output_path = config.OUTPUTS_DIR / "evaluation.json"
    if _is_complete(output_path) and not force:
        print(f"Evaluation already present at {output_path}; skipping.")
        return

    results = evaluate(sample_size=150)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


STAGES = {
    "data": ("Download NHTSA recall notices", run_data_stage),
    "features": ("Build plain-language training data", run_features_stage),
    "train": ("Fine-tune the LoRA adapter", run_train_stage),
    "evaluate": ("Evaluate against untuned baselines", run_evaluate_stage),
}


def parse_arguments() -> argparse.Namespace:
    """Parse the setup script's command-line options."""
    parser = argparse.ArgumentParser(description="Set up the RecallClear project end to end.")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGE_ORDER,
        default=list(STAGE_ORDER),
        help="Which stages to run, in order.",
    )
    parser.add_argument("--force", action="store_true", help="Re-run stages even if output exists.")
    return parser.parse_args()


def main() -> int:
    """Run the requested pipeline stages and report timings."""
    arguments = parse_arguments()
    config.ensure_directories()
    selected = [stage for stage in STAGE_ORDER if stage in arguments.stages]

    for index, stage in enumerate(selected, start=1):
        description, runner = STAGES[stage]
        _stage_heading(description, index, len(selected))
        started = time.time()
        try:
            runner(arguments.force)
        except Exception as error:  # surface the failing stage clearly
            print(f"\nStage '{stage}' failed: {error}", file=sys.stderr)
            return 1
        print(f"-- {stage} finished in {time.time() - started:.1f}s")

    print("\nSetup complete. Start the app with:  python main.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

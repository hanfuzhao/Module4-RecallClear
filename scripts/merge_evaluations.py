"""Merge the v2 tuned-only evaluation into the full three-system results file.

The base and few-shot baselines use the untuned model, which did not change
between adapter versions, and both evaluations run on the identical stratified
sample (fixed seed). Re-running the baselines would cost ~25 minutes to compute
byte-identical numbers, so the v2 run scored only the tuned system and this
script splices it into the v1 results file.

    python -m scripts.merge_evaluations
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts import config


def merge(full_path: Path, tuned_path: Path, output_path: Path) -> dict:
    """Splice the tuned system from ``tuned_path`` into ``full_path`` results."""
    full = json.loads(full_path.read_text(encoding="utf-8"))
    tuned_only = json.loads(tuned_path.read_text(encoding="utf-8"))

    sample_check = (full["test_examples_total"], tuned_only["test_examples_total"])
    if sample_check[0] != sample_check[1]:
        raise SystemExit(f"Sample mismatch between runs: {sample_check}")

    full["systems"]["tuned"] = tuned_only["systems"]["tuned"]
    full["adapter_note"] = (
        "base and few_shot were measured with adapter v1 loaded but disabled "
        "(identical to the untuned model); tuned reflects adapter v2, trained "
        "with rare-class oversampling."
    )
    output_path.write_text(json.dumps(full, indent=2), encoding="utf-8")
    return full


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Merge tuned-only results into the full file.")
    parser.add_argument("--full", type=Path, default=config.OUTPUTS_DIR / "evaluation.json")
    parser.add_argument("--tuned", type=Path, default=config.OUTPUTS_DIR / "evaluation_v2_tuned.json")
    parser.add_argument("--output", type=Path, default=config.OUTPUTS_DIR / "evaluation.json")
    args = parser.parse_args()

    merged = merge(args.full, args.tuned, args.output)
    tuned = merged["systems"]["tuned"]
    print(f"tuned v2: format {tuned['format_compliance']:.0%} | "
          f"macro-F1 {tuned['urgency_macro_f1']:.3f} | grade {tuned['grade_level']}")
    print(f"merged results written to {args.output}")


if __name__ == "__main__":
    main()

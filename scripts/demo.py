"""Print a before / after comparison for a single held-out recall notice.

This is the terminal version of what the web app shows, and the quickest way to
sanity-check an adapter after training:

    python -m scripts.demo                       # first held-out example
    python -m scripts.demo --campaign 23V123000  # a live notice from NHTSA
    python -m scripts.demo --index 4 --modes base tuned
"""

from __future__ import annotations

import argparse
import textwrap

from scripts import config
from scripts.app_service import DemoLibrary
from scripts.model import RecallExplainer
from scripts.plain_language import flesch_kincaid_grade, format_notice, jargon_rate
from scripts.recall_lookup import CampaignNotFoundError, fetch_campaign

RULE = "=" * 78


def _print_block(title: str, body: str) -> None:
    """Print a titled, wrapped block of text."""
    print(f"\n{title}\n{'-' * len(title)}")
    for paragraph in body.split("\n"):
        print(textwrap.fill(paragraph, width=78) if paragraph else "")


def _print_metrics(label: str, text: str, seconds: float | None = None) -> None:
    """Print the readability measurements for one piece of text."""
    timing = f" | {seconds:.1f}s" if seconds is not None else ""
    print(
        f"  [{label}] reading grade {flesch_kincaid_grade(text):.1f} | "
        f"jargon/100 words {jargon_rate(text):.2f} | {len(text.split())} words{timing}"
    )


def resolve_notice(arguments: argparse.Namespace) -> tuple[str, str]:
    """Return ``(title, notice_text)`` for whichever source the user chose."""
    if arguments.campaign:
        record = fetch_campaign(arguments.campaign)
        title = f"{record['campaign_number']} — {record['manufacturer']}"
        return title, format_notice(record)

    examples = DemoLibrary(count=max(arguments.index + 1, 1)).examples()
    if not examples:
        raise SystemExit("No held-out examples found. Run `make features` first.")
    example = examples[min(arguments.index, len(examples) - 1)]
    return f"{example['campaign_number']} — {example['manufacturer']}", example["notice"]


def main() -> None:
    """Command-line entry point for the demo."""
    parser = argparse.ArgumentParser(description="Show a before/after recall rewrite.")
    parser.add_argument("--index", type=int, default=0, help="Which held-out example to use.")
    parser.add_argument("--campaign", type=str, default=None, help="NHTSA campaign number to fetch live.")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=[RecallExplainer.MODE_BASE, RecallExplainer.MODE_TUNED],
        choices=list(RecallExplainer.MODES),
    )
    arguments = parser.parse_args()

    try:
        title, notice = resolve_notice(arguments)
    except (CampaignNotFoundError, ValueError) as error:
        raise SystemExit(f"Could not load that notice: {error}") from error

    print(RULE)
    print(f"RecallClear demo — {title}")
    print(RULE)
    _print_block("ORIGINAL NOTICE (what the owner receives)", notice)
    _print_metrics("original", notice)

    explainer = RecallExplainer()
    for mode in arguments.modes:
        import time

        started = time.time()
        output = explainer.explain(notice, mode=mode, max_new_tokens=config.MAX_NEW_TOKENS)
        elapsed = time.time() - started

        heading = {
            RecallExplainer.MODE_BASE: "BEFORE — base model, no adapter",
            RecallExplainer.MODE_FEW_SHOT: "BASELINE — base model with two examples in the prompt",
            RecallExplainer.MODE_TUNED: "AFTER — LoRA fine-tuned",
        }[mode]
        _print_block(heading, output)
        _print_metrics(mode, output, elapsed)

    print()


if __name__ == "__main__":
    main()

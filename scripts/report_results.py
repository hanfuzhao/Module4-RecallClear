"""Render evaluation results into the README.

python -m scripts.report_results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts import config
from scripts.plain_language import URGENCY_PARK_OUTSIDE, URGENCY_STOP_DRIVING

START_MARKER = "<!-- RESULTS:START -->"
END_MARKER = "<!-- RESULTS:END -->"

SYSTEM_LABELS = {
    "base": "base (zero-shot)",
    "few_shot": "base + 2 examples",
    "tuned": "**fine-tuned**",
}


METRIC_ROWS = (
    ("format_compliance", "Card format produced correctly", "percent"),
    ("urgency_macro_f1", "Urgency triage (macro-F1)", "ratio"),
    ("grade_level", "Reading grade level (lower is better)", "number"),
    ("jargon_rate", "Jargon per 100 words (lower is better)", "number"),
    ("free_repair_stated", "States the repair is free", "percent"),
    ("phone_hallucination", "Invented a phone number (lower is better)", "percent"),
    ("grounding", "Grounding in the source notice", "ratio"),
    ("mean_output_words", "Output length (words)", "number"),
    ("mean_prompt_tokens", "Prompt tokens per request", "number"),
    ("seconds_per_card", "Seconds per card", "number"),
)


def _format(value: object, style: str) -> str:
    """Render one metric value for the results table."""
    if value is None:
        return "—"
    if style == "percent":
        return f"{float(value) * 100:.0f}%"
    if style == "ratio":
        return f"{float(value):.2f}"
    return f"{float(value):g}"


def build_results_table(results: dict) -> str:
    """Build the markdown block inserted between the README result markers."""
    systems = results.get("systems", {})
    present = [name for name in ("base", "few_shot", "tuned") if name in systems]
    if not present:
        return "_No evaluation results yet — run `make evaluate`._"

    lines: list[str] = []
    header = "| | " + " | ".join(SYSTEM_LABELS[name] for name in present) + " |"
    lines.append(header)
    lines.append("|---|" + "---|" * len(present))

    for key, label, style in METRIC_ROWS:
        cells = [_format(systems[name].get(key), style) for name in present]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    reference = results.get("reference_cards", {})
    if reference:
        lines.append("")
        lines.append(
            f"The original notices score **{reference.get('notice_grade_level')}** on reading "
            f"grade and **{reference.get('notice_jargon_rate')}** on jargon; the rule-built "
            f"reference cards score **{reference.get('grade_level')}** and "
            f"**{reference.get('jargon_rate')}**."
        )

    lines.append("")
    lines.append("### Rare, high-stakes classes")
    lines.append("")
    ceiling = results.get("label_ceiling", {})
    lines.append("| | support | notices with explicit warning text | best achievable recall |")
    lines.append("|---|---|---|---|")
    for label in (URGENCY_STOP_DRIVING, URGENCY_PARK_OUTSIDE):
        entry = ceiling.get(label, {})
        best = entry.get("max_achievable_recall")
        lines.append(
            f"| {label} | {entry.get('support', 0)} | "
            f"{entry.get('notices_with_explicit_warning', 0)} | "
            f"{'—' if best is None else f'{float(best) * 100:.0f}%'} |"
        )

    lines.append("")
    lines.append("| | " + " | ".join(SYSTEM_LABELS[name] for name in present) + " |")
    lines.append("|---|" + "---|" * len(present))
    for label in (URGENCY_STOP_DRIVING, URGENCY_PARK_OUTSIDE):
        cells = []
        for name in present:
            entry = systems[name].get("urgency_per_class", {}).get(label, {})
            recall = entry.get("recall")
            cells.append("—" if recall is None else f"{float(recall) * 100:.0f}%")
        lines.append(f"| recall, {label} | " + " | ".join(cells) + " |")

    downgrade_cells = []
    for name in present:
        confusion = systems[name].get("urgency_confusion", {})
        rate = confusion.get("safety_downgrade_rate")
        downgrade_cells.append("—" if rate is None else f"{float(rate) * 100:.0f}%")
    lines.append("| safety downgrades (lower is better) | " + " | ".join(downgrade_cells) + " |")

    return "\n".join(lines)


def update_readme(results: dict, readme_path: Path) -> bool:
    """Replace the marked results block in the README. Returns True if changed."""
    text = readme_path.read_text(encoding="utf-8")
    if START_MARKER not in text or END_MARKER not in text:
        raise SystemExit(f"{readme_path} is missing the {START_MARKER} / {END_MARKER} markers.")

    head, _, remainder = text.partition(START_MARKER)
    _, _, tail = remainder.partition(END_MARKER)
    updated = f"{head}{START_MARKER}\n\n{build_results_table(results)}\n\n{END_MARKER}{tail}"

    if updated == text:
        return False
    readme_path.write_text(updated, encoding="utf-8")
    return True


def main() -> None:
    """Command-line entry point for results reporting."""
    parser = argparse.ArgumentParser(description="Write evaluation results into the README.")
    parser.add_argument("--results", type=Path, default=config.OUTPUTS_DIR / "evaluation.json")
    parser.add_argument("--readme", type=Path, default=config.PROJECT_ROOT / "README.md")
    parser.add_argument("--print-only", action="store_true")
    arguments = parser.parse_args()

    if not arguments.results.exists():
        raise SystemExit(f"No results at {arguments.results}. Run `make evaluate` first.")
    results = json.loads(arguments.results.read_text(encoding="utf-8"))

    if arguments.print_only:
        print(build_results_table(results))
        return

    changed = update_readme(results, arguments.readme)
    print("README updated." if changed else "README already up to date.")


if __name__ == "__main__":
    main()

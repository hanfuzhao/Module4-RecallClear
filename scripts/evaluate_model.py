"""Evaluate the fine-tuned model against two untuned baselines.

Three systems are compared on the held-out brands:

* ``base``      -- Qwen2.5-0.5B-Instruct, zero-shot, same instruction.
* ``few_shot``  -- the same untuned model with two worked examples prepended.
                   This is the honest baseline: prompting alone can teach the
                   format, so the fine-tune has to beat it on more than layout.
* ``tuned``     -- the same model with the LoRA adapter enabled.

Metrics
-------
format_compliance   all five sections present, in the required order
urgency_accuracy    exact match against NHTSA's gold urgency label
urgency_macro_f1    macro-F1, which is what matters under 97/3 class imbalance
grade_level         Flesch-Kincaid reading grade of the generated card
jargon_rate         technical terms per 100 words
free_repair_stated  says the repair costs the owner nothing (always true of recalls)
phone_hallucination share of cards inventing a phone number absent from the notice
grounding           share of card content words that appear in the source notice
prompt_tokens       average prompt cost, the efficiency argument for tuning

Run directly:
    python -m scripts.evaluate_model --sample 150
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

from scripts import config
from scripts.make_dataset import read_jsonl, write_jsonl
from scripts.model import RecallExplainer
from scripts.plain_language import (
    CARD_SECTIONS,
    URGENCY_LEVELS,
    URGENCY_PARK_OUTSIDE,
    URGENCY_STOP_DRIVING,
    extract_urgency,
    flesch_kincaid_grade,
    jargon_rate,
    parse_card,
)

_PHONE = re.compile(r"\b1?[-\s]?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}\b")
_WORD = re.compile(r"[a-z][a-z'-]{3,}")
_FREE_REPAIR = re.compile(r"\bfree\b|\bno cost\b|\bnothing\b|\bno charge\b|\$0", re.IGNORECASE)

# Wording that shows the notice itself carries NHTSA's stop-driving or
# park-outside advice. Used to measure the ceiling on urgency recall.
_WARNING_LANGUAGE = re.compile(
    r"do not drive|not to drive|stop driving|park (?:it )?outside|"
    r"away from (?:structures|buildings|homes)|do not park",
    re.IGNORECASE,
)

STOPWORDS = {
    "this", "that", "with", "from", "your", "will", "have", "they", "them", "their",
    "than", "then", "there", "these", "those", "been", "being", "were", "what",
    "when", "which", "while", "would", "could", "should", "about", "into", "more",
    "some", "such", "only", "also", "very", "most", "must", "make", "made", "does",
    "cost", "costs", "free", "call", "dealer", "dealers", "recall", "repair",
    "owner", "owners", "vehicle", "vehicles", "car", "cars",
}


# --------------------------------------------------------------------------- #
# Per-card metrics
# --------------------------------------------------------------------------- #


def has_valid_format(card_text: str) -> bool:
    """True when all five sections are present in the required order."""
    sections = parse_card(card_text)
    if set(sections) != set(CARD_SECTIONS):
        return False
    positions = [card_text.upper().find(section) for section in CARD_SECTIONS]
    return all(position >= 0 for position in positions) and positions == sorted(positions)


def states_free_repair(card_text: str) -> bool:
    """True when the card tells the owner the recall repair costs nothing."""
    section = parse_card(card_text).get("WHAT IT COSTS", "")
    return bool(_FREE_REPAIR.search(section))


def hallucinated_phone(card_text: str, notice: str) -> bool:
    """True when the card states a phone number that the notice does not contain."""
    notice_digits = {re.sub(r"\D", "", match) for match in _PHONE.findall(notice)}
    for match in _PHONE.findall(card_text):
        if re.sub(r"\D", "", match) not in notice_digits:
            return True
    return False


def grounding_score(card_text: str, notice: str) -> float:
    """Share of the card's content words that also appear in the source notice.

    A crude but useful faithfulness proxy: a card built from the notice scores
    high, a card that drifts into invented detail scores low. Boilerplate the
    card is *supposed* to add (dealer, free, recall) is excluded via STOPWORDS.
    """
    card_words = {word for word in _WORD.findall(card_text.lower()) if word not in STOPWORDS}
    if not card_words:
        return 0.0
    notice_words = set(_WORD.findall(notice.lower()))
    return round(len(card_words & notice_words) / len(card_words), 4)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def macro_f1(gold: list[str], predicted: list[str | None], labels: tuple[str, ...]) -> float:
    """Macro-averaged F1 over the given labels, counting misses as wrong."""
    scores = []
    for label in labels:
        true_positive = sum(1 for g, p in zip(gold, predicted) if g == label and p == label)
        false_positive = sum(1 for g, p in zip(gold, predicted) if g != label and p == label)
        false_negative = sum(1 for g, p in zip(gold, predicted) if g == label and p != label)
        if true_positive == 0:
            scores.append(0.0)
            continue
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        scores.append(2 * precision * recall / (precision + recall))
    return round(sum(scores) / len(scores), 4)


def per_class_recall(gold: list[str], predicted: list[str | None]) -> dict[str, dict]:
    """Recall for each urgency level, with support counts."""
    report: dict[str, dict] = {}
    for label in URGENCY_LEVELS:
        support = sum(1 for value in gold if value == label)
        hits = sum(1 for g, p in zip(gold, predicted) if g == label and p == label)
        report[label] = {
            "support": support,
            "recall": round(hits / support, 4) if support else None,
        }
    return report


def urgency_confusion(gold: list[str], predicted: list[str | None]) -> dict:
    """Report the confusion pairs, and count safety-relevant downgrades.

    Averaged scores hide the one error that carries a real-world cost: calling a
    notice that NHTSA flagged as do-not-drive or park-outside something less
    urgent. That count is surfaced on its own.
    """
    severity = {level: index for index, level in enumerate(URGENCY_LEVELS)}
    pairs = Counter(
        (gold_label, predicted_label or "NOT STATED")
        for gold_label, predicted_label in zip(gold, predicted)
    )

    downgrades = 0
    for gold_label, predicted_label in zip(gold, predicted):
        if gold_label not in (URGENCY_STOP_DRIVING, URGENCY_PARK_OUTSIDE):
            continue
        # A missing prediction is treated as a downgrade: no warning was given.
        if predicted_label is None or severity[predicted_label] > severity[gold_label]:
            downgrades += 1

    high_stakes = sum(
        1 for label in gold if label in (URGENCY_STOP_DRIVING, URGENCY_PARK_OUTSIDE)
    )
    return {
        "pairs": {f"{gold_label} -> {predicted_label}": count for (gold_label, predicted_label), count in pairs.most_common()},
        "safety_downgrades": downgrades,
        "high_stakes_examples": high_stakes,
        "safety_downgrade_rate": round(downgrades / high_stakes, 4) if high_stakes else None,
    }


def score_generations(rows: list[dict], generations: list[str], elapsed: float) -> dict:
    """Compute the full metric set for one system's generations."""
    gold_urgency = [row["urgency"] for row in rows]
    predicted_urgency = [extract_urgency(text) for text in generations]

    valid_format = [has_valid_format(text) for text in generations]
    count = len(rows)

    return {
        "examples": count,
        "format_compliance": round(sum(valid_format) / count, 4),
        "urgency_stated": round(sum(1 for value in predicted_urgency if value) / count, 4),
        "urgency_accuracy": round(
            sum(1 for g, p in zip(gold_urgency, predicted_urgency) if g == p) / count, 4
        ),
        "urgency_macro_f1": macro_f1(gold_urgency, predicted_urgency, URGENCY_LEVELS),
        "urgency_per_class": per_class_recall(gold_urgency, predicted_urgency),
        "urgency_confusion": urgency_confusion(gold_urgency, predicted_urgency),
        "grade_level": round(sum(flesch_kincaid_grade(text) for text in generations) / count, 2),
        "jargon_rate": round(sum(jargon_rate(text) for text in generations) / count, 3),
        "free_repair_stated": round(sum(states_free_repair(text) for text in generations) / count, 4),
        "phone_hallucination": round(
            sum(hallucinated_phone(text, row["notice"]) for text, row in zip(generations, rows)) / count,
            4,
        ),
        "grounding": round(
            sum(grounding_score(text, row["notice"]) for text, row in zip(generations, rows)) / count, 4
        ),
        "mean_output_words": round(sum(len(text.split()) for text in generations) / count, 1),
        "seconds_per_card": round(elapsed / count, 2),
    }


def label_ceiling(rows: list[dict]) -> dict:
    """Measure how often the gold urgency is even inferable from the notice text.

    NHTSA's do-not-drive and park-outside flags are database metadata, not
    always words in the notice. Where the notice carries no warning language, no
    text-only model can recover the label; this reports that ceiling so the
    urgency numbers are read honestly.
    """
    report = {}
    for label in (URGENCY_STOP_DRIVING, URGENCY_PARK_OUTSIDE):
        subset = [row for row in rows if row["urgency"] == label]
        with_evidence = sum(1 for row in subset if _WARNING_LANGUAGE.search(row["notice"]))
        report[label] = {
            "support": len(subset),
            "notices_with_explicit_warning": with_evidence,
            "max_achievable_recall": round(with_evidence / len(subset), 4) if subset else None,
        }
    return report


# --------------------------------------------------------------------------- #
# Sampling and orchestration
# --------------------------------------------------------------------------- #


def stratified_sample(rows: list[dict], sample_size: int, seed: int = config.RANDOM_SEED) -> list[dict]:
    """Sample the test set while keeping every rare urgency class represented.

    A uniform sample of 150 from a 97 %-majority test set would contain almost
    no do-not-drive cases, so the rare classes are taken in full first.
    """
    rng = random.Random(seed)
    rare = [row for row in rows if row["urgency"] in (URGENCY_STOP_DRIVING, URGENCY_PARK_OUTSIDE)]
    common = [row for row in rows if row not in rare]

    rng.shuffle(common)
    remaining = max(sample_size - len(rare), 0)
    selected = rare + common[:remaining]
    rng.shuffle(selected)
    return selected


def evaluate(
    sample_size: int = 150,
    modes: tuple[str, ...] = RecallExplainer.MODES,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
    batch_size: int = 8,
) -> dict:
    """Run every system over a sample of the held-out test set."""
    test_rows = read_jsonl(config.SPLIT_PATHS["test"])
    rows = stratified_sample(test_rows, sample_size)
    notices = [row["notice"] for row in rows]
    print(f"Evaluating {len(rows)} held-out notices "
          f"({Counter(row['urgency'] for row in rows)})")

    explainer = RecallExplainer()
    results: dict = {
        "base_model": config.BASE_MODEL_ID,
        "adapter": str(config.ADAPTER_DIR),
        "test_examples_total": len(test_rows),
        "held_out_brands": config.HELD_OUT_BRANDS,
        "label_ceiling": label_ceiling(rows),
        "systems": {},
    }

    generations_by_mode: dict[str, list[str]] = {}
    for mode in modes:
        print(f"  generating with mode={mode} ...", flush=True)
        started = time.time()
        generations = explainer.explain_batch(
            notices, mode=mode, max_new_tokens=max_new_tokens, batch_size=batch_size
        )
        elapsed = time.time() - started
        generations_by_mode[mode] = generations

        scores = score_generations(rows, generations, elapsed)
        scores["mean_prompt_tokens"] = round(
            sum(explainer.prompt_token_count(notice, mode) for notice in notices) / len(notices), 1
        )
        results["systems"][mode] = scores
        print(f"    format {scores['format_compliance']:.2%} | "
              f"urgency macro-F1 {scores['urgency_macro_f1']:.3f} | "
              f"grade {scores['grade_level']}", flush=True)

    results["reference_cards"] = {
        "grade_level": round(sum(flesch_kincaid_grade(row["card_text"]) for row in rows) / len(rows), 2),
        "jargon_rate": round(sum(jargon_rate(row["card_text"]) for row in rows) / len(rows), 3),
        "notice_grade_level": round(sum(flesch_kincaid_grade(row["notice"]) for row in rows) / len(rows), 2),
        "notice_jargon_rate": round(sum(jargon_rate(row["notice"]) for row in rows) / len(rows), 3),
    }

    comparisons = [
        {
            "campaign_number": row["campaign_number"],
            "manufacturer": row["manufacturer"],
            "subject": row["subject"],
            "gold_urgency": row["urgency"],
            "notice": row["notice"],
            "reference_card": row["card_text"],
            **{f"output_{mode}": generations_by_mode[mode][index] for mode in modes},
        }
        for index, row in enumerate(rows)
    ]
    write_jsonl(comparisons, config.OUTPUTS_DIR / "before_after.jsonl")

    return results


def main() -> None:
    """Command-line entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate RecallClear against baselines.")
    parser.add_argument("--sample", type=int, default=150, help="Held-out notices to score.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=config.MAX_NEW_TOKENS)
    parser.add_argument("--output", type=Path, default=config.OUTPUTS_DIR / "evaluation.json")
    args = parser.parse_args()
    config.ensure_directories()

    results = evaluate(
        sample_size=args.sample, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size
    )
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["systems"], indent=2))
    print(f"\nSaved metrics to {args.output}")


if __name__ == "__main__":
    main()

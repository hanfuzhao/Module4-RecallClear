"""Turn raw NHTSA recall notices into supervised plain-language training data.

Pipeline: raw JSONL -> reference card -> quality gate -> de-duplication ->
brand-grouped train / validation / test splits.

The test split is grouped by manufacturer (see ``config.HELD_OUT_BRANDS``): the
model never sees a single notice from those brands during training, so test
scores measure generalisation rather than memorisation.

Run directly:
    python -m scripts.build_features
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from scripts import config
from scripts.make_dataset import read_jsonl, write_jsonl
from scripts.plain_language import (
    build_card,
    flesch_kincaid_grade,
    format_notice,
    jargon_rate,
)
from scripts.prompts import build_training_messages

# Corporate suffixes stripped when reducing a manufacturer string to a brand.
_LEGAL_SUFFIX = re.compile(
    r"\b(inc|llc|ltd|corp|corporation|company|co|gmbh|ag|sa|nv|plc|lp|holdings?|"
    r"group|usa|u\.s\.a|america|north|american|of|motors?|motor|automotive|"
    r"vehicles?|industries|manufacturing|mfg)\b\.?",
    re.IGNORECASE,
)


def brand_key(manufacturer: str) -> str:
    """Reduce a manufacturer string to a coarse brand key.

    ``"Subaru of America, Inc."`` and ``"Subaru Corporation"`` both become
    ``"SUBARU"``, which is what the held-out split is grouped on.
    """
    name = re.sub(r"\([^)]*\)", " ", manufacturer or "")
    name = _LEGAL_SUFFIX.sub(" ", name)
    name = re.sub(r"[^A-Za-z0-9 ]+", " ", name)
    tokens = [token for token in name.upper().split() if token]
    return tokens[0] if tokens else "UNKNOWN"


def is_held_out(manufacturer: str) -> bool:
    """True when a manufacturer belongs to the held-out test brands."""
    key = brand_key(manufacturer)
    return any(brand in key or key in brand for brand in config.HELD_OUT_BRANDS)


def _content_fingerprint(record: dict) -> str:
    """Hash the defect text so re-issued campaigns are not counted twice."""
    blob = f"{record.get('summary', '')}|{record.get('consequence', '')}"
    normalised = re.sub(r"\W+", " ", blob.lower()).strip()
    return hashlib.sha1(normalised.encode("utf-8")).hexdigest()


def build_examples(records: list[dict]) -> list[dict]:
    """Build one training example per usable recall notice."""
    examples: list[dict] = []
    seen_fingerprints: set[str] = set()
    rejected_gate = 0
    rejected_duplicate = 0

    for record in records:
        fingerprint = _content_fingerprint(record)
        if fingerprint in seen_fingerprints:
            rejected_duplicate += 1
            continue

        card = build_card(record)
        if card is None:
            rejected_gate += 1
            continue

        seen_fingerprints.add(fingerprint)
        notice = format_notice(record)
        examples.append(
            {
                "campaign_number": record.get("campaign_number"),
                "manufacturer": record.get("manufacturer"),
                "brand": brand_key(record.get("manufacturer", "")),
                "subject": record.get("subject"),
                "component": record.get("component"),
                "units_affected": record.get("units_affected"),
                "park_it": bool(record.get("park_it")),
                "park_outside": bool(record.get("park_outside")),
                "notice": notice,
                "card_text": card.to_text(),
                "card": card.to_dict(),
                "urgency": card.urgency,
                "messages": build_training_messages(notice, card.to_text()),
            }
        )

    print(f"  quality gate rejected {rejected_gate} notices (defect or consequence too short)")
    print(f"  de-duplication removed {rejected_duplicate} repeated campaigns")
    return examples


def split_examples(examples: list[dict]) -> dict[str, list[dict]]:
    """Split into train / validation / test, holding the test brands out entirely."""
    test = [row for row in examples if is_held_out(row["manufacturer"] or "")]
    trainable = [row for row in examples if not is_held_out(row["manufacturer"] or "")]

    rng = random.Random(config.RANDOM_SEED)
    rng.shuffle(trainable)
    validation_size = max(1, int(len(trainable) * config.VALIDATION_FRACTION))
    validation = trainable[:validation_size]
    train = trainable[validation_size:]
    if config.MAX_TRAIN_EXAMPLES:
        train = train[: config.MAX_TRAIN_EXAMPLES]

    return {"train": train, "validation": validation, "test": test}


def summarise(splits: dict[str, list[dict]]) -> dict:
    """Compute the corpus statistics reported in the README and the pitch."""
    stats: dict = {"splits": {}}
    for name, rows in splits.items():
        if not rows:
            stats["splits"][name] = {"count": 0}
            continue
        notice_grades = [flesch_kincaid_grade(row["notice"]) for row in rows]
        card_grades = [flesch_kincaid_grade(row["card_text"]) for row in rows]
        stats["splits"][name] = {
            "count": len(rows),
            "urgency_distribution": dict(Counter(row["urgency"] for row in rows).most_common()),
            "distinct_brands": len({row["brand"] for row in rows}),
            "gold_do_not_drive": sum(1 for row in rows if row["park_it"]),
            "gold_fire_risk_when_parked": sum(1 for row in rows if row["park_outside"]),
            "mean_notice_grade_level": round(sum(notice_grades) / len(notice_grades), 2),
            "mean_card_grade_level": round(sum(card_grades) / len(card_grades), 2),
            "mean_notice_jargon_rate": round(
                sum(jargon_rate(row["notice"]) for row in rows) / len(rows), 2
            ),
            "mean_card_jargon_rate": round(
                sum(jargon_rate(row["card_text"]) for row in rows) / len(rows), 2
            ),
        }
    return stats


def main() -> None:
    """Command-line entry point for the dataset-construction step."""
    parser = argparse.ArgumentParser(description="Build the RecallClear training data.")
    parser.add_argument("--input", type=Path, default=config.RAW_RECALLS_PATH)
    parser.add_argument("--stats-output", type=Path, default=config.OUTPUTS_DIR / "dataset_stats.json")
    args = parser.parse_args()
    config.ensure_directories()

    records = read_jsonl(args.input)
    print(f"Loaded {len(records)} raw recall notices from {args.input}")

    examples = build_examples(records)
    print(f"Built {len(examples)} plain-language cards")

    splits = split_examples(examples)
    for name, rows in splits.items():
        write_jsonl(rows, config.SPLIT_PATHS[name])
        print(f"  {name:11s} {len(rows):5d} -> {config.SPLIT_PATHS[name]}")

    write_jsonl(examples, config.CARD_DATASET_PATH)

    stats = summarise(splits)
    stats["held_out_brands"] = config.HELD_OUT_BRANDS
    args.stats_output.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

"""Service layer that the web application builds on."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from scripts import config
from scripts.make_dataset import read_jsonl
from scripts.model import RecallExplainer
from scripts.plain_language import (
    CARD_SECTIONS,
    detect_text_warnings,
    URGENCY_LEVELS,
    build_card,
    flesch_kincaid_grade,
    format_notice,
    jargon_rate,
    parse_card,
)


URGENCY_RANK = {level: index for index, level in enumerate(URGENCY_LEVELS)}


class ExplainerService:
    """Loads the fine-tuned model once and serves plain-language cards."""

    def __init__(
        self,
        adapter_source: str = config.APP_ADAPTER_SOURCE,
        base_model_id: str = config.BASE_MODEL_ID,
        max_new_tokens: int = config.APP_MAX_NEW_TOKENS,
    ) -> None:
        self.adapter_source = adapter_source
        self.base_model_id = base_model_id
        self.max_new_tokens = max_new_tokens
        self._explainer: RecallExplainer | None = None
        self._load_lock = threading.Lock()
        self._generate_lock = threading.Lock()


    @property
    def is_loaded(self) -> bool:
        """Whether the weights are already in memory."""
        return self._explainer is not None

    def explainer(self) -> RecallExplainer:
        """Return the shared explainer, loading it on first call."""
        if self._explainer is None:
            with self._load_lock:
                if self._explainer is None:
                    import os

                    import torch


                    allocated = int(os.environ.get("OMP_NUM_THREADS", "0") or 0)
                    torch.set_num_threads(allocated if allocated > 0 else max(1, os.cpu_count() or 1))
                    self._explainer = RecallExplainer(
                        adapter_path=self.adapter_source, base_model_id=self.base_model_id
                    )
        return self._explainer


    def explain(self, notice: str, mode: str = RecallExplainer.MODE_TUNED) -> dict:
        """Generate one card and return it with its readability measurements."""
        explainer = self.explainer()
        started = time.time()
        with self._generate_lock:
            raw_output = explainer.explain(notice, mode=mode, max_new_tokens=self.max_new_tokens)
        elapsed = time.time() - started

        sections = parse_card(raw_output)
        return {
            "mode": mode,
            "raw": raw_output,
            "sections": {section: sections.get(section, "") for section in CARD_SECTIONS},
            "missing_sections": [section for section in CARD_SECTIONS if section not in sections],
            "well_formed": len(sections) == len(CARD_SECTIONS),
            "grade_level": flesch_kincaid_grade(raw_output),
            "jargon_rate": jargon_rate(raw_output),
            "seconds": round(elapsed, 2),
            "prompt_tokens": explainer.prompt_token_count(notice, mode),
        }

    def compare(self, notice: str) -> dict:
        """Run the untuned base model and the fine-tuned model on one notice."""
        return {
            "base": self.explain(notice, mode=RecallExplainer.MODE_BASE),
            "tuned": self.explain(notice, mode=RecallExplainer.MODE_TUNED),
        }


    @staticmethod
    def notice_from_record(record: dict) -> str:
        """Render a looked-up recall record as notice text for the model."""
        return format_notice(record)

    @staticmethod
    def notice_metrics(notice: str) -> dict:
        """Readability of the original notice, shown next to the model output."""
        return {
            "grade_level": flesch_kincaid_grade(notice),
            "jargon_rate": jargon_rate(notice),
            "words": len(notice.split()),
        }

    @staticmethod
    def text_warnings(notice: str) -> dict:
        """Detect do-not-drive / park-outside wording in raw notice text."""
        return detect_text_warnings(notice)

    @staticmethod
    def official_warnings(record: dict) -> dict:
        """Return NHTSA's own owner warnings for a record."""
        return {
            "do_not_drive": bool(record.get("park_it")),
            "fire_risk_when_parked": bool(record.get("park_outside")),
        }

    @staticmethod
    def reference_card(record: dict) -> str | None:
        """Return the rule-built reference card for a record, when buildable."""
        card = build_card(record)
        return card.to_text() if card else None


class DemoLibrary:
    """Curated held-out notices offered as one-click examples in the UI."""

    def __init__(self, path: Path | None = None, count: int = config.DEMO_EXAMPLE_COUNT) -> None:
        self.path = path or config.SPLIT_PATHS["test"]
        self.count = count
        self._examples: list[dict] | None = None

    def examples(self) -> list[dict]:
        """Return demo notices, preferring a spread of urgency levels."""
        if self._examples is not None:
            return self._examples

        if not self.path.exists():
            self._examples = []
            return self._examples

        rows = read_jsonl(self.path)
        rows.sort(key=lambda row: (URGENCY_RANK.get(row["urgency"], 99), row["campaign_number"]))

        chosen: list[dict] = []
        seen_brands: set[str] = set()
        for row in rows:
            brand = row.get("brand", "")

            if brand in seen_brands and len(chosen) < self.count:
                continue
            seen_brands.add(brand)
            chosen.append(
                {
                    "campaign_number": row["campaign_number"],
                    "manufacturer": row["manufacturer"],
                    "subject": row["subject"],
                    "component": row["component"],
                    "urgency": row["urgency"],
                    "notice": row["notice"],
                }
            )
            if len(chosen) >= self.count:
                break

        self._examples = chosen
        return self._examples


def load_evaluation_summary(path: Path | None = None) -> dict:
    """Load the headline evaluation numbers for display in the UI."""
    path = path or config.OUTPUTS_DIR / "evaluation.json"
    if not path.exists():
        return {}
    try:
        results = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    systems = results.get("systems", {})
    return {
        "base_model": results.get("base_model"),
        "examples": systems.get("tuned", {}).get("examples"),
        "format_base": systems.get("base", {}).get("format_compliance"),
        "format_few_shot": systems.get("few_shot", {}).get("format_compliance"),
        "format_tuned": systems.get("tuned", {}).get("format_compliance"),
        "grade_base": systems.get("base", {}).get("grade_level"),
        "grade_tuned": systems.get("tuned", {}).get("grade_level"),
        "grade_notice": results.get("reference_cards", {}).get("notice_grade_level"),
        "macro_f1_base": systems.get("base", {}).get("urgency_macro_f1"),
        "macro_f1_few_shot": systems.get("few_shot", {}).get("urgency_macro_f1"),
        "macro_f1_tuned": systems.get("tuned", {}).get("urgency_macro_f1"),
        "prompt_tokens_few_shot": systems.get("few_shot", {}).get("mean_prompt_tokens"),
        "prompt_tokens_tuned": systems.get("tuned", {}).get("mean_prompt_tokens"),
    }

"""Tests for dataset construction, evaluation metrics, and the web API.

These cover the parts of the system that do not need model weights, so the
suite runs in a couple of seconds and is safe to run in CI.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from scripts.build_features import brand_key, build_examples, is_held_out, split_examples
from scripts.evaluate_model import (
    grounding_score,
    hallucinated_phone,
    has_valid_format,
    macro_f1,
    states_free_repair,
    stratified_sample,
)
from scripts.plain_language import URGENCY_FIX_SOON, URGENCY_STOP_DRIVING
from scripts.prompts import build_few_shot_messages, build_training_messages, build_zero_shot_messages
from scripts.recall_lookup import normalise_campaign_number

WELL_FORMED_CARD = (
    "WHAT'S WRONG: A bolt in the suspension can come loose.\n"
    "WHAT COULD HAPPEN: The wheel can move out of place and you could crash.\n"
    "HOW URGENT: GET IT FIXED SOON - This problem can lead to a crash, fire, or injury.\n"
    "WHAT TO DO: Book the free repair with your dealer in the next few weeks.\n"
    "WHAT IT COSTS: Nothing. Recall repairs are always free."
)


def make_record(campaign: str, manufacturer: str, **overrides) -> dict:
    """Build a minimal but valid recall record for tests."""
    record = {
        "campaign_number": campaign,
        "manufacturer": manufacturer,
        "subject": "Suspension Bolt May Loosen",
        "component": "SUSPENSION",
        "summary": (
            f"{manufacturer} is recalling certain 2021 vehicles. A bolt in the rear "
            "suspension may have been tightened incorrectly and can come loose over time."
        ),
        "consequence": "A loose bolt can let the wheel move out of position, increasing the risk of a crash.",
        "remedy": "Dealers will tighten or replace the bolt, free of charge.",
        "park_it": False,
        "park_outside": False,
    }
    record.update(overrides)
    return record


class BrandKeyTests(unittest.TestCase):
    """Reducing messy manufacturer strings to a stable brand key."""

    def test_legal_suffixes_are_stripped(self) -> None:
        self.assertEqual(brand_key("Subaru of America, Inc."), "SUBARU")

    def test_parenthetical_alias_is_ignored(self) -> None:
        self.assertEqual(brand_key("Chrysler (FCA US, LLC)"), "CHRYSLER")

    def test_two_entities_of_one_brand_share_a_key(self) -> None:
        self.assertEqual(brand_key("Volvo Trucks North America"), brand_key("Volvo Car USA, LLC"))

    def test_empty_manufacturer_is_handled(self) -> None:
        self.assertEqual(brand_key(""), "UNKNOWN")

    def test_held_out_brands_are_recognised(self) -> None:
        self.assertTrue(is_held_out("Subaru of America, Inc."))
        self.assertFalse(is_held_out("Ford Motor Company"))


class SplitTests(unittest.TestCase):
    """Dataset construction and the held-out-by-brand split."""

    def setUp(self) -> None:
        self.records = [
            make_record("23V001000", "Ford Motor Company"),
            make_record("23V002000", "Honda (American Honda Motor Co.)"),
            make_record("23V003000", "Subaru of America, Inc."),
            make_record("23V004000", "Tesla, Inc."),
        ]

    def test_every_valid_record_becomes_an_example(self) -> None:
        self.assertEqual(len(build_examples(self.records)), 4)

    def test_duplicate_notices_are_removed(self) -> None:
        duplicated = self.records + [make_record("23V009000", "Ford Motor Company")]
        # The fifth record repeats Ford's defect text verbatim.
        self.assertEqual(len(build_examples(duplicated)), 4)

    def test_held_out_brands_land_in_test_only(self) -> None:
        splits = split_examples(build_examples(self.records))
        test_brands = {row["brand"] for row in splits["test"]}
        train_brands = {row["brand"] for row in splits["train"]} | {
            row["brand"] for row in splits["validation"]
        }
        self.assertTrue(test_brands <= {"SUBARU", "TESLA"})
        self.assertFalse(test_brands & train_brands, "held-out brands leaked into training")

    def test_example_carries_the_gold_flags(self) -> None:
        examples = build_examples([make_record("23V010000", "Ford Motor Company", park_it=True)])
        self.assertTrue(examples[0]["park_it"])
        self.assertEqual(examples[0]["urgency"], URGENCY_STOP_DRIVING)


class PromptTests(unittest.TestCase):
    """The prompt contract shared by training, evaluation, and the app."""

    def test_zero_shot_prompt_has_two_turns(self) -> None:
        messages = build_zero_shot_messages("NOTICE TEXT")
        self.assertEqual([message["role"] for message in messages], ["system", "user"])

    def test_few_shot_prompt_adds_demonstrations(self) -> None:
        messages = build_few_shot_messages("NOTICE TEXT")
        self.assertEqual(sum(1 for message in messages if message["role"] == "assistant"), 2)

    def test_few_shot_prompt_is_longer_than_zero_shot(self) -> None:
        zero = sum(len(message["content"]) for message in build_zero_shot_messages("N"))
        few = sum(len(message["content"]) for message in build_few_shot_messages("N"))
        self.assertGreater(few, zero * 3)

    def test_training_example_ends_with_the_target_card(self) -> None:
        messages = build_training_messages("NOTICE", WELL_FORMED_CARD)
        self.assertEqual(messages[-1]["role"], "assistant")
        self.assertEqual(messages[-1]["content"], WELL_FORMED_CARD)

    def test_notice_is_embedded_in_the_instruction(self) -> None:
        messages = build_zero_shot_messages("UNIQUE-NOTICE-MARKER")
        self.assertIn("UNIQUE-NOTICE-MARKER", messages[-1]["content"])


class MetricTests(unittest.TestCase):
    """Evaluation metrics."""

    def test_well_formed_card_passes_the_format_check(self) -> None:
        self.assertTrue(has_valid_format(WELL_FORMED_CARD))

    def test_missing_section_fails_the_format_check(self) -> None:
        truncated = "\n".join(WELL_FORMED_CARD.split("\n")[:3])
        self.assertFalse(has_valid_format(truncated))

    def test_out_of_order_sections_fail_the_format_check(self) -> None:
        lines = WELL_FORMED_CARD.split("\n")
        shuffled = "\n".join([lines[1], lines[0]] + lines[2:])
        self.assertFalse(has_valid_format(shuffled))

    def test_free_repair_is_detected(self) -> None:
        self.assertTrue(states_free_repair(WELL_FORMED_CARD))

    def test_paid_repair_claim_is_flagged(self) -> None:
        card = WELL_FORMED_CARD.replace("Nothing. Recall repairs are always free.", "About $450.")
        self.assertFalse(states_free_repair(card))

    def test_invented_phone_number_is_caught(self) -> None:
        notice = "Owners may contact Ford at 1-866-436-7332."
        self.assertTrue(hallucinated_phone("Call 1-800-000-0000 now.", notice))

    def test_quoted_phone_number_is_not_flagged(self) -> None:
        notice = "Owners may contact Ford at 1-866-436-7332."
        self.assertFalse(hallucinated_phone("Call 1-866-436-7332 now.", notice))

    def test_grounding_rewards_words_from_the_notice(self) -> None:
        notice = "The suspension bolt may loosen and the wheel may move."
        grounded = grounding_score("WHAT'S WRONG: The suspension bolt may loosen.", notice)
        invented = grounding_score("WHAT'S WRONG: The battery coolant pump exploded.", notice)
        self.assertGreater(grounded, invented)

    def test_macro_f1_is_one_for_perfect_predictions(self) -> None:
        gold = [URGENCY_STOP_DRIVING, URGENCY_FIX_SOON]
        self.assertEqual(macro_f1(gold, list(gold), (URGENCY_STOP_DRIVING, URGENCY_FIX_SOON)), 1.0)

    def test_macro_f1_punishes_always_predicting_the_majority(self) -> None:
        gold = [URGENCY_FIX_SOON] * 9 + [URGENCY_STOP_DRIVING]
        predicted = [URGENCY_FIX_SOON] * 10
        score = macro_f1(gold, predicted, (URGENCY_STOP_DRIVING, URGENCY_FIX_SOON))
        self.assertLess(score, 0.6)

    def test_unparsed_predictions_count_as_wrong(self) -> None:
        gold = [URGENCY_FIX_SOON, URGENCY_FIX_SOON]
        self.assertEqual(macro_f1(gold, [None, None], (URGENCY_FIX_SOON,)), 0.0)


class StratifiedSampleTests(unittest.TestCase):
    """Sampling that keeps the rare, high-stakes classes visible."""

    def setUp(self) -> None:
        self.rows = [
            {"urgency": URGENCY_FIX_SOON, "campaign_number": f"23V{index:03d}000", "notice": "n"}
            for index in range(200)
        ] + [
            {"urgency": URGENCY_STOP_DRIVING, "campaign_number": "23V900000", "notice": "n"},
            {"urgency": URGENCY_STOP_DRIVING, "campaign_number": "23V901000", "notice": "n"},
        ]

    def test_rare_class_is_always_included(self) -> None:
        sample = stratified_sample(self.rows, sample_size=20)
        self.assertEqual(sum(1 for row in sample if row["urgency"] == URGENCY_STOP_DRIVING), 2)

    def test_sample_size_is_respected(self) -> None:
        self.assertEqual(len(stratified_sample(self.rows, sample_size=20)), 20)

    def test_sampling_is_deterministic(self) -> None:
        first = [row["campaign_number"] for row in stratified_sample(self.rows, 15)]
        second = [row["campaign_number"] for row in stratified_sample(self.rows, 15)]
        self.assertEqual(first, second)


class CampaignNumberTests(unittest.TestCase):
    """Normalising the campaign numbers people type."""

    def test_canonical_form_is_unchanged(self) -> None:
        self.assertEqual(normalise_campaign_number("23V123000"), "23V123000")

    def test_short_form_is_padded(self) -> None:
        self.assertEqual(normalise_campaign_number("23v123"), "23V123000")

    def test_separators_are_ignored(self) -> None:
        self.assertEqual(normalise_campaign_number(" 23-V-123 "), "23V123000")

    def test_nonsense_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalise_campaign_number("not-a-recall")


class WebApiTests(unittest.TestCase):
    """The HTTP surface, with the model stubbed out."""

    def setUp(self) -> None:
        import main

        main.app.config.update(TESTING=True)
        self.client = main.app.test_client()
        self.main = main

    def test_health_reports_ok_without_loading_weights(self) -> None:
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_index_renders(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"RecallClear", response.data)

    def test_explain_rejects_an_empty_notice(self) -> None:
        response = self.client.post("/api/explain", json={"notice": "   "})
        self.assertEqual(response.status_code, 400)

    def test_explain_rejects_an_oversized_notice(self) -> None:
        response = self.client.post("/api/explain", json={"notice": "x" * 20000})
        self.assertEqual(response.status_code, 413)

    def test_explain_returns_the_generated_card(self) -> None:
        stub = {"raw": WELL_FORMED_CARD, "sections": {}, "well_formed": True, "seconds": 0.1}
        with patch.object(self.main.explainer_service, "explain", return_value=stub):
            response = self.client.post("/api/explain", json={"notice": "A recall notice."})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["tuned"]["raw"], WELL_FORMED_CARD)

    def test_lookup_rejects_a_malformed_campaign_number(self) -> None:
        response = self.client.post("/api/lookup", json={"campaign_number": "hello"})
        self.assertEqual(response.status_code, 400)

    def test_lookup_reports_a_missing_campaign(self) -> None:
        from scripts.recall_lookup import CampaignNotFoundError

        with patch("main.fetch_campaign", side_effect=CampaignNotFoundError("23V999000")):
            response = self.client.post("/api/lookup", json={"campaign_number": "23V999000"})
        self.assertEqual(response.status_code, 404)

    def test_lookup_returns_official_warnings(self) -> None:
        record = make_record("23V123000", "Ford Motor Company", park_it=True)
        with patch("main.fetch_campaign", return_value=record):
            response = self.client.post("/api/lookup", json={"campaign_number": "23V123000"})
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["official_warnings"]["do_not_drive"])

    def test_explain_accepts_an_explicit_mode(self) -> None:
        stub = {"raw": WELL_FORMED_CARD, "sections": {}, "well_formed": True, "seconds": 0.1}
        with patch.object(self.main.explainer_service, "explain", return_value=stub) as mocked:
            response = self.client.post(
                "/api/explain", json={"notice": "A recall notice.", "mode": "few_shot"}
            )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["mode"], "few_shot")
        self.assertEqual(payload["result"]["raw"], WELL_FORMED_CARD)
        mocked.assert_called_once_with("A recall notice.", mode="few_shot")

    def test_explain_rejects_an_unknown_mode(self) -> None:
        response = self.client.post(
            "/api/explain", json={"notice": "A recall notice.", "mode": "gpt7"}
        )
        self.assertEqual(response.status_code, 400)

    def test_explain_reports_text_warnings(self) -> None:
        stub = {"raw": WELL_FORMED_CARD, "sections": {}, "well_formed": True, "seconds": 0.1}
        notice = "Owners are advised not to drive these vehicles until repaired."
        with patch.object(self.main.explainer_service, "explain", return_value=stub):
            response = self.client.post("/api/explain", json={"notice": notice, "mode": "tuned"})
        self.assertTrue(response.get_json()["text_warnings"]["do_not_drive"])

    def test_unknown_api_path_returns_json(self) -> None:
        response = self.client.get("/api/does-not-exist")
        self.assertEqual(response.status_code, 404)
        self.assertIn("error", response.get_json())


if __name__ == "__main__":
    unittest.main()

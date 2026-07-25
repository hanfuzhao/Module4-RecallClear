"""Tests for the notice-to-card transformation and its readability metrics."""

from __future__ import annotations

import unittest

from scripts.plain_language import (
    CARD_SECTIONS,
    URGENCY_FIX_SOON,
    URGENCY_PARK_OUTSIDE,
    URGENCY_SCHEDULE,
    URGENCY_STOP_DRIVING,
    build_card,
    extract_defect,
    extract_fix,
    extract_phone,
    extract_urgency,
    flesch_kincaid_grade,
    shorten,
    split_sentences,
    format_notice,
    jargon_rate,
    parse_card,
    simplify_jargon,
    triage_urgency,
)

CRASH_RISK_RECORD = {
    "campaign_number": "21V986000",
    "manufacturer": "Ford Motor Company (Ford)",
    "subject": "Driveshaft May Fracture",
    "component": "POWER TRAIN:DRIVELINE:DRIVESHAFT",
    "summary": (
        "Ford Motor Company (Ford) is recalling certain 2021-2022 F-150 vehicles. "
        "Underbody heat and noise insulators may loosen and contact the aluminum "
        "driveshaft, which could damage the driveshaft and cause it to fracture."
    ),
    "consequence": (
        "A fractured driveshaft can cause a loss of drive power, or a loss of vehicle "
        "control if the driveshaft contacts the ground. Any of these scenarios can "
        "increase the risk of a crash."
    ),
    "remedy": (
        "Dealers will inspect and repair the driveshaft as necessary, free of charge. "
        "Owner notification letters were mailed February 4, 2022. Owners may contact "
        "Ford customer service at 1-866-436-7332."
    ),
    "park_it": False,
    "park_outside": False,
}

LABEL_ONLY_RECORD = {
    "campaign_number": "22V456000",
    "manufacturer": "Kia America, Inc.",
    "subject": "Incorrect Tire Information Label",
    "component": "LABEL",
    "summary": (
        "Kia America, Inc. (Kia) is recalling certain 2022 Carnival vehicles. The tire "
        "and loading information label may list an incorrect tire size."
    ),
    "consequence": "An incorrect tire size listed on the label may result in an owner installing the wrong size tire.",
    "remedy": "Dealers will replace the tire and loading information label, free of charge.",
    "park_it": False,
    "park_outside": False,
}


class TriageUrgencyTests(unittest.TestCase):
    """The four-level urgency ladder."""

    def test_do_not_drive_flag_wins(self) -> None:
        record = dict(CRASH_RISK_RECORD, park_it=True)
        level, reason = triage_urgency(record)
        self.assertEqual(level, URGENCY_STOP_DRIVING)
        self.assertIn("not to drive", reason)

    def test_fire_risk_flag_maps_to_park_outside(self) -> None:
        record = dict(CRASH_RISK_RECORD, park_outside=True)
        self.assertEqual(triage_urgency(record)[0], URGENCY_PARK_OUTSIDE)

    def test_do_not_drive_outranks_park_outside(self) -> None:
        record = dict(CRASH_RISK_RECORD, park_it=True, park_outside=True)
        self.assertEqual(triage_urgency(record)[0], URGENCY_STOP_DRIVING)

    def test_crash_language_maps_to_fix_soon(self) -> None:
        self.assertEqual(triage_urgency(CRASH_RISK_RECORD)[0], URGENCY_FIX_SOON)

    def test_no_harm_language_maps_to_schedule(self) -> None:
        self.assertEqual(triage_urgency(LABEL_ONLY_RECORD)[0], URGENCY_SCHEDULE)

    def test_string_flags_are_accepted(self) -> None:
        """The API returns booleans, but exports sometimes carry "true" strings."""
        record = dict(CRASH_RISK_RECORD, park_it="true")
        self.assertEqual(triage_urgency(record)[0], URGENCY_STOP_DRIVING)


class ExtractionTests(unittest.TestCase):
    """Pulling the useful spans out of NHTSA's boilerplate."""

    def test_defect_drops_the_recall_population_sentence(self) -> None:
        defect = extract_defect(CRASH_RISK_RECORD["summary"])
        self.assertNotIn("is recalling", defect)
        self.assertIn("insulators may loosen", defect)

    def test_fix_drops_dates_and_phone_numbers(self) -> None:
        fix = extract_fix(CRASH_RISK_RECORD["remedy"])
        self.assertIn("inspect and repair", fix)
        self.assertNotIn("1-866", fix)
        self.assertNotIn("February", fix)

    def test_phone_number_is_recovered(self) -> None:
        self.assertEqual(extract_phone(CRASH_RISK_RECORD), "1-866-436-7332")

    def test_missing_phone_number_returns_empty_string(self) -> None:
        self.assertEqual(extract_phone(LABEL_ONLY_RECORD), "")


class SimplifyJargonTests(unittest.TestCase):
    """The plain-language lexicon."""

    def test_multi_word_terms_are_replaced(self) -> None:
        self.assertIn("the airbag computer", simplify_jargon("The restraint control module may fail."))

    def test_longest_match_wins(self) -> None:
        result = simplify_jargon("The high-pressure fuel pump may fail.")
        self.assertIn("high-pressure fuel pump", result)

    def test_replacement_is_case_insensitive(self) -> None:
        self.assertIn("rust", simplify_jargon("CORROSION may occur."))

    def test_words_inside_other_words_are_left_alone(self) -> None:
        """"deploy" must not corrupt "deployment" handling or partial words."""
        self.assertNotIn("fireed", simplify_jargon("redeployed"))


class BuildCardTests(unittest.TestCase):
    """End-to-end card construction."""

    def test_card_has_all_five_sections(self) -> None:
        card = build_card(CRASH_RISK_RECORD)
        self.assertIsNotNone(card)
        parsed = parse_card(card.to_text())
        self.assertEqual(set(parsed), set(CARD_SECTIONS))

    def test_card_states_the_repair_is_free(self) -> None:
        card = build_card(CRASH_RISK_RECORD)
        self.assertIn("free", card.what_it_costs.lower())

    def test_card_is_easier_to_read_than_the_notice(self) -> None:
        card = build_card(CRASH_RISK_RECORD)
        notice = format_notice(CRASH_RISK_RECORD)
        self.assertLess(flesch_kincaid_grade(card.to_text()), flesch_kincaid_grade(notice))

    def test_card_contains_less_jargon_than_the_notice(self) -> None:
        card = build_card(CRASH_RISK_RECORD)
        notice = format_notice(CRASH_RISK_RECORD)
        self.assertLess(jargon_rate(card.to_text()), jargon_rate(notice))

    def test_stop_driving_card_tells_the_owner_not_to_drive(self) -> None:
        card = build_card(dict(CRASH_RISK_RECORD, park_it=True))
        self.assertIn("do not drive", card.what_to_do.lower())

    def test_quality_gate_rejects_an_empty_notice(self) -> None:
        self.assertIsNone(build_card({"summary": "", "consequence": "", "remedy": ""}))

    def test_quality_gate_rejects_a_stub_notice(self) -> None:
        stub = {"summary": "Ford is recalling certain vehicles.", "consequence": "Risk.", "remedy": ""}
        self.assertIsNone(build_card(stub))


class ParseCardTests(unittest.TestCase):
    """Reading sections back out of generated text."""

    def test_round_trip_through_parse(self) -> None:
        card = build_card(CRASH_RISK_RECORD)
        parsed = parse_card(card.to_text())
        self.assertEqual(parsed["WHAT'S WRONG"], card.whats_wrong)

    def test_markdown_decoration_is_tolerated(self) -> None:
        text = (
            "**WHAT'S WRONG:** The bolt is loose.\n"
            "- **WHAT COULD HAPPEN:** The wheel may come off.\n"
            "**HOW URGENT:** GET IT FIXED SOON - crash risk.\n"
            "**WHAT TO DO:** Call the dealer.\n"
            "**WHAT IT COSTS:** Nothing."
        )
        parsed = parse_card(text)
        self.assertEqual(set(parsed), set(CARD_SECTIONS))
        self.assertEqual(parsed["WHAT'S WRONG"], "The bolt is loose.")

    def test_urgency_is_read_from_the_urgency_section_only(self) -> None:
        text = (
            "WHAT'S WRONG: You should stop driving nothing here.\n"
            "WHAT COULD HAPPEN: A crash.\n"
            "HOW URGENT: SCHEDULE IT - labelling only.\n"
            "WHAT TO DO: Book it.\n"
            "WHAT IT COSTS: Nothing."
        )
        self.assertEqual(extract_urgency(text), URGENCY_SCHEDULE)

    def test_missing_urgency_section_returns_none(self) -> None:
        self.assertIsNone(extract_urgency("Some prose with no card structure at all."))


class BoilerplateRegressionTests(unittest.TestCase):
    """Regressions for defects found by inspecting generated cards.

    Each of these produced visibly broken output in the training targets before
    it was fixed, which is worse than a crash: the model would have learned to
    reproduce the mistake.
    """

    PORSCHE = {
        "manufacturer": "Porsche Cars North America, Inc.",
        "summary": (
            "Porsche Cars North America, Inc. (Porsche) is recalling certain model year "
            "2015 Porsche 918 vehicles.  The front lower control arms may fracture."
        ),
        "consequence": "The fracture of a lower control arm while driving increases the risk of a crash.",
        "remedy": (
            "Porsche will notify owners, and dealers will replace the front lower control "
            "arms, free of charge.  Owners may contact Porsche at 1-800-767-7243."
        ),
        "park_it": True,
        "park_outside": False,
    }

    def test_abbreviation_does_not_end_a_sentence(self) -> None:
        """"Inc." must not split, or the company name becomes its own sentence."""
        sentences = split_sentences(
            "Porsche Cars North America, Inc. (Porsche) is recalling cars.  The arms may fail."
        )
        self.assertEqual(len(sentences), 2)
        self.assertIn("is recalling", sentences[0])

    def test_company_name_never_leaks_into_the_defect(self) -> None:
        card = build_card(self.PORSCHE)
        self.assertNotIn("Porsche Cars North America", card.whats_wrong)
        self.assertTrue(card.whats_wrong.startswith("The front lower control arms"))

    def test_remedy_lead_in_is_removed(self) -> None:
        """"X will notify owners, and dealers will Y" must reduce to just Y."""
        card = build_card(self.PORSCHE)
        self.assertNotIn("will notify owners", card.what_to_do)
        self.assertIn("The dealer will replace the front lower control arms", card.what_to_do)

    def test_spliced_action_is_not_recapitalised(self) -> None:
        card = build_card(self.PORSCHE)
        self.assertNotIn("The dealer will Replace", card.what_to_do)

    def test_additionally_included_is_treated_as_boilerplate(self) -> None:
        defect = extract_defect(
            "Subaru is recalling certain 2010-2012 Legacy vehicles.  "
            "Additionally included are certain 2013 Outback vehicles that received new fobs.  "
            "If the fob is dropped, the engine may start unexpectedly."
        )
        self.assertNotIn("Additionally included", defect)
        self.assertIn("engine may start unexpectedly", defect)

    def test_shorten_keeps_whole_sentences(self) -> None:
        text = "The bolt may loosen over time. The wheel can then move out of position."
        self.assertEqual(shorten(text, max_words=12), "The bolt may loosen over time.")

    def test_shorten_never_returns_a_bare_fragment(self) -> None:
        text = "If the fob is dropped. The engine may unexpectedly start and run for fifteen minutes."
        self.assertTrue(shorten(text, max_words=10).endswith("."))

    def test_noun_phrase_substitutions_keep_grammar(self) -> None:
        """Replacements for noun phrases must stay noun phrases."""
        result = simplify_jargon("This can cause a loss of drive power.")
        self.assertIn("a loss of engine power", result)
        self.assertNotIn("a the car", result)


class ReadabilityTests(unittest.TestCase):
    """The self-contained readability metrics."""

    def test_grade_level_rises_with_complexity(self) -> None:
        simple = "The bolt is loose. Call your dealer. The fix is free."
        complex_text = (
            "Insufficient torque retention of the fastener may precipitate "
            "progressive degradation of the structural interface, thereby "
            "increasing the probability of a collision event."
        )
        self.assertLess(flesch_kincaid_grade(simple), flesch_kincaid_grade(complex_text))

    def test_empty_text_scores_zero(self) -> None:
        self.assertEqual(flesch_kincaid_grade(""), 0.0)

    def test_jargon_rate_counts_lexicon_terms(self) -> None:
        self.assertGreater(jargon_rate("The restraint control module failed."), 0)

    def test_plain_text_has_no_jargon(self) -> None:
        self.assertEqual(jargon_rate("The bolt is loose and may come off."), 0.0)


if __name__ == "__main__":
    unittest.main()

"""Prompt construction shared by training, evaluation, and the web app."""

from __future__ import annotations

from scripts.plain_language import CARD_SECTIONS, URGENCY_LEVELS

SYSTEM_PROMPT = (
    "You are RecallClear, an assistant that rewrites official vehicle safety "
    "recall notices into plain language for car owners."
)

_SECTION_LIST = ", ".join(CARD_SECTIONS)
_URGENCY_LIST = " / ".join(URGENCY_LEVELS)

INSTRUCTION = (
    "Rewrite this vehicle recall notice for the car's owner.\n"
    f"Use exactly these five labelled lines, in this order: {_SECTION_LIST}.\n"
    f"HOW URGENT must start with one of: {_URGENCY_LIST}.\n"
    "Write short everyday sentences. Do not invent facts that are not in the notice.\n\n"
    "NOTICE:\n{notice}"
)


FEW_SHOT_EXAMPLES: tuple[tuple[str, str], ...] = (
    (
        "Manufacturer: Chrysler (FCA US, LLC)\n"
        "NHTSA Campaign Number: 19V123000\n"
        "Component: AIR BAGS:FRONTAL:DRIVER SIDE INFLATOR MODULE\n"
        "Summary: Chrysler (FCA US, LLC) is recalling certain 2008-2010 Dodge Ram 2500 vehicles. "
        "These vehicles are equipped with a driver frontal air bag that may be susceptible to "
        "moisture intrusion which, over time, could cause the inflator to rupture.\n"
        "Consequence: An inflator rupture may result in metal fragments striking the driver or "
        "other occupants, resulting in serious injury or death.\n"
        "Remedy: Dealers will replace the driver frontal air bag inflator, free of charge. "
        "Owners may contact Chrysler customer service at 1-800-853-1403.",
        "WHAT'S WRONG: Moisture can get into the driver's airbag inflator and make it burst when "
        "the airbag fires.\n"
        "WHAT COULD HAPPEN: Metal pieces can be thrown at the driver or passengers, causing "
        "serious injury or death.\n"
        "HOW URGENT: STOP DRIVING - NHTSA has told owners not to drive this vehicle until it is "
        "fixed.\n"
        "WHAT TO DO: Do not drive the car. Call your dealer now and ask for the recall repair. "
        "The dealer will replace the driver's airbag inflator. You can also call Chrysler at "
        "1-800-853-1403.\n"
        "WHAT IT COSTS: Nothing. Recall repairs are always free, no matter the car's age or mileage.",
    ),
    (
        "Manufacturer: Kia America, Inc.\n"
        "NHTSA Campaign Number: 22V456000\n"
        "Component: LABEL\n"
        "Summary: Kia America, Inc. (Kia) is recalling certain 2022 Carnival vehicles. The tire "
        "and loading information label may list an incorrect tire size. As such, these vehicles "
        "fail to comply with the requirements of Federal Motor Vehicle Safety Standard number 110.\n"
        "Consequence: An incorrect tire size listed on the label may result in an owner "
        "installing the wrong size tire.\n"
        "Remedy: Dealers will replace the tire and loading information label, free of charge. "
        "Owners may contact Kia customer service at 1-800-333-4542.",
        "WHAT'S WRONG: The tire information sticker shows the wrong tire size, so the car does "
        "not meet the federal safety rule.\n"
        "WHAT COULD HAPPEN: An owner could buy and fit the wrong size tires.\n"
        "HOW URGENT: SCHEDULE IT - This is a safety-rule or labeling problem with no direct "
        "crash risk.\n"
        "WHAT TO DO: Book the free repair with your dealer at your next service visit. The "
        "dealer will replace the tire and loading information label. You can also call Kia at "
        "1-800-333-4542.\n"
        "WHAT IT COSTS: Nothing. Recall repairs are always free, no matter the car's age or mileage.",
    ),
)


def build_zero_shot_messages(notice: str) -> list[dict]:
    """Return chat messages for the zero-shot prompt used in training and demo."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": INSTRUCTION.format(notice=notice)},
    ]


def build_few_shot_messages(notice: str) -> list[dict]:
    """Return chat messages that prepend two worked examples to the instruction."""
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example_notice, example_card in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": INSTRUCTION.format(notice=example_notice)})
        messages.append({"role": "assistant", "content": example_card})
    messages.append({"role": "user", "content": INSTRUCTION.format(notice=notice)})
    return messages


def build_training_messages(notice: str, card_text: str) -> list[dict]:
    """Return the full supervised example, prompt plus target card."""
    return build_zero_shot_messages(notice) + [{"role": "assistant", "content": card_text}]

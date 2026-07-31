"""Turn an official NHTSA recall notice into a plain-language owner card."""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict


JARGON_LEXICON: dict[str, str] = {

    "electronic stability control": "the anti-skid system",
    "supplemental restraint system": "the airbag system",
    "occupant classification system": "the sensor that decides if the airbag should fire",
    "restraint control module": "the airbag computer",
    "powertrain control module": "the engine computer",
    "body control module": "the car's main electrical computer",
    "electronic control unit": "a control computer",
    "electronic control module": "a control computer",
    "instrument panel cluster": "the dashboard display",
    "telematics control unit": "the built-in cellular unit",
    "battery energy control module": "the battery computer",
    "high-voltage battery": "the main drive battery",
    "printed circuit board": "circuit board",
    "short circuit": "an electrical short",
    "wiring harness": "wiring bundle",
    "ground connection": "electrical ground",
    "software calibration": "software setting",
    "over-the-air": "wireless",

    "anti-lock brake": "anti-lock brake",
    "hydraulic control unit": "the brake pressure unit",
    "brake caliper": "brake clamp",
    "master cylinder": "the main brake cylinder",
    "brake booster": "the brake power assist",
    "tie rod": "steering link",
    "steering knuckle": "the part that connects the wheel to the steering",
    "steering gear": "steering box",
    "control arm": "suspension arm",
    "ball joint": "suspension pivot joint",
    "lower control arm": "lower suspension arm",
    "strut assembly": "shock absorber unit",
    "electric power steering": "the electric power steering",

    "power train": "drivetrain",
    "powertrain": "drivetrain",
    "driveshaft": "drive shaft",
    "drive shaft": "drive shaft",
    "half shaft": "axle shaft",
    "transmission control module": "the transmission computer",
    "torque converter": "the transmission coupling",
    "crankshaft": "crankshaft",
    "connecting rod": "engine rod",
    "camshaft": "camshaft",
    "timing chain": "timing chain",
    "fuel rail": "fuel pipe",
    "fuel pump assembly": "the fuel pump",
    "high-pressure fuel pump": "the high-pressure fuel pump",
    "fuel delivery module": "the fuel pump unit",
    "exhaust gas recirculation": "the exhaust recycling system",
    "catalytic converter": "the exhaust cleaner",
    "turbocharger": "turbo",

    "inflator": "airbag inflator",
    "pretensioner": "seat-belt tightener",
    "seat belt anchorage": "the seat-belt mounting point",
    "child restraint anchor": "child-seat anchor",
    "lower anchors and tethers for children": "child-seat anchors",
    "latch striker": "door latch catch",
    "door latch assembly": "the door latch",
    "windshield wiper motor": "the wiper motor",
    "rearview camera": "backup camera",
    "rear view camera": "backup camera",
    "back-up camera": "backup camera",
    "head restraint": "headrest",

    "federal motor vehicle safety standard": "the federal safety rule",
    "fmvss": "the federal safety rule",
    "does not conform to": "does not meet",
    "fails to conform to": "does not meet",
    "noncompliance": "rule violation",
    "non-compliance": "rule violation",
    "as necessary": "if needed",
    "free of charge": "at no cost to you",
    "authorized dealer": "dealer",
    "owner notification letters": "owner letters",
    "remedy": "fix",
    "defect": "problem",
    "malfunction": "stop working correctly",
    "inadvertent": "unintended",
    "inadvertently": "unexpectedly",
    "deployment": "firing",
    "deploy": "fire",
    "may become detached": "can come loose",
    "become detached": "come loose",
    "detach": "come loose",
    "fracture": "break",
    "degrade": "wear out",
    "degradation": "wearing out",
    "ingress": "getting in",
    "water ingress": "water getting in",
    "corrode": "rust",
    "corrosion": "rust",
    "seize": "lock up",
    "insufficient": "not enough",
    "excessive": "too much",
    "premature": "early",
    "subsequent": "later",
    "utilize": "use",
    "in the event of": "if there is",
    "increase the risk of a crash": "make a crash more likely",
    "increasing the risk of a crash": "making a crash more likely",
    "increase the risk of injury": "make injury more likely",
    "increasing the risk of injury": "making injury more likely",


    "loss of motive power": "loss of engine power",
    "loss of drive power": "loss of engine power",
    "loss of vehicle control": "loss of control of the car",
    "unintended acceleration": "the car speeding up on its own",
    "engine stall": "the engine shutting off",
    "stall": "shut off",
    "thermal event": "fire",
    "thermal runaway": "a battery fire",
    "vehicle fire": "a fire",
    "in certain circumstances": "sometimes",
    "under certain conditions": "sometimes",
    "prior to": "before",
}


JARGON_TERMS: tuple[str, ...] = tuple(
    term
    for term in JARGON_LEXICON
    if len(term.split()) >= 2 or term in {"fmvss", "inflator", "pretensioner", "noncompliance"}
)


URGENCY_STOP_DRIVING = "STOP DRIVING"
URGENCY_PARK_OUTSIDE = "PARK OUTSIDE"
URGENCY_FIX_SOON = "GET IT FIXED SOON"
URGENCY_SCHEDULE = "SCHEDULE IT"

URGENCY_LEVELS = (
    URGENCY_STOP_DRIVING,
    URGENCY_PARK_OUTSIDE,
    URGENCY_FIX_SOON,
    URGENCY_SCHEDULE,
)


_TEXT_WARNING_STOP = re.compile(
    r"do not drive|not to drive|stop driving|should not be driven|"
    r"cease (?:driving|use)|discontinue (?:driving|use)|park (?:the vehicle )?immediately",
    re.IGNORECASE,
)
_TEXT_WARNING_PARK_OUTSIDE = re.compile(
    r"park (?:it |the vehicle |vehicles? )?outside|park outdoors|"
    r"away from (?:structures|buildings|homes|other vehicles)|do not park (?:in|inside|near)|"
    r"risk of fire (?:even )?(?:while|when) (?:parked|the vehicle is parked)",
    re.IGNORECASE,
)


def detect_text_warnings(notice: str) -> dict[str, bool]:
    """Detect NHTSA's top-level owner warnings in the notice text itself."""
    text = notice or ""
    return {
        "do_not_drive": bool(_TEXT_WARNING_STOP.search(text)),
        "fire_risk_when_parked": bool(_TEXT_WARNING_PARK_OUTSIDE.search(text)),
    }


_HARM_PATTERNS = re.compile(
    r"\b(crash|fire|burn|injur|death|fatal|collision|loss of (?:vehicle )?control|"
    r"lose control|stall|roll ?away|fail to (?:deploy|inflate)|not deploy|"
    r"reduce(?:d)? visibility|struck|strike a pedestrian)\w*",
    re.IGNORECASE,
)

_URGENCY_REASONS = {
    URGENCY_STOP_DRIVING: "NHTSA has told owners not to drive this vehicle until it is fixed.",
    URGENCY_PARK_OUTSIDE: "There is a fire risk even when parked, so keep the car away from buildings.",
    URGENCY_FIX_SOON: "This problem can lead to a crash, fire, or injury.",
    URGENCY_SCHEDULE: "This is a safety-rule or labeling problem with no direct crash risk.",
}


def triage_urgency(record: dict) -> tuple[str, str]:
    """Return the ``(level, reason)`` pair for a recall record."""
    if _as_bool(record.get("park_it")):
        return URGENCY_STOP_DRIVING, _URGENCY_REASONS[URGENCY_STOP_DRIVING]
    if _as_bool(record.get("park_outside")):
        return URGENCY_PARK_OUTSIDE, _URGENCY_REASONS[URGENCY_PARK_OUTSIDE]
    consequence = record.get("consequence") or ""
    if _HARM_PATTERNS.search(consequence):
        return URGENCY_FIX_SOON, _URGENCY_REASONS[URGENCY_FIX_SOON]
    return URGENCY_SCHEDULE, _URGENCY_REASONS[URGENCY_SCHEDULE]


def _as_bool(value: object) -> bool:
    """Coerce the API's mixed true/"true"/None booleans to a real bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return False


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


_ABBREVIATIONS = (
    "Inc", "Co", "Corp", "Ltd", "LLC", "L.L.C", "Mfg", "Bros", "Div",
    "No", "Nos", "St", "Ave", "Rd", "Dr", "Mr", "Mrs", "Ms", "Jr", "Sr",
    "U.S", "U.S.A", "Approx", "approx", "vs", "etc", "e.g", "i.e", "Ph.D",
)
_PROTECT_ABBREVIATION = re.compile(
    r"\b(" + "|".join(re.escape(item) for item in _ABBREVIATIONS) + r")\.(?=\s)"
)
_ABBREVIATION_PLACEHOLDER = "․"


_POPULATION_SENTENCE = re.compile(
    r"\bis recalling\b|\bare recalling\b|\bhas recalled\b|\bhave recalled\b|"
    r"\brecalling certain\b|\bare being recalled\b|\bis being recalled\b|"
    r"\badditionally,? included\b|\balso included\b|\bincluded in this recall\b|"
    r"\bthis recall (?:also )?(?:includes|covers|expands)\b|\brecall population\b",
    re.IGNORECASE,
)


_DEFECT_VERB = re.compile(
    r"\b(may|can|could|will|might|is|are|was|were|has|have|do(?:es)?\s+not|"
    r"fail|fails|failed|lack|lacks|incorrect|improper)\b",
    re.IGNORECASE,
)
_PHONE = re.compile(r"\b1[-\s]?\d{3}[-\s]?\d{3}[-\s]?\d{4}\b")
_CONTACT_SENTENCE = re.compile(
    r"owners? may (?:also )?contact|notification letters?|recall began|"
    r"number for this recall|expected to begin|interim (?:letter|notification)",
    re.IGNORECASE,
)
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, protecting abbreviations such as "Inc."."""
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    if not cleaned:
        return []
    protected = _PROTECT_ABBREVIATION.sub(rf"\1{_ABBREVIATION_PLACEHOLDER}", cleaned)
    parts = _SENTENCE_SPLIT.split(protected)
    return [
        part.replace(_ABBREVIATION_PLACEHOLDER, ".").strip() for part in parts if part.strip()
    ]


_ARTICLE_COLLISION = re.compile(r"\b(a|an|the)\s+(a|an|the)\b", re.IGNORECASE)


def simplify_jargon(text: str) -> str:
    """Replace technical terms with everyday equivalents from the lexicon."""
    result = text

    for term in sorted(JARGON_LEXICON, key=len, reverse=True):
        replacement = JARGON_LEXICON[term]
        if replacement.lower() == term.lower():
            continue
        pattern = re.compile(rf"(?<![\w-]){re.escape(term)}(?![\w-])", re.IGNORECASE)
        result = pattern.sub(replacement, result)
    return _ARTICLE_COLLISION.sub(r"\2", result)


def shorten(text: str, max_words: int = 34) -> str:
    """Trim text to a word budget without leaving a sentence fragment."""
    words = text.split()
    if len(words) <= max_words:
        return text

    sentences = split_sentences(text)
    if len(sentences) > 1:
        kept: list[str] = []
        used = 0
        for sentence in sentences:
            length = len(sentence.split())
            if kept and used + length > max_words:
                break
            kept.append(sentence)
            used += length
        if kept and used <= max_words:
            return " ".join(kept)
        text = sentences[0]
        words = text.split()
        if len(words) <= max_words:
            return text

    clipped = " ".join(words[:max_words])
    for marker in (", which", ", and", ", or", ";", ","):
        head, sep, _tail = clipped.rpartition(marker)
        if sep and len(head.split()) >= max_words // 2:
            clipped = head
            break
    return clipped.rstrip(" ,;") + "."


def _tidy(text: str, capitalise: bool = True) -> str:
    """Normalise whitespace, capitalisation, and terminal punctuation."""
    cleaned = re.sub(r"\s+", " ", text).strip().strip(",;")
    if not cleaned:
        return ""
    if capitalise:
        cleaned = cleaned[0].upper() + cleaned[1:]
    if not cleaned.endswith((".", "!", "?")):
        cleaned += "."
    return cleaned


def count_syllables(word: str) -> int:
    """Approximate the syllable count of an English word."""
    word = word.lower()
    groups = re.findall(r"[aeiouy]+", word)
    total = len(groups)
    if word.endswith("e") and not word.endswith(("le", "ee", "ye")) and total > 1:
        total -= 1
    return max(total, 1)


def flesch_kincaid_grade(text: str) -> float:
    """Return the Flesch-Kincaid grade level of a passage (0.0 if empty)."""
    sentences = split_sentences(text)
    words = _WORD.findall(text or "")
    if not sentences or not words:
        return 0.0
    syllables = sum(count_syllables(word) for word in words)
    words_per_sentence = len(words) / len(sentences)
    syllables_per_word = syllables / len(words)
    return round(0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59, 2)


def jargon_rate(text: str) -> float:
    """Return jargon terms per 100 words -- lower is more owner-friendly."""
    words = _WORD.findall(text or "")
    if not words:
        return 0.0
    lowered = (text or "").lower()
    hits = sum(len(re.findall(rf"(?<![\w-]){re.escape(term)}(?![\w-])", lowered)) for term in JARGON_TERMS)
    return round(100.0 * hits / len(words), 2)


def extract_defect(summary: str) -> str:
    """Pull the defect description out of a Summary, dropping recall boilerplate."""
    candidates = [
        sentence
        for sentence in split_sentences(summary)
        if not _POPULATION_SENTENCE.search(sentence)
        and len(sentence.split()) >= 5
        and _DEFECT_VERB.search(sentence)
    ]
    if not candidates:
        return ""
    return " ".join(candidates[:2])


def extract_vehicles(summary: str) -> str:
    """Return the 'which vehicles' sentence from a Summary, if present."""
    for sentence in split_sentences(summary):
        if _POPULATION_SENTENCE.search(sentence):
            return sentence
    return ""


def extract_fix(remedy: str) -> str:
    """Return the repair action from a Remedy, without dates or phone numbers."""
    sentences = [s for s in split_sentences(remedy) if not _CONTACT_SENTENCE.search(s)]
    if not sentences:
        return ""
    return sentences[0]


def extract_phone(record: dict) -> str:
    """Return the manufacturer's owner phone number if the notice includes one."""
    for field in ("remedy",):
        match = _PHONE.search(record.get(field) or "")
        if match:
            return re.sub(r"\s", "-", match.group(0))
    return ""


CARD_SECTIONS = (
    "WHAT'S WRONG",
    "WHAT COULD HAPPEN",
    "HOW URGENT",
    "WHAT TO DO",
    "WHAT IT COSTS",
)


@dataclass
class RecallCard:
    """The five-section owner-facing summary of a recall notice."""

    whats_wrong: str
    what_could_happen: str
    urgency: str
    urgency_reason: str
    what_to_do: str
    what_it_costs: str

    def to_text(self) -> str:
        """Render the card in the exact format the model is trained to emit."""
        return "\n".join(
            [
                f"WHAT'S WRONG: {self.whats_wrong}",
                f"WHAT COULD HAPPEN: {self.what_could_happen}",
                f"HOW URGENT: {self.urgency} - {self.urgency_reason}",
                f"WHAT TO DO: {self.what_to_do}",
                f"WHAT IT COSTS: {self.what_it_costs}",
            ]
        )

    def to_dict(self) -> dict:
        """Return the card as a plain dictionary."""
        return asdict(self)


def _build_action_steps(record: dict, urgency: str, fix: str) -> str:
    """Compose the owner's to-do sentence for a given urgency level."""
    phone = extract_phone(record)
    manufacturer = _short_manufacturer(record.get("manufacturer") or "the manufacturer")

    if urgency == URGENCY_STOP_DRIVING:
        opening = "Do not drive the car. Call your dealer now and ask for the recall repair"
    elif urgency == URGENCY_PARK_OUTSIDE:
        opening = "Park outside, away from buildings, until the repair is done. Book it with your dealer now"
    elif urgency == URGENCY_FIX_SOON:
        opening = "Book the free repair with your dealer in the next few weeks"
    else:
        opening = "Book the free repair with your dealer at your next service visit"

    fix_clause = f" The dealer will {_lowercase_first(fix)}" if fix else ""
    contact = f" You can also call {manufacturer} at {phone}." if phone else ""
    return _tidy(opening) + fix_clause + contact


_REMEDY_LEAD_IN = re.compile(
    r"^.*?\b(?:authorized\s+|certified\s+)?(?:dealers?|technicians?|service centers?)\s+will\s+",
    re.IGNORECASE,
)
_REMEDY_NOTIFY_ONLY = re.compile(
    r"^[^,.]*\bwill\s+notify\s+owners?\b[,.]?\s*", re.IGNORECASE
)


def _strip_remedy_lead_in(sentence: str) -> str:
    """Return just the repair action from a Remedy sentence."""
    text = sentence.strip()
    if _REMEDY_LEAD_IN.search(text):
        return _REMEDY_LEAD_IN.sub("", text, count=1).strip()

    return _REMEDY_NOTIFY_ONLY.sub("", text, count=1).strip()


def _lowercase_first(sentence: str) -> str:
    """Reduce a Remedy sentence to a lowercase action phrase."""
    text = simplify_jargon(_strip_remedy_lead_in(sentence))
    text = re.sub(r",?\s*at no cost to you\.?$", ".", text, flags=re.IGNORECASE)
    text = re.sub(r",?\s*free of charge\.?$", ".", text, flags=re.IGNORECASE)
    text = text.strip()
    if not text:
        return ""
    if text[:2].isupper():
        return _tidy(text, capitalise=False)
    return _tidy(text[0].lower() + text[1:], capitalise=False)


def _short_manufacturer(name: str) -> str:
    """Trim 'Ford Motor Company (Ford)' style names down to the brand."""
    inner = re.search(r"\(([^)]+)\)", name)
    if inner:
        return inner.group(1).strip()
    return re.sub(r",?\s+(Inc|LLC|Corp|Corporation|Company|Co)\.?$", "", name).strip()


def build_card(record: dict) -> RecallCard | None:
    """Build the reference plain-language card, or ``None`` if the notice is unusable."""
    defect = extract_defect(record.get("summary") or "")
    consequence = record.get("consequence") or ""


    if len(defect.split()) < 5 or len(consequence.split()) < 5:
        return None

    urgency, reason = triage_urgency(record)
    fix = extract_fix(record.get("remedy") or "")

    whats_wrong = _tidy(shorten(simplify_jargon(defect), max_words=34))
    what_could_happen = _tidy(shorten(simplify_jargon(consequence), max_words=30))

    return RecallCard(
        whats_wrong=whats_wrong,
        what_could_happen=what_could_happen,
        urgency=urgency,
        urgency_reason=reason,
        what_to_do=_build_action_steps(record, urgency, fix),
        what_it_costs="Nothing. Recall repairs are always free, no matter the car's age or mileage.",
    )


_SECTION_HEADING = re.compile(
    r"^\s*(?:[-*#>\d.\s]*)?(" + "|".join(re.escape(section) for section in CARD_SECTIONS) + r")\s*:\s*",
    re.IGNORECASE | re.MULTILINE,
)


def parse_card(text: str) -> dict[str, str]:
    """Extract the five card sections from a generated string."""
    cleaned = (text or "").replace("**", "").replace("__", "")
    matches = list(_SECTION_HEADING.finditer(cleaned))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(cleaned)
        heading = match.group(1).upper()
        body = cleaned[match.end() : end].strip()
        if heading not in sections and body:
            sections[heading] = body
    return sections


def extract_urgency(text: str) -> str | None:
    """Return the urgency level asserted in a generated card, if any."""
    section = parse_card(text).get("HOW URGENT")
    if not section:
        return None
    upper = section.upper()
    for level in URGENCY_LEVELS:
        if upper.startswith(level):
            return level
    for level in URGENCY_LEVELS:
        if level in upper:
            return level
    return None


def format_notice(record: dict) -> str:
    """Render a recall record as the raw notice text fed to the model."""
    parts = [
        f"Manufacturer: {record.get('manufacturer') or 'Unknown'}",
        f"NHTSA Campaign Number: {record.get('campaign_number') or 'Unknown'}",
        f"Subject: {record.get('subject') or 'Unknown'}",
        f"Component: {record.get('component') or 'Unknown'}",
        f"Summary: {(record.get('summary') or '').strip()}",
        f"Consequence: {(record.get('consequence') or '').strip()}",
        f"Remedy: {(record.get('remedy') or '').strip()}",
    ]
    return "\n".join(parts)

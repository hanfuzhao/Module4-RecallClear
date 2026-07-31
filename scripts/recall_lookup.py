"""Look up a live recall notice by its NHTSA campaign number.

Uses the public NHTSA recalls API (https://api.nhtsa.gov), which returns the
"""

from __future__ import annotations

import re

import requests

from scripts import config

NHTSA_CAMPAIGN_URL = "https://api.nhtsa.gov/recalls/campaignNumber"


_CAMPAIGN_PATTERN = re.compile(r"^\s*(\d{2})\s*([VETC])\s*(\d{3,6})\s*$", re.IGNORECASE)


class CampaignNotFoundError(LookupError):
    """Raised when NHTSA has no campaign with the requested number."""


def normalise_campaign_number(raw: str) -> str:
    """Return a campaign number in NHTSA's canonical ``23V123000`` form."""
    match = _CAMPAIGN_PATTERN.match((raw or "").replace("-", "").replace(" ", ""))
    if not match:
        raise ValueError(
            "Campaign numbers look like 23V123000: two digits, a letter, then digits."
        )
    year, kind, sequence = match.groups()
    return f"{year}{kind.upper()}{sequence.ljust(6, '0')}"


def fetch_campaign(raw_campaign_number: str, timeout: int = 15) -> dict:
    """Fetch one campaign from NHTSA and return it in this project's schema."""
    campaign_number = normalise_campaign_number(raw_campaign_number)
    response = requests.get(
        NHTSA_CAMPAIGN_URL,
        params={"campaignNumber": campaign_number},
        timeout=timeout,
        headers={"User-Agent": "RecallClear/1.0 (course project)"},
    )
    if response.status_code == 400:
        raise CampaignNotFoundError(campaign_number)
    response.raise_for_status()

    rows = response.json().get("results") or []
    if not rows:
        raise CampaignNotFoundError(campaign_number)

    first = rows[0]
    vehicles = sorted(
        {
            f"{row.get('ModelYear', '')} {row.get('Make', '')} {row.get('Model', '')}".strip()
            for row in rows
            if row.get("Make")
        }
    )
    return {
        "campaign_number": first.get("NHTSACampaignNumber") or campaign_number,
        "manufacturer": first.get("Manufacturer") or "",
        "subject": first.get("Component") or "",
        "component": first.get("Component") or "",
        "summary": (first.get("Summary") or "").strip(),
        "consequence": (first.get("Consequence") or "").strip(),
        "remedy": (first.get("Remedy") or "").strip(),
        "park_it": bool(first.get("parkIt")),
        "park_outside": bool(first.get("parkOutSide")),
        "affected_vehicles": vehicles[:12],
        "report_received_date": first.get("ReportReceivedDate") or "",
        "source_url": f"https://www.nhtsa.gov/recalls?nhtsaId={campaign_number}",
    }


def is_lookup_enabled() -> bool:
    """Whether live lookup is permitted in this deployment."""
    return config.ENABLE_LIVE_LOOKUP

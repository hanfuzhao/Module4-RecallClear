"""Download real vehicle safety-recall notices published by NHTSA.

https://data.transportation.gov/d/6axg-epim
python -m scripts.make_dataset
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

from scripts import config


FIELD_MAP = {
    "nhtsa_id": "campaign_number",
    "manufacturer": "manufacturer",
    "subject": "subject",
    "component": "component",
    "defect_summary": "summary",
    "consequence_summary": "consequence",
    "corrective_action": "remedy",
    "potentially_affected": "units_affected",
    "report_received_date": "report_received_date",
}


class RecallsDataClient:
    """Paged reader for the Socrata "Recalls Data" resource."""

    def __init__(
        self,
        resource_url: str = config.RECALLS_RESOURCE_URL,
        page_size: int = config.RECALLS_PAGE_SIZE,
        timeout: int = config.REQUEST_TIMEOUT_SECONDS,
        retries: int = config.REQUEST_RETRIES,
    ) -> None:
        self.resource_url = resource_url
        self.page_size = page_size
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "RecallClear/1.0 (course project)"})

    def _get(self, params: dict) -> list[dict]:
        """Issue one Socrata query, retrying transient failures with backoff."""
        for attempt in range(self.retries):
            try:
                response = self.session.get(self.resource_url, params=params, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as error:
                if attempt == self.retries - 1:
                    raise RuntimeError(f"Socrata query failed: {error}") from error
                time.sleep(config.REQUEST_BACKOFF_SECONDS * (2**attempt))
        return []

    def count(self, where: str) -> int:
        """Return how many rows match a filter, for progress reporting."""
        rows = self._get({"$select": "count(1)", "$where": where})
        return int(rows[0]["count_1"]) if rows else 0

    def iter_rows(self, where: str):
        """Yield every matching row, one page at a time."""
        offset = 0
        while True:
            page = self._get(
                {
                    "$where": where,
                    "$limit": self.page_size,
                    "$offset": offset,
                    "$order": "nhtsa_id",
                }
            )
            if not page:
                return
            yield from page
            offset += len(page)
            if len(page) < self.page_size:
                return


def build_where_clause(start_year: int, recall_type: str = "Vehicle") -> str:
    """Build the Socrata filter for vehicle recalls issued from a given year."""
    return (
        f"recall_type='{recall_type}' AND report_received_date >= '{start_year}-01-01' "
        "AND defect_summary IS NOT NULL AND consequence_summary IS NOT NULL"
    )


def normalise_row(row: dict) -> dict:
    """Map one Socrata row onto this project's canonical record schema."""
    record = {target: (row.get(source) or "").strip() for source, target in FIELD_MAP.items()}

    record["park_it"] = (row.get("do_not_drive") or "").strip().lower() == "yes"
    record["park_outside"] = (row.get("fire_risk_when_parked") or "").strip().lower() == "yes"
    return record


def collect_recalls(start_year: int = config.RECALL_START_YEAR) -> list[dict]:
    """Download every vehicle recall notice issued from ``start_year`` onward."""
    client = RecallsDataClient()
    where = build_where_clause(start_year)

    expected = client.count(where)
    print(f"Downloading {expected} vehicle recall notices issued since {start_year} ...")

    records: list[dict] = []
    for row in client.iter_rows(where):
        records.append(normalise_row(row))
        if len(records) % 2000 == 0:
            print(f"  ... {len(records)}/{expected}", flush=True)

    if expected and abs(len(records) - expected) > max(5, expected * 0.01):
        print(
            f"  ! downloaded {len(records)} rows but expected about {expected}",
            file=sys.stderr,
        )
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    """Write records to a JSON Lines file, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSON Lines file into a list of dictionaries."""
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    """Command-line entry point for the raw-data download step."""
    parser = argparse.ArgumentParser(description="Download NHTSA vehicle recall notices.")
    parser.add_argument("--start-year", type=int, default=config.RECALL_START_YEAR)
    parser.add_argument("--output", type=Path, default=config.RAW_RECALLS_PATH)
    args = parser.parse_args()
    config.ensure_directories()

    records = collect_recalls(args.start_year)
    write_jsonl(records, args.output)
    print(f"Saved {len(records)} recall notices to {args.output}")


if __name__ == "__main__":
    main()

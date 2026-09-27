"""Fetch all Federal Register documents published since 2026-09-01 (public API, no key)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "raw" / "fedreg" / "since_2026-09-01.jsonl"
FIELDS = ["document_number", "title", "abstract", "type", "agencies", "publication_date", "html_url",
          "raw_text_url", "action", "dates"]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    params = [("conditions[publication_date][gte]", "2026-09-01"), ("per_page", "1000"), ("order", "oldest")]
    params += [("fields[]", f) for f in FIELDS]
    url, rows = "https://www.federalregister.gov/api/v1/documents.json", []
    while url:
        r = httpx.get(url, params=params, timeout=120)
        r.raise_for_status()
        d = r.json()
        rows += d.get("results", [])
        url, params = d.get("next_page_url"), None
        time.sleep(1)
    OUT.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    print(len(rows), "documents")


if __name__ == "__main__":
    main()

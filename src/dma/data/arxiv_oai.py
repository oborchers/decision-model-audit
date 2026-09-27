"""Bulk arXiv metadata via OAI-PMH (oaipmh.arxiv.org), the interface arXiv provides for harvesting.

The primary category is the first entry of <categories>; this is cross-checked against the
search API's arxiv:primary_category on a sample in build_s1.py.
"""
from __future__ import annotations

import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw" / "arxiv"
URL = "https://oaipmh.arxiv.org/oai"
NS = {"o": "http://www.openarchives.org/OAI/2.0/", "a": "http://arxiv.org/OAI/arXiv/"}


def harvest(name: str, set_: str, frm: str, until: str | None = None, id_prefix: str | None = None,
            max_pages: int = 200) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    params = {"verb": "ListRecords", "metadataPrefix": "arXiv", "set": set_, "from": frm}
    if until:
        params["until"] = until
    rows, pages = {}, 0
    while pages < max_pages:
        for attempt in range(6):
            try:
                r = httpx.get(URL, params=params, timeout=300)
                if r.status_code == 200:
                    break
                wait = int(r.headers.get("Retry-After", 20))
            except httpx.HTTPError:
                wait = 20
            print(f"{name}: retry {attempt} (wait {wait}s)", flush=True)
            time.sleep(wait)
        root = ET.fromstring(r.content)
        for rec in root.iter("{http://www.openarchives.org/OAI/2.0/}record"):
            m = rec.find(".//a:arXiv", NS)
            if m is None:
                continue
            aid = m.findtext("a:id", namespaces=NS)
            if id_prefix and not aid.startswith(id_prefix):
                continue
            rows[aid] = {
                "arxiv_id": aid,
                "created": m.findtext("a:created", namespaces=NS),
                "title": " ".join((m.findtext("a:title", namespaces=NS) or "").split()),
                "abstract": " ".join((m.findtext("a:abstract", namespaces=NS) or "").split()),
                "categories": (m.findtext("a:categories", namespaces=NS) or "").split(),
            }
        pages += 1
        tok = root.find(".//o:resumptionToken", NS)
        print(f"{name}: page {pages}, kept {len(rows)}", flush=True)
        if tok is None or not (tok.text or "").strip():
            break
        params = {"verb": "ListRecords", "resumptionToken": tok.text.strip()}
        time.sleep(3)
    (RAW / f"{name}.jsonl").write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in rows.values()))
    print(f"{name}: done, {len(rows)} records", flush=True)


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "eval":
        harvest("oai_cs_2609", "cs", "2026-09-01", id_prefix="2609.")
    elif what == "ood":
        for s in ["math", "physics:astro-ph", "physics:cond-mat", "q-bio", "econ"]:
            harvest(f"oai_{s.replace(':', '_')}_2609", s, "2026-09-01", id_prefix="2609.", max_pages=4)
    elif what == "train":
        for m in range(1, 13):
            harvest(f"oai_cs_2024_{m:02d}", "cs", f"2024-{m:02d}-10", f"2024-{m:02d}-17", id_prefix=f"24{m:02d}.",
                    max_pages=14)

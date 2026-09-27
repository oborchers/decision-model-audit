"""Fetch arXiv metadata via the official API (export.arxiv.org), respecting the 3 s rate limit.

Writes raw records to data/raw/arxiv/<name>.jsonl. Sampling happens in build_s1.py.
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
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom",
      "os": "http://a9.com/-/spec/opensearch/1.1/"}
API = "https://export.arxiv.org/api/query"


def query(search: str, start: int = 0, max_results: int = 500) -> tuple[list[dict], int]:
    params = {"search_query": search, "start": start, "max_results": max_results,
              "sortBy": "submittedDate", "sortOrder": "ascending"}
    for attempt in range(6):
        try:
            r = httpx.get(API, params=params, timeout=180)
            if r.status_code == 200 and "<feed" in r.text:
                break
        except httpx.HTTPError:
            pass
        print(f"retry {attempt} {search} start={start}", flush=True); time.sleep(10 * (attempt + 1))
    else:
        raise RuntimeError(f"arXiv API failed for {search} start={start}")
    root = ET.fromstring(r.text)
    total = int(root.find("os:totalResults", NS).text)
    out = []
    for e in root.findall("a:entry", NS):
        aid = e.find("a:id", NS).text.rsplit("/abs/", 1)[-1]
        out.append({
            "arxiv_id": aid,
            "title": " ".join(e.find("a:title", NS).text.split()),
            "abstract": " ".join(e.find("a:summary", NS).text.split()),
            "published": e.find("a:published", NS).text,
            "updated": e.find("a:updated", NS).text,
            "primary_category": e.find("arxiv:primary_category", NS).attrib["term"],
            "categories": [c.attrib["term"] for c in e.findall("a:category", NS)],
        })
    return out, total


def fetch(name: str, search: str, limit: int) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    f = RAW / f"{name}.jsonl"
    seen, rows, start = set(), [], 0
    while start < limit:
        batch, total = query(search, start, min(200, limit - start))
        time.sleep(3.5)
        if not batch:
            break
        for b in batch:
            if b["arxiv_id"] not in seen:
                seen.add(b["arxiv_id"]); rows.append(b)
        start += len(batch)
        if start >= total:
            break
    f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{name}: {len(rows)} records (query total {total})", flush=True)


EVAL_CATS = ["cs.CL", "cs.CV", "cs.CR", "cs.RO", "cs.DB", "cs.SE", "cs.HC", "cs.DC"]
OOD = ["astro-ph", "math", "q-bio", "econ", "cond-mat"]
WINDOW = "submittedDate:[202609010000 TO 202609272359]"

if __name__ == "__main__":
    what = sys.argv[1]
    if what == "eval":
        for c in EVAL_CATS:
            fetch(f"eval_{c}", f"cat:{c} AND {WINDOW}", 3000)
    elif what == "ood":
        for c in OOD:
            fetch(f"ood_{c}", f"cat:{c}* AND {WINDOW}", 500)
    elif what == "train":
        for c in EVAL_CATS:
            for m in range(1, 13):
                end = f"2024{m:02d}{[31,29,31,30,31,30,31,31,30,31,30,31][m-1]}2359"
                fetch(f"train_{c}_{m:02d}", f"cat:{c} AND submittedDate:[2024{m:02d}010000 TO {end}]", 120)

"""Quality check: OAI first category versus the primary subject shown on the arXiv abstract page."""
import json
import random
import re
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[3]
items = [json.loads(l) for l in (ROOT / "data/s1_arxiv/main.jsonl").read_text().splitlines()]
sample = random.Random(20260927).sample(items, 60)
page = {}
for i in sample:
    aid = i["meta"]["arxiv_id"]
    for attempt in range(4):
        try:
            r = httpx.get(f"https://arxiv.org/abs/{aid}", timeout=60, follow_redirects=True)
            if r.status_code == 200:
                m = re.search(r'primary-subject">[^<]*\(([^)]+)\)', r.text)
                d = re.search(r"Submitted on (\d+ \w+ \d{4})", r.text)
                page[aid] = (m.group(1) if m else None, d.group(1) if d else None)
                break
        except httpx.HTTPError:
            pass
        time.sleep(3 * (attempt + 1))
    time.sleep(1)
agree = sum(page.get(i["meta"]["arxiv_id"], (None,))[0] == i["label"] for i in sample)
res = {"sample": len(sample), "found": sum(v[0] is not None for v in page.values()), "primary_agree": agree,
       "submitted_dates": sorted({v[1] for v in page.values() if v[1]}),
       "disagreements": [(i["meta"]["arxiv_id"], i["label"], page.get(i["meta"]["arxiv_id"])) for i in sample
                         if page.get(i["meta"]["arxiv_id"], (None,))[0] != i["label"]]}
(ROOT / "data/s1_arxiv/primary_check.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))

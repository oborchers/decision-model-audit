"""German versus English on the same documents (part 2): FWF project summaries.

Source: FWF Open API (Austrian Science Fund), CC0, public read key at https://openapi.fwf.ac.at/fwfkey/.
Every project has a proposal summary in German and in English written by the applicants, and research fields
(ÖFOS) with percentage shares. Label = the dominant research field (share of at least 60%).

Selection: both summaries present, different, each in the expected language (stop-word heuristic), each 80 to
600 words; approval from 2019-01-01; programme not "Book Publications"; eight fields; 75 projects per field
(fixed seed), five more per field as pilot. Class names in the prompt come from the FWF's own German and English
field names, so no translation of ours enters the labels.

Not fresh: summaries appear months after approval, so models may have seen them. The paired design (same
project, both languages) keeps that from biasing the language difference.

uv run python -m dma.data.build_fwf
"""
from __future__ import annotations

import gzip
import hashlib
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw" / "fwf"
OUT = ROOT / "data" / "fwf"
SEED = 20261002
API = "https://openapi.fwf.ac.at"
UA = {"User-Agent": "decision-model-audit (research benchmark; https://github.com/oborchers/decision-model-audit)"}
FIELDS = ["Mathematics", "Physics, Astronomy", "Chemistry", "Biology", "Computer Sciences", "Geosciences",
          "History, Archaeology", "Linguistics and Literature"]
DE_STOP = {"und", "der", "die", "das", "ist", "nicht", "mit", "für", "von", "werden", "wird", "eine", "ein", "auf", "zu"}
EN_STOP = {"the", "and", "of", "is", "to", "in", "for", "with", "this", "are", "be", "will", "on", "that", "by"}


def fetch() -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / f"projects_{time.strftime('%Y-%m-%d')}.jsonl.gz"
    if out.exists():
        return out
    key = httpx.get(f"{API}/fwfkey/", headers=UA, timeout=30).text.strip()
    h = {**UA, "Authorization": f"Bearer {key}"}
    rows, offset, total = [], 0, None
    while total is None or offset < total:
        d = httpx.get(f"{API}/indexes/projects/documents", params={"limit": 1000, "offset": offset},
                      headers=h, timeout=120).json()
        total = d["total"]
        rows += d["results"]
        offset += 1000
        time.sleep(2)
    with gzip.open(out, "wt") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


def lang_ok(text: str, stop: set) -> bool:
    w = re.findall(r"[a-zäöüß]+", text.lower())
    return len(w) > 0 and sum(x in stop for x in w) / len(w) > 0.08


def dominant(fields_en: str) -> tuple[str, int] | None:
    best = None
    for name, pct in re.findall(r"([^;]+?) \((\d+)%\)", fields_en or ""):
        name, pct = name.strip(), int(pct)
        if best is None or pct > best[1]:
            best = (name, pct)
    return best


def main():
    raw = fetch()
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    rows = [json.loads(l) for l in gzip.open(raw, "rt")]
    by, de_names = defaultdict(list), {}
    for r in rows:
        de, en = (r.get("_str.prproposalsummary.de") or "").strip(), (r.get("_str.prproposalsummary.en") or "").strip()
        de, en = " ".join(de.split()), " ".join(en.split())
        dom = dominant(r.get("_str.researchfields.en"))
        if (not de or not en or de == en or not lang_ok(de, DE_STOP) or not lang_ok(en, EN_STOP)
                or not 80 <= len(de.split()) <= 600 or not 80 <= len(en.split()) <= 600
                or (r.get("_date.approvaldate") or "") < "2019-01-01" or r.get("_str.program.en") == "Book Publications"
                or dom is None or dom[1] < 60 or dom[0] not in FIELDS):
            continue
        fe, fd = r.get("_list.researchfields.en") or [], r.get("_list.researchfields.de") or []
        if dom[0] in fe and len(fe) == len(fd):
            de_names[dom[0]] = fd[fe.index(dom[0])]
        by[dom[0]].append({"id": r["_str.grantdoi"].split("/")[-1], "doi": r["_str.grantdoi"], "de": de, "en": en,
                           "label": dom[0], "share": dom[1], "approved": r.get("_date.approvaldate"),
                           "program": r.get("_str.program.en")})
    rng = random.Random(SEED)
    pilot, main_ = [], []
    for f in FIELDS:
        pool = sorted(by[f], key=lambda x: x["id"])
        rng.shuffle(pool)
        pilot += pool[:5]
        main_ += pool[5:80]
    rng.shuffle(pilot)
    rng.shuffle(main_)
    OUT.mkdir(parents=True, exist_ok=True)
    labels_en = {f: f"Research field: {f}." for f in FIELDS}
    labels_de = {f: f"Forschungsfeld: {de_names[f]}." for f in FIELDS}
    (OUT / "task_en.json").write_text(json.dumps({"name": "fwf", "type": "choice",
        "instructions": "Which research field does this research project primarily belong to?", "labels": labels_en}, indent=1, ensure_ascii=False) + "\n")
    (OUT / "task_de.json").write_text(json.dumps({"name": "fwf", "type": "choice",
        "instructions": "Zu welchem Forschungsfeld gehört dieses Forschungsprojekt hauptsächlich?", "labels": labels_de}, indent=1, ensure_ascii=False) + "\n")
    for split, part in (("pilot", pilot), ("main", main_)):
        for lang in ("de", "en"):
            with (OUT / f"{split}_{lang}.jsonl").open("w") as fh:
                for x in part:
                    fh.write(json.dumps({"item_id": f"fwf-{x['id']}", "task": "fwf", "text": x[lang], "label": x["label"],
                                         "meta": {"lang": lang, "doi": x["doi"], "share": x["share"], "approved": x["approved"],
                                                  "program": x["program"], "words": len(x[lang].split())}},
                                        ensure_ascii=False) + "\n")
    stats = {"raw_file": raw.name, "raw_sha256": digest, "projects": len(rows), "eligible_per_field": {f: len(by[f]) for f in FIELDS},
             "pilot": len(pilot), "main": len(main_), "seed": SEED, "german_field_names": de_names}
    (OUT / "build_stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(stats, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

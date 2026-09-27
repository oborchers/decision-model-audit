"""Build S2 (Federal Register issuing agency) with masking, near-duplicate removal and leakage audit."""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw" / "fedreg" / "since_2026-09-01.jsonl"
OUT = ROOT / "data" / "s2_fedreg"
SEED = 20260927

AGENCIES = {
    "federal-aviation-administration": ("FAA", "Aviation safety: aircraft, airworthiness, airspace and airports."),
    "international-trade-administration": ("ITA", "International trade enforcement: antidumping and countervailing duties, trade promotion."),
    "national-park-service": ("NPS", "Management of national parks, monuments, historic places and cultural resources."),
    "food-and-drug-administration": ("FDA", "Safety of food, drugs, medical devices, biologics and tobacco products."),
    "environmental-protection-agency": ("EPA", "Environmental protection: air and water quality, pesticides, chemicals and waste."),
    "national-oceanic-and-atmospheric-administration": ("NOAA", "Oceans, fisheries, marine mammals, weather and atmosphere."),
}
MASK_TERMS = [
    "Federal Aviation Administration", "FAA", "International Trade Administration", "ITA",
    "Enforcement and Compliance", "National Park Service", "NPS", "Food and Drug Administration", "FDA",
    "Environmental Protection Agency", "EPA", "National Oceanic and Atmospheric Administration", "NOAA",
    "National Marine Fisheries Service", "NMFS", "Department of Transportation", "DOT", "Department of Commerce",
    "Commerce", "Department of the Interior", "Interior", "Department of Health and Human Services", "HHS",
    "the Agency", "Agency", "Administrator", "the Administration", "the Service", "the Department", "we", "We",
]
MASK_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, MASK_TERMS[:-2]), key=len, reverse=True)) + r")\b")
LEAK_RE = re.compile(r"\b(FAA|ITA|NPS|FDA|EPA|NOAA|NMFS|Aviation Administration|Trade Administration|"
                     r"Park Service|Drug Administration|Protection Agency|Oceanic)\b", re.I)


def leaf(agencies: list[dict]) -> str | None:
    slugs = [a.get("slug") for a in agencies if a.get("slug")]
    parents = {a.get("parent_id") for a in agencies}
    leaves = [a.get("slug") for a in agencies if a.get("id") not in parents and a.get("slug")]
    return leaves[-1] if len(set(leaves)) == 1 else None


def main() -> None:
    rows = [json.loads(l) for l in RAW.read_text().splitlines()]
    items = []
    for r in rows:
        ab = r.get("abstract") or ""
        if len(ab.split()) < 30:
            continue
        a = leaf(r["agencies"])
        if a not in AGENCIES:
            continue
        text = f"{r['title']}\n\n{ab}"
        masked = MASK_RE.sub("[AGENCY]", text)
        items.append({"doc": r["document_number"], "agency": a, "type": r["type"], "text": masked,
                      "url": r["html_url"], "date": r["publication_date"], "leak": bool(LEAK_RE.search(masked))})
    # near-duplicate removal within agency: keep one per cluster with cosine >= 0.9
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit([i["text"] for i in items])
    keep, dropped = [], 0
    rng = random.Random(SEED)
    rng.shuffle(items)
    kept_vecs = []
    for it in items:
        v = vec.transform([it["text"]])
        if any(cosine_similarity(v, k)[0, 0] >= 0.9 for k, a in kept_vecs if a == it["agency"]):
            dropped += 1
            continue
        kept_vecs.append((v, it["agency"]))
        keep.append(it)
    by = {a: [i for i in keep if i["agency"] == a] for a in AGENCIES}
    OUT.mkdir(parents=True, exist_ok=True)
    pilot, main_ = [], []
    for a, lst in by.items():
        pilot += lst[:5]
        main_ += lst[5:]
    def to_item(i):
        return {"item_id": f"fr-{i['doc']}", "task": "s2_fedreg", "text": i["text"], "label": AGENCIES[i["agency"]][0],
                "meta": {"url": i["url"], "date": i["date"], "doc_type": i["type"], "residual_leak": i["leak"]}}
    for name, lst in [("pilot", pilot), ("main", main_)]:
        (OUT / f"{name}.jsonl").write_text("".join(json.dumps(to_item(i), ensure_ascii=False) + "\n" for i in lst))
    task = {"name": "s2_fedreg", "type": "choice",
            "instructions": "Which U.S. federal agency issued this Federal Register document? Agency names are masked as [AGENCY].",
            "labels": {v[0]: v[1] for v in AGENCIES.values()},
            "none_label": {"none": "None of the listed agencies issued this document."}}
    (OUT / "task.json").write_text(json.dumps(task, indent=1))
    stats = {"candidates": len(items), "near_duplicates_dropped": dropped,
             "per_agency_kept": {AGENCIES[a][0]: len(l) for a, l in by.items()},
             "pilot": len(pilot), "main": len(main_),
             "residual_leak_rate_main": round(sum(i["meta"]["residual_leak"] for i in map(to_item, main_)) / len(main_), 4)}
    (OUT / "build_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()

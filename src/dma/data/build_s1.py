"""Build S1 (arXiv primary category), P4 (no fitting label) and the 2024 training set for tfidf."""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw" / "arxiv"
OUT = ROOT / "data" / "s1_arxiv"
P4 = ROOT / "data" / "probes"
SEED = 20260927

LABELS = {
    "cs.CL": "Computation and Language: natural language processing, computational linguistics, speech and language models.",
    "cs.CV": "Computer Vision and Pattern Recognition: image and video analysis, visual recognition, image generation.",
    "cs.CR": "Cryptography and Security: cryptography, authentication, attacks and defenses, privacy.",
    "cs.RO": "Robotics: robot control, manipulation, locomotion, and planning and perception for robots.",
    "cs.DB": "Databases: data management, data models, query languages and query processing.",
    "cs.SE": "Software Engineering: design, testing, verification, maintenance and tooling of software.",
    "cs.HC": "Human-Computer Interaction: user interfaces, user studies, interaction design, human factors.",
    "cs.DC": "Distributed, Parallel, and Cluster Computing: distributed systems, parallel computation, cloud and clusters.",
}
PARA0 = {
    "cs.CL": "Research on human language: processing, understanding or generating text and speech with computers.",
    "cs.CV": "Research on making computers see: analysing, recognising or synthesising images and video.",
    "cs.CR": "Research on protecting systems and data: encryption, security vulnerabilities, attacks and privacy.",
    "cs.RO": "Research on physical robots: how they move, grasp, plan and perceive their surroundings.",
    "cs.DB": "Research on storing and querying data: database systems, data models and query evaluation.",
    "cs.SE": "Research on how software is built and maintained: development processes, testing and program analysis.",
    "cs.HC": "Research on people using technology: interface design, usability and studies with users.",
    "cs.DC": "Research on computing across many machines or cores: distributed, parallel and cloud systems.",
}
PARA1 = {k: v.split(":")[0] for k, v in LABELS.items()}  # names only
NONE = {"none": "None of the listed categories fits this paper."}
P4_ARCHIVES = {"astro-ph", "math", "q-bio", "econ", "cond-mat"}
OOD_FILES = ["oai_math_2609", "oai_physics_astro-ph_2609", "oai_physics_cond-mat_2609", "oai_q-bio_2609", "oai_econ_2609"]


def load(name: str) -> list[dict]:
    f = RAW / f"{name}.jsonl"
    return [json.loads(l) for l in f.read_text().splitlines()] if f.exists() else []


def fresh(r: dict) -> bool:
    return r["arxiv_id"].startswith("2609.") and (r.get("created") or "") >= "2026-09-01" \
        and len(r["abstract"].split()) >= 50


def to_item(r: dict, task: str) -> dict:
    cats = r["categories"]
    return {"item_id": f"ax-{r['arxiv_id']}", "task": task, "text": f"{r['title']}\n\n{r['abstract']}",
            "label": cats[0] if task == "s1_arxiv" else "none",
            "meta": {"arxiv_id": r["arxiv_id"], "created": r["created"], "categories": cats,
                     "ambiguous": any(c in LABELS for c in cats[1:]), "url": f"https://arxiv.org/abs/{r['arxiv_id']}"}}


def build_eval() -> None:
    rows = [r for r in load("oai_cs_2609") if fresh(r) and r["categories"][0] in LABELS]
    by = {c: [r for r in rows if r["categories"][0] == c] for c in LABELS}
    rng = random.Random(SEED)
    pilot, main = [], []
    for c, lst in by.items():
        lst = sorted(lst, key=lambda r: r["arxiv_id"])
        rng.shuffle(lst)
        pilot += lst[:5]
        main += lst[5:55]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, lst in [("pilot", pilot), ("main", main)]:
        (OUT / f"{name}.jsonl").write_text("".join(json.dumps(to_item(r, "s1_arxiv"), ensure_ascii=False) + "\n"
                                                   for r in lst))
    task = {"name": "s1_arxiv", "type": "choice",
            "instructions": "Which arXiv category is the primary category of this computer science paper?",
            "labels": LABELS, "none_label": NONE, "paraphrases": [PARA0, PARA1]}
    (OUT / "task.json").write_text(json.dumps(task, indent=1))
    stats = {"fresh_primary_in_set": {c: len(v) for c, v in by.items()}, "pilot": len(pilot), "main": len(main),
             "ambiguous_share_main": round(sum(to_item(r, "s1_arxiv")["meta"]["ambiguous"] for r in main) / len(main), 4)}
    (OUT / "build_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


def build_p4() -> None:
    pool = []
    for f in OOD_FILES:
        pool += [r for r in load(f) if fresh(r) and r["categories"][0].split(".")[0] in P4_ARCHIVES
                 and not any(c.startswith(("cs.", "eess.", "stat.")) for c in r["categories"])]
    rng = random.Random(SEED)
    fam = Counter(r["categories"][0].split(".")[0] for r in pool)
    pool = sorted(pool, key=lambda r: r["arxiv_id"])
    rng.shuffle(pool)
    # up to 20 per top-level archive for spread
    per, picked = Counter(), []
    for r in pool:
        a = r["categories"][0].split(".")[0]
        if per[a] < 22:
            per[a] += 1
            picked.append(r)
    pilot, main = picked[:10], picked[10:110]
    for name, lst in [("pilot", pilot), ("main", main)]:
        items = [to_item(r, "p4_no_fit") for r in lst]
        for it in items:
            it["meta"]["family"] = it["meta"]["categories"][0].split(".")[0]
        (P4 / f"p4_no_fit.{name}.jsonl").write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items))
    t = json.loads((OUT / "task.json").read_text())
    t["name"] = "p4_no_fit"
    (P4 / "p4_no_fit.task.json").write_text(json.dumps(t, indent=1))
    print("P4 pool by archive", dict(fam), "picked", dict(per), "pilot", len(pilot), "main", len(main))


def build_train() -> None:
    rows = []
    for m in range(1, 13):
        rows += [r for r in load(f"oai_cs_2024_{m:02d}") if r["categories"][0] in LABELS
                 and r["arxiv_id"].startswith(f"24{m:02d}.") and len(r["abstract"].split()) >= 50]
    rng = random.Random(SEED)
    rows = sorted({r["arxiv_id"]: r for r in rows}.values(), key=lambda r: r["arxiv_id"])
    rng.shuffle(rows)
    per, out = Counter(), []
    for r in rows:
        c = r["categories"][0]
        if per[c] < 400:
            per[c] += 1
            out.append(to_item(r, "s1_arxiv"))
    (OUT / "train_2024.jsonl").write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in out))
    print("train per class", dict(per))


if __name__ == "__main__":
    {"eval": build_eval, "p4": build_p4, "train": build_train}[sys.argv[1]]()

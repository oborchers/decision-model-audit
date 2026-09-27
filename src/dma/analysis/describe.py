"""Descriptive statistics and quality checks for every dataset (results/data_quality.json and .md)."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[3]
SETS = {
    "s1_arxiv": ["data/s1_arxiv/pilot.jsonl", "data/s1_arxiv/main.jsonl"],
    "s1_train_2024": ["data/s1_arxiv/train_2024.jsonl"],
    "s2_fedreg": ["data/s2_fedreg/pilot.jsonl", "data/s2_fedreg/main.jsonl"],
    "p1_calibration": ["data/probes/p1_calibration.pilot.jsonl", "data/probes/p1_calibration.main.jsonl"],
    "p2_minimal_pairs": ["data/probes/p2_minimal_pairs.pilot.jsonl", "data/probes/p2_minimal_pairs.main.jsonl"],
    "p3_long_input": ["data/probes/p3_long_input.pilot.jsonl", "data/probes/p3_long_input.main.jsonl"],
    "p4_no_fit": ["data/probes/p4_no_fit.pilot.jsonl", "data/probes/p4_no_fit.main.jsonl"],
}
S1_NAME_RE = re.compile(r"\b(natural language processing|computer vision|cryptograph\w*|robot\w*|database\w*|"
                        r"software engineering|human-computer interaction|distributed (system|comput)\w*)\b", re.I)


def load(p):
    f = ROOT / p
    return [json.loads(l) for l in f.read_text().splitlines()] if f.exists() else []


def lengths(items):
    w = np.array([len(i["text"].split()) for i in items])
    return {"mean": round(float(w.mean()), 1), "median": float(np.median(w)), "p5": float(np.percentile(w, 5)),
            "p95": float(np.percentile(w, 95)), "min": int(w.min()), "max": int(w.max())}


def near_dups(items, thr=0.9):
    if len(items) < 2:
        return 0
    X = TfidfVectorizer(ngram_range=(1, 2)).fit_transform([i["text"] for i in items])
    S = cosine_similarity(X)
    np.fill_diagonal(S, 0)
    return int((S >= thr).any(axis=1).sum())


def main():
    out, all_ids = {}, Counter()
    for name, files in SETS.items():
        splits = {Path(f).name: load(f) for f in files}
        items = [i for v in splits.values() for i in v]
        if not items:
            continue
        for i in items:
            all_ids[i["item_id"]] += 1
        d = {"n": {k: len(v) for k, v in splits.items()}, "words": lengths(items),
             "labels": dict(Counter(str(i["label"]) for i in items)),
             "exact_duplicate_texts": len(items) - len({i["text"] for i in items}),
             "near_duplicate_items_cos>=0.9": near_dups(items) if name.startswith(("s1", "s2", "p4")) else None,
             "pilot_main_overlap": len({i["item_id"] for i in splits.get("pilot.jsonl", [])} &
                                       {i["item_id"] for i in splits.get("main.jsonl", [])})}
        if name == "s1_arxiv":
            d["ambiguous_share"] = round(np.mean([i["meta"]["ambiguous"] for i in items]), 4)
            d["label_topic_phrase_in_text_share"] = round(np.mean([bool(S1_NAME_RE.search(i["text"])) for i in items]), 4)
            d["n_categories_per_item"] = dict(Counter(len(i["meta"]["categories"]) for i in items))
        if name == "s2_fedreg":
            d["residual_leak_share"] = round(np.mean([i["meta"]["residual_leak"] for i in items]), 4)
            d["doc_types"] = dict(Counter(i["meta"]["doc_type"] for i in items))
        if name == "p3_long_input":
            d["cells"] = dict(Counter(f"{i['meta']['length']}@{i['meta']['position']}" for i in items))
        if name == "p1_calibration":
            d["p_levels"] = dict(Counter(round(i["meta"]["p"], 2) for i in items))
        out[name] = d
    ev = {i["item_id"] for f in SETS["s1_arxiv"] for i in load(f)}
    tr = {i["item_id"] for i in load(SETS["s1_train_2024"][0])}
    p4 = {i["item_id"] for f in SETS["p4_no_fit"] for i in load(f)}
    out["cross_checks"] = {"s1_train_eval_overlap": len(ev & tr), "s1_p4_overlap": len(ev & p4),
                           "duplicate_item_ids_across_all": sum(1 for v in all_ids.values() if v > 1)}
    pc = ROOT / "data/s1_arxiv/primary_check.json"
    if pc.exists():
        c = json.loads(pc.read_text())
        out["cross_checks"]["s1_primary_category_check"] = {k: c[k] for k in ("sample", "found", "primary_agree")}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/data_quality.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

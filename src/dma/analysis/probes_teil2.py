"""Probes P5 to P7 (part 2, post hoc): per-item options (P5), source and claim (P6), counting score (P7).
Writes results/probes_teil2.json.

Accuracy with Wilson interval, ECE (equal mass) and AURC on the returned confidence, paired comparison against Jev
(exact McNemar, Holm over all systems of the probe). P6 adds balanced accuracy, accuracy per kind and AUROC of
p(yes); P7 adds within-one-level accuracy. The embedding baseline's probabilities are not calibrated, so no ECE.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from dma.analysis.metrics import aurc, ece_equal_mass, holm, mcnemar_exact, wilson

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "results/raw/main"
PROBES = {"p5": "p5_titles", "p6": "p6_claims", "p7": "p7_count"}
UNCALIBRATED = {"emb-qwen3-8b-cos"}


def load(p: str) -> dict:
    last = {}
    f = RAW / f"{p}.teil2.jsonl"
    for l in f.read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            last[(r["system"], r["item_id"])] = r
    by = defaultdict(dict)
    for (s, i), r in last.items():
        by[s][i] = r
    return by


def main():
    out = {}
    for p, name in PROBES.items():
        items = {json.loads(l)["item_id"]: json.loads(l) for l in (ROOT / f"data/probes/{name}.main.jsonl").read_text().splitlines()}
        ids = sorted(items)
        by = load(p)
        res, correct = {}, {}
        for s, rows in sorted(by.items()):
            if set(rows) != set(ids):
                res[s] = {"note": f"incomplete: {len(rows)} of {len(ids)} items"}
                continue
            ok = np.array([rows[i]["valid"] and str(rows[i]["pred"]) == str(items[i]["label"]) for i in ids])
            correct[s] = ok
            conf = np.array([float(rows[i]["confidence"]) if rows[i]["valid"] and rows[i]["confidence"] is not None else 0.0 for i in ids])
            d = {"n": len(ids), "invalid": int(sum(not rows[i]["valid"] for i in ids)), "accuracy": round(float(ok.mean()), 4),
                 "acc_ci95": [round(x, 4) for x in wilson(int(ok.sum()), len(ids))], "aurc": round(aurc(conf, ok), 4)}
            if s not in UNCALIBRATED:
                d["ece"] = round(ece_equal_mass(conf, ok.astype(float)), 4)
            if p == "p6":
                gold = np.array([bool(items[i]["label"]) for i in ids])
                py = np.array([float(rows[i]["probs"].get("yes", 0.5)) if rows[i]["valid"] else 0.5 for i in ids])
                d["balanced_accuracy"] = round(float((ok[gold].mean() + ok[~gold].mean()) / 2), 4)
                d["auroc_p_yes"] = round(float(roc_auc_score(gold, py)), 4)
                kinds = defaultdict(list)
                for i, k in zip(ids, ok):
                    kinds[items[i]["meta"]["kind"]].append(k)
                d["accuracy_by_kind"] = {k: round(float(np.mean(v)), 4) for k, v in kinds.items()}
            if p == "p7":
                w1 = [rows[i]["valid"] and abs(int(rows[i]["pred"]) - int(items[i]["label"])) <= 1 for i in ids]
                d["within_one_level"] = round(float(np.mean(w1)), 4)
            res[s] = d
        if "jev" in correct:
            pv = {s: mcnemar_exact(correct["jev"], c)["p"] for s, c in correct.items() if s != "jev"}
            for s, ph in holm(pv).items():
                m = mcnemar_exact(correct["jev"], correct[s])
                res[s]["vs_jev"] = {"delta_acc": round(float(correct[s].mean() - correct["jev"].mean()), 4),
                                    "mcnemar": {**m, "p_holm": round(ph, 5)}}
        out[p] = res
    (ROOT / "results/probes_teil2.json").write_text(json.dumps(out, indent=1) + "\n")
    for p, res in out.items():
        print(p, {s: d.get("accuracy") for s, d in res.items()})


if __name__ == "__main__":
    main()

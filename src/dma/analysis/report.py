"""Main analysis (protocol section 5). Writes results/summary.json and results/REPORT.md.

uv run python -m dma.analysis.report [--raw results/raw/main]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from dma.analysis.metrics import (aurc, bootstrap_ci, brier_top, ece_equal_mass, holm, macro_f1, mcnemar_exact,
                                  risk_at_coverage, wilson)

ROOT = Path(__file__).resolve().parents[3]
SPLIT = "main"
EXTRA = {"clef", "clef-flash", "eikos-4b", "semif-4b", "kev-0.8b", "tfidf-bal", "emb-qwen3-8b", "emb-qwen3-8b-unbal", "emb-oai-3l", "emb-oai-3l-unbal"}  # post hoc systems: separate Holm family
CLEF = {"clef", "clef-flash"}  # added 2026-10-02 in their own Holm family, so earlier adjusted p values stay unchanged


def load_rows(raw: Path, name: str) -> list[dict]:
    rows = []
    for f in [raw / f"{name}.jsonl", raw / f"{name}.local.jsonl", raw / f"{name}.extra.jsonl", raw / f"{name}.supervised.jsonl", raw / f"{name}.clef.jsonl"]:
        if f.exists():
            rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    # keep the last row per (system, variant, item) so reruns supersede earlier rows
    last, lat = {}, {}
    for r in rows:
        k = (r["system"], r["variant"], r["item_id"])
        last[k] = r
        if r.get("latency_s") is not None and not r.get("cached"):
            lat[k] = r["latency_s"]  # latency of the original, non-cached request
    for k, r in last.items():
        if r.get("latency_s") is None and k in lat:
            r["latency_s"] = lat[k]
    return list(last.values())


def items_of(path: str) -> dict:
    path = path.replace(".main.jsonl", f".{SPLIT}.jsonl").replace("/main.jsonl", f"/{SPLIT}.jsonl")
    return {json.loads(l)["item_id"]: json.loads(l) for l in (ROOT / path).read_text().splitlines()}


def group(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["system"], r["variant"])].append(r)
    return g


def pctl(x, q):
    x = [v for v in x if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return round(float(np.percentile(x, q)), 3) if x else None


def choice_block(rows, labels, item_meta=None, subset=None):
    rows = sorted(rows, key=lambda r: r["item_id"])
    if subset is not None:
        rows = [r for r in rows if subset(r)]
    n = len(rows)
    if not n:
        return None
    valid = [r for r in rows if r["valid"]]
    correct = np.array([r["valid"] and r["pred"] == r["gold"] for r in rows])
    conf = np.array([float(r["confidence"]) if r["valid"] and r["confidence"] is not None else 0.0 for r in rows])
    conf = np.clip(conf, 0, 1)
    k = int(correct.sum())
    lo, hi = wilson(k, n)
    out = {
        "n": n, "invalid": n - len(valid), "accuracy": round(k / n, 4), "acc_ci95": [round(lo, 4), round(hi, 4)],
        "macro_f1": round(macro_f1([r["gold"] for r in rows], [r["pred"] for r in rows], labels), 4),
        "brier_top": round(brier_top(conf, correct.astype(float)), 4),
        "aurc": round(aurc(conf, correct.astype(float)), 4),
        "risk_at_80": round(risk_at_coverage(conf, correct.astype(float), 0.8), 4),
        "conf_eq_1_share": round(float(np.mean(conf >= 0.9999)), 4),
        "mean_conf": round(float(conf.mean()), 4),
        "cost_per_1k_usd": round(1000 * np.mean([r["cost_usd"] or 0 for r in rows]), 4),
        "latency_p50_s": pctl([r["latency_s"] for r in rows], 50),
        "latency_p95_s": pctl([r["latency_s"] for r in rows], 95),
    }
    if n >= 200:
        out["ece"] = round(ece_equal_mass(conf, correct.astype(float)), 4)
        out["ece_ci95"] = [round(v, 4) for v in bootstrap_ci(lambda c, y: ece_equal_mass(c, y), conf,
                                                             correct.astype(float), n=2000)]
    return out, {r["item_id"]: (bool(c), float(p)) for r, c, p in zip(rows, correct, conf)}


def paired_vs(ref: dict, other: dict) -> dict:
    ids = sorted(set(ref) & set(other))
    a = np.array([ref[i][0] for i in ids])
    b = np.array([other[i][0] for i in ids])
    ca = np.array([ref[i][1] for i in ids])
    cb = np.array([other[i][1] for i in ids])
    mc = mcnemar_exact(a, b)
    d_acc = float(b.mean() - a.mean())
    ci = bootstrap_ci(lambda x, y: y.mean() - x.mean(), a.astype(float), b.astype(float), n=10000)
    r80 = lambda c1, y1, c2, y2: risk_at_coverage(c2, y2, .8) - risk_at_coverage(c1, y1, .8)
    d_r80 = r80(ca, a.astype(float), cb, b.astype(float))
    ci_r80 = bootstrap_ci(r80, ca, a.astype(float), cb, b.astype(float), n=10000)
    return {"n_paired": len(ids), "delta_acc_vs_jev": round(d_acc, 4), "delta_acc_ci95": [round(x, 4) for x in ci],
            "delta_risk80_vs_jev": round(d_r80, 4), "delta_risk80_ci95": [round(x, 4) for x in ci_r80],
            "mcnemar": mc}


def analyse_choice_source(name, raw, task_path, items_path, ambiguous_key=None):
    task = json.loads((ROOT / task_path).read_text())
    labels = list(task["labels"])
    items = items_of(items_path)
    g = group(load_rows(raw, name))
    res, per_item = {}, {}
    for (s, v), rows in sorted(g.items()):
        if v not in ("choice", "rationale", "reversed", "para0", "para1", "yesno", "choice_none"):
            continue
        b = choice_block(rows, labels)
        if b is None:
            continue
        res[f"{s}/{v}"], per_item[(s, v)] = b
        if ambiguous_key:
            ub = choice_block(rows, labels, subset=lambda r: not items[r["item_id"]]["meta"][ambiguous_key])
            if ub:
                res[f"{s}/{v}"]["unambiguous"] = {k: ub[0][k] for k in ("n", "accuracy", "acc_ci95", "risk_at_80")}
    # paired comparisons against jev/choice
    if ("jev", "choice") in per_item:
        ref = per_item[("jev", "choice")]
        pv, comps = {}, {}
        for (s, v), pi in per_item.items():
            if (s, v) == ("jev", "choice") or v not in ("choice", "rationale"):
                continue
            comps[f"{s}/{v}"] = paired_vs(ref, pi)
            pv[f"{s}/{v}"] = comps[f"{s}/{v}"]["mcnemar"]["p"]
        pre = {k: p for k, p in pv.items() if k.split("/")[0] not in EXTRA and k.endswith("/choice")}
        rat = {k: p for k, p in pv.items() if k.endswith("/rationale")}
        post = {k: p for k, p in pv.items() if k.split("/")[0] in EXTRA - CLEF}
        clef = {k: p for k, p in pv.items() if k.split("/")[0] in CLEF}
        for fam, pvals in (("pre-registered systems", pre), ("rationale variants", rat), ("post hoc", post),
                           ("post hoc, Clef (October 2026)", clef)):
            for k, p in holm(pvals).items():
                comps[k]["mcnemar"]["p_holm"] = round(p, 5)
                comps[k]["holm_family"] = fam
        res["_paired_vs_jev_choice"] = comps
    # stability of jev variants relative to jev/choice
    stab = {}
    base = {r["item_id"]: r["pred"] for r in g.get(("jev", "choice"), [])}
    for v in ("reversed", "para0", "para1", "yesno"):
        rows = g.get(("jev", v), [])
        if rows and base:
            stab[f"jev/{v}"] = {"agreement_with_choice": round(np.mean([base.get(r["item_id"]) == r["pred"] for r in rows]), 4)}
            if v == "yesno":
                sums = [r.get("yes_sum") for r in rows if r.get("yes_sum") is not None]
                stab["jev/yesno"]["yes_sum"] = {"median": pctl(sums, 50), "p5": pctl(sums, 5), "p95": pctl(sums, 95),
                                                "share_outside_0.9_1.1": round(float(np.mean([(x < .9) | (x > 1.1) for x in sums])), 4)}
    for s in {k[0] for k in g}:
        if (s, "yesno") in g and (s, "choice") in g and s != "jev":
            b2 = {r["item_id"]: r["pred"] for r in g[(s, "choice")]}
            stab[f"{s}/yesno"] = {"agreement_with_choice": round(np.mean([b2.get(r["item_id"]) == r["pred"] for r in g[(s, "yesno")]]), 4)}
    res["_stability"] = stab
    # rationale cost and latency relative to label-only
    rat = {}
    for s in {k[0] for k in g}:
        if (s, "rationale") in g and (s, "choice") in g:
            c0 = [r["cost_usd"] or 0 for r in g[(s, "choice")]]
            c1 = [r["cost_usd"] or 0 for r in g[(s, "rationale")]]
            t0 = [r["usage"]["completion_tokens"] for r in g[(s, "choice")] if r.get("usage")]
            t1 = [r["usage"]["completion_tokens"] for r in g[(s, "rationale")] if r.get("usage")]
            rat[s] = {"cost_ratio": round(np.mean(c1) / max(np.mean(c0), 1e-12), 2),
                      "output_tokens_mean": [round(float(np.mean(t0)), 1), round(float(np.mean(t1)), 1)],
                      "latency_p50_s": [pctl([r["latency_s"] for r in g[(s, "choice")]], 50),
                                        pctl([r["latency_s"] for r in g[(s, "rationale")]], 50)]}
    # exploratory: quote fidelity of rationales
    import re
    qre = re.compile(r"[\u201c\"]([^\u201d\"]{6,}?)[\u201d\"]|'([^']{6,}?)'")
    for s in list(rat):
        found, total = 0, 0
        for r in g[(s, "rationale")]:
            txt = items[r["item_id"]]["text"].lower()
            for a, b in qre.findall(r.get("reasoning") or ""):
                q = (a or b).strip().lower().rstrip(".,")
                total += 1
                found += q in txt
        rat[s]["quotes_total"] = total
        rat[s]["quote_fidelity"] = round(found / total, 4) if total else None
        rat[s]["items_with_quote"] = round(np.mean([bool(qre.search(r.get("reasoning") or "")) for r in g[(s, "rationale")]]), 4)
    # Q6: paired within-system test, rationale vs label only (Holm across the rationale systems)
    pv6 = {}
    for s in list(rat):
        a = {r["item_id"]: r["valid"] and r["pred"] == r["gold"] for r in g[(s, "choice")]}
        b = {r["item_id"]: r["valid"] and r["pred"] == r["gold"] for r in g[(s, "rationale")]}
        ids = sorted(set(a) & set(b))
        x = np.array([a[i] for i in ids]); y = np.array([b[i] for i in ids])
        mc = mcnemar_exact(x, y)
        ci = bootstrap_ci(lambda u, v: v.mean() - u.mean(), x.astype(float), y.astype(float), n=10000)
        rat[s]["delta_acc_rationale_vs_label"] = round(float(y.mean() - x.mean()), 4)
        rat[s]["delta_ci95"] = [round(c, 4) for c in ci]
        rat[s]["mcnemar"] = mc
        pv6[s] = mc["p"]
    for s, p in holm(pv6).items():
        rat[s]["mcnemar"]["p_holm"] = round(p, 5)
    res["_rationale"] = rat
    # exploratory: exclude items where all API LLM choice runs agree on the same non-gold label
    llm = [k for k in g if k[1] == "choice" and k[0] in ("luna", "flash", "haiku", "sonnet")]
    if len(llm) >= 3:
        preds = defaultdict(set)
        cnt = defaultdict(int)
        for k in llm:
            for r in g[k]:
                preds[r["item_id"]].add(r["pred"]); cnt[r["item_id"]] += 1
        gold = {r["item_id"]: r["gold"] for k in llm for r in g[k]}
        drop = {i for i, p in preds.items() if cnt[i] == len(llm) and len(p) == 1 and gold[i] not in p}
        sens = {"excluded_items": len(drop)}
        for (s, v), rows in g.items():
            if v == "choice":
                keep = [r for r in rows if r["item_id"] not in drop]
                sens[s] = round(np.mean([r["valid"] and r["pred"] == r["gold"] for r in keep]), 4)
        res["_exploratory_consensus_exclusion"] = sens
        # exploratory: bounds on attainable accuracy (label = author's choice among overlapping categories)
        systems = llm + ([("jev", "choice")] if ("jev", "choice") in g else [])
        corr = defaultdict(dict)
        for k in systems:
            for r in g[k]:
                corr[r["item_id"]][k[0]] = (r["pred"] == r["gold"], r["pred"])
        full = {i: c for i, c in corr.items() if len(c) == len(systems)}
        # majority vote; ties broken by the order of `systems` (first system's label among the tied labels wins)
        def vote(c):
            labs = [c[k[0]][1] for k in systems]
            best = max(labs.count(l) for l in labs)
            return next(l for l in labs if labs.count(l) == best)
        votes = [vote(c) for c in full.values()]
        n_ties = sum(1 for c in full.values()
                     if sorted([ [p for _, p in c.values()].count(l) for l in set(p for _, p in c.values())])[-2:] in ([2, 2],)
                     )
        golds = [gold[i] for i in full]
        res["_exploratory_bounds"] = {
            "systems": [k[0] for k in systems], "n": len(full),
            "oracle_any_correct": round(np.mean([any(x for x, _ in c.values()) for c in full.values()]), 4),
            "all_wrong": round(np.mean([not any(x for x, _ in c.values()) for c in full.values()]), 4),
            "all_wrong_same_label": len(drop) if len(systems) == len(llm) else sum(
                1 for c in full.values() if not any(x for x, _ in c.values()) and len({p for _, p in c.values()}) == 1),
            "majority_vote_accuracy": round(np.mean([v == gd for v, gd in zip(votes, golds)]), 4),
            "majority_vote_ties_2_2": n_ties, "tie_rule": "first system in order luna, flash, haiku, sonnet, jev",
            "note": "indicative only; not a measured ceiling (no independent adjudication)"}
    return res


def analyse_p1(raw):
    items = items_of("data/probes/p1_calibration.main.jsonl")
    out = {}
    for (s, v), rows in sorted(group(load_rows(raw, "p1")).items()):
        rows = [r for r in rows if r["item_id"] in items]
        ok = [r for r in rows if r["valid"]]
        p = np.array([items[r["item_id"]]["meta"]["p"] for r in ok])
        q = np.array([r["probs"]["yes"] for r in ok])
        fam = np.array([items[r["item_id"]]["meta"]["family"] + str(items[r["item_id"]]["meta"]["total"]) for r in ok])
        err = np.abs(q - p)
        out[s] = {"n": len(rows), "invalid": len(rows) - len(ok), "mae": round(float(err.mean()), 4),
                  "mae_ci95_cluster": [round(x, 4) for x in bootstrap_ci(lambda e: e.mean(), err, clusters=fam, n=5000)],
                  "max_abs_error": round(float(err.max()), 4),
                  "share_abs_error_gt_0.1": round(float(np.mean(err > 0.1)), 4),
                  "by_level": {str(round(l, 1)): round(float(q[np.isclose(p, l, atol=0.051)].mean()), 3)
                               for l in np.arange(0.1, 1.0, 0.1)}}
    return out


def analyse_p2(raw):
    items = items_of("data/probes/p2_minimal_pairs.main.jsonl")
    out = {}
    for (s, v), rows in sorted(group(load_rows(raw, "p2")).items()):
        rows = [r for r in rows if r["item_id"] in items]
        corr = {r["item_id"]: r["valid"] and r["pred"] == r["gold"] for r in rows}
        pairs = defaultdict(list)
        for i, c in corr.items():
            pairs[items[i]["meta"]["pair"]].append(c)
        full = [all(v) for v in pairs.values() if len(v) == 2]
        k = sum(corr.values())
        out[s] = {"n": len(rows), "accuracy": round(k / len(rows), 4), "acc_ci95": [round(x, 4) for x in wilson(k, len(rows))],
                  "pairs": len(full), "pair_consistency": round(float(np.mean(full)), 4) if full else None,
                  "errors": sorted(i for i, c in corr.items() if not c)}
    return out


def analyse_p3(raw):
    items = items_of("data/probes/p3_long_input.main.jsonl")
    out = {}
    for (s, v), rows in sorted(group(load_rows(raw, "p3")).items()):
        rows = [r for r in rows if r["item_id"] in items]
        cells = defaultdict(list)
        for r in rows:
            m = items[r["item_id"]]["meta"]
            ok = r["valid"] and r["pred"] == r["gold"]
            cells[(m["length"], m["position"], r["gold"])].append((ok, r["probs"].get("yes") if r["valid"] else None))
        by_len = defaultdict(list)
        for (L, pos, gold), v2 in cells.items():
            by_len[L] += [x[0] for x in v2]
        pos_recall = {f"{L}@{pos}": round(float(np.mean([x[0] for x in v2])), 2)
                      for (L, pos, gold), v2 in sorted(cells.items()) if gold}
        neg_spec = {str(L): round(float(np.mean([x[0] for (L2, pos, g), v2 in cells.items() if L2 == L and not g
                                                  for x in v2])), 2) for L in sorted({k[0] for k in cells})}
        out[s] = {"n": len(rows), "invalid": sum(not r["valid"] for r in rows),
                  "accuracy_by_length": {str(L): round(float(np.mean(v2)), 3) for L, v2 in sorted(by_len.items())},
                  "recall_by_length_position": pos_recall, "specificity_by_length": neg_spec,
                  "errors": [(r["error"].get("body", "") if isinstance(r.get("error"), dict) else str(r.get("error", "")))[:160] for r in rows if not r["valid"]][:3]}
    return out


def analyse_p4(raw):
    items = items_of("data/probes/p4_no_fit.main.jsonl")
    out = {}
    g = group(load_rows(raw, "p4"))
    s1 = group(load_rows(raw, "s1"))
    for s in sorted({k[0] for k in g}):
        forced = [r for r in g.get((s, "choice"), []) if r["item_id"] in items and r["valid"]]
        wn = [r for r in g.get((s, "choice_none"), []) if r["item_id"] in items]
        conf = np.array([float(r["confidence"]) for r in forced]) if forced else np.array([])
        k = sum(r["valid"] and r["pred"] == "none" for r in wn)
        out[s] = {"forced_n": len(forced), "forced_mean_conf": round(float(conf.mean()), 4) if len(conf) else None,
                  "forced_share_conf_ge_0.9": round(float(np.mean(conf >= 0.9)), 4) if len(conf) else None,
                  "with_none_n": len(wn), "chose_none": round(k / len(wn), 4) if wn else None,
                  "chose_none_ci95": [round(x, 4) for x in wilson(k, len(wn))] if wn else None}
        inset = s1.get((s, "choice_none"), [])
        if inset:  # wrong "none" on S1 items where a label fits; P4 detection is uninterpretable without it
            fn = sum(r["valid"] and r["pred"] == "none" for r in inset)
            out[s]["s1_false_none"] = round(fn / len(inset), 4)
            out[s]["s1_false_none_ci95"] = [round(x, 4) for x in wilson(fn, len(inset))]
    return out


def analyse_ood(raw):
    """Post hoc: AUROC of each system's top confidence for separating S1 (a label fits) from P4 (none fits)."""
    from sklearn.metrics import roc_auc_score
    s1 = group(load_rows(raw, "s1"))
    p4 = group(load_rows(raw, "p4"))
    out = {}
    for (s, v) in sorted(set(s1) & set(p4)):
        if v != "choice":
            continue
        a = [float(r["confidence"]) for r in s1[(s, v)] if r["valid"] and r["confidence"] is not None]
        b = [float(r["confidence"]) for r in p4[(s, v)] if r["valid"] and r["confidence"] is not None]
        if len(a) < 50 or len(b) < 50:
            continue
        y = np.array([1] * len(a) + [0] * len(b))
        x = np.array(a + b)
        auc = roc_auc_score(y, x)
        ci = bootstrap_ci(lambda yy, xx: roc_auc_score(yy, xx) if 0 < yy.sum() < len(yy) else np.nan, y, x, n=2000)
        out[s] = {"n_in": len(a), "n_out": len(b), "auroc": round(float(auc), 4), "auroc_ci95": [round(c, 4) for c in ci],
                  "post_hoc": True}
    return out


def analyse_degenerate(raw):
    """Post hoc: how informative are the returned confidences (distinct values, saturation)."""
    out = {}
    for name in ("s1", "s2"):
        for (s, v), rows in sorted(group(load_rows(raw, name)).items()):
            if v != "choice":
                continue
            c = np.array([float(r["confidence"]) for r in rows if r["valid"] and r["confidence"] is not None])
            if not len(c):
                continue
            out[f"{name}/{s}"] = {"n": len(c), "distinct_values": int(len(np.unique(np.round(c, 6)))),
                                  "share_eq_1": round(float(np.mean(c >= 0.9999)), 4),
                                  "share_ge_0.99": round(float(np.mean(c >= 0.99)), 4),
                                  "std": round(float(c.std()), 4)}
    return out


def analyse_latency():
    """Isolated latency run (protocol changelog): local at batch 1, API at concurrency 1, cache bypassed."""
    out = {}
    for f, kind in (("local.jsonl", "local"), ("api_c1.jsonl", "api"), ("api_c1.clef.jsonl", "api")):
        p = ROOT / "results/raw/latency" / f
        if not p.exists():
            continue
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        by = defaultdict(list)
        for r in rows:
            if r["valid"] and r.get("latency_s") is not None and not r.get("cached"):
                by[r["system"]].append(r["latency_s"])
        for s, x in by.items():
            if s in out:  # several files per system are pooled
                x = x + out[s]["_x"]
            out[s] = {"_x": x, "kind": kind, "n": len(x), "p50_s": pctl(x, 50), "p95_s": pctl(x, 95)}
    for v in out.values():
        v.pop("_x")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/raw/main")
    ap.add_argument("--split", default="main")
    a = ap.parse_args()
    raw = ROOT / a.raw
    global SPLIT
    SPLIT = a.split
    summary = {
        "s1_arxiv": analyse_choice_source("s1", raw, "data/s1_arxiv/task.json", "data/s1_arxiv/main.jsonl", "ambiguous"),
        "s2_fedreg": analyse_choice_source("s2", raw, "data/s2_fedreg/task.json", "data/s2_fedreg/main.jsonl"),
        "p1_calibration": analyse_p1(raw), "p2_minimal_pairs": analyse_p2(raw),
        "p3_long_input": analyse_p3(raw), "p4_no_fit": analyse_p4(raw),
        "post_hoc_ood_auroc": analyse_ood(raw), "post_hoc_degenerate": analyse_degenerate(raw),
        "latency_isolated": analyse_latency(),
    }
    (ROOT / f"results/summary{'' if SPLIT == 'main' else '_' + SPLIT}.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()

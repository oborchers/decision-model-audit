"""Post hoc gap-closing analyses: LLM stability, batching, repeatability. Writes results/gaps.json."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dma.analysis.metrics import bootstrap_ci, mcnemar_exact, wilson
from dma.analysis.report import group, load_rows, pctl

ROOT = Path(__file__).resolve().parents[3]
G = ROOT / "results/raw/gaps"


def rows(path):
    """Rows of a gaps file plus its `.clef.jsonl` sibling (post hoc Clef runs, October 2026)."""
    out = []
    for p in (Path(path), Path(path).with_suffix(".clef.jsonl")):
        if p.exists():
            out += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return out


def main():
    main_s1 = group(load_rows(ROOT / "results/raw/main", "s1"))
    base = {s: {r["item_id"]: r for r in main_s1[(s, "choice")]} for s in ("jev", "luna", "flash", "haiku", "clef", "clef-flash") if (s, "choice") in main_s1}
    out = {"stability": {}, "batch": {}, "repeat": {}}
    # stability
    st = {}
    for r in rows(G / "s1_stability.jsonl"):
        st.setdefault((r["system"], r["variant"]), {})[r["item_id"]] = r
    for (s, v), d in sorted(st.items()):
        agree = np.mean([d[i]["pred"] == base[s][i]["pred"] for i in d])
        acc = np.mean([d[i]["valid"] and d[i]["pred"] == d[i]["gold"] for i in d])
        out["stability"][f"{s}/{v}"] = {"n": len(d), "agreement_with_choice": round(float(agree), 4), "accuracy": round(float(acc), 4)}
    # batch
    bt = {}
    for r in rows(G / "s1_batch.jsonl"):
        bt.setdefault(r["system"], {})[r["item_id"]] = r
    blat_rows = rows(G / "s1_batch_latency.jsonl")
    for s, d in bt.items():
        ids = sorted(set(d) & set(base[s]))
        a = np.array([base[s][i]["valid"] and base[s][i]["pred"] == base[s][i]["gold"] for i in ids])
        b = np.array([d[i]["valid"] and d[i]["pred"] == d[i]["gold"] for i in ids])
        k = int(b.sum())
        single_cost = np.mean([base[s][i]["cost_usd"] or 0 for i in ids])
        batch_cost = np.mean([d[i]["cost_usd"] or 0 for i in ids])
        own = [r["batch_latency_s"] for r in blat_rows if r["system"] == s and r.get("batch_latency_s")]  # one value per row, as in the original analysis
        blat = own or sorted({d[i]["batch_latency_s"] for i in ids if d[i].get("batch_latency_s")})
        out["batch"][s] = {"n": len(ids), "accuracy_single": round(float(a.mean()), 4), "accuracy_batch10": round(float(b.mean()), 4),
                           "acc_batch_ci95": [round(x, 4) for x in wilson(k, len(ids))],
                           "delta": round(float(b.mean() - a.mean()), 4),
                           "delta_ci95": [round(x, 4) for x in bootstrap_ci(lambda x, y: y.mean() - x.mean(), a.astype(float), b.astype(float), n=10000)],
                           "mcnemar": mcnemar_exact(a, b),
                           "agreement_with_single": round(float(np.mean([d[i]["pred"] == base[s][i]["pred"] for i in ids])), 4),
                           "cost_per_1k_single": round(1000 * single_cost, 4), "cost_per_1k_batch10": round(1000 * batch_cost, 4),
                           "latency_per_request_p50_s": pctl(blat, 50), "latency_per_decision_p50_s": round(pctl(blat, 50) / 10, 3) if blat else None,
                           "n_requests_latency": len(blat)}
    # repeat
    rp = {}
    for r in rows(G / "s1_repeat.jsonl"):
        rp.setdefault(r["system"], {})[r["item_id"]] = r
    for s, d in rp.items():
        ids = sorted(set(d) & set(base[s]))
        same = [d[i]["pred"] == base[s][i]["pred"] for i in ids]
        dc = [abs(float(d[i]["confidence"] or 0) - float(base[s][i]["confidence"] or 0)) for i in ids]
        out["repeat"][s] = {"n": len(ids), "label_agreement": round(float(np.mean(same)), 4),
                            "conf_abs_diff_mean": round(float(np.mean(dc)), 4), "conf_abs_diff_max": round(float(np.max(dc)), 4),
                            "share_conf_identical": round(float(np.mean([x < 1e-9 for x in dc])), 4),
                            "acc_run1": round(float(np.mean([base[s][i]["pred"] == base[s][i]["gold"] for i in ids])), 4),
                            "acc_run2": round(float(np.mean([d[i]["pred"] == d[i]["gold"] for i in ids])), 4)}
    (ROOT / "results/gaps.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

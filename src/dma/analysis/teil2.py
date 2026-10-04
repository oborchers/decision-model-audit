"""Part 2 analyses (post hoc, October 2026). Writes results/teil2.json.

1. Automation rate on S1: the largest share of items a system may decide alone while the expected error among
   them stays at or below 2% or 5% (tie-aware risk at coverage, coverage grid 1/400), from the returned
   `confidence` and, for decision models, from the top probability. Bootstrap interval over papers.
2. FWF German versus English: paired accuracy for English text (en), German text with English task (de-en) and
   German text with German task (de); difference de minus en with bootstrap interval and exact McNemar, Holm
   over systems.
3. Interface behaviour per decision model (S1 choice): share of `confidence` values that differ from the top
   probability, share of exact 1.0, distinct values, invalid answers per probe.
4. Stability on S1: agreement of reversed order, paraphrased labels and the yes/no form with the own choice run.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from dma.analysis.metrics import bootstrap_ci, holm, mcnemar_exact, risk_at_coverage
from dma.analysis.report import group, load_rows

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "results/raw/main"
DECISION = {"jev", "clef", "clef-flash", "d1", "solar", "mercury", "tev1", "kev-4b-api", "pplx-decider", "glide",
            "strands-2b", "apus-4b", "apus-9b", "clm-8b", "decision2-kai-0.6b", "decision2-nox-4b"}


def risk_curve(conf: np.ndarray, correct: np.ndarray) -> np.ndarray:
    """Tie-aware risk at coverage k/n for k = 1..n, identical to metrics.risk_at_coverage, in O(n log n)."""
    err = 1.0 - correct
    order = np.argsort(-conf, kind="stable")
    c, e = conf[order], err[order]
    n = len(c)
    out = np.empty(n)
    start = 0
    cum_err = 0.0
    while start < n:
        end = start
        while end < n and c[end] == c[start]:
            end += 1
        block_mean = e[start:end].mean()
        for k in range(start + 1, end + 1):
            out[k - 1] = (cum_err + (k - start) * block_mean) / k
        cum_err += e[start:end].sum()
        start = end
    return out


def max_coverage(conf: np.ndarray, correct: np.ndarray, target: float) -> float:
    r = risk_curve(conf, correct)
    ok = np.where(r <= target + 1e-12)[0]
    return float((ok.max() + 1) / len(conf)) if len(ok) else 0.0


def automation(per):
    out = {}
    for (s, v), rows in sorted(per.items()):
        if v != "choice" or len(rows) < 400:
            continue
        rows = sorted(rows, key=lambda r: r["item_id"])
        ok = np.array([r["valid"] and r["pred"] == r["gold"] for r in rows], dtype=float)
        fields = {"confidence": np.array([float(r["confidence"]) if r["valid"] and r["confidence"] is not None else 0.0 for r in rows])}
        if s in DECISION and all(len(r["probs"]) > 1 for r in rows if r["valid"]):
            fields["top_probability"] = np.array([max(r["probs"].values()) if r["valid"] and r["probs"] else 0.0 for r in rows])
        d = {"accuracy": round(float(ok.mean()), 4)}
        for name, c in fields.items():
            for t in (0.02, 0.05):
                cov = max_coverage(c, ok, t)
                ci = bootstrap_ci(lambda cc, oo: max_coverage(cc, oo, t), c, ok, n=2000)
                d[f"{name}_coverage_at_{int(t * 100)}pct_error"] = round(cov, 4)
                d[f"{name}_coverage_at_{int(t * 100)}pct_error_ci95"] = [round(x, 4) for x in ci]
        out[s] = d
    return out


def fwf():
    rows = {}
    for c in ("en", "de-en", "de"):
        p = RAW / f"fwf_{c}.teil2.jsonl"
        for l in p.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                rows[(c, r["system"], r["item_id"])] = r
    items = {json.loads(l)["item_id"] for l in (ROOT / "data/fwf/main_en.jsonl").read_text().splitlines()}
    ids = sorted(items)
    systems = sorted({k[1] for k in rows})
    out, pv = {}, {}
    for s in systems:
        if not all((c, s, i) in rows for c in ("en", "de-en", "de") for i in ids):
            out[s] = {"note": "incomplete"}
            continue
        a = {c: np.array([rows[(c, s, i)]["valid"] and rows[(c, s, i)]["pred"] == rows[(c, s, i)]["gold"] for i in ids]) for c in ("en", "de-en", "de")}
        tok = {c: float(np.median([(rows[(c, s, i)].get("usage") or {}).get("input_tokens") or 0 for i in ids])) for c in ("en", "de")}
        d = {"n": len(ids), **{f"acc_{c}": round(float(a[c].mean()), 4) for c in a}}
        for c in ("de-en", "de"):
            d[f"delta_{c}_minus_en"] = round(float(a[c].mean() - a["en"].mean()), 4)
            d[f"delta_{c}_minus_en_ci95"] = [round(x, 4) for x in bootstrap_ci(lambda x, y: y.mean() - x.mean(), a["en"].astype(float), a[c].astype(float), n=10000)]
            d[f"mcnemar_{c}"] = mcnemar_exact(a["en"], a[c])
        d["median_input_tokens"] = tok
        pv[s] = d["mcnemar_de"]["p"]
        out[s] = d
    for s, p in holm(pv).items():
        out[s]["mcnemar_de"]["p_holm"] = round(p, 5)
    return out


def interface(per):
    out = {}
    for (s, v), rows in sorted(per.items()):
        if v != "choice" or s not in DECISION:
            continue
        ok = [r for r in rows if r["valid"] and r["probs"] and r["confidence"] is not None]
        if not ok:
            continue
        diff = [abs(r["confidence"] - max(r["probs"].values())) > 1e-3 for r in ok]
        c = np.array([r["confidence"] for r in ok])
        out[s] = {"n": len(ok), "confidence_differs_from_top_probability": round(float(np.mean(diff)), 4),
                  "share_confidence_eq_1": round(float(np.mean(c >= 0.9999)), 4),
                  "distinct_confidence_values": int(len(np.unique(np.round(c, 6))))}
    inval = defaultdict(dict)
    for name in ("s1", "s2", "p1", "p2", "p3", "p4"):
        for (s, v), rows in group(load_rows(RAW, name)).items():
            if s in DECISION:
                n_bad = sum(not r["valid"] for r in rows)
                if n_bad:
                    inval[s][f"{name}/{v}"] = f"{n_bad}/{len(rows)}"
    for s in out:
        out[s]["invalid"] = inval.get(s, {})
    return out


def stability(per):
    """Agreement of each S1 variant with the system's own choice run, as in part 1 (`report.py`, `_stability`):
    share of items with the same prediction; an invalid answer counts as disagreement."""
    out = {}
    for s in sorted(DECISION | {"luna"}):
        base = {r["item_id"]: r["pred"] for r in per.get((s, "choice"), []) if r["valid"]}
        if not base:
            continue
        d = {}
        for v in ("reversed", "para0", "para1", "yesno"):
            rows = per.get((s, v), [])
            if rows:
                d[v] = round(float(np.mean([r["valid"] and base.get(r["item_id"]) == r["pred"] for r in rows])), 4)
        if d:
            out[s] = d
    return out


def main():
    per = group(load_rows(RAW, "s1"))
    res = {"automation_s1": automation(per), "fwf_de_vs_en": fwf(), "interface_s1": interface(per), "stability_s1": stability(per)}
    (ROOT / "results/teil2.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: list(v) for k, v in res.items()}))
    print(figure_risk_coverage())



def figure_risk_coverage(path=ROOT / "results/figures/teil2_risk_coverage.png"):
    """Risk-coverage curves on S1 for the decision models of both parts, GPT-6 Luna and the embedding classifier."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    per = group(load_rows(RAW, "s1"))
    show = ["jev", "clef", "glide", "pplx-decider", "kev-4b-api", "d1", "solar", "strands-2b", "apus-9b", "luna", "emb-qwen3-8b"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in show:
        rows = per.get((s, "choice"))
        if not rows or len(rows) < 400:
            continue
        rows = sorted(rows, key=lambda r: r["item_id"])
        ok = np.array([r["valid"] and r["pred"] == r["gold"] for r in rows], dtype=float)
        c = np.array([float(r["confidence"]) if r["valid"] and r["confidence"] is not None else 0.0 for r in rows])
        r = risk_curve(c, ok)
        cov = np.arange(1, len(r) + 1) / len(r)
        style = dict(lw=2.6, color="#d95f02") if s == "jev" else dict(lw=1.3, ls="--" if s in ("luna", "emb-qwen3-8b") else "-")
        ax.plot(cov, r, label=s, **style)
    ax.axhline(0.05, color="#888", lw=0.8, ls=":")
    ax.set_xlabel("coverage (share of S1 papers decided automatically)")
    ax.set_ylabel("error rate among decided papers")
    ax.set_ylim(0, 0.3)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("S1 risk-coverage from the returned confidence (ties at random)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return path


if __name__ == "__main__":
    main()

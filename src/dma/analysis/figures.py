"""Figures for results/figures/: risk-coverage (S1), reliability (S1), cost vs accuracy (S1), long input (P3).

uv run python -m dma.analysis.figures
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dma.analysis.metrics import risk_at_coverage
from dma.analysis.report import EXTRA, group, load_rows

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "results/raw/main"
OUT = ROOT / "results/figures"
API = {"jev", "luna", "flash", "haiku", "sonnet"}
NAMES = {"jev": "Jev 1.13", "luna": "GPT-6 Luna", "flash": "Gemini 3.5 Flash-Lite", "haiku": "Claude Haiku 4.5",
         "sonnet": "Claude Sonnet 5", "tfidf": "TF-IDF + LR (2024 labels)", "gliner": "GLiNER2.5-Decide",
         "gliner-1b": "GLiNER2.5-Decide-1B", "nli": "DeBERTa-v3 NLI", "gliclass": "GLiClass v3", "qwen-lp": "Qwen3.5-4B logprobs",
         "laya": "Laya", "laya-long": "Laya long", "gliner-long": "GLiNER2.5 chunked", "eikos-4b": "Eikos-4B",
         "semif-4b": "SemIf 4B", "kev-0.8b": "Kev-0.8B"}


def style(s):
    if s == "jev":
        return dict(color="#d95f02", lw=2.6, zorder=5)
    if s in API:
        return dict(color="#1b9e77", lw=1.4, alpha=0.9)
    if s in EXTRA:
        return dict(color="#7570b3", lw=1.2, ls="--", alpha=0.9)
    return dict(color="#666666", lw=1.2, alpha=0.8)


def s1_choice():
    g = group(load_rows(RAW, "s1"))
    return {s: sorted(rows, key=lambda r: r["item_id"]) for (s, v), rows in g.items() if v == "choice"}


def risk_coverage(data):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cov = np.linspace(0.05, 1, 96)
    for s, rows in sorted(data.items()):
        c = np.array([float(r["confidence"] or 0) for r in rows])
        y = np.array([float(r["valid"] and r["pred"] == r["gold"]) for r in rows])
        ax.plot(cov, [risk_at_coverage(c, y, k) for k in cov], label=NAMES.get(s, s), **style(s))
    ax.set_xlabel("Coverage (share of cases decided automatically)")
    ax.set_ylabel("Error rate among automated cases")
    ax.set_title("S1 arXiv: risk–coverage (deferral by each system's own confidence)")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "s1_risk_coverage.png", dpi=180)


def reliability(data, systems=("jev", "luna", "sonnet", "tfidf", "gliner", "eikos-4b")):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=1)
    for s in systems:
        if s not in data:
            continue
        rows = data[s]
        c = np.array([float(r["confidence"] or 0) for r in rows])
        y = np.array([float(r["valid"] and r["pred"] == r["gold"]) for r in rows])
        order = np.argsort(c, kind="stable")
        xs, ys = [], []
        for idx in np.array_split(order, 10):
            xs.append(c[idx].mean()); ys.append(y[idx].mean())
        ax.plot(xs, ys, marker="o", ms=3, label=NAMES.get(s, s), **style(s))
    ax.set_xlabel("Mean confidence (10 equal-mass bins)")
    ax.set_ylabel("Observed accuracy")
    ax.set_title("S1 arXiv: reliability")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "s1_reliability.png", dpi=180)


def cost_accuracy(summary):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, v in summary["s1_arxiv"].items():
        s, var = k.split("/") if "/" in k else (k, "")
        if var != "choice" or s not in API:
            continue
        ax.scatter(v["cost_per_1k_usd"], v["accuracy"], s=40, **{kk: vv for kk, vv in style(s).items() if kk in ("color", "zorder")})
        ax.errorbar(v["cost_per_1k_usd"], v["accuracy"], yerr=[[v["accuracy"] - v["acc_ci95"][0]], [v["acc_ci95"][1] - v["accuracy"]]],
                    color=style(s)["color"], lw=1, capsize=2)
        ax.annotate(NAMES[s], (v["cost_per_1k_usd"], v["accuracy"]), fontsize=8, xytext=(5, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("USD per 1,000 decisions (log scale, OpenRouter)")
    ax.set_ylabel("Accuracy (95% CI)")
    ax.set_title("S1 arXiv: accuracy against cost, API systems")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "s1_cost_accuracy.png", dpi=180)


def long_input(summary):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s, v in sorted(summary["p3_long_input"].items()):
        L = sorted(int(x) for x in v["accuracy_by_length"])
        ax.plot(L, [v["accuracy_by_length"][str(x)] for x in L], marker="o", ms=3, label=NAMES.get(s, s), **style(s))
    ax.set_xscale("log")
    ax.set_xticks([500, 2000, 8000, 24000], ["500", "2k", "8k", "24k"])
    ax.set_xlabel("Input length (tokens, approx.)")
    ax.set_ylabel("Accuracy (refused inputs count as errors)")
    ax.set_title("P3: one decisive sentence in long filler")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "p3_long_input.png", dpi=180)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = json.loads((ROOT / "results/summary.json").read_text())
    data = s1_choice()
    risk_coverage(data)
    reliability(data)
    cost_accuracy(summary)
    long_input(summary)
    print("written:", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()

"""Post hoc learning curve for supervised baselines on S1 (see protocol changelog)."""
from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from dma.analysis.metrics import aurc
from dma.runners.supervised import EMB, embed
from dma.tasks import load_items

ROOT = Path(__file__).resolve().parents[3]
SIZES = [40, 80, 160, 400, 800, 1600, 2806]
REPS = 10
C = {"emb-qwen3-8b": 30, "emb-oai-3l": 10, "tfidf": 100}
REF = {"Jev 1.13": 0.8625, "GPT-6 Luna": 0.8875}


def sample(train, n, seed):
    if n >= len(train):
        return list(range(len(train)))
    rng = random.Random(seed)
    by = {}
    for i, r in enumerate(train):
        by.setdefault(r["label"], []).append(i)
    per = n // len(by)
    idx = []
    for lab in sorted(by):
        pool = by[lab][:]
        rng.shuffle(pool)
        idx += pool[:per]
    rest = [i for i in range(len(train)) if i not in set(idx)]
    rng.shuffle(rest)
    idx += rest[: n - len(idx)]
    return idx


def main():
    train = load_items(ROOT / "data/s1_arxiv/train_2024.jsonl")
    test = load_items(ROOT / "data/s1_arxiv/main.jsonl")
    ytr = np.array([r["label"] for r in train])
    yte = np.array([r["label"] for r in test])
    feats = {}
    for name, model in EMB.items():
        feats[name] = (embed(model, [r["text"] for r in train])[0], embed(model, [r["text"] for r in test])[0])
    feats["tfidf"] = None
    out = {}
    for name in ("emb-qwen3-8b", "emb-oai-3l", "tfidf"):
        res = {}
        for n in SIZES:
            accs, aurcs = [], []
            for rep in range(1 if n >= len(train) else REPS):
                idx = sample(train, n, 1000 * n + rep)
                if len(set(ytr[idx])) < 8:
                    continue
                if name == "tfidf":
                    clf = make_pipeline(TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=1 if n < 400 else 2),
                                        LogisticRegression(C=C[name], max_iter=3000, class_weight="balanced"))
                    clf.fit([train[i]["text"] for i in idx], ytr[idx])
                    P = clf.predict_proba([r["text"] for r in test]); classes = clf.classes_
                else:
                    Xtr, Xte = feats[name]
                    clf = LogisticRegression(C=C[name], max_iter=3000, class_weight="balanced").fit(Xtr[idx], ytr[idx])
                    P = clf.predict_proba(Xte); classes = clf.classes_
                pred = classes[P.argmax(1)]
                corr = (pred == yte).astype(float)
                accs.append(corr.mean())
                aurcs.append(aurc(P.max(1), corr))
            res[n] = {"reps": len(accs), "acc_mean": round(float(np.mean(accs)), 4),
                      "acc_p2_5": round(float(np.percentile(accs, 2.5)), 4), "acc_p97_5": round(float(np.percentile(accs, 97.5)), 4),
                      "aurc_mean": round(float(np.mean(aurcs)), 4)}
            print(name, n, res[n], flush=True)
        out[name] = res
    (ROOT / "results/learning_curve.json").write_text(json.dumps(out, indent=1))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = {"emb-qwen3-8b": "Qwen3-Embedding-8B + LR", "emb-oai-3l": "OpenAI 3-large + LR", "tfidf": "TF-IDF + LR"}
    for name, res in out.items():
        ns = sorted(res)
        m = [res[n]["acc_mean"] for n in ns]
        ax.plot(ns, m, marker="o", ms=3, label=labels[name])
        ax.fill_between(ns, [res[n]["acc_p2_5"] for n in ns], [res[n]["acc_p97_5"] for n in ns], alpha=0.15)
    for lab, v in REF.items():
        ax.axhline(v, ls="--", lw=1, color="#d95f02" if "Jev" in lab else "#1b9e77")
        ax.annotate(f"{lab} (zero-shot)", (40, v), xytext=(0, 3), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks(SIZES, [str(n) for n in SIZES])
    ax.set_xlabel("Labelled training examples (2024 arXiv, about equal per class)")
    ax.set_ylabel("Accuracy on S1 (Sept 2026)")
    ax.set_title("How many labels until a simple classifier matches Jev?")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "results/figures/s1_learning_curve.png", dpi=180)


if __name__ == "__main__":
    main()

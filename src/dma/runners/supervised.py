"""Post hoc supervised baselines on S1: TF-IDF and API embeddings + logistic regression, with and without
class weighting. Same temporal split as the pre-registered TF-IDF baseline (train: 2024 arXiv, test: main set
from September 2026); C chosen by 5-fold cross-validation on the training data only.

uv run python -m dma.runners.supervised
Writes results/raw/main/{s1,p4}.supervised.jsonl
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import normalize

from dma.client import cached_call
from dma.tasks import load_items, result_row

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/raw/main"
EMB = {"emb-qwen3-8b": "qwen/qwen3-embedding-8b", "emb-oai-3l": "openai/text-embedding-3-large"}
SEED = 20260927


def embed(model: str, texts: list[str], batch: int = 64) -> tuple[np.ndarray, float]:
    vecs, cost = [], 0.0
    for i in range(0, len(texts), batch):
        rec = cached_call("/v1/embeddings", {"model": model, "input": texts[i:i + batch]})
        resp = rec["response"]
        if "error" in resp or "data" not in resp:
            raise RuntimeError(f"embedding failed: {str(resp)[:300]}")
        data = sorted(resp["data"], key=lambda d: d["index"])
        vecs += [d["embedding"] for d in data]
        if not rec["cached"]:
            cost += (resp.get("usage") or {}).get("cost") or 0
    return normalize(np.array(vecs, dtype=np.float32)), cost


def fit(X, y, balanced: bool):
    lr = LogisticRegression(max_iter=3000, class_weight="balanced" if balanced else None)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    gs = GridSearchCV(lr, {"C": [0.1, 0.3, 1, 3, 10, 30]}, cv=cv, scoring="neg_log_loss", n_jobs=1)
    gs.fit(X, y)
    return gs


def rows_for(system, variant, clf, X_items, items, classes, info):
    t0 = time.perf_counter()
    P = clf.predict_proba(X_items)
    per = (time.perf_counter() - t0) / max(1, len(items))
    out = []
    for it, p in zip(items, P):
        probs = {c: float(v) for c, v in zip(classes, p)}
        pred = max(probs, key=probs.get)
        out.append(result_row(it, system, variant, pred=pred, probs=probs, confidence=probs[pred], latency_s=per,
                              cost_usd=0.0, extra=info))
    return out


def main():
    train = load_items(ROOT / "data/s1_arxiv/train_2024.jsonl")
    test = load_items(ROOT / "data/s1_arxiv/main.jsonl")
    p4 = load_items(ROOT / "data/probes/p4_no_fit.main.jsonl")
    ytr = [r["label"] for r in train]
    s1_rows, p4_rows, summary = [], [], {}
    # TF-IDF with class weighting (the unweighted version is the pre-registered `tfidf`)
    pipe = make_pipeline(TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2),
                         LogisticRegression(max_iter=3000, class_weight="balanced"))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    gs = GridSearchCV(pipe, {"logisticregression__C": [1, 3, 10, 30, 100]}, cv=cv, scoring="neg_log_loss", n_jobs=1)
    gs.fit([r["text"] for r in train], ytr)
    info = {"model_id": "tfidf+lr balanced", "best_C": gs.best_params_["logisticregression__C"], "n_train": len(train)}
    s1_rows += rows_for("tfidf-bal", "choice", gs, [r["text"] for r in test], test, list(gs.classes_), info)
    p4_rows += rows_for("tfidf-bal", "choice", gs, [r["text"] for r in p4], p4, list(gs.classes_), info)
    summary["tfidf-bal"] = info
    # API embeddings
    for name, model in EMB.items():
        Xtr, c1 = embed(model, [r["text"] for r in train])
        Xte, c2 = embed(model, [r["text"] for r in test])
        Xp4, c3 = embed(model, [r["text"] for r in p4])
        emb_cost = c1 + c2 + c3
        for balanced in (True, False):
            sysname = name + ("" if balanced else "-unbal")
            clf = fit(Xtr, ytr, balanced)
            info = {"model_id": model, "embedding_dim": int(Xtr.shape[1]), "best_C": clf.best_params_["C"],
                    "class_weight": "balanced" if balanced else None, "n_train": len(train),
                    "embedding_cost_usd_total": round(emb_cost, 5),
                    "embedding_cost_per_1k_test_usd": round(1000 * c2 / len(test), 5) if c2 else None}
            s1_rows += rows_for(sysname, "choice", clf, Xte, test, list(clf.classes_), info)
            p4_rows += rows_for(sysname, "choice", clf, Xp4, p4, list(clf.classes_), info)
            summary[sysname] = info
    for f, rows in (("s1", s1_rows), ("p4", p4_rows)):
        (OUT / f"{f}.supervised.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for s, info in summary.items():
        acc = np.mean([r["pred"] == r["gold"] for r in s1_rows if r["system"] == s])
        print(s, round(float(acc), 4), info)


if __name__ == "__main__":
    main()

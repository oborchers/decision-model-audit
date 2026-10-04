"""P5 training-free embedding baseline (part 2): cosine similarity of abstract and title embeddings.

The classical approach when options change per item. probs = softmax(20 * cosine) over the item's five titles;
the factor only orders and spreads the scores and is not fitted, so these probabilities are not calibrated
(accuracy and AURC are reported, ECE is not).

uv run python -m dma.runners.p5_embed --items data/probes/p5_titles.main.jsonl --out results/raw/main/p5.teil2.jsonl
"""
from __future__ import annotations

import argparse

import numpy as np

from dma.runners.supervised import EMB, embed
from dma.tasks import load_items, result_row, write_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--system", default="emb-qwen3-8b")
    a = ap.parse_args()
    items = load_items(a.items)
    texts = [it["text"] for it in items] + [t for it in items for t in it["options"].values()]
    V, cost = embed(EMB[a.system], texts)
    n = len(items)
    rows, j = [], n
    for i, it in enumerate(items):
        keys = list(it["options"])
        T = V[j:j + len(keys)]
        j += len(keys)
        cos = T @ V[i]
        z = np.exp(20 * (cos - cos.max()))
        p = z / z.sum()
        probs = {k: float(v) for k, v in zip(keys, p)}
        pred = keys[int(cos.argmax())]
        rows.append(result_row(it, f"{a.system}-cos", "choice", pred=pred, probs=probs, confidence=probs[pred],
                               cost_usd=cost / n, extra={"cosine": {k: float(c) for k, c in zip(keys, cos)}}))
    write_rows(a.out, rows)
    acc = np.mean([r["pred"] == r["gold"] for r in rows])
    print(f"{a.system}-cos: accuracy {acc:.4f} on {n}, embedding cost ${cost:.4f}")


if __name__ == "__main__":
    main()

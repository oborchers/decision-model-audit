"""Plain encoder baseline for the fine-tuning comparison (post hoc, part 2): ModernBERT-large with a standard
sequence-classification head, fine-tuned on the same S1 training labels as Laya (whose backbone is
ModernBERT-large). Isolates whether Laya's decision pretraining adds anything once labels exist.

uv run python -m dma.runners.encoder_finetune --labels 400 --seed 1 --device cuda --out results/raw/main/s1.teil2.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import torch

from dma.runners.laya_finetune import dev_split, sample_balanced
from dma.tasks import Task, load_items, result_row, write_rows

MODEL = "answerdotai/ModernBERT-large"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="", help="suffix of the system name, e.g. -dec-lr3e-05")
    ap.add_argument("--dev-only", action="store_true", help="evaluate on validation papers from train_2024, never on S1 main")
    a = ap.parse_args()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    task = Task.load("data/s1_arxiv/task.json")
    labels = list(task.labels)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(labels)).to(a.device)
    pool = load_items("data/s1_arxiv/train_2024.jsonl")
    train = sample_balanced(pool, a.labels, labels, rng)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * ((len(train) + a.batch - 1) // a.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, steps // 10)) * max(0.0, 1 - s / steps))
    model.train()
    t0 = time.time()
    for ep in range(a.epochs):
        rng.shuffle(train)
        for i in range(0, len(train), a.batch):
            chunk = train[i:i + a.batch]
            enc = tok([r["text"] for r in chunk], truncation=True, max_length=a.max_len, padding=True, return_tensors="pt").to(a.device)
            y = torch.tensor([labels.index(r["label"]) for r in chunk], device=a.device)
            with torch.autocast(a.device, dtype=torch.bfloat16, enabled=a.device == "cuda"):
                loss = model(**enc, labels=y).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    train_s = time.time() - t0
    model.eval()
    items = dev_split(pool, train, labels) if a.dev_only else load_items("data/s1_arxiv/main.jsonl")
    rows = []
    with torch.no_grad():
        for it in items:
            ts = time.time()
            enc = tok([it["text"]], truncation=True, max_length=a.max_len, return_tensors="pt").to(a.device)
            p = torch.softmax(model(**enc).logits.float(), -1)[0].cpu().numpy()
            probs = {l: float(v) for l, v in zip(labels, p)}
            pred = labels[int(p.argmax())]
            rows.append(result_row(it, f"mbert-ft{len(train)}{a.tag}-s{a.seed}", "choice", pred=pred, probs=probs,
                                   confidence=probs[pred], latency_s=time.time() - ts, cost_usd=0.0,
                                   extra={"train_labels": len(train), "epochs": a.epochs, "lr": a.lr, "seed": a.seed,
                                          "train_seconds": round(train_s, 1)}))
    acc = float(np.mean([r["pred"] == r["gold"] for r in rows]))
    print(json.dumps({"system": f"mbert-ft{len(train)}{a.tag}-s{a.seed}", "epochs": a.epochs, "eval": "dev" if a.dev_only else "s1_main",
                      "accuracy": round(acc, 4), "train_seconds": round(train_s, 1)}), flush=True)
    if a.out and not a.dev_only:
        write_rows(a.out, rows)


if __name__ == "__main__":
    main()

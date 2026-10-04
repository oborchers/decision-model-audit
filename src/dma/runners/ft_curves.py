"""Longer fine-tuning with learning curves (post hoc, part 2, protocol changelog 2026-10-03).

The 3-epoch runs left ModernBERT-large undertrained at 400 labels (mean confidence 0.58 for one seed). Here each
seed trains once for many epochs (bf16 autocast on CUDA for both models) at a constant learning rate after one epoch of linear warmup, and after every
epoch records mean training loss, validation loss and accuracy, and S1 main accuracy (descriptive only). The epoch
is selected on validation accuracy alone (ties: earlier epoch); its weights are restored and S1 main rows are
written through the normal prediction path (Laya: the vendor's `system_one`).

Validation: 400 balanced papers from train_2024 outside the training sample (400 labels), or, when all labels
are wanted, held out first so that 2,406 papers remain for training.

Learning rate: a small grid per model at 400 labels, chosen by mean best validation accuracy over seeds; the
run with all labels uses that rate.

uv run python -m dma.runners.ft_curves --model mbert --labels 400 --epochs 20 --lr 3e-5 --seed 1 --device cuda
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from dma.runners.laya_finetune import DEV_SEED, build, dev_split, sample_balanced
from dma.tasks import Task, load_items, result_row, write_rows

MBERT = "answerdotai/ModernBERT-large"
ROOT = Path(__file__).resolve().parents[3]


class Mbert:
    batch, lr = 16, 2e-5

    def __init__(self, labels, device):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.labels, self.device = labels, device
        self.tok = AutoTokenizer.from_pretrained(MBERT)
        self.model = AutoModelForSequenceClassification.from_pretrained(MBERT, num_labels=len(labels)).to(device)

    def prep(self, rows):
        return [(r["text"], self.labels.index(r["label"])) for r in rows]

    def logits(self, chunk):
        enc = self.tok([c[0] for c in chunk], truncation=True, max_length=512, padding=True, return_tensors="pt").to(self.device)
        with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"):
            return self.model(**enc).logits.float()

    def rows(self, items, system, extra):
        out = []
        self.model.eval()
        with torch.no_grad():
            for it in items:
                ts = time.time()
                p = torch.softmax(self.logits([(it["text"], 0)]), -1)[0].cpu().numpy()
                probs = {l: float(v) for l, v in zip(self.labels, p)}
                pred = self.labels[int(p.argmax())]
                out.append(result_row(it, system, "choice", pred=pred, probs=probs, confidence=probs[pred],
                                      latency_s=time.time() - ts, cost_usd=0.0, extra=extra))
        return out


class Laya:
    batch, lr = 8, 1e-5

    def __init__(self, labels, device):
        from laya import Agent
        from laya.common import collate_items
        self.labels, self.device, self.collate = labels, device, collate_items
        self.task = Task.load("data/s1_arxiv/task.json")
        self.agent = Agent("convaiinnovations/laya", device=device)
        self.agent.amp_enabled = False
        self.model = self.agent.model
        self.pad = self.agent.tok.pad_token_id

    def prep(self, rows):
        return [(build(self.agent, self.task, r["text"]), self.labels.index(r["label"])) for r in rows]

    def logits(self, chunk):
        b = self.collate([[c[0]] for c in chunk], self.pad)
        with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"):
            lg, _ = self.model(*(b[k].to(self.device) for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")))
        return lg[:, :len(self.labels)].float()

    def rows(self, items, system, extra):
        out = []
        self.model.eval()
        q = {"q": {"type": "choice", "instructions": self.task.instructions, "criteria": self.task.labels}}
        for it in items:
            ts = time.time()
            ans = self.agent.system_one(it["text"], q)["answers"]["q"]
            out.append(result_row(it, system, "choice", pred=ans["choice"], probs=ans["probabilities"],
                                  confidence=ans["confidence"], latency_s=time.time() - ts, cost_usd=0.0,
                                  extra={"answer_confidence": ans.get("answer_confidence"), **extra}))
        return out


def evaluate(m, data):
    m.model.eval()
    losses, ok = [], []
    with torch.no_grad():
        for i in range(0, len(data), 32):
            chunk = data[i:i + 32]
            lg = m.logits(chunk)
            y = torch.tensor([c[1] for c in chunk], device=m.device)
            losses.append(torch.nn.functional.cross_entropy(lg, y, reduction="sum").item())
            ok += (lg.argmax(-1) == y).tolist()
    return sum(losses) / len(data), float(np.mean(ok))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mbert", "laya"], required=True)
    ap.add_argument("--labels", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lr", type=float, default=None, help="override the model default (Laya 1e-5, ModernBERT 2e-5)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/raw/main/s1.ft.jsonl")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    task = Task.load("data/s1_arxiv/task.json")
    labels = list(task.labels)
    pool = load_items("data/s1_arxiv/train_2024.jsonl")
    if a.labels + 400 > len(pool):
        dev = sample_balanced(pool, 400, labels, random.Random(DEV_SEED))
        held = {r["item_id"] for r in dev}
        train = [r for r in pool if r["item_id"] not in held]
    else:
        train = sample_balanced(pool, a.labels, labels, rng)
        dev = dev_split(pool, train, labels)
    test = load_items("data/s1_arxiv/main.jsonl")
    m = Mbert(labels, a.device) if a.model == "mbert" else Laya(labels, a.device)
    if a.lr:
        m.lr = a.lr
    tr, dv, te = m.prep(train), m.prep(dev), m.prep(test)
    opt = torch.optim.AdamW(m.model.parameters(), lr=m.lr, weight_decay=0.01)
    per_epoch = (len(tr) + m.batch - 1) // m.batch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / per_epoch))
    system = f"{a.model}-ft{len(train)}-long-lr{m.lr:g}-s{a.seed}"
    curve = {"system": system, "train_labels": len(train), "dev": len(dev), "batch": m.batch, "lr": m.lr,
             "schedule": "linear warmup over epoch 1, then constant", "step_loss": [], "epochs": []}
    best, best_state, t0 = None, None, time.time()
    for ep in range(1, a.epochs + 1):
        m.model.train()
        rng.shuffle(tr)
        ep_loss = []
        for i in range(0, len(tr), m.batch):
            chunk = tr[i:i + m.batch]
            loss = torch.nn.functional.cross_entropy(m.logits(chunk), torch.tensor([c[1] for c in chunk], device=m.device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            ep_loss.append(loss.item())
        curve["step_loss"] += [round(x, 4) for x in ep_loss]
        dl, da = evaluate(m, dv)
        tl, ta = evaluate(m, te)
        e = {"epoch": ep, "train_loss": round(float(np.mean(ep_loss)), 4), "dev_loss": round(dl, 4), "dev_acc": round(da, 4),
             "test_loss": round(tl, 4), "test_acc": round(ta, 4), "elapsed_s": round(time.time() - t0, 1)}
        curve["epochs"].append(e)
        print(json.dumps(e), flush=True)
        if best is None or da > best["dev_acc"]:
            best, best_state = e, {k: v.detach().to("cpu", copy=True) for k, v in m.model.state_dict().items()}
    m.model.load_state_dict(best_state)
    curve["selected_epoch"] = best["epoch"]
    rows = m.rows(test, system, {"train_labels": len(train), "selected_epoch": best["epoch"], "epochs_trained": a.epochs,
                                 "seed": a.seed, "lr": m.lr, "batch": m.batch})
    acc = float(np.mean([r["pred"] == r["gold"] for r in rows]))
    curve["test_acc_selected_rows"] = round(acc, 4)
    out = ROOT / "results/ft_curves"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{system}.json").write_text(json.dumps(curve) + "\n")
    write_rows(a.out, rows)
    print(json.dumps({"system": system, "selected_epoch": best["epoch"], "dev_acc": best["dev_acc"], "accuracy": round(acc, 4)}), flush=True)


if __name__ == "__main__":
    main()

"""Fine-tune Laya on the S1 training labels (post hoc, part 2, optional): does a decision model trained on the
same labels beat the embedding classifier of part 1?

The installed `laya` package ships inference only, so this is our own minimal loop: the vendor's own encoding
(`Agent._encode_state`, `collate_items`) builds each choice question over the eight S1 labels, the model's
option logits are trained with cross-entropy against the 2024 label, AdamW, full fine-tuning on MPS in FP32.
Evaluation then runs the vendor's normal `predict` path on the 400 S1 papers (with the vendor temperature).

uv run python -m dma.runners.laya_finetune --labels 400 --epochs 3 --out results/raw/main/s1.teil2.jsonl
uv run python -m dma.runners.laya_finetune --steps 20 --timing-only      # mini test
"""
from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import torch

from dma.tasks import Task, load_items, result_row, write_rows

SEED = 20261003


def build(agent, task: Task, text: str):
    q = {"q": {"type": "choice", "instructions": task.instructions, "criteria": task.labels}}
    internal = {"q": agent._to_internal(q["q"])}
    return agent._encode_state(text, ["q"], internal)[0]


def sample_balanced(rows, n, labels, rng):
    if n >= len(rows):
        return rows[:]
    per = {l: [r for r in rows if r["label"] == l] for l in labels}
    k = n // len(labels)
    out = []
    for l in labels:
        rng.shuffle(per[l])
        out += per[l][:k]
    rest = [r for r in rows if r not in out]
    rng.shuffle(rest)
    return out + rest[: n - len(out)]

DEV_SEED = 777


def dev_split(pool, train, labels, n=400):
    """Validation papers for choosing the epoch count: balanced, from train_2024 outside the training sample."""
    used = {r["item_id"] for r in train}
    return sample_balanced([r for r in pool if r["item_id"] not in used], n, labels, random.Random(DEV_SEED))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--steps", type=int, default=0, help="stop after this many steps (mini test)")
    ap.add_argument("--timing-only", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--decay", action="store_true", help="linear warmup over 10%% of steps, then linear decay to zero")
    ap.add_argument("--tag", default="", help="suffix of the system name, e.g. -dec-lr3e-05")
    ap.add_argument("--dev-only", action="store_true", help="evaluate on validation papers from train_2024, never on S1 main")
    a = ap.parse_args()
    from laya import Agent
    from laya.common import collate_items
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    task = Task.load("data/s1_arxiv/task.json")
    labels = list(task.labels)
    agent = Agent("convaiinnovations/laya", device=a.device)
    agent.amp_enabled = False
    model = agent.model
    model.train()
    pool = load_items("data/s1_arxiv/train_2024.jsonl")
    train = sample_balanced(pool, a.labels, labels, rng)
    data = []
    for r in train:
        it = build(agent, task, r["text"])
        data.append((it, labels.index(r["label"])))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    pad = agent.tok.pad_token_id
    n_steps = a.epochs * ((len(data) + a.batch - 1) // a.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, n_steps // 10)) * max(0.0, 1 - s / n_steps)) if a.decay else None
    step, t0, times = 0, time.time(), []
    for ep in range(a.epochs):
        rng.shuffle(data)
        for i in range(0, len(data), a.batch):
            chunk = data[i:i + a.batch]
            b = collate_items([[c[0]] for c in chunk], pad)
            ts = time.time()
            logits, _ = model(b["input_ids"].to(a.device), b["attention_mask"].to(a.device), b["marker_pos"].to(a.device),
                              b["marker_mask"].to(a.device), b["qtype"].to(a.device))
            loss = torch.nn.functional.cross_entropy(logits[:, :len(labels)].float(),
                                                     torch.tensor([c[1] for c in chunk], device=a.device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            if sched:
                sched.step()
            torch.cuda.synchronize() if a.device == "cuda" else torch.mps.synchronize()
            times.append(time.time() - ts)
            step += 1
            if step % 10 == 0:
                print(json.dumps({"epoch": ep, "step": step, "loss": round(loss.item(), 4),
                                  "s_per_step": round(float(np.mean(times[-10:])), 3),
                                  "gpu_gb": round((torch.cuda.max_memory_allocated() if a.device == "cuda" else torch.mps.driver_allocated_memory()) / 1e9, 2)}), flush=True)
            if a.steps and step >= a.steps:
                break
        if a.steps and step >= a.steps:
            break
    total_steps = a.epochs * ((len(data) + a.batch - 1) // a.batch)
    est = float(np.median(times)) * total_steps / 60
    print(json.dumps({"labels": len(data), "steps_done": step, "median_s_per_step": round(float(np.median(times)), 3),
                      "estimated_minutes_full": round(est, 1), "elapsed_s": round(time.time() - t0, 1)}), flush=True)
    if a.timing_only:
        return
    model.eval()
    items = dev_split(pool, train, labels) if a.dev_only else load_items("data/s1_arxiv/main.jsonl")
    rows = []
    for it in items:
        ts = time.time()
        res = agent.system_one(it["text"], {"q": {"type": "choice", "instructions": task.instructions, "criteria": task.labels}})
        ans = res["answers"]["q"]
        rows.append(result_row(it, f"laya-ft{len(data)}{a.tag}-s{a.seed}", "choice", pred=ans["choice"], probs=ans["probabilities"],
                               confidence=ans["confidence"], latency_s=time.time() - ts, cost_usd=0.0,
                               extra={"answer_confidence": ans.get("answer_confidence"), "train_labels": len(data),
                                      "epochs": a.epochs, "lr": a.lr, "seed": a.seed, "decay": a.decay}))
    acc = np.mean([r["pred"] == r["gold"] for r in rows])
    print(json.dumps({"system": f"laya-ft{len(data)}{a.tag}-s{a.seed}", "epochs": a.epochs, "eval": "dev" if a.dev_only else "s1_main",
                      "accuracy": round(float(acc), 4)}), flush=True)
    if a.out and not a.dev_only:
        write_rows(a.out, rows)


if __name__ == "__main__":
    main()

"""Fine-tuning comparison (post hoc, part 2, descriptive). Writes results/finetune.json and two figures.

Per configuration (model, training labels, recipe, learning rate): S1 main accuracy over seeds (mean, min, max),
selected epochs, best validation accuracy, ECE (equal mass) and automation rate at 5% error from the returned
confidence. Curves: mean training loss, validation loss and accuracy, S1 main accuracy per epoch. Overfitting is
read as the gap between the epoch of lowest validation loss and the epoch of highest validation accuracy, and the
rise of validation loss after its minimum. Reference points: Jev zero-shot and the embedding classifier from part 1.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from dma.analysis.metrics import ece_equal_mass
from dma.analysis.teil2 import max_coverage

ROOT = Path(__file__).resolve().parents[3]
ROWS = ROOT / "results/raw/main/s1.ft.jsonl"
CURVES = ROOT / "results/ft_curves"
NAME = re.compile(r"^(laya|mbert)-ft(\d+)(?:-(long|dec)-lr([0-9.e+-]+))?-s(\d+)$")


def configs():
    by = defaultdict(lambda: defaultdict(dict))
    for l in ROWS.read_text().splitlines():
        r = json.loads(l)
        m = NAME.match(r["system"])
        key = (m[1], int(m[2]), m[3] or "3ep", float(m[4]) if m[4] else None)
        by[key][int(m[5])][r["item_id"]] = r
    return by


def summarize():
    out = []
    for (model, n, recipe, lr), seeds in sorted(configs().items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3] or 0)):
        accs, eces, autos, sel, dev = [], [], [], [], []
        for s, rows in sorted(seeds.items()):
            rs = [rows[i] for i in sorted(rows)]
            ok = np.array([r["valid"] and r["pred"] == r["gold"] for r in rs], dtype=float)
            conf = np.array([float(r["confidence"]) for r in rs])
            accs.append(ok.mean())
            eces.append(ece_equal_mass(conf, ok))
            autos.append(max_coverage(conf, ok, 0.05))
            if recipe == "long":
                c = json.loads((CURVES / f"{model}-ft{n}-long-lr{lr:g}-s{s}.json").read_text())
                sel.append(c["selected_epoch"])
                dev.append(max(e["dev_acc"] for e in c["epochs"]))
        d = {"model": model, "train_labels": n, "recipe": recipe, "lr": lr, "seeds": len(accs),
             "acc_mean": round(float(np.mean(accs)), 4), "acc_min": round(float(min(accs)), 4), "acc_max": round(float(max(accs)), 4),
             "acc_per_seed": [round(float(a), 4) for a in accs], "ece_mean": round(float(np.mean(eces)), 4),
             "automation_at_5pct_mean": round(float(np.mean(autos)), 4)}
        if sel:
            d |= {"selected_epochs": sel, "best_dev_acc_mean": round(float(np.mean(dev)), 4)}
        out.append(d)
    return out


def overfitting():
    out = {}
    for f in sorted(CURVES.glob("*.json")):
        c = json.loads(f.read_text())
        ep = c["epochs"]
        dl = [e["dev_loss"] for e in ep]
        da = [e["dev_acc"] for e in ep]
        i_loss, i_acc = int(np.argmin(dl)), int(np.argmax(da))
        out[c["system"]] = {"epoch_min_dev_loss": i_loss + 1, "epoch_max_dev_acc": i_acc + 1, "min_dev_loss": dl[i_loss],
                            "final_dev_loss": dl[-1], "train_loss_final": ep[-1]["train_loss"],
                            "dev_acc_final_minus_best": round(da[-1] - da[i_acc], 4)}
    return out


def figure(n_label, lrs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex="col")
    colors = {"mbert": "#3A6F8A", "laya": "#B83A2A"}
    for model, lr in lrs.items():
        for f in sorted(CURVES.glob(f"{model}-ft{n_label}-long-lr{lr:g}-s*.json")):
            c = json.loads(f.read_text())
            x = [e["epoch"] for e in c["epochs"]]
            kw = dict(color=colors[model], alpha=0.8, lw=1.2)
            lab = f"{model} lr {lr:g}" if f.name.endswith("-s1.json") else None
            ax[0, 0].plot(x, [e["train_loss"] for e in c["epochs"]], label=lab, **kw)
            ax[0, 1].plot(x, [e["dev_loss"] for e in c["epochs"]], **kw)
            ax[1, 0].plot(x, [e["dev_acc"] for e in c["epochs"]], **kw)
            ax[1, 1].plot(x, [e["test_acc"] for e in c["epochs"]], **kw)
            ax[1, 0].plot(c["selected_epoch"], c["epochs"][c["selected_epoch"] - 1]["dev_acc"], "o", color=colors[model], ms=5)
    for a, t in zip(ax.flat, ["training loss", "validation loss", "validation accuracy", "S1 main accuracy (descriptive)"]):
        a.set_title(t, fontsize=10)
        a.grid(alpha=0.3)
    ax[0, 0].set_yscale("log")
    ref = {"400": 0.8502, "2406": 0.8625}.get(str(n_label))
    for a in (ax[1, 0], ax[1, 1]):
        a.axhline(0.8625, color="#d95f02", ls=":", lw=1, label="Jev zero-shot 86.3%")
        if n_label == 400:
            a.axhline(0.8502, color="#A16B10", ls="--", lw=1, label="embedding + LR, 400 labels (part 1)")
        a.set_ylim(0.5, 0.92)
    ax[1, 1].legend(fontsize=8, loc="lower right")
    ax[0, 0].legend(fontsize=8)
    for a in ax[1]:
        a.set_xlabel("epoch")
    fig.suptitle(f"S1 fine-tuning curves, {n_label} training labels, 3 seeds each (dot: epoch chosen on validation)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return path


def main():
    summ = summarize()
    chosen = {}
    for model in ("mbert", "laya"):
        grid = [d for d in summ if d["model"] == model and d["train_labels"] == 400 and d["recipe"] == "long"]
        if grid:
            chosen[model] = max(grid, key=lambda d: (d["best_dev_acc_mean"], -d["lr"]))["lr"]
    res = {"configs": summ, "chosen_lr": chosen, "overfitting": overfitting(),
           "reference": {"jev_zero_shot": 0.8625, "embedding_lr_part1": {"400": 0.8502, "2806": 0.8625}}}
    (ROOT / "results/finetune.json").write_text(json.dumps(res, indent=1) + "\n")
    for d in summ:
        print(d["model"], d["train_labels"], d["recipe"], d["lr"], d["acc_per_seed"], "mean", d["acc_mean"], "ECE", d["ece_mean"],
              "auto5", d["automation_at_5pct_mean"], d.get("selected_epochs", ""))
    print("chosen", chosen)
    figdir = ROOT / "results/figures"
    print(figure(400, chosen, figdir / "teil2_finetune_curves_400.png"))
    print(figure(2406, chosen, figdir / "teil2_finetune_curves_2406.png"))


if __name__ == "__main__":
    main()

"""Task and item formats shared by every system.

A task file (JSON) defines the decision contract:
    {"name": "s1_arxiv", "type": "choice" | "noul",
     "instructions": "...",
     "labels": {"cs.CL": "description", ...},          # choice only
     "none_label": {"none": "description"},             # optional, for the choice_none variant
     "paraphrases": [{"cs.CL": "...", ...}, ...]}       # optional alternative descriptions

An item file (JSONL) holds one decision per line:
    {"item_id": "...", "text": "...", "label": "cs.CL" | true | false,
     "task": "s1_arxiv", "meta": {...}}

For noul tasks the item may carry its own "question" (probes) and the label is a bool;
for P1 the meta holds the true probability "p".

Variants a runner may be asked for:
    choice      single choice question over the task labels
    choice_none choice with the none_label added
    reversed    choice with the label order reversed
    para<k>     choice with paraphrase k of the descriptions
    yesno       one yes/no question per label (choice tasks only); probs = per-label yes probability
    noul        the item's yes/no question (noul tasks)
    rationale   LLM only: reasoning before label
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Task:
    name: str
    type: str
    instructions: str
    labels: dict[str, str] = field(default_factory=dict)
    none_label: dict[str, str] = field(default_factory=dict)
    paraphrases: list[dict[str, str]] = field(default_factory=list)
    levels: list[str] = field(default_factory=list)  # score tasks (P7): ordered level descriptions

    @classmethod
    def load(cls, path: str | Path) -> "Task":
        d = json.loads(Path(path).read_text())
        return cls(**{k: d[k] for k in d if k in cls.__dataclass_fields__})

    def label_map(self, variant: str, item: dict | None = None) -> dict[str, str]:
        """Ordered label -> description for a choice variant. Items with their own "options" (P5) use those."""
        if item is not None and item.get("options"):
            return dict(item["options"])
        if variant == "choice_none":
            return {**self.labels, **self.none_label}
        if variant == "reversed":
            return dict(reversed(list(self.labels.items())))
        if variant.startswith("para"):
            return self.paraphrases[int(variant[4:])]
        return dict(self.labels)


def load_items(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def write_rows(path: str | Path, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def result_row(item: dict, system: str, variant: str, *, pred=None, probs=None, confidence=None,
               latency_s=None, cost_usd=None, valid=True, extra=None) -> dict:
    """One prediction. For noul, pred is a bool and probs = {"yes": p}."""
    return {
        "item_id": item["item_id"], "task": item["task"], "system": system, "variant": variant,
        "gold": item["label"], "pred": pred, "probs": probs or {}, "confidence": confidence,
        "latency_s": latency_s, "cost_usd": cost_usd, "valid": valid, **(extra or {}),
    }

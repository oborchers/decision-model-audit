"""Constructed probes with labels by construction (protocol section 2, S3).

P1 calibration: stated-probability scenarios, gold = true probability p.
P2 minimal pairs: routing messages where one phrase flips the label.
P3 long input: a decisive sentence inside Federal Register filler at controlled length and position.
P4 is built from arXiv in build_s1.py.

Templates are fixed slot grammars authored in this repository by the orchestrating model
(Claude), see protocol changelog. No model under test generated any text.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "probes"
SEED = 20260927

# ---------- P1 ----------
P1_TEMPLATES = {
    "urn": ("An urn contains {k} red balls and {m} blue balls and nothing else. One ball is drawn uniformly at random.",
            "Is the drawn ball red?"),
    "raffle": ("{n} raffle tickets were sold and exactly {k} of them are winning tickets. Maria holds one ticket, "
               "chosen uniformly at random from all tickets sold.", "Does Maria hold a winning ticket?"),
    "wheel": ("A fair wheel is divided into {n} equal sectors. {k} of the sectors are green and the rest are white. "
              "The wheel is spun once.", "Does the wheel stop on a green sector?"),
}
TOTALS = [10, 20, 50, 100, 1000]


def p1() -> list[dict]:
    items = []
    for fam, (ctx, q) in P1_TEMPLATES.items():
        for tenth in range(1, 10):
            p = tenth / 10
            for n in TOTALS:
                k = round(p * n)
                items.append({"item_id": f"p1-{fam}-{tenth}-{n}", "task": "p1_calibration",
                              "text": ctx.format(k=k, m=n - k, n=n), "question": q,
                              "label": None, "meta": {"p": k / n, "family": fam, "total": n}})
    return items


# ---------- P2 ----------
P2_PAIRS = [
    ("I want to cancel my subscription.", "cancel",
     "I don't want to cancel my subscription, I just want to pause it for two months.", "pause"),
    ("Please cancel my plan at the end of the month.", "cancel",
     "Please don't cancel my plan, but move me to the bigger plan at the end of the month.", "upgrade"),
    ("Why was I charged twice this month?", "billing",
     "I was charged correctly this month, but I'd like to move up to the premium tier.", "upgrade"),
    ("Stop my membership permanently.", "cancel",
     "Stop my membership for the summer only, I'll be back in September.", "pause"),
    ("I need a refund for last month's invoice.", "billing",
     "I don't need a refund, just end everything from now on.", "cancel"),
    ("Put my account on hold while I travel.", "pause",
     "Close my account for good, I won't be back.", "cancel"),
    ("Can you move me to the annual premium plan?", "upgrade",
     "Can you explain the charge on my annual plan invoice?", "billing"),
    ("I'd like to keep my subscription but skip next month.", "pause",
     "I'd like to end my subscription and not be charged next month.", "cancel"),
    ("My card was declined and I'm confused about the invoice.", "billing",
     "My card is fine, I want more seats for my team.", "upgrade"),
    ("Terminate the contract effective immediately.", "cancel",
     "Don't terminate the contract, just freeze it until January.", "pause"),
]
P2_FRAMES = [
    "{s}",
    "Hi support team, {s} Thanks, Alex",
    "Hello, this is regarding account 48213. {s}",
    "Quick question from a long-time customer: {s} Best regards",
]
P2_LABELS = {
    "cancel": "The customer wants to end the subscription or contract permanently.",
    "pause": "The customer wants to temporarily pause, freeze or skip the subscription and keep it.",
    "billing": "The customer has a question or problem about charges, invoices, refunds or payment.",
    "upgrade": "The customer wants a larger plan, a higher tier or more seats.",
}


def p2() -> list[dict]:
    items = []
    for i, (a, la, b, lb) in enumerate(P2_PAIRS):
        for j, frame in enumerate(P2_FRAMES):
            for side, (s, lab) in enumerate([(a, la), (b, lb)]):
                items.append({"item_id": f"p2-{i}-{j}-{'ab'[side]}", "task": "p2_minimal_pairs",
                              "text": frame.format(s=s), "label": lab,
                              "meta": {"pair": f"{i}-{j}", "family": f"pair{i}"}})
    return items


# ---------- P3 ----------
P3_POSITIVE = [
    "Please cancel my subscription at the end of this month.",
    "I would like to cancel my subscription effective next billing cycle.",
    "Kindly terminate my subscription, I no longer wish to be a customer.",
    "Consider this my request to cancel the subscription on my account.",
    "I am writing to cancel my subscription as of the first of next month.",
]
P3_LENGTHS = [500, 2000, 8000, 24000]  # approximate tokens (words * 1.33)
P3_POSITIONS = [0.05, 0.5, 0.95]
P3_QUESTION = "Does this text contain a request from a customer to cancel their subscription?"
FILLER_EXCLUDE = {"federal-aviation-administration", "international-trade-administration", "national-park-service",
                  "food-and-drug-administration", "environmental-protection-agency",
                  "national-oceanic-and-atmospheric-administration"}


def p3() -> list[dict]:
    rows = [json.loads(l) for l in (ROOT / "data/raw/fedreg/since_2026-09-01.jsonl").read_text().splitlines()]
    paras = []
    for r in rows:
        ab = r.get("abstract") or ""
        slugs = {a.get("slug") for a in r["agencies"]}
        if len(ab.split()) >= 40 and not slugs & FILLER_EXCLUDE and "cancel" not in ab.lower() \
                and "terminat" not in ab.lower():
            paras.append(ab)
    rng = random.Random(SEED)
    items = []
    for L in P3_LENGTHS:
        for pos in P3_POSITIONS:
            for v, sent in enumerate(P3_POSITIVE):
                rng.shuffle(paras)
                words_needed, filler = int(L / 1.33), []
                for p in paras:
                    filler.append(p)
                    if sum(len(x.split()) for x in filler) >= words_needed:
                        break
                for present in (True, False):
                    parts = list(filler)
                    if present:
                        idx = min(len(parts), round(pos * len(parts)))
                        parts.insert(idx, sent)
                    items.append({"item_id": f"p3-{L}-{int(pos*100)}-{v}-{'y' if present else 'n'}",
                                  "task": "p3_long_input", "text": "\n\n".join(parts), "question": P3_QUESTION,
                                  "label": present,
                                  "meta": {"length": L, "position": pos, "variant": v,
                                           "words": sum(len(x.split()) for x in parts), "family": f"v{v}"}})
    return items


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = {
        "p1_calibration": {"name": "p1_calibration", "type": "noul",
                           "instructions": "Answer the question about the described random event."},
        "p2_minimal_pairs": {"name": "p2_minimal_pairs", "type": "choice",
                             "instructions": "Route this customer message to the matching request type.",
                             "labels": P2_LABELS},
        "p3_long_input": {"name": "p3_long_input", "type": "noul", "instructions": P3_QUESTION},
    }
    for name, t in tasks.items():
        (OUT / f"{name}.task.json").write_text(json.dumps(t, indent=1))
    rng = random.Random(SEED)
    for name, items in [("p1_calibration", p1()), ("p2_minimal_pairs", p2()), ("p3_long_input", p3())]:
        # pilot: a fixed random subset; for P2 whole pairs
        if name == "p2_minimal_pairs":
            pairs = sorted({i["meta"]["pair"] for i in items}); rng.shuffle(pairs)
            pilot_keys = set(pairs[:3])
            pilot = [i for i in items if i["meta"]["pair"] in pilot_keys]
        else:
            ids = [i["item_id"] for i in items]; rng.shuffle(ids)
            k = 8 if name == "p1_calibration" else 6
            pilot_ids = set(ids[:k]); pilot = [i for i in items if i["item_id"] in pilot_ids]
        main_ = [i for i in items if i not in pilot]
        for split, lst in [("pilot", pilot), ("main", main_)]:
            (OUT / f"{name}.{split}.jsonl").write_text("".join(json.dumps(i) + "\n" for i in lst))
        print(name, "pilot", len(pilot), "main", len(main_))


if __name__ == "__main__":
    main()

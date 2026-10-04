"""Probes P5 to P7 (post hoc, part 2): decisions a fixed-label classifier cannot make. Labels by construction.
Built as P5a, P5b and P5c and renamed on 2026-10-03 (protocol changelog); items and labels unchanged.

P5   per-item options: an abstract and five titles from the same category; pick its own title. The four
     distractors are the lexically closest papers of the category (TF-IDF), so titles overlap in vocabulary.
P6   source and claim: is a sentence supported by the abstract? Supported (own sentence), unsupported
     (sentence from the lexically closest abstract of the same category), unsupported-hard (own sentence with one
     number changed).
P7   score: count events of one type in a JSON log of 40 to 120 events, answer on four ordered levels. Counts
     cluster at level boundaries, and near-miss event names (login_failed_mfa for login_failed) must not count.
Version 2 after the pilot: version 1 (random distractors, random other abstract, 12 to 30 events) was solved
10 of 10 by every system and could not separate them.

Source: arXiv cs papers created on or after 2026-09-01 in the eight S1 categories, not in S1 main or pilot,
same freshness rule as build_s1.py. The P7 log grammar is a fixed template authored in this repository by
the orchestrating model (Claude); no model under test generated any text.
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "probes"
SEED = 20261002
CATS = ["cs.CL", "cs.CV", "cs.CR", "cs.RO", "cs.DB", "cs.SE", "cs.HC", "cs.DC"]
LETTERS = "ABCDE"


def fresh_pool() -> dict[str, list[dict]]:
    used = set()
    for f in ("main", "pilot"):
        used |= {json.loads(l)["meta"]["arxiv_id"] for l in (ROOT / f"data/s1_arxiv/{f}.jsonl").read_text().splitlines()}
    by = defaultdict(list)
    for l in (ROOT / "data/raw/arxiv/oai_cs_2609.jsonl").read_text().splitlines():
        r = json.loads(l)
        cats = r["categories"].split() if isinstance(r["categories"], str) else r["categories"]
        if (r["arxiv_id"].startswith("2609.") and (r.get("created") or "") >= "2026-09-01"
                and len(r["abstract"].split()) >= 50 and cats[0] in CATS and r["arxiv_id"] not in used):
            r["abstract"] = " ".join(r["abstract"].split())
            r["title"] = " ".join(r["title"].split())
            by[cats[0]].append(r)
    for c in by:
        by[c].sort(key=lambda r: r["arxiv_id"])
    return by


def neighbours(by) -> dict[str, list[str]]:
    """Per paper, the other papers of its category ordered by TF-IDF cosine similarity of the abstracts.
    Lexical similarity, independent of every model under test, gives hard negatives."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    out = {}
    for c, rows in by.items():
        X = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=1).fit_transform([r["abstract"] for r in rows])
        S = (X @ X.T).toarray()
        for i, r in enumerate(rows):
            order = S[i].argsort()[::-1]
            out[r["arxiv_id"]] = [rows[j]["arxiv_id"] for j in order if j != i]
    return out


def sentences(text: str) -> list[str]:
    out = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s for s in out if 8 <= len(s.split()) <= 45]


NUM = re.compile(r"(?<![\w.-])(\d+(?:\.\d+)?)(%|\s?percent|x\b|×)?(?![\w-])")


def perturb_number(s: str, rng: random.Random) -> str | None:
    cands = [m for m in NUM.finditer(s) if not re.fullmatch(r"(19|20)\d\d", m.group(1)) and m.group(1) not in {"0", "1"}]
    if not cands:
        return None
    m = rng.choice(cands)
    v = m.group(1)
    if "." in v:
        whole, frac = v.split(".")
        d = int(frac[-1])
        new_frac = frac[:-1] + str((d + rng.choice([3, 4, 5, 6])) % 10)
        new = f"{whole}.{new_frac}"
    else:
        n = int(v)
        new = str(n * 2 if n < 10 else n + max(1, round(n * rng.choice([0.3, 0.4, 0.5]))))
    if new == v:
        return None
    return s[:m.start(1)] + new + s[m.end(1):]


def p5_titles(by, rng, nb, idx) -> list[dict]:
    items = []
    for c in CATS:
        pool = by[c][:]
        rng.shuffle(pool)
        for r in pool[:25]:
            others = [idx[a] for a in nb[r["arxiv_id"]][:4]]  # four lexically closest papers of the category
            titles = [r["title"]] + [o["title"] for o in others]
            rng.shuffle(titles)
            opts = {LETTERS[i]: t for i, t in enumerate(titles)}
            gold = LETTERS[titles.index(r["title"])]
            items.append({"item_id": f"p5-{r['arxiv_id']}", "task": "p5_titles", "text": r["abstract"],
                          "options": opts, "label": gold,
                          "meta": {"arxiv_id": r["arxiv_id"], "category": c,
                                   "distractors": [o["arxiv_id"] for o in others]}})
    return items


def p6_claims(by, rng, used_ids, nb, idx) -> list[dict]:
    items, kinds = [], ["supported", "other_abstract", "number_changed"]
    for c in CATS:
        pool = [r for r in by[c] if r["arxiv_id"] not in used_ids]
        rng.shuffle(pool)
        want = {k: 9 for k in kinds}  # 27 per category, 216 in total
        for r in pool:
            if not any(want.values()):
                break
            sents = sentences(r["abstract"])
            if len(sents) < 3:
                continue
            k = rng.choice([k for k in kinds if want[k]])
            if k == "supported":
                claim, label = rng.choice(sents[1:]), True
            elif k == "other_abstract":
                other = idx[nb[r["arxiv_id"]][0]]  # lexically closest abstract of the category
                os_ = sentences(other["abstract"])[1:]
                if not os_:
                    continue
                claim, label = rng.choice(os_), False
            else:
                opts = [s for s in sents if perturb_number(s, random.Random(0)) is not None]
                if not opts:
                    continue
                s = rng.choice(opts)
                claim, label = perturb_number(s, rng), False
                if claim is None:
                    continue
            want[k] -= 1
            items.append({"item_id": f"p6-{r['arxiv_id']}", "task": "p6_claims", "text": r["abstract"],
                          "question": f'Is the following statement supported by the text? Statement: "{claim}"',
                          "label": label, "meta": {"arxiv_id": r["arxiv_id"], "category": c, "kind": k, "claim": claim}})
    return items


EVENT_TYPES = ["login_failed", "login_ok", "file_download", "password_reset", "permission_change"]
NEAR_MISS = {"login_failed": "login_failed_mfa", "login_ok": "login_ok_sso", "file_download": "file_download_failed",
             "password_reset": "password_reset_requested", "permission_change": "permission_change_reverted"}
LEVELS = ["No such event", "1 to 2 such events", "3 to 5 such events", "6 or more such events"]
LEVEL_OF = lambda n: 0 if n == 0 else 1 if n <= 2 else 2 if n <= 5 else 3  # noqa: E731


def p7_count(rng) -> list[dict]:
    items, i = [], 0
    targets = {0: [0], 1: [1, 2, 2], 2: [3, 5, 5], 3: [6, 6, 7]}  # weighted towards level boundaries
    for level in range(4):
        for _ in range(38):
            target = rng.choice(EVENT_TYPES)
            n_target = rng.choice(targets[level])
            n_total = rng.randint(40, 120)
            others = [t for t in EVENT_TYPES if t != target] + [NEAR_MISS[target]] * 2  # near-miss names must not count
            events = [target] * n_target + [rng.choice(others) for _ in range(n_total - n_target)]
            rng.shuffle(events)
            log = [{"time": f"2026-09-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}",
                    "user": f"u{rng.randint(100, 140)}", "event": e} for e in events]
            log.sort(key=lambda e: e["time"])
            items.append({"item_id": f"p7-{i:03d}", "task": "p7_count", "text": json.dumps({"events": log}),
                          "question": f'How many "{target}" events does the log contain?', "label": LEVEL_OF(n_target),
                          "meta": {"target": target, "count": n_target, "total": n_total}})
            i += 1
    rng.shuffle(items)
    return items


TASKS = {
    "p5_titles": {"name": "p5_titles", "type": "choice",
                   "instructions": "Which of the listed titles belongs to the paper whose abstract is given?",
                   "labels": {}},
    "p6_claims": {"name": "p6_claims", "type": "noul",
                   "instructions": "Decide whether the statement is supported by the given text."},
    "p7_count": {"name": "p7_count", "type": "score",
                  "instructions": "Count the events of the requested type in the JSON log.", "levels": LEVELS},
}


def main():
    rng = random.Random(SEED)
    by = fresh_pool()
    nb = neighbours(by)
    idx = {r["arxiv_id"]: r for rows in by.values() for r in rows}
    a = p5_titles(by, rng, nb, idx)
    b = p6_claims(by, rng, {it["meta"]["arxiv_id"] for it in a}, nb, idx)
    c = p7_count(rng)
    rng.shuffle(a)
    rng.shuffle(b)
    for name, items in (("p5_titles", a), ("p6_claims", b), ("p7_count", c)):
        (OUT / f"{name}.task.json").write_text(json.dumps(TASKS[name], indent=1) + "\n")
        pilot, main_ = items[:10], items[10:]
        for split, part in (("pilot", pilot), ("main", main_)):
            (OUT / f"{name}.{split}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in part))
        print(name, "pilot", len(pilot), "main", len(main_))
    print("pool per category", {k: len(v) for k, v in by.items()})


if __name__ == "__main__":
    main()

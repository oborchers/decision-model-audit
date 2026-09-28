"""Runners for API systems (protocol section 3/4): jev, nano, haiku, flash.

uv run python -m dma.runners.api --system jev --task T --items I --variant choice --out O [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor

from dma.client import chat, decisions
from dma.tasks import Task, load_items, result_row, write_rows

LLM = {
    "luna": ("openai/gpt-6-luna", "OpenAI", {"reasoning": {"effort": "none"}}),
    "nano": ("openai/gpt-5.4-nano", "OpenAI", {"reasoning": {"effort": "none"}}),
    "haiku": ("anthropic/claude-haiku-4.5", "Anthropic", {"temperature": 0}),
    "flash": ("google/gemini-3.5-flash-lite", "Google AI Studio", {"temperature": 0, "reasoning": {"effort": "minimal"}}),
    "flash31": ("google/gemini-3.1-flash-lite", "Google AI Studio", {"temperature": 0, "reasoning": {"effort": "none"}}),
    "sonnet": ("anthropic/claude-sonnet-5", "Anthropic", {"reasoning": {"effort": "none"}}),
}
JEV_MODEL = "typesafe/jev-1.13"


def options_text(labels: dict[str, str]) -> str:
    return "\n".join(f"- {k}: {v}" for k, v in labels.items())


# ---------------- Jev ----------------
def run_jev(task: Task, item: dict, variant: str) -> dict:
    if task.type == "noul" or variant == "noul":
        q = {"q": {"type": "noul", "instructions": item.get("question") or task.instructions}}
        rec = decisions(JEV_MODEL, item["text"], q)
        resp = rec["response"]
        if "error" in resp:
            return result_row(item, "jev", variant, valid=False, extra={"error": resp["error"]})
        p = resp["answers"]["q"]["noul"]
        return result_row(item, "jev", "noul", pred=p >= 0.5, probs={"yes": p}, confidence=max(p, 1 - p),
                          latency_s=None if rec["cached"] else rec["latency_s"], cost_usd=resp["usage"]["cost"],
                          extra={"gen_id": resp.get("id"), "usage": resp.get("usage"), "cached": rec["cached"]})
    if variant == "yesno":
        labels = task.label_map("choice")
        keys = {f"l{i}": lab for i, lab in enumerate(labels)}
        qs = {k: {"type": "noul", "instructions": f"{task.instructions} Is the correct answer \"{lab}\" "
                                                  f"({labels[lab]})?"} for k, lab in keys.items()}
        rec = decisions(JEV_MODEL, item["text"], qs)
        resp = rec["response"]
        if "error" in resp:
            return result_row(item, "jev", variant, valid=False, extra={"error": resp["error"]})
        probs = {keys[k]: resp["answers"][k]["noul"] for k in keys}
        pred = max(probs, key=probs.get)
        s = sum(probs.values()) or 1.0
        return result_row(item, "jev", variant, pred=pred, probs=probs, confidence=probs[pred] / s,
                          latency_s=None if rec["cached"] else rec["latency_s"], cost_usd=resp["usage"]["cost"],
                          extra={"gen_id": resp.get("id"), "yes_sum": s, "cached": rec["cached"]})
    labels = task.label_map(variant)
    q = {"q": {"type": "choice", "instructions": task.instructions, "criteria": labels}}
    rec = decisions(JEV_MODEL, item["text"], q)
    resp = rec["response"]
    if "error" in resp:
        return result_row(item, "jev", variant, valid=False, extra={"error": resp["error"]})
    a = resp["answers"]["q"]
    return result_row(item, "jev", variant, pred=a["choice"], probs=a["probabilities"], confidence=a["confidence"],
                      latency_s=None if rec["cached"] else rec["latency_s"], cost_usd=resp["usage"]["cost"],
                      extra={"gen_id": resp.get("id"), "usage": resp.get("usage"), "cached": rec["cached"]})


# ---------------- LLMs ----------------
def _schema(props: dict, required: list[str]) -> dict:
    return {"type": "json_schema", "json_schema": {"name": "decision", "strict": True, "schema": {
        "type": "object", "additionalProperties": False, "properties": props, "required": required}}}


def run_llm(system: str, task: Task, item: dict, variant: str) -> dict:
    model, provider, extra = LLM[system]
    if task.type == "noul":
        question = item.get("question") or task.instructions
        sys_msg = (f"{task.instructions}\n\nQuestion: {question}\n\nReturn your answer (yes or no) and "
                   "probability_yes, the probability from 0 to 1 that the correct answer is yes.")
        fmt = _schema({"answer": {"type": "string", "enum": ["yes", "no"]}, "probability_yes": {"type": "number"}},
                      ["answer", "probability_yes"])
    else:
        labels = task.label_map("choice" if variant == "rationale" else variant)
        sys_msg = f"{task.instructions}\n\nOptions:\n{options_text(labels)}\n\n"
        props = {"label": {"type": "string", "enum": list(labels)}, "confidence": {"type": "number"}}
        req = ["label", "confidence"]
        if variant == "rationale":
            sys_msg += ("First write a brief rationale (at most three sentences) that quotes the phrases of the text "
                        "your decision rests on. Then return the single best option and confidence, the probability "
                        "from 0 to 1 that your chosen option is correct.")
            props = {"reasoning": {"type": "string"}, **props}
            req = ["reasoning", *req]
        else:
            sys_msg += "Return the single best option and confidence, the probability from 0 to 1 that it is correct."
        fmt = _schema(props, req)
    body = {"model": model, "messages": [{"role": "system", "content": sys_msg}, {"role": "user", "content": item["text"]}],
            "response_format": fmt, "max_tokens": 600 if variant == "rationale" else 150, "usage": {"include": True},
            "provider": {"order": [provider], "allow_fallbacks": False, "require_parameters": True}, **extra}
    rec = chat(body)
    resp = rec["response"]
    lat = None if rec["cached"] else rec["latency_s"]
    if "error" in resp or not resp.get("choices"):
        return result_row(item, system, variant, valid=False, latency_s=lat, extra={"error": resp.get("error", resp)})
    usage = resp.get("usage", {})
    try:
        out = json.loads(resp["choices"][0]["message"]["content"])
    except (json.JSONDecodeError, TypeError) as e:
        return result_row(item, system, variant, valid=False, latency_s=lat, cost_usd=usage.get("cost"),
                          extra={"error": f"unparseable: {e}", "raw": resp["choices"][0]["message"].get("content")})
    x = {"gen_id": resp.get("id"), "usage": usage, "provider": resp.get("provider"), "cached": rec["cached"]}
    if task.type == "noul":
        p = float(out["probability_yes"])
        return result_row(item, system, "noul", pred=out["answer"] == "yes", probs={"yes": p},
                          confidence=max(p, 1 - p), latency_s=lat, cost_usd=usage.get("cost"), extra=x)
    if variant == "rationale":
        x["reasoning"] = out.get("reasoning")
    c = float(out["confidence"])
    return result_row(item, system, variant, pred=out["label"], probs={out["label"]: c}, confidence=c,
                      latency_s=lat, cost_usd=usage.get("cost"), extra=x)


# ---------------- batches: several items in one request (post hoc gap closing) ----------------
def run_batch(system: str, task: Task, items: list[dict], variant: str) -> list[dict]:
    """One request for len(items) decisions. Jev: one state with all papers, one choice question per paper.
    LLM: all papers in one user message, JSON array of labels and confidences. Cost and latency per request
    are divided evenly across its items (recorded as cost_usd / latency_s per row, plus batch totals)."""
    labels = task.label_map("choice")
    k = len(items)
    if system == "jev":
        state = {"papers": {f"p{i}": it["text"] for i, it in enumerate(items)}}
        qs = {f"q{i}": {"type": "choice", "instructions": f"{task.instructions} The paper is `papers.p{i}`.",
                        "criteria": labels} for i in range(k)}
        rec = decisions(JEV_MODEL, state, qs)
        resp = rec["response"]
        if "error" in resp:
            return [result_row(it, system, variant, valid=False, extra={"error": resp["error"]}) for it in items]
        cost, lat = resp["usage"]["cost"], (None if rec["cached"] else rec["latency_s"])
        rows = []
        for i, it in enumerate(items):
            a = resp["answers"].get(f"q{i}")
            if not a:
                rows.append(result_row(it, system, variant, valid=False, extra={"error": "missing answer"}))
                continue
            rows.append(result_row(it, system, variant, pred=a["choice"], probs=a["probabilities"],
                                   confidence=a["confidence"], latency_s=lat / k if lat else None, cost_usd=cost / k,
                                   extra={"batch_size": k, "batch_cost": cost, "batch_latency_s": lat,
                                          "cached": rec["cached"]}))
        return rows
    model, provider, extra = LLM[system]
    sys_msg = (f"{task.instructions}\n\nOptions:\n{options_text(labels)}\n\nYou receive {k} papers, numbered 0 to "
               f"{k - 1}. For each paper, in order, return the single best option and confidence, the probability "
               "from 0 to 1 that it is correct.")
    user = "\n\n".join(f"### Paper {i}\n{it['text']}" for i, it in enumerate(items))
    item_schema = {"type": "object", "additionalProperties": False,
                   "properties": {"paper": {"type": "integer"}, "label": {"type": "string", "enum": list(labels)},
                                  "confidence": {"type": "number"}}, "required": ["paper", "label", "confidence"]}
    fmt = _schema({"decisions": {"type": "array", "items": item_schema}}, ["decisions"])
    body = {"model": model, "messages": [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}],
            "response_format": fmt, "max_tokens": 60 * k + 100, "usage": {"include": True},
            "provider": {"order": [provider], "allow_fallbacks": False, "require_parameters": True}, **extra}
    rec = chat(body)
    resp = rec["response"]
    if "error" in resp or not resp.get("choices"):
        return [result_row(it, system, variant, valid=False, extra={"error": resp.get("error", resp)}) for it in items]
    usage = resp.get("usage", {})
    cost, lat = usage.get("cost") or 0, (None if rec["cached"] else rec["latency_s"])
    try:
        dec = {d["paper"]: d for d in json.loads(resp["choices"][0]["message"]["content"])["decisions"]}
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return [result_row(it, system, variant, valid=False, extra={"error": f"unparseable: {e}"}) for it in items]
    rows = []
    for i, it in enumerate(items):
        d = dec.get(i)
        if not d:
            rows.append(result_row(it, system, variant, valid=False, extra={"error": "missing paper"}))
            continue
        c = float(d["confidence"])
        rows.append(result_row(it, system, variant, pred=d["label"], probs={d["label"]: c}, confidence=c,
                               latency_s=lat / k if lat else None, cost_usd=cost / k,
                               extra={"batch_size": k, "batch_cost": cost, "batch_latency_s": lat, "cached": rec["cached"]}))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--variant", default="choice")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-words", type=int, default=0, help="skip items longer than this (cost cap)")
    a = ap.parse_args()
    task, items = Task.load(a.task), load_items(a.items)
    if a.max_words:
        items = [i for i in items if len(i["text"].split()) <= a.max_words]
    if a.variant.startswith("batch"):
        k = int(a.variant[5:])
        chunks = [items[i:i + k] for i in range(0, len(items), k)]
        with ThreadPoolExecutor(a.workers) as ex:
            rows = [r for rs in ex.map(lambda ch: run_batch(a.system, task, ch, a.variant), chunks) for r in rs]
    else:
        fn = (lambda it: run_jev(task, it, a.variant)) if a.system == "jev" else \
             (lambda it: run_llm(a.system, task, it, a.variant))
        with ThreadPoolExecutor(a.workers) as ex:
            rows = list(ex.map(fn, items))
    write_rows(a.out, rows)
    ok = sum(r["valid"] for r in rows)
    cost = sum((r["cost_usd"] or 0) for r in rows if not r.get("cached"))
    print(f"{a.system}/{a.variant}: {ok}/{len(rows)} valid, new cost ${cost:.4f}")


if __name__ == "__main__":
    main()

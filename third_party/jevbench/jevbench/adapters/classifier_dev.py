"""classifier.dev adapter, fast tier: public zero-shot classification over HTTP (production API).

Interface (https://classifier.dev/openapi.json and /llms.txt, read 2026-09-19):
  POST https://classifier.dev/v1/classify
  {"input": <text>, "labels": [...], "instructions": "<extra criteria>", "tier": "fast"}
  -> {"results": [{"label", "confidence", "scores": {label: p, ...}, "model", "ms"}], ...}
No key (free tier, per-IP limits 3,000/minute and 20,000/day on fast). Its own benchmark page says the fast tier
is Jev (TypeSafe's decision model), so this row shows what the wrapper adds or costs.

classifier.dev has one primitive, single-label choice, so every JevBench record is sent as one:
  choice  labels = the option keys
  noul    labels = ["yes", "no"]
  score   labels = the level texts (unique and <= 200 characters on every item), mapped back to level indices;
          "'urgent bug' classifies better than 'p0'" is the API's own advice, so index names are not used
  instructions = the question's instructions plus the full rubric (every option or level with its description),
          the same rubric every other adapter sends, never less of it
  input   = the state (JSON states serialized)
Distribution: `scores` ("probability per label, sums to 1 for single-label"), native. If the service withholds
scores ("unscored": input does not read as natural language) the item is scored on its label alone.

Politeness: one request at a time; a 429 is answered by waiting its Retry-After (bounded) and retrying the same
bytes at most 3 times; no retries on anything else. Latency is the runner's wall clock (network included).
"""

from __future__ import annotations

import json
import time

from .base import DecisionResult, http_post_json


def cd_labels(task):
    """(labels sent, map back to JevBench label)."""
    qtype, crit = task.question["type"], task.question.get("criteria")
    if qtype == "noul":
        return ["yes", "no"], {"yes": "yes", "no": "no"}
    if qtype == "score":
        texts = [str(d).strip() for d in crit]
        if len(set(texts)) == len(texts) and all(0 < len(x) <= 200 for x in texts):
            return texts, {x: str(i) for i, x in enumerate(texts)}
        idx = [str(i) for i in range(len(crit))]
        return idx, {i: i for i in idx}
    keys = list(crit)
    return keys, {k: k for k in keys}


def cd_instructions(task) -> str:
    q, qtype = task.question, task.question["type"]
    crit = q.get("criteria")
    lines = [q["instructions"].strip()]
    if qtype == "noul":
        crit = crit or {}
        if crit.get("true") or crit.get("false"):
            lines += ["", "Options:", f"- yes: {crit.get('true') or 'yes'}", f"- no: {crit.get('false') or 'no'}"]
    elif qtype == "score":
        lines += ["", "Levels, lowest to highest:"] + [f"- {str(d).strip()}" for d in crit]
    elif any(v for v in crit.values()):
        lines += ["", "Options:"] + [f"- {k}: {v or k}" for k, v in crit.items()]
    return "\n".join(lines)


class ClassifierDevAdapter:
    name = "classifier_dev"
    cost_basis = "published_plan_price"

    def __init__(self, endpoint=None, model=None, key_env="", timeout_s=120.0,
                 price_input_per_m=None, price_output_per_m=None):
        self.endpoint = (endpoint or "https://classifier.dev").rstrip("/")
        self.model = model or "fast"
        self.key_env = key_env
        self.timeout_s = timeout_s
        self.price_input_per_m = price_input_per_m
        self.price_output_per_m = price_output_per_m

    def build_request(self, task) -> dict:
        labels, _ = cd_labels(task)
        state = task.state if isinstance(task.state, str) else json.dumps(task.state, ensure_ascii=False)
        return {"input": state, "labels": labels, "instructions": cd_instructions(task), "tier": self.model}

    def run(self, task) -> DecisionResult:
        body = self.build_request(task)
        _, back = cd_labels(task)
        headers = {"Content-Type": "application/json", "Accept": "application/json",
                   "User-Agent": "jevbench/1.2 (+https://github.com/fstandhartinger/jevbench)"}
        retries, waited = 0, 0.0
        while True:
            try:
                status, parsed, latency = http_post_json(f"{self.endpoint}/v1/classify", body, headers, self.timeout_s)
            except ConnectionError as e:
                return DecisionResult(adapter=self.name, ok=False, error=str(e), probs_source="native",
                                      model=self.model, request_body=body)
            if status == 429 and retries < 3:
                ra = 30.0
                if isinstance(parsed, dict):
                    ra = float(parsed.get("retry_after") or parsed.get("retryAfter") or ra)
                ra = min(max(ra, 5.0), 120.0)
                time.sleep(ra)
                retries += 1
                waited += ra
                continue
            break
        res = DecisionResult(adapter=self.name, ok=False, status=status, latency_s=latency, probs_source="native",
                             model=self.model, raw=parsed, request_body=body)
        if status != 200 or not isinstance(parsed, dict):
            res.error = f"HTTP {status}: {str(parsed)[:300]}"
            return res
        try:
            r = parsed["results"][0]
            res.model = r.get("model") or parsed.get("model") or self.model
            res.usage = {"classifications": (parsed.get("usage") or {}).get("classifications"),
                         "server_ms": r.get("ms"), **({"retries_429": retries, "waited_s": waited} if retries else {})}
            scores = r.get("scores")
            if isinstance(scores, dict) and scores:
                probs = {back[k]: float(v) for k, v in scores.items()}
                if set(probs) != set(back.values()):
                    raise ValueError(f"score labels {sorted(scores)} do not match the labels sent")
                res.probs = probs
            else:
                res.probs_source = "label_only_no_calibrated_distribution"
                res.label = back.get(r.get("label"))
                res.usage["unscored"] = r.get("unscored") or "no scores returned"
        except (KeyError, IndexError, TypeError, ValueError) as e:
            res.error = f"answer parse failed: {e}"
            return res
        res.ok = True
        return res

    def reserve_estimate(self, task) -> float:
        return 0.0

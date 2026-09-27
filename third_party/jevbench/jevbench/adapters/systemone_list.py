"""Native adapter for the list-shaped `/decide` flavour of the Jev wire format.

Several open rebuilds keep Jev's three primitives but send questions as a list
with an `id`, and name the option/level field differently:

  POST {endpoint}/decide
    {"state": ..., "questions": [
        {"id": "decision", "type": "choice", "instructions": ..., "options": {...}},
        {"id": "decision", "type": "score",  "instructions": ..., "levels": [...]},
        {"id": "decision", "type": "noul",   "instructions": ..., "criteria": {...}}]}
  -> {"answers": [{"id": "decision", ...}], "latency_ms": ..., "model": ...}

Answers carry the same fields as the canonical format, so the mapping to
exact-label probabilities is identical to `TypeSafeAdapter`.
"""

from __future__ import annotations

import json

from .base import DecisionResult, http_post_json


class SystemOneListAdapter:
    name = "systemone_list"

    def __init__(self, endpoint, model=None, key_env="", timeout_s=120.0,
                 price_input_per_m=None, price_output_per_m=None, path="/decide"):
        self.endpoint = endpoint.rstrip("/")
        self.model = model or ""
        self.key_env = key_env
        self.timeout_s = timeout_s
        self.price_input_per_m = price_input_per_m
        self.price_output_per_m = price_output_per_m
        self.path = path

    def build_request(self, task) -> dict:
        q = {"id": "decision", "type": task.question["type"],
             "instructions": task.question["instructions"]}
        crit = task.question.get("criteria")
        if task.question["type"] == "choice":
            q["options"] = crit
        elif task.question["type"] == "score":
            q["levels"] = crit
        elif crit is not None:
            q["criteria"] = crit
        return {"state": task.state, "questions": [q]}

    def run(self, task) -> DecisionResult:
        import os

        key = os.environ.get(self.key_env, "") if self.key_env else ""
        body = self.build_request(task)
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        try:
            status, parsed, latency = http_post_json(
                self.endpoint + self.path, body, headers, self.timeout_s
            )
        except ConnectionError as e:
            return DecisionResult(adapter=self.name, ok=False, error=str(e),
                                  probs_source="native", model=self.model,
                                  request_body=body)
        res = DecisionResult(adapter=self.name, ok=False, status=status,
                             latency_s=latency, probs_source="native",
                             model=self.model, raw=parsed, request_body=body)
        if status != 200 or not isinstance(parsed, dict):
            res.error = f"HTTP {status}: {str(parsed)[:300]}"
            return res
        answers = parsed.get("answers")
        if not isinstance(answers, list) or len(answers) != 1:
            res.error = "expected exactly one answer in a list"
            return res
        ans = answers[0]
        res.model = parsed.get("model") or self.model
        res.usage = parsed.get("usage") or {}
        qtype = task.question["type"]
        if not isinstance(ans, dict) or ans.get("type") != qtype:
            res.error = "native answer type mismatch"
            return res
        try:
            if qtype == "noul":
                p = ans["noul"]
                if isinstance(p, bool) or not isinstance(p, (int, float)):
                    raise ValueError("noul must be numeric")
                p = float(p)
                if not (0.0 <= p <= 1.0):
                    raise ValueError(f"noul out of range: {p}")
                res.probs = {"yes": p, "no": 1.0 - p}
            else:
                probs = ans.get("probabilities")
                if not isinstance(probs, dict):
                    raise ValueError(f"{qtype} answer missing probabilities")
                if qtype == "choice" and ans.get("choice") not in task.labels:
                    raise ValueError("invalid native choice")
                res.probs = probs
        except (KeyError, TypeError, ValueError) as e:
            res.error = f"answer parse failed: {e}"
            return res
        res.ok = True
        return res

    def reserve_estimate(self, task) -> float:
        if self.price_input_per_m is None or self.price_output_per_m is None:
            return None
        return (100_000 * self.price_input_per_m / 1e6
                + 4_000 * self.price_output_per_m / 1e6)

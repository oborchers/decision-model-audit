"""Client for `r2_serve.py`: runs a library-only entrant's in-process adapter on a remote GPU.

The answer key never leaves our side: the task is sent without `expected` and
without provenance. The server runs the very same adapter code as an in-process run
(SemIfDirectAdapter / So1DeciderAdapter), so only the network hop is added, which the
runner's wall-clock latency includes - the same rule as every other remote entrant.
"""

from __future__ import annotations

from dataclasses import asdict

from .base import DecisionResult, http_post_json


class RemoteInprocAdapter:
    name = "remote_inproc"
    cost_basis = "self_hosted_gpu"

    def __init__(self, endpoint=None, model=None, key_env="", timeout_s=120.0,
                 price_input_per_m=None, price_output_per_m=None, revision=None):
        self.endpoint = (endpoint or "").rstrip("/")
        self.model = model or ""
        self.timeout_s = timeout_s
        self.price_input_per_m = price_input_per_m
        self.price_output_per_m = price_output_per_m

    def run(self, task) -> DecisionResult:
        body = {"task": {**asdict(task), "expected": None, "provenance": {}}}
        res = DecisionResult(adapter=self.name, ok=False, probs_source="native", model=self.model)
        try:
            status, parsed, latency = http_post_json(f"{self.endpoint}/run", body,
                                                     {"Content-Type": "application/json"}, self.timeout_s)
        except ConnectionError as e:
            res.error = str(e)
            return res
        res.status, res.latency_s = status, latency
        if status != 200 or not isinstance(parsed, dict):
            res.error = f"HTTP {status}: {str(parsed)[:300]}"
            return res
        res.ok = bool(parsed.get("ok"))
        res.probs = parsed.get("probs")
        res.error = parsed.get("error")
        res.usage = parsed.get("usage") or {}
        res.model = parsed.get("model") or self.model
        res.raw = parsed.get("raw")
        if isinstance(res.raw, dict):
            res.raw.setdefault("runtime", {})
            res.raw["runtime"] = dict(res.raw["runtime"] or {}, server_latency_s=parsed.get("latency_s"))
        return res

    def reserve_estimate(self, task) -> float:
        return 0.0

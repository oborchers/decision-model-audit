"""SystemOne-style endpoints outside OpenRouter and Workers AI (part 2, 2026-10).

Local servers that speak POST /v1/systemone (Strands Decider, Open-Jev, CLM) and hosted APIs with the same body
(Perplexity /v1/decisions, Fastino /v1/systemone). Same on-disk cache as the other clients; responses are
normalised to the Jev shape with usage.cost (0 for local servers).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import httpx
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".cache" / "ext"

# system -> (url, model name, auth: None | (env var, header, prefix), USD per 1M input tokens)
PROVIDERS = {
    "strands-2b": (os.environ.get("DMA_STRANDS_URL", "http://127.0.0.1:8401/v1/systemone"), "strands-decider-2B-hobson-v19", None, 0.0),
    "openjev-2b": (os.environ.get("DMA_OPENJEV_URL", "http://127.0.0.1:8402/v1/systemone"), "Open-Jev-2B", None, 0.0),
    "apus-4b": (os.environ.get("DMA_APUS4_URL", "http://127.0.0.1:8404/v1/systemone"), "APUS-OpenJev-v1-4B-MLX-4bit", None, 0.0),
    "apus-9b": (os.environ.get("DMA_APUS9_URL", "http://127.0.0.1:8405/v1/systemone"), "APUS-OpenJev-v1-9B-MLX-4bit", None, 0.0),
    "decision2-nox-4b": (os.environ.get("DMA_D2NOX_URL", "http://127.0.0.1:8406/v1/systemone"), "Decision-2.0-Nox-4B", None, 0.0),
    "clef-selfhost": (os.environ.get("DMA_CLEF_URL", "http://127.0.0.1:8408/v1/systemone"), "clef", None, 0.0),
    "clef-flash-selfhost": (os.environ.get("DMA_CLEFF_URL", "http://127.0.0.1:8409/v1/systemone"), "clef-flash", None, 0.0),
    "clef-selfhost-64k": (os.environ.get("DMA_CLEF_URL", "http://127.0.0.1:8408/v1/systemone"), "clef", None, 0.0),
    "clef-flash-selfhost-64k": (os.environ.get("DMA_CLEFF_URL", "http://127.0.0.1:8409/v1/systemone"), "clef-flash", None, 0.0),
    "decision2-kai-0.6b": (os.environ.get("DMA_D2KAI_URL", "http://127.0.0.1:8407/v1/systemone"), "Decision-2.0-Kai-0.6B", None, 0.0),
    "clm-8b": (os.environ.get("DMA_CLM_URL", "http://127.0.0.1:8403/v1/systemone"), "CLM-v0.1-8B", None, 0.0),
    "pplx-decider": ("https://api.perplexity.ai/v1/decisions", "pplx-decider-v1-27b",
                     ("PERPLEXITY_API_KEY", "Authorization", "Bearer "), 0.04),
    "glide": ("https://api.fastino.ai/v1/systemone", "fastino/GLiDE", ("FASTINO_API_KEY", "X-API-Key", ""), 0.30),
}
_client = httpx.Client(timeout=300.0)
LEDGER = ROOT / "results" / "raw" / "teil2" / "spend_ext.jsonl"
BUDGET = float(os.environ.get("DMA_EXT_BUDGET", "1.0"))  # USD, hard cap per paid provider (part 2)
_spent: dict | None = None


class BudgetExceeded(RuntimeError):
    pass


def _spend(system: str, add: float = 0.0) -> float:
    global _spent
    if _spent is None:
        _spent = {}
        if LEDGER.exists():
            for l in LEDGER.read_text().splitlines():
                if l.strip():
                    e = json.loads(l)
                    _spent[e["system"]] = _spent.get(e["system"], 0.0) + e["cost"]
    if add:
        _spent[system] = _spent.get(system, 0.0) + add
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with LEDGER.open("a") as f:
            f.write(json.dumps({"ts": time.time(), "system": system, "cost": add}) + "\n")
    return _spent.get(system, 0.0)


def _key(var: str) -> str:
    f = os.environ.get("DMA_EXT_KEY_FILE")
    k = os.environ.get(var) or (dotenv_values(f).get(var) if f else None)
    if not k:
        raise RuntimeError(f"Set {var}, or DMA_EXT_KEY_FILE to a dotenv file that contains it.")
    return k


def _post(system: str, body: dict, retries: int = 4) -> tuple[dict, float]:
    url, _, auth, price = PROVIDERS[system]
    headers = {"Content-Type": "application/json"}
    if auth:
        headers[auth[1]] = auth[2] + _key(auth[0])
    last = None
    for attempt in range(retries):
        if price and _spend(system) >= BUDGET:
            raise BudgetExceeded(f"{system} spend reached cap {BUDGET} USD")
        t0 = time.perf_counter()
        try:
            r = _client.post(url, json=body, headers=headers)
            lat = time.perf_counter() - t0
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}: {r.text[:300]}"
                time.sleep(2 ** attempt)
                continue
            if r.status_code >= 400:
                return {"error": {"status": r.status_code, "body": r.text[:2000]}}, lat
            d = r.json()
            u = d.setdefault("usage", {})
            tok = u.get("input_tokens") or u.get("prompt_tokens") or 0
            u.setdefault("cost", tok * price / 1e6)
            if price:
                _spend(system, u["cost"])
            return d, lat
        except (httpx.HTTPError, ValueError) as e:
            last = repr(e)
            time.sleep(2 ** attempt)
    return {"error": {"status": None, "body": last}}, float("nan")


def decisions(system: str, state, questions: dict) -> dict:
    body = {"model": PROVIDERS[system][1], "state": state, "questions": questions}
    key = hashlib.sha256((system + json.dumps(body, sort_keys=False)).encode()).hexdigest()
    f = CACHE / key[:2] / f"{key}.json"
    if f.exists() and not os.environ.get("DMA_NO_CACHE"):
        rec = json.loads(f.read_text())
        rec["cached"] = True
        return rec
    resp, lat = _post(system, body)
    rec = {"system": system, "request": body, "response": resp, "latency_s": lat, "ts": time.time()}
    err = resp.get("error")
    transient = err and (err.get("status") is None or err.get("status", 0) >= 500)
    if not transient and not (os.environ.get("DMA_NO_CACHE") and f.exists()):
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(rec))
    rec["cached"] = False
    return rec

"""OpenRouter access with an on-disk request cache.

Every request body is hashed; the raw response is stored under .cache/ and
appended to results/raw/<run>.jsonl by the runners. A repeated request never
costs money twice, and every published number can be traced to a raw response.
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
CACHE = ROOT / ".cache" / "openrouter"
KEY_PATH = Path("~/.config/agent-os/openrouter.env").expanduser()
BASE = "https://openrouter.ai/api"

_key: str | None = None


def _api_key() -> str:
    global _key
    if _key is None:
        _key = os.environ.get("OPENROUTER_API_KEY") or dotenv_values(KEY_PATH)["OPENROUTER_API_KEY"]
    return _key


def _post(path: str, body: dict, timeout: float = 120.0, retries: int = 4) -> tuple[dict, float]:
    last = None
    for attempt in range(retries):
        t0 = time.perf_counter()
        try:
            r = httpx.post(
                f"{BASE}{path}",
                json=body,
                headers={"Authorization": f"Bearer {_api_key()}", "X-Title": "decision-model-audit"},
                timeout=timeout,
            )
            latency = time.perf_counter() - t0
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}: {r.text[:300]}"
                time.sleep(2 ** attempt)
                continue
            if r.status_code >= 400:
                return {"error": {"status": r.status_code, "body": r.text[:2000]}}, latency
            return r.json(), latency
        except httpx.HTTPError as e:
            last = repr(e)
            time.sleep(2 ** attempt)
    return {"error": {"status": None, "body": last}}, float("nan")


def cached_call(path: str, body: dict) -> dict:
    """Return {'response', 'latency_s', 'cached'}; errors are cached too only if deterministic (4xx)."""
    key = hashlib.sha256((path + json.dumps(body, sort_keys=True)).encode()).hexdigest()
    f = CACHE / key[:2] / f"{key}.json"
    if f.exists():
        rec = json.loads(f.read_text())
        rec["cached"] = True
        return rec
    resp, latency = _post(path, body)
    rec = {"path": path, "request": body, "response": resp, "latency_s": latency, "ts": time.time()}
    err = resp.get("error") if isinstance(resp, dict) else None
    transient = err and (err.get("status") is None or err.get("status", 0) >= 500)
    if not transient:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(rec))
    rec["cached"] = False
    return rec


def decisions(model: str, state, questions: dict) -> dict:
    return cached_call("/alpha/decisions", {"model": model, "state": state, "questions": questions})


def chat(body: dict) -> dict:
    return cached_call("/v1/chat/completions", body)


def credits() -> dict:
    r = httpx.get(f"{BASE}/v1/credits", headers={"Authorization": f"Bearer {_api_key()}"}, timeout=30)
    d = r.json()["data"]
    return {"total": d["total_credits"], "used": d["total_usage"], "remaining": d["total_credits"] - d["total_usage"]}

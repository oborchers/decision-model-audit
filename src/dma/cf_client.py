"""Cloudflare Workers AI access for the Clef decision models, with the same on-disk cache as client.py.

Token and account come from CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID or from a dotenv file named by DMA_CF_KEY_FILE.
Responses are normalised to the Jev shape: {"answers", "usage": {"input_tokens", "cost"}}.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

import httpx
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".cache" / "cloudflare"
PRICE_PER_M_INPUT = {"clef": 0.24, "clef-flash": 0.09}  # USD, Workers AI catalogue 2026-10-02

LEDGER = ROOT / "results" / "raw" / "clef" / "spend.jsonl"
BUDGET_TOTAL = float(os.environ.get("DMA_CF_BUDGET", "5.0"))  # USD, hard cap for the whole Clef programme
BLOCK = os.environ.get("DMA_CF_BLOCK", "unlabelled")
BLOCK_BUDGET = float(os.environ.get("DMA_CF_BLOCK_BUDGET", "inf"))  # USD, cap for the current block


class BudgetExceeded(RuntimeError):
    pass


_spend: dict | None = None
_spend_lock = threading.Lock()


def _ledger() -> dict:
    global _spend
    if _spend is None:
        _spend = {"total": 0.0, "block": 0.0}
        if LEDGER.exists():
            for l in LEDGER.read_text().splitlines():
                if l.strip():
                    e = json.loads(l)
                    _spend["total"] += e["cost"]
                    _spend["block"] += e["cost"] if e["block"] == BLOCK else 0.0
    return _spend


def _check_budget() -> None:
    with _spend_lock:
        s = _ledger()
        if s["total"] >= BUDGET_TOTAL:
            raise BudgetExceeded(f"total spend {s['total']:.4f} USD reached cap {BUDGET_TOTAL} USD")
        if s["block"] >= BLOCK_BUDGET:
            raise BudgetExceeded(f"block {BLOCK} spend {s['block']:.4f} USD reached cap {BLOCK_BUDGET} USD")


def _book(model: str, tokens: int, cost: float) -> None:
    with _spend_lock:
        s = _ledger()
        s["total"] += cost
        s["block"] += cost
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with LEDGER.open("a") as f:
            f.write(json.dumps({"ts": time.time(), "block": BLOCK, "model": model, "input_tokens": tokens, "cost": cost}) + "\n")


def spend() -> dict:
    with _spend_lock:
        return dict(_ledger())


_key: str | None = None
_lock = threading.Lock()
_client = httpx.Client(timeout=180.0)


def _setting(name: str) -> str | None:
    f = os.environ.get("DMA_CF_KEY_FILE")
    return os.environ.get(name) or (dotenv_values(f).get(name) if f else None)


def _base() -> str:
    account = _setting("CLOUDFLARE_ACCOUNT_ID")
    if not account:
        raise RuntimeError("Set CLOUDFLARE_ACCOUNT_ID, or put it into the DMA_CF_KEY_FILE dotenv file.")
    return f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/@cf/cloudflare"


def _api_key() -> str:
    global _key
    with _lock:
        if _key is None:
            _key = _setting("CLOUDFLARE_API_TOKEN")
            if not _key:
                raise RuntimeError("Set CLOUDFLARE_API_TOKEN, or DMA_CF_KEY_FILE to a dotenv file that contains it.")
    return _key


def _post(model: str, body: dict, retries: int = 5) -> tuple[dict, float]:
    last = None
    for attempt in range(retries):
        _check_budget()
        t0 = time.perf_counter()
        try:
            r = _client.post(f"{_base()}/{model}", json=body, headers={"Authorization": f"Bearer {_api_key()}"})
            latency = time.perf_counter() - t0
            if r.status_code in (403, 429) and any(c in r.text for c in ('"code":4006', '"code":5035')):
                raise BudgetExceeded(f"Workers AI plan quota: {r.text[:200]}")  # never retry a quota or plan refusal
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}: {r.text[:300]}"
                time.sleep(2 ** attempt)
                continue
            d = r.json()
            if r.status_code >= 400 or not d.get("success"):
                return {"error": {"status": r.status_code, "body": json.dumps(d.get("errors") or d)[:2000]}}, latency
            res = d["result"]
            tok = (res.get("usage") or {}).get("input_tokens", 0)
            res.setdefault("usage", {})["cost"] = tok * PRICE_PER_M_INPUT[model] / 1e6
            _book(model, tok, res["usage"]["cost"])
            return res, latency
        except (httpx.HTTPError, ValueError) as e:
            last = repr(e)
            time.sleep(2 ** attempt)
    return {"error": {"status": None, "body": last}}, float("nan")


def decisions(model: str, state, questions: dict) -> dict:
    """Return {'response', 'latency_s', 'cached'} like client.decisions."""
    body = {"model": model, "state": state, "questions": questions}
    key = hashlib.sha256((model + json.dumps(body, sort_keys=False)).encode()).hexdigest()
    f = CACHE / key[:2] / f"{key}.json"
    if f.exists() and not os.environ.get("DMA_NO_CACHE"):
        rec = json.loads(f.read_text())
        rec["cached"] = True
        return rec
    resp, latency = _post(model, body)
    rec = {"model": model, "request": body, "response": resp, "latency_s": latency, "ts": time.time()}
    err = resp.get("error")
    transient = err and (err.get("status") is None or err.get("status", 0) >= 500)
    if not transient and not (os.environ.get("DMA_NO_CACHE") and f.exists()):
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(rec))
    rec["cached"] = False
    return rec

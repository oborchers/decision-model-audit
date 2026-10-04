"""Local POST /v1/systemone shim for APUS-OpenJev MLX checkpoints (part 2, 2026-10).

APUS uses its own request contract (primitive, state, instructions, criteria as a list, 2 to 16 candidates,
labels A to P, at most 8,192 prompt tokens, no truncation). This shim translates the Jev body used for every
other decision model, one question at a time, with the vendor's own prompt renderer and scoring
(`openjev_contracts.py`, `examples/openjev_mlx.py` in the checkpoint):
    choice -> primitive choice over the criteria
    noul   -> primitive noul with the canonical yes/no candidates; noul = p(yes)
    score  -> primitive choice over the level descriptions (the vendor's score_level is not an ordinal API)
A JSON state is serialised with json.dumps. Inputs over 8,192 tokens return HTTP 400 and count as invalid.
Requests are processed one at a time (Metal is not thread-safe).

uv run python -m dma.local_servers.apus --model <checkpoint dir> --port 8404 --name apus-4b
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LOCK = threading.Lock()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8404)
    ap.add_argument("--name", default="apus")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.model) / "examples"))
    sys.path.insert(0, a.model)
    from openjev_contracts import BINARY_CRITERIA  # noqa: E402
    from openjev_mlx import OpenJevMLX  # noqa: E402
    engine = OpenJevMLX(a.model)

    def answer(state: str, qid: str, q: dict) -> tuple[dict, int]:
        rec = {"id": qid, "group_id": qid, "state": state, "instructions": q.get("instructions") or qid}
        if q["type"] == "noul":
            rec.update(primitive="noul", criteria=BINARY_CRITERIA)
        elif q["type"] == "score":
            rec.update(primitive="choice", criteria=[{"id": str(i), "description": d} for i, d in enumerate(q["criteria"])])
        else:
            rec.update(primitive="choice", criteria=[{"id": k, "description": v} for k, v in q["criteria"].items()])
        r = engine.decide(rec)
        p = r["probabilities"]
        if q["type"] == "noul":
            return {"type": "noul", "noul": p["yes"]}, r["prompt_tokens"]
        best = max(p, key=p.get)
        out = {"type": q["type"], "probabilities": p, "confidence": p[best]}
        if q["type"] == "score":
            out["score"] = sum(int(k) * v for k, v in p.items())
        else:
            out["choice"] = best
        return out, r["prompt_tokens"]

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state = body["state"] if isinstance(body["state"], str) else json.dumps(body["state"])
            try:
                with LOCK:
                    answers, tokens = {}, 0
                    for qid, q in body["questions"].items():
                        answers[qid], t = answer(state, qid, q)
                        tokens += t
                code, out = 200, {"model": a.name, "answers": answers, "usage": {"input_tokens": tokens, "cost": 0.0}}
            except ValueError as e:
                code, out = 400, {"error": str(e)}
            data = json.dumps(out).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    print(f"serving {a.model} as {a.name} on 127.0.0.1:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()

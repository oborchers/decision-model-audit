"""Local POST /v1/systemone shim for Hugging Face decision models that ship a `system_one` method
(Decision 2.0 of vLLM Semantic Router, part 2, 2026-10). The model's own method builds the prompt and scores the
options; this shim only exposes it over HTTP with the Jev request body. One request at a time (MPS).

uv run python -m dma.local_servers.hf_system_one --model vllm-sr/Decision-2.0-Nox-4B --revision <sha> --port 8406
"""
from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOCK = threading.Lock()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--port", type=int, default=8406)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    import torch
    from transformers import AutoModel
    model = AutoModel.from_pretrained(a.model, revision=a.revision, trust_remote_code=True)  # numerics fixed by the vendor runtime
    if a.device != "auto":
        model.to(a.device)
    model.eval()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            try:
                with LOCK, torch.inference_mode():
                    res = model.system_one(state=body["state"], questions=body["questions"])
                res = dict(res)
                res.setdefault("usage", {}).setdefault("cost", 0.0)
                code, out = 200, res
            except (ValueError, RuntimeError) as e:
                code, out = 400, {"error": str(e)[:500]}
            data = json.dumps(out, default=float).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    print(f"serving {a.model}@{a.revision} on 127.0.0.1:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()

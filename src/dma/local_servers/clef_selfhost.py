"""Local POST /v1/systemone server for Cloudflare Clef and Clef-flash with open weights (part 2, self-hosted check,
2026-10-04). Uses the vendor's own `load_release_model` and `systemone` from the model repository, so prompt
construction, truncation and scoring are Cloudflare's; this file only exposes them over HTTP with the Jev body.
One request at a time. `--max-length` raises the vendor's default truncation limit; requests may carry `images_b64`
(base64-encoded images), which are decoded to PIL images and passed on as the vendor's `images` field.

uv run python -m dma.local_servers.clef_selfhost --model Cloudflare/clef --revision <sha> --port 8408
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOCK = threading.Lock()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--port", type=int, default=8408)
    ap.add_argument("--max-length", type=int, default=16384, help="vendor default 16384; the backbone supports far more")
    a = ap.parse_args()
    import torch
    from huggingface_hub import snapshot_download
    path = snapshot_download(a.model, revision=a.revision)
    sys.path.insert(0, path)
    from joint_schema_model import load_release_model, systemone
    model, processor = load_release_model(path, device="cuda")
    model.eval()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            try:
                if "images_b64" in body:
                    import base64, io
                    from PIL import Image
                    body["images"] = [Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB") for b in body.pop("images_b64")]
                with LOCK, torch.inference_mode():
                    res = dict(systemone(model, processor, body, max_length=a.max_length))
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

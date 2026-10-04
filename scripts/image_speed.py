"""Exploratory image check for self-hosted Clef (part 2, protocol 2026-10-04): how fast and how accurate is one
image classification request? 50 photos from the Imagenette validation split (5 per class, seed 20261004), one
choice question over the ten class names, sent one at a time to the local server. Not part of any comparison.

uv run python scripts/image_speed.py --system clef-selfhost --port 8408
"""
import argparse, base64, io, json, random, statistics, time
from pathlib import Path
import httpx
import pandas as pd
from huggingface_hub import hf_hub_download

NAMES = ["tench", "English springer", "cassette player", "chain saw", "church", "French horn", "garbage truck",
         "gas pump", "golf ball", "parachute"]
ap = argparse.ArgumentParser()
ap.add_argument("--system", required=True)
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--resize", default=None, help="WxH, e.g. 320x240 (ViZDoom default screen size, check for a later game experiment)")
ap.add_argument("--out", default="results/raw/extra/images.selfhost.jsonl")
a = ap.parse_args()
f = hf_hub_download("Multimodal-Fatima/Imagenette_validation", "data/validation-00000-of-00001-b518380f1876c09a.parquet",
                    repo_type="dataset")
df = pd.read_parquet(f, columns=["image", "label", "id"])
rng = random.Random(20261004)
pick = []
for lab in range(10):
    ids = list(df.index[df["label"] == lab])
    rng.shuffle(ids)
    pick += ids[:5]
criteria = {n: f"The photo mainly shows a {n}." for n in NAMES}
rows = []
client = httpx.Client(timeout=300)
for i in pick:
    img = df.at[i, "image"]["bytes"]
    from PIL import Image
    if a.resize:
        rw, rh = map(int, a.resize.split("x"))
        buf = io.BytesIO()
        Image.open(io.BytesIO(img)).convert("RGB").resize((rw, rh), Image.BILINEAR).save(buf, format="PNG")
        img = buf.getvalue()
    w, h = Image.open(io.BytesIO(img)).size
    body = {"model": a.system, "state": "Classify the attached photo.", "images_b64": [base64.b64encode(img).decode()],
            "questions": {"q": {"type": "choice", "instructions": "What does the photo mainly show?", "criteria": criteria}}}
    t0 = time.time()
    r = client.post(f"http://127.0.0.1:{a.port}/v1/systemone", json=body)
    dt = time.time() - t0
    ans = r.json().get("answers", {}).get("q", {}) if r.status_code == 200 else {}
    rows.append({"system": a.system, "resize": a.resize or "native", "item_id": f"imagenette-{int(df.at[i, 'id'])}", "gold": NAMES[int(df.at[i, 'label'])],
                 "pred": ans.get("choice"), "confidence": ans.get("confidence"), "valid": r.status_code == 200,
                 "latency_s": dt, "width": w, "height": h, "usage": r.json().get("usage") if r.status_code == 200 else None})
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with open(a.out, "a") as fh:
    for row in rows:
        fh.write(json.dumps(row) + "\n")
lat = sorted(r["latency_s"] for r in rows)
print(json.dumps({"system": a.system, "resize": a.resize or "native", "n": len(rows), "valid": sum(r["valid"] for r in rows),
                  "accuracy": round(sum(r["pred"] == r["gold"] for r in rows) / len(rows), 3),
                  "latency_p50_s": round(statistics.median(lat), 3), "latency_p95_s": round(lat[int(0.95 * len(lat)) - 1], 3),
                  "median_size": f"{int(statistics.median(r['width'] for r in rows))}x{int(statistics.median(r['height'] for r in rows))}"}), flush=True)

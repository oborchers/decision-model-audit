#!/usr/bin/env bash
# Second self-hosted Clef run (part 2, protocol 2026-10-04): long input with the truncation limit raised to 65,536,
# S1 with the "none" option, and an exploratory image speed check. Ran on Runpod with UV_NO_SYNC=1, torch 2.11.0+cu128.
cd "$(dirname "$0")/.."
export UV_NO_SYNC=1 DMA_WORKERS=1
run() { uv run python -m dma.runners.api "$@" --workers 1; }
T1=data/s1_arxiv/task.json; P=data/probes
for spec in "Cloudflare/clef 2f3de3dd85f379784083b0814d997ab627200f0c 8408 clef-selfhost" "Cloudflare/clef-flash 17f0b0ad64efb65d273590632833508766b2aae6 8409 clef-flash-selfhost"; do
  set -- $spec; M=$1; R=$2; PORT=$3; S=$4
  uv run python -m dma.local_servers.clef_selfhost --model $M --revision $R --port $PORT --max-length 65536 > logs/selfhost2_$S.log 2>&1 &
  PID=$!
  for i in $(seq 1 240); do grep -q serving logs/selfhost2_$S.log && break; sleep 5; done
  run --system ${S}-64k --task $P/p3_long_input.task.json --items $P/p3_long_input.main.jsonl --variant noul --out results/raw/main/p3.selfhost.jsonl
  run --system $S --task $T1 --items data/s1_arxiv/main.jsonl --variant choice_none --out results/raw/main/s1.selfhost.jsonl
  uv run python scripts/image_speed.py --system $S --port $PORT
  uv run python scripts/image_speed.py --system $S --port $PORT --resize 320x240
  kill $PID; sleep 10
  echo "$S DONE"
done
echo SELFHOST2 DONE

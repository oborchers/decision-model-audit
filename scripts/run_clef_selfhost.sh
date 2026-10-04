#!/usr/bin/env bash
# Clef and Clef-flash with open weights on one rented GPU (part 2, protocol 2026-10-04): checks whether the silent
# truncation seen through Workers AI belongs to the hosted path. Rows go to *.selfhost.jsonl, separate from part 1.
# Ran on Runpod with UV_NO_SYNC=1 and torch 2.11.0+cu128.
cd "$(dirname "$0")/.."
export UV_NO_SYNC=1 DMA_WORKERS=1
run() { uv run python -m dma.runners.api "$@" --workers 1; }
T1=data/s1_arxiv/task.json; P=data/probes; G=results/raw/gaps
for spec in "Cloudflare/clef 2f3de3dd85f379784083b0814d997ab627200f0c 8408 clef-selfhost" "Cloudflare/clef-flash 17f0b0ad64efb65d273590632833508766b2aae6 8409 clef-flash-selfhost"; do
  set -- $spec; M=$1; R=$2; PORT=$3; S=$4
  uv run python -m dma.local_servers.clef_selfhost --model $M --revision $R --port $PORT > logs/selfhost_$S.log 2>&1 &
  PID=$!
  for i in $(seq 1 240); do grep -q serving logs/selfhost_$S.log && break; sleep 5; done
  run --system $S --task $T1 --items data/s1_arxiv/pilot.jsonl --variant choice --out results/raw/pilot/s1.selfhost.jsonl
  run --system $S --task $T1 --items data/s1_arxiv/main.jsonl --variant choice --out results/raw/main/s1.selfhost.jsonl
  run --system $S --task $P/p3_long_input.task.json --items $P/p3_long_input.main.jsonl --variant noul --out results/raw/main/p3.selfhost.jsonl
  run --system $S --task $T1 --items data/s1_arxiv/main_shuffled.jsonl --variant batch10 --out $G/s1_batch.selfhost.jsonl
  DMA_NO_CACHE=1 run --system $S --task $T1 --items data/s1_arxiv/latency30.jsonl --variant choice --out results/raw/latency/api_c1.selfhost.jsonl
  kill $PID; sleep 10
  echo "$S DONE"
done
echo SELFHOST DONE

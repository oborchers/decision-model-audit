#!/usr/bin/env bash
# Isolated latency measurement (protocol changelog): one process at a time, nothing else loaded.
set -uo pipefail
cd "$(dirname "$0")/.."
LT=results/raw/latency
mkdir -p $LT
S1="--task data/s1_arxiv/task.json --items data/s1_arxiv/main.jsonl"
for s in laya gliclass nli gliner gliner-1b qwen-lp; do
  uv run python -m dma.runners.local --system $s $S1 --variant choice --limit 50 --out $LT/local.jsonl 2>&1 | grep -E '^\{"system"|Error'
done
uv run python -m dma.runners.local --system tfidf $S1 --variant choice --limit 50 --train data/s1_arxiv/train_2024.jsonl --out $LT/local.jsonl 2>&1 | grep -E '^\{"system"|Error'
for s in eikos-4b semif-4b kev-0.8b; do
  uv run python -m dma.runners.local_extra --system $s $S1 --variant choice --limit 50 --out $LT/local.jsonl 2>&1 | grep -E '^\{"system"|Error'
done
for s in jev luna flash haiku sonnet; do
  DMA_NO_CACHE=1 uv run python -m dma.runners.api --system $s --task data/s1_arxiv/task.json --items data/s1_arxiv/latency30.jsonl --variant choice --out $LT/api_c1.jsonl --workers 1
done
uv run python -c "from dma.client import credits; print(credits())"
echo "LATENCY DONE $(date)"

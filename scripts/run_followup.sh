#!/usr/bin/env bash
# Runs after the main local lanes: additional (post hoc) decision models, then isolated latency.
set -uo pipefail
cd "$(dirname "$0")/.."
R=results/raw/main
while pgrep -f run_main_local.sh >/dev/null; do sleep 60; done
echo "MAIN LANES FINISHED $(date)"
X() { uv run python -m dma.runners.local_extra "$@" 2>&1 | grep -E '^\{"system"|Traceback|Error' ; }
S1="--task data/s1_arxiv/task.json --items data/s1_arxiv/main.jsonl"
S2="--task data/s2_fedreg/task.json --items data/s2_fedreg/main.jsonl"
P1="--task data/probes/p1_calibration.task.json --items data/probes/p1_calibration.main.jsonl"
P2="--task data/probes/p2_minimal_pairs.task.json --items data/probes/p2_minimal_pairs.main.jsonl"
P3="--task data/probes/p3_long_input.task.json --items data/probes/p3_long_input.main.jsonl"
P4="--task data/probes/p4_no_fit.task.json --items data/probes/p4_no_fit.main.jsonl"
for s in eikos-4b semif-4b kev-0.8b kev-4b; do
  for v in choice choice_none reversed para0 para1 yesno; do X --system $s $S1 --variant $v --out $R/s1.extra.jsonl; done
  for v in choice choice_none yesno; do X --system $s $S2 --variant $v --out $R/s2.extra.jsonl; done
  X --system $s $P1 --variant noul --out $R/p1.extra.jsonl
  X --system $s $P2 --variant choice --out $R/p2.extra.jsonl
  for v in choice choice_none; do X --system $s $P4 --variant $v --out $R/p4.extra.jsonl; done
  X --system $s $P3 --variant noul --out $R/p3.extra.jsonl
  echo "EXTRA $s DONE $(date)"
done
# Isolated latency: one process at a time, nothing else loaded
LT=results/raw/latency
mkdir -p $LT
for s in laya gliclass nli gliner gliner-1b qwen-lp; do
  uv run python -m dma.runners.local --system $s $S1 --variant choice --limit 50 --out $LT/local.jsonl 2>&1 | grep -E '^\{"system"|Error'
done
uv run python -m dma.runners.local --system tfidf $S1 --variant choice --limit 50 --train data/s1_arxiv/train_2024.jsonl --out $LT/local.jsonl 2>&1 | grep -E '^\{"system"|Error'
for s in eikos-4b semif-4b kev-0.8b kev-4b; do X --system $s $S1 --variant choice --limit 50 --out $LT/local.jsonl; done
for s in jev luna flash haiku sonnet; do
  DMA_NO_CACHE=1 uv run python -m dma.runners.api --system $s --task data/s1_arxiv/task.json --items data/s1_arxiv/latency30.jsonl --variant choice --out $LT/api_c1.jsonl --workers 1
done
uv run python -c "from dma.client import credits; print(credits())"
echo "FOLLOWUP DONE $(date)"

#!/usr/bin/env bash
# Main evaluation run for local systems. Two lanes run in parallel:
#   lane qwen:  qwen-lp on everything (MLX)
#   lane torch: all torch/sklearn systems (MPS; gliner family on CPU for P3 long inputs)
# Usage: scripts/run_main_local.sh qwen|torch
set -uo pipefail
cd "$(dirname "$0")/.."
R=results/raw/main
mkdir -p $R
L() { uv run python -m dma.runners.local "$@" 2>&1 | grep -E '^\{"system"|Traceback|Error' ; }
S1="--task data/s1_arxiv/task.json --items data/s1_arxiv/main.jsonl"
S2="--task data/s2_fedreg/task.json --items data/s2_fedreg/main.jsonl"
P1="--task data/probes/p1_calibration.task.json --items data/probes/p1_calibration.main.jsonl"
P2="--task data/probes/p2_minimal_pairs.task.json --items data/probes/p2_minimal_pairs.main.jsonl"
P3="--task data/probes/p3_long_input.task.json --items data/probes/p3_long_input.main.jsonl"
P4="--task data/probes/p4_no_fit.task.json --items data/probes/p4_no_fit.main.jsonl"

common() {  # $1 = system, rest = extra args
  local s=$1; shift
  for v in choice choice_none reversed para0 para1 yesno; do L --system $s $S1 --variant $v --out $R/s1.local.jsonl "$@"; done
  for v in choice choice_none yesno; do L --system $s $S2 --variant $v --out $R/s2.local.jsonl "$@"; done
  L --system $s $P1 --variant noul --out $R/p1.local.jsonl "$@"
  L --system $s $P2 --variant choice --out $R/p2.local.jsonl "$@"
  for v in choice choice_none; do L --system $s $P4 --variant $v --out $R/p4.local.jsonl "$@"; done
}

case "${1:-}" in
  qwen)
    common qwen-lp
    L --system qwen-lp $P3 --variant noul --out $R/p3.local.jsonl
    ;;
  torch)
    for s in laya gliclass nli gliner gliner-1b; do common $s; done
    for v in choice choice_none reversed para0 para1; do
      L --system tfidf $S1 --variant $v --train data/s1_arxiv/train_2024.jsonl --out $R/s1.local.jsonl; done
    L --system tfidf --task data/probes/p4_no_fit.task.json --items data/probes/p4_no_fit.main.jsonl --variant choice --train data/s1_arxiv/train_2024.jsonl --out $R/p4.local.jsonl
    for s in laya laya-long nli gliclass; do L --system $s $P3 --variant noul --out $R/p3.local.jsonl; done
    for s in gliner gliner-1b gliner-long; do L --system $s $P3 --variant noul --out $R/p3.local.jsonl --device cpu; done
    ;;
  *) echo "usage: $0 qwen|torch"; exit 1;;
esac
echo "LANE $1 DONE"

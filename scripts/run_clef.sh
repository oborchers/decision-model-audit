#!/usr/bin/env bash
# Post hoc Clef / Clef-flash programme (protocol changelog 2026-10-02). Same calls as Jev in run_main_api.sh,
# run_followup.sh and the gap runs. Needs CLOUDFLARE_API_TOKEN or DMA_CF_KEY_FILE.
# Usage: scripts/run_clef.sh <block> [model]   block = pilot | main | gaps | latency
# Spend caps: DMA_CF_BUDGET (total, default 5 USD), DMA_CF_BLOCK_BUDGET (this block).
set -euo pipefail
cd "$(dirname "$0")/.."
BLOCK=$1; MODELS=${2:-"clef-flash clef"}
export DMA_CF_BLOCK="$BLOCK${2:+-$2}"
run() { uv run python -m dma.runners.api "$@"; }
T1=data/s1_arxiv/task.json; T2=data/s2_fedreg/task.json; P=data/probes
case $BLOCK in
  pilot|main)
    R=results/raw/$BLOCK; mkdir -p $R; D=$BLOCK
    for m in $MODELS; do
      for v in choice choice_none reversed para0 para1 yesno; do run --system $m --task $T1 --items data/s1_arxiv/$D.jsonl --variant $v --out $R/s1.clef.jsonl; done
      for v in choice choice_none yesno; do run --system $m --task $T2 --items data/s2_fedreg/$D.jsonl --variant $v --out $R/s2.clef.jsonl; done
      run --system $m --task $P/p1_calibration.task.json --items $P/p1_calibration.$D.jsonl --variant noul --out $R/p1.clef.jsonl
      run --system $m --task $P/p2_minimal_pairs.task.json --items $P/p2_minimal_pairs.$D.jsonl --variant choice --out $R/p2.clef.jsonl
      for v in choice choice_none; do run --system $m --task $P/p4_no_fit.task.json --items $P/p4_no_fit.$D.jsonl --variant $v --out $R/p4.clef.jsonl; done
      run --system $m --task $P/p3_long_input.task.json --items $P/p3_long_input.$D.jsonl --variant noul --out $R/p3.clef.jsonl --workers 4
    done ;;
  gaps)
    G=results/raw/gaps
    for m in $MODELS; do
      DMA_NO_CACHE=1 run --system $m --task $T1 --items data/s1_arxiv/repeat100.jsonl --variant choice --out $G/s1_repeat.clef.jsonl
      run --system $m --task $T1 --items data/s1_arxiv/main_shuffled.jsonl --variant batch10 --out $G/s1_batch.clef.jsonl
      DMA_NO_CACHE=1 run --system $m --task $T1 --items data/s1_arxiv/shuffled100.jsonl --variant batch10 --out $G/s1_batch_latency.clef.jsonl --workers 1
    done ;;
  latency)
    for m in $MODELS; do
      DMA_NO_CACHE=1 run --system $m --task $T1 --items data/s1_arxiv/latency30.jsonl --variant choice --out results/raw/latency/api_c1.clef.jsonl --workers 1
    done ;;
esac
uv run python -c "from dma.cf_client import spend; print('SPEND', spend())"

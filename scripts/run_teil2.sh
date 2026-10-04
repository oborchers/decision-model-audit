#!/usr/bin/env bash
# Part 2 (protocol changelog 2026-10-02): new decision models and probes P5 to P7.
# Usage: scripts/run_teil2.sh <block> [systems]
#   block = pilot | main | gaps | latency | probes-pilot | probes | fwf-pilot | fwf
# Needs OPENROUTER_API_KEY or DMA_KEY_FILE; for clef systems also DMA_CF_KEY_FILE.
# Spend caps: DMA_OR_BUDGET (default 4 USD), DMA_OR_BLOCK_BUDGET; ledger is active because DMA_OR_BLOCK is set.
set -euo pipefail
cd "$(dirname "$0")/.."
BLOCK=$1; SYSTEMS=${2:-"d1 solar tev1 kev-4b-api"}
export DMA_OR_BLOCK="teil2-$BLOCK-${SYSTEMS// /_}" DMA_CF_BLOCK="teil2-$BLOCK"
run() { uv run python -m dma.runners.api "$@" ${DMA_WORKERS:+--workers $DMA_WORKERS}; }  # DMA_WORKERS=1 for local servers (MPS is not thread-safe)
T1=data/s1_arxiv/task.json; T2=data/s2_fedreg/task.json; P=data/probes
case $BLOCK in
  pilot|main)
    R=results/raw/$BLOCK; mkdir -p $R; D=$BLOCK
    for s in $SYSTEMS; do
      if [ "$s" = mercury ]; then  # rate-limited free tier: reduced programme
        for v in choice choice_none; do run --system $s --task $T1 --items data/s1_arxiv/$D.jsonl --variant $v --out $R/s1.teil2.jsonl --workers 2; done
        for v in choice choice_none; do run --system $s --task $P/p4_no_fit.task.json --items $P/p4_no_fit.$D.jsonl --variant $v --out $R/p4.teil2.jsonl --workers 2; done
        continue
      fi
      for v in choice choice_none reversed para0 para1 yesno; do run --system $s --task $T1 --items data/s1_arxiv/$D.jsonl --variant $v --out $R/s1.teil2.jsonl; done
      for v in choice choice_none yesno; do run --system $s --task $T2 --items data/s2_fedreg/$D.jsonl --variant $v --out $R/s2.teil2.jsonl; done
      run --system $s --task $P/p1_calibration.task.json --items $P/p1_calibration.$D.jsonl --variant noul --out $R/p1.teil2.jsonl
      run --system $s --task $P/p2_minimal_pairs.task.json --items $P/p2_minimal_pairs.$D.jsonl --variant choice --out $R/p2.teil2.jsonl
      for v in choice choice_none; do run --system $s --task $P/p4_no_fit.task.json --items $P/p4_no_fit.$D.jsonl --variant $v --out $R/p4.teil2.jsonl; done
      run --system $s --task $P/p3_long_input.task.json --items $P/p3_long_input.$D.jsonl --variant noul --out $R/p3.teil2.jsonl --workers 4
    done ;;
  probes-pilot|probes)  # P5 to P7
    D=$([ "$BLOCK" = probes ] && echo main || echo pilot); R=results/raw/$D; mkdir -p $R
    for s in $SYSTEMS; do
      W=$([ "$s" = mercury ] && echo 2 || echo 8)
      run --system $s --task $P/p5_titles.task.json --items $P/p5_titles.$D.jsonl --variant choice --out $R/p5.teil2.jsonl --workers $W
      run --system $s --task $P/p6_claims.task.json --items $P/p6_claims.$D.jsonl --variant noul --out $R/p6.teil2.jsonl --workers $W
      run --system $s --task $P/p7_count.task.json --items $P/p7_count.$D.jsonl --variant score --out $R/p7.teil2.jsonl --workers $W
    done ;;
  fwf-pilot|fwf)
    D=$([ "$BLOCK" = fwf ] && echo main || echo pilot); R=results/raw/$D; mkdir -p $R; F=data/fwf
    for s in $SYSTEMS; do
      run --system $s --task $F/task_en.json --items $F/${D}_en.jsonl --variant choice --out $R/fwf_en.teil2.jsonl
      run --system $s --task $F/task_en.json --items $F/${D}_de.jsonl --variant choice --out $R/fwf_de-en.teil2.jsonl
      run --system $s --task $F/task_de.json --items $F/${D}_de.jsonl --variant choice --out $R/fwf_de.teil2.jsonl
    done ;;
  gaps)
    G=results/raw/gaps
    for s in $SYSTEMS; do
      DMA_NO_CACHE=1 run --system $s --task $T1 --items data/s1_arxiv/repeat100.jsonl --variant choice --out $G/s1_repeat.teil2.jsonl
      run --system $s --task $T1 --items data/s1_arxiv/main_shuffled.jsonl --variant batch10 --out $G/s1_batch.teil2.jsonl
      DMA_NO_CACHE=1 run --system $s --task $T1 --items data/s1_arxiv/shuffled100.jsonl --variant batch10 --out $G/s1_batch_latency.teil2.jsonl --workers 1
    done ;;
  latency)
    for s in $SYSTEMS; do
      DMA_NO_CACHE=1 run --system $s --task $T1 --items data/s1_arxiv/latency30.jsonl --variant choice --out results/raw/latency/api_c1.teil2.jsonl --workers 1
    done ;;
esac
uv run python -c "from dma.client import spend, credits; print('OR SPEND', spend(), 'BALANCE', credits())"

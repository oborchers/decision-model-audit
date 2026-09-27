#!/usr/bin/env bash
# Main evaluation run for API systems (protocol v1 + changelog). Idempotent via request cache.
set -euo pipefail
cd "$(dirname "$0")/.."
R=results/raw/main
mkdir -p $R
run() { uv run python -m dma.runners.api "$@"; }
S1="--task data/s1_arxiv/task.json --items data/s1_arxiv/main.jsonl"
S2="--task data/s2_fedreg/task.json --items data/s2_fedreg/main.jsonl"
P1="--task data/probes/p1_calibration.task.json --items data/probes/p1_calibration.main.jsonl"
P2="--task data/probes/p2_minimal_pairs.task.json --items data/probes/p2_minimal_pairs.main.jsonl"
P3="--task data/probes/p3_long_input.task.json --items data/probes/p3_long_input.main.jsonl"
P4="--task data/probes/p4_no_fit.task.json --items data/probes/p4_no_fit.main.jsonl"
# Jev: all variants
for v in choice choice_none reversed para0 para1 yesno; do run --system jev $S1 --variant $v --out $R/s1.jsonl; done
for v in choice choice_none yesno; do run --system jev $S2 --variant $v --out $R/s2.jsonl; done
# LLMs
for s in luna flash haiku; do
  run --system $s $S1 --variant choice --out $R/s1.jsonl
  run --system $s $S1 --variant rationale --out $R/s1.jsonl
  run --system $s $S2 --variant choice --out $R/s2.jsonl
done
run --system sonnet $S1 --variant choice --out $R/s1.jsonl --workers 4
# Probes
for s in jev luna flash haiku; do
  run --system $s $P1 --variant noul --out $R/p1.jsonl
  run --system $s $P2 --variant choice --out $R/p2.jsonl
  for v in choice choice_none; do run --system $s $P4 --variant $v --out $R/p4.jsonl; done
done
for s in jev luna flash; do run --system $s $P3 --variant noul --out $R/p3.jsonl; done
run --system haiku $P3 --variant noul --out $R/p3.jsonl --max-words 6100
uv run python -c "from dma.client import credits; print(credits())"

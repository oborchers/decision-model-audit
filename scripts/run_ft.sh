#!/usr/bin/env bash
# Fine-tuning comparison on the pod (post hoc, part 2): Laya and plain ModernBERT-large, 400 and all 2,806 labels, 3 seeds.
# Ran on one NVIDIA A40 (Runpod) with UV_NO_SYNC=1 and torch 2.11.0+cu128; see notes/protocol.md, fine-tuning entries.
cd "$(dirname "$0")/.."
export UV_NO_SYNC=1
mkdir -p "${FTLOGS:-logs/ft}"
for n in 400 2806; do for s in 1 2 3; do
  .venv/bin/python -u -m dma.runners.encoder_finetune --labels $n --seed $s --device cuda --out results/raw/main/s1.ft.jsonl > ${FTLOGS:-logs/ft}/mbert_${n}_s$s.log 2>&1
  echo "mbert $n s$s exit $?: $(grep '^{"system' ${FTLOGS:-logs/ft}/mbert_${n}_s$s.log)"
  .venv/bin/python -u -m dma.runners.laya_finetune --labels $n --seed $s --epochs 3 --batch 8 --device cuda --out results/raw/main/s1.ft.jsonl > ${FTLOGS:-logs/ft}/laya_${n}_s$s.log 2>&1
  echo "laya $n s$s exit $?: $(grep '^{"system' ${FTLOGS:-logs/ft}/laya_${n}_s$s.log)"
done; done
echo FT DONE

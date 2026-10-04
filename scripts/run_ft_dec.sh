#!/usr/bin/env bash
# Same short recipe for both models with all labels (post hoc, protocol 2026-10-04).
# Ran on one NVIDIA A40 (Runpod) with UV_NO_SYNC=1 and torch 2.11.0+cu128; see notes/protocol.md, fine-tuning entries.
cd "$(dirname "$0")/.."
export UV_NO_SYNC=1
mkdir -p "${FTLOGS:-logs/ft}"
for s in 1 2 3; do
  .venv/bin/python -u -m dma.runners.encoder_finetune --labels 2806 --seed $s --epochs 3 --lr 3e-5 --tag=-dec-lr3e-05 --device cuda --out results/raw/main/s1.ft.jsonl > ${FTLOGS:-logs/ft}/dec_mbert_s$s.log 2>&1
  echo "dec mbert s$s exit $?: $(grep '^{"system' ${FTLOGS:-logs/ft}/dec_mbert_s$s.log)"
  .venv/bin/python -u -m dma.runners.laya_finetune --labels 2806 --seed $s --epochs 3 --batch 8 --lr 3e-5 --decay --tag=-dec-lr3e-05 --device cuda --out results/raw/main/s1.ft.jsonl > ${FTLOGS:-logs/ft}/dec_laya_s$s.log 2>&1
  echo "dec laya s$s exit $?: $(grep '^{"system' ${FTLOGS:-logs/ft}/dec_laya_s$s.log)"
done
echo DEC DONE

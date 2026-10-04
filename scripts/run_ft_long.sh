#!/usr/bin/env bash
# Longer fine-tuning with curves and a learning-rate grid (post hoc). One stream per model, both on the same GPU.
# Ran on one NVIDIA A40 (Runpod) with UV_NO_SYNC=1 and torch 2.11.0+cu128; see notes/protocol.md, fine-tuning entries.
# Usage: run_ft_long.sh mbert "1e-5 3e-5 8e-5"   |   run_ft_long.sh laya "3e-6 1e-5 3e-5"
cd "$(dirname "$0")/.."
export UV_NO_SYNC=1
M=$1; LRS=$2; mkdir -p "${FTLOGS:-logs/ft}"
for lr in $LRS; do for s in 1 2 3; do
  .venv/bin/python -u -m dma.runners.ft_curves --model $M --labels 400 --epochs 15 --lr $lr --seed $s --device cuda > ${FTLOGS:-logs/ft}/long_${M}_400_lr${lr}_s$s.log 2>&1
  echo "$M 400 lr$lr s$s exit $?: $(grep '^{"system' ${FTLOGS:-logs/ft}/long_${M}_400_lr${lr}_s$s.log)"
done; done
LR=$(.venv/bin/python scripts/pick_lr.py $M); echo "$M picked lr $LR"
for s in 1 2 3; do
  .venv/bin/python -u -m dma.runners.ft_curves --model $M --labels 2806 --epochs 8 --lr $LR --seed $s --device cuda > ${FTLOGS:-logs/ft}/long_${M}_2806_s$s.log 2>&1
  echo "$M 2806 lr$LR s$s exit $?: $(grep '^{"system' ${FTLOGS:-logs/ft}/long_${M}_2806_s$s.log)"
done
echo "LONG $M DONE"

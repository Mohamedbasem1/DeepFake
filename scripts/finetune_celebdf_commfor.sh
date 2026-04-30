#!/usr/bin/env bash
set -euo pipefail

python scripts/finetune_commfor.py \
  --train-root data/finetune/frames \
  --output models/commfor_finetuned.pt \
  --epochs "${EPOCHS:-2}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --freeze-backbone \
  --use-amp \
  --limit-per-class "${LIMIT_PER_CLASS:-7000}"


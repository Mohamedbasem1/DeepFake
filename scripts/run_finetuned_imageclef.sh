#!/usr/bin/env bash
set -euo pipefail

python scripts/run_inference.py \
  --config configs/image_detection_commfor_finetuned.yaml \
  --input data/raw/imageclef_detection/Data/Images_Detection \
  --output outputs/images_detection_submission_finetuned.csv

wc -l outputs/images_detection_submission_finetuned.csv


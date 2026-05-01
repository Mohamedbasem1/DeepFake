#!/usr/bin/env bash
set -euo pipefail

# Runs the model family from sky787770/DeepFake-Image-Detection on our data:
# VGG16, VGG19, InceptionV3, and ResNet50.
#
# Usage:
#   bash scripts/run_sky_classic_models.sh train
#   bash scripts/run_sky_classic_models.sh predict
#   bash scripts/run_sky_classic_models.sh merge
#   bash scripts/run_sky_classic_models.sh all

MODE="${1:-all}"
TRAIN_ROOT="${TRAIN_ROOT:-data/finetune/frames}"
TEST_ROOT="${TEST_ROOT:-data/raw/imageclef_detection/Data/Images_Detection}"
LIMIT_PER_CLASS="${LIMIT_PER_CLASS:-15000}"
EPOCHS="${EPOCHS:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"

train_one() {
  local name="$1"
  local image_size="$2"
  local batch_size="$3"
  local lr="$4"
  local output="models/sky_${name}.pt"

  python scripts/train_timm_image_classifier.py \
    --train-root "${TRAIN_ROOT}" \
    --output "${output}" \
    --model-name "${name}" \
    --image-size "${image_size}" \
    --epochs "${EPOCHS}" \
    --batch-size "${batch_size}" \
    --lr "${lr}" \
    --use-amp \
    --num-workers "${NUM_WORKERS}" \
    --limit-per-class "${LIMIT_PER_CLASS}"
}

predict_one() {
  local name="$1"
  local batch_size="$2"
  local checkpoint="models/sky_${name}.pt"
  local output="outputs/images_detection_scores_sky_${name}.csv"

  python scripts/predict_timm_image_classifier.py \
    --checkpoint "${checkpoint}" \
    --input "${TEST_ROOT}" \
    --output "${output}" \
    --batch-size "${batch_size}" \
    --num-workers "${NUM_WORKERS}" \
    --use-amp \
    --tta-hflip
}

train_all() {
  train_one "vgg16" 224 32 2e-5
  train_one "vgg19" 224 24 2e-5
  train_one "inception_v3" 299 32 2e-5
  train_one "resnet50" 224 64 3e-5
}

predict_all() {
  predict_one "vgg16" 64
  predict_one "vgg19" 48
  predict_one "inception_v3" 64
  predict_one "resnet50" 96
}

merge_all() {
  python scripts/merge_multiple_scores.py \
    --mode avg \
    --threshold 0.5 \
    --inputs \
      outputs/images_detection_scores_sky_vgg16.csv \
      outputs/images_detection_scores_sky_vgg19.csv \
      outputs/images_detection_scores_sky_inception_v3.csv \
      outputs/images_detection_scores_sky_resnet50.csv \
    --output outputs/images_detection_submission_sky_classic_avg.csv \
    --debug-output outputs/images_detection_sky_classic_avg_debug.csv

  python scripts/prepare_submission_upload.py \
    --input outputs/images_detection_submission_sky_classic_avg.csv \
    --output-csv outputs/upload/images_detection_submission_sky_classic_avg.csv \
    --output-zip outputs/upload/images_detection_submission_sky_classic_avg.zip
}

case "${MODE}" in
  train)
    train_all
    ;;
  predict)
    predict_all
    ;;
  merge)
    merge_all
    ;;
  all)
    train_all
    predict_all
    merge_all
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Use: train, predict, merge, or all" >&2
    exit 2
    ;;
esac

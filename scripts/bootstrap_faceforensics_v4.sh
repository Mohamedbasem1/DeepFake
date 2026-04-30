#!/usr/bin/env bash
set -euo pipefail

SCRIPT_URL="${SCRIPT_URL:-https://kaldir.vc.in.tum.de/faceforensics_download_v4.py}"
SCRIPT_PATH="${SCRIPT_PATH:-data/external/faceforensics_download_v4.py}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/external/FaceForensics_v4}"
DATASET_TYPE="${DATASET_TYPE:-compressed}"
FRAMES_PER_VIDEO="${FRAMES_PER_VIDEO:-8}"
SAMPLE_ONLY="${SAMPLE_ONLY:-0}"

python -m pip install -r requirements-lightning.txt
mkdir -p data/external data/finetune/frames "$OUTPUT_ROOT"

if [ ! -f "$SCRIPT_PATH" ]; then
  python scripts/download_file.py \
    --url "$SCRIPT_URL" \
    --output "$SCRIPT_PATH"
fi

sample_flag=()
if [ "$SAMPLE_ONLY" = "1" ]; then
  sample_flag=(--sample_only)
fi

echo "Downloading FaceForensics v4 dataset_type=$DATASET_TYPE"
printf '\n\n' | python "$SCRIPT_PATH" \
  "$OUTPUT_ROOT" \
  -d "$DATASET_TYPE" \
  --not_mask \
  "${sample_flag[@]}"

echo "Extracting frames from FaceForensics v4..."
if [ -d "$OUTPUT_ROOT/FaceForensics_${DATASET_TYPE}" ]; then
  if [ -d "$OUTPUT_ROOT/FaceForensics_${DATASET_TYPE}/train/original" ] || \
     [ -d "$OUTPUT_ROOT/FaceForensics_${DATASET_TYPE}/val/original" ] || \
     [ -d "$OUTPUT_ROOT/FaceForensics_${DATASET_TYPE}/test/original" ]; then
    python scripts/extract_video_frames.py \
      --input "$OUTPUT_ROOT/FaceForensics_${DATASET_TYPE}" \
      --output data/finetune/frames \
      --dataset-name FaceForensics_v4 \
      --label-from-parent \
      --frames-per-video "$FRAMES_PER_VIDEO"
  else
    echo "Downloaded folder exists but original/altered subfolders were not found."
  fi
else
  echo "Expected folder not found: $OUTPUT_ROOT/FaceForensics_${DATASET_TYPE}"
fi

echo "Frame counts:"
echo -n "real: "
find data/finetune/frames -path "*/real/*" -type f | wc -l
echo -n "fake: "
find data/finetune/frames -path "*/fake/*" -type f | wc -l


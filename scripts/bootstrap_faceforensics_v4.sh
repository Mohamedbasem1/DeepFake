#!/usr/bin/env bash
set -euo pipefail

SCRIPT_URL="${SCRIPT_URL:-https://kaldir.vc.in.tum.de/faceforensics_download_v4.py}"
SCRIPT_PATH="${SCRIPT_PATH:-data/external/faceforensics_download_v4.py}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/external/FaceForensics_v4}"
FF_COMPRESSION="${FF_COMPRESSION:-c23}"
FF_SERVER="${FF_SERVER:-EU2}"
FF_DATASETS="${FF_DATASETS:-original Deepfakes Face2Face FaceSwap NeuralTextures}"
FRAMES_PER_VIDEO="${FRAMES_PER_VIDEO:-8}"
NUM_VIDEOS="${NUM_VIDEOS:-}"

python -m pip install -r requirements-lightning.txt
mkdir -p data/external data/finetune/frames "$OUTPUT_ROOT"

if [ ! -f "$SCRIPT_PATH" ]; then
  python scripts/download_file.py \
    --url "$SCRIPT_URL" \
    --output "$SCRIPT_PATH"
fi

num_videos_args=()
if [ -n "$NUM_VIDEOS" ]; then
  num_videos_args=(-n "$NUM_VIDEOS")
fi

for dataset in $FF_DATASETS; do
  echo "Downloading FaceForensics++ dataset=$dataset compression=$FF_COMPRESSION server=$FF_SERVER"
  yes "" | python "$SCRIPT_PATH" \
    "$OUTPUT_ROOT" \
    -d "$dataset" \
    -c "$FF_COMPRESSION" \
    -t videos \
    --server "$FF_SERVER" \
    "${num_videos_args[@]}"
done

echo "Extracting frames from FaceForensics++..."
python scripts/extract_video_frames.py \
  --input "$OUTPUT_ROOT" \
  --output data/finetune/frames \
  --dataset-name FaceForensics_v4_${FF_COMPRESSION} \
  --label-from-parent \
  --frames-per-video "$FRAMES_PER_VIDEO"

echo "Frame counts:"
echo -n "real: "
find data/finetune/frames -path "*/real/*" -type f | wc -l
echo -n "fake: "
find data/finetune/frames -path "*/fake/*" -type f | wc -l


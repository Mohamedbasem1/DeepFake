#!/usr/bin/env bash
set -euo pipefail

IMAGECLEF_URL="${IMAGECLEF_URL:-https://ai4media-bench.aimultimedialab.ro/datasets/download/9161afbc-ff45-41f5-ba85-60285e301df8/}"
CELEBDF_GDRIVE_ID="${CELEBDF_GDRIVE_ID:-1iLx76wsbi9itnkxSqz9BVBl4ZvnbIazj}"
FRAMES_PER_VIDEO="${FRAMES_PER_VIDEO:-8}"

echo "Installing Python dependencies..."
python -m pip install -r requirements-lightning.txt
python -m pip install -U gdown

mkdir -p data data/external data/raw/imageclef_detection data/finetune/frames

echo "Downloading ImageCLEF detection data..."
if [ ! -s data/ImageCLEF2026-DeepFakeDetection-Tes.zip ]; then
  python scripts/download_file.py \
    --url "$IMAGECLEF_URL" \
    --output data/ImageCLEF2026-DeepFakeDetection-Tes.zip
else
  echo "ImageCLEF zip already exists, skipping."
fi

echo "Extracting ImageCLEF image detection files..."
if [ ! -d data/raw/imageclef_detection/Data/Images_Detection ]; then
  python scripts/prepare_data.py \
    --zip data/ImageCLEF2026-DeepFakeDetection-Tes.zip \
    --output data/raw/imageclef_detection \
    --prefix Data/Images_Detection/
else
  echo "ImageCLEF images already extracted, skipping."
fi

echo "Downloading Celeb-DF v2..."
if [ ! -s data/external/Celeb-DF-v2.zip ]; then
  gdown "$CELEBDF_GDRIVE_ID" -O data/external/Celeb-DF-v2.zip
else
  echo "Celeb-DF zip already exists, skipping."
fi

echo "Extracting Celeb-DF v2..."
if [ ! -d data/external/Celeb-DF/Celeb-synthesis ]; then
  python - <<'PY'
import zipfile
from pathlib import Path

zip_path = Path("data/external/Celeb-DF-v2.zip")
output = Path("data/external/Celeb-DF")
output.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zip_path) as archive:
    archive.extractall(output)
print(f"Extracted {zip_path} to {output}")
PY
else
  echo "Celeb-DF already extracted, skipping."
fi

echo "Preparing Celeb-DF frames..."
python scripts/extract_video_frames.py \
  --input data/external/Celeb-DF/Celeb-real \
  --output data/finetune/frames \
  --dataset-name CelebDF_CelebReal \
  --label real \
  --frames-per-video "$FRAMES_PER_VIDEO"

python scripts/extract_video_frames.py \
  --input data/external/Celeb-DF/YouTube-real \
  --output data/finetune/frames \
  --dataset-name CelebDF_YouTubeReal \
  --label real \
  --frames-per-video "$FRAMES_PER_VIDEO"

python scripts/extract_video_frames.py \
  --input data/external/Celeb-DF/Celeb-synthesis \
  --output data/finetune/frames \
  --dataset-name CelebDF_Synthesis \
  --label fake \
  --frames-per-video "$FRAMES_PER_VIDEO"

echo "Frame counts:"
echo -n "real: "
find data/finetune/frames -path "*/real/*" -type f | wc -l
echo -n "fake: "
find data/finetune/frames -path "*/fake/*" -type f | wc -l

echo "Ready. Next fine-tune with:"
echo "python scripts/finetune_commfor.py --train-root data/finetune/frames --output models/commfor_finetuned.pt --epochs 2 --batch-size 32 --freeze-backbone --use-amp --limit-per-class 7000"


#!/usr/bin/env bash
set -euo pipefail

FF_COMPRESSION="${FF_COMPRESSION:-c23}"
FF_SERVER="${FF_SERVER:-EU2}"
FF_DATASETS="${FF_DATASETS:-original Deepfakes Face2Face FaceSwap NeuralTextures}"
FF_DOWNLOAD_SCRIPT="${FF_DOWNLOAD_SCRIPT:-}"
FF_DOWNLOAD_SCRIPT_URL="${FF_DOWNLOAD_SCRIPT_URL:-}"

DEEPERFORENSICS_GDRIVE_ID="${DEEPERFORENSICS_GDRIVE_ID:-1s3KwYyTIXT78VzkRazn9QDPuNh18TWe-}"
DEEPERFORENSICS_DOWNLOAD="${DEEPERFORENSICS_DOWNLOAD:-1}"
FRAMES_PER_VIDEO="${FRAMES_PER_VIDEO:-8}"

python -m pip install -r requirements-lightning.txt
python -m pip install -U gdown tqdm

mkdir -p data/external data/finetune/frames

download_faceforensics() {
  local script_path="data/external/FaceForensics_download.py"

  if [ -n "$FF_DOWNLOAD_SCRIPT" ]; then
    cp "$FF_DOWNLOAD_SCRIPT" "$script_path"
  elif [ -n "$FF_DOWNLOAD_SCRIPT_URL" ]; then
    python scripts/download_file.py \
      --url "$FF_DOWNLOAD_SCRIPT_URL" \
      --output "$script_path"
  elif [ -f "$script_path" ]; then
    echo "Using existing FaceForensics script: $script_path"
  else
    echo "Skipping FaceForensics++ download."
    echo "Set FF_DOWNLOAD_SCRIPT=/path/to/download.py or FF_DOWNLOAD_SCRIPT_URL='approved_link_from_email'."
    return 0
  fi

  for dataset in $FF_DATASETS; do
    echo "Downloading FaceForensics++ dataset=$dataset compression=$FF_COMPRESSION server=$FF_SERVER"
    python "$script_path" \
      data/external/FaceForensics \
      --server "$FF_SERVER" \
      -d "$dataset" \
      -c "$FF_COMPRESSION" \
      -t videos
  done
}

download_deeperforensics() {
  if [ "$DEEPERFORENSICS_DOWNLOAD" != "1" ]; then
    echo "Skipping DeeperForensics download because DEEPERFORENSICS_DOWNLOAD=$DEEPERFORENSICS_DOWNLOAD"
    return 0
  fi

  mkdir -p data/external/DeeperForensics_downloads data/external/DeeperForensics-1.0
  echo "Downloading DeeperForensics from Google Drive id=$DEEPERFORENSICS_GDRIVE_ID"
  echo "If this id is a folder, gdown --folder will be used; if it fails, download manually or use rclone."
  if ! gdown --folder "https://drive.google.com/drive/folders/${DEEPERFORENSICS_GDRIVE_ID}" \
    -O data/external/DeeperForensics_downloads; then
    echo "Folder download failed. Trying direct file download..."
    gdown "$DEEPERFORENSICS_GDRIVE_ID" -O data/external/DeeperForensics_downloads/deeperforensics_download
  fi

  echo "Extracting DeeperForensics archives where possible..."
  python - <<'PY'
import zipfile
from pathlib import Path

root = Path("data/external/DeeperForensics_downloads")
out = Path("data/external/DeeperForensics-1.0")
out.mkdir(parents=True, exist_ok=True)
archives = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".zip"]
print(f"zip archives found: {len(archives)}")
for archive_path in archives:
    print(f"extracting {archive_path}")
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(out)
print(f"extracted to {out}")
PY
}

extract_faceforensics_frames() {
  local root="data/external/FaceForensics"
  if [ ! -d "$root" ]; then
    echo "No FaceForensics folder found, skipping frame extraction."
    return 0
  fi

  if [ -d "$root/original_sequences/youtube/$FF_COMPRESSION/videos" ]; then
    python scripts/extract_video_frames.py \
      --input "$root/original_sequences/youtube/$FF_COMPRESSION/videos" \
      --output data/finetune/frames \
      --dataset-name FaceForensics_Original_${FF_COMPRESSION} \
      --label real \
      --frames-per-video "$FRAMES_PER_VIDEO"
  fi

  for method in Deepfakes Face2Face FaceSwap NeuralTextures FaceShifter DeepFakeDetection; do
    if [ -d "$root/manipulated_sequences/$method/$FF_COMPRESSION/videos" ]; then
      python scripts/extract_video_frames.py \
        --input "$root/manipulated_sequences/$method/$FF_COMPRESSION/videos" \
        --output data/finetune/frames \
        --dataset-name FaceForensics_${method}_${FF_COMPRESSION} \
        --label fake \
        --frames-per-video "$FRAMES_PER_VIDEO"
    fi
  done
}

extract_deeperforensics_frames() {
  local root="data/external/DeeperForensics-1.0"
  if [ ! -d "$root" ]; then
    echo "No DeeperForensics folder found, skipping frame extraction."
    return 0
  fi

  if [ -d "$root/manipulated_videos" ]; then
    python scripts/extract_video_frames.py \
      --input "$root/manipulated_videos" \
      --output data/finetune/frames \
      --dataset-name DeeperForensics \
      --label fake \
      --frames-per-video "$FRAMES_PER_VIDEO"
  else
    echo "No DeeperForensics manipulated_videos folder found after extraction."
  fi
}

download_faceforensics
download_deeperforensics
extract_faceforensics_frames
extract_deeperforensics_frames

echo "Frame counts after FaceForensics++/DeeperForensics prep:"
echo -n "real: "
find data/finetune/frames -path "*/real/*" -type f | wc -l
echo -n "fake: "
find data/finetune/frames -path "*/fake/*" -type f | wc -l

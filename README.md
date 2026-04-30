# ImageCLEF 2026 Deepfake Detection

Starter workspace for the ImageCLEF 2026 deepfake detection task.

The detection subtask is binary classification: decide whether each media item is
`real` or `deepfake`. The organizers do not provide a training split for this
subtask, so this repository starts with a robust inference/submission pipeline
and a lightweight artifact baseline that can be replaced or ensembled with
stronger pretrained detectors.

## Project Layout

```text
configs/default.yaml          Default inference and submission settings
data/raw/                     Put downloaded test/dev media here
outputs/                      Predictions and submissions
scripts/run_inference.py      Thin script wrapper around the package CLI
src/deepfake_detector/        Package code
tests/                        Smoke tests for pipeline behavior
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
```

For a minimal smoke test, only `numpy`, `pillow`, `pyyaml`, and `pytest` are
needed. Install `requirements-vision.txt` later when we add video decoding or
stronger `torch`/`timm` model backends.

## Run Inference

The downloaded detection archive contains:

- `Data/Images_Detection/` with 24,404 `.png` files
- `Data/Audio_Detection/` with 11,520 `.wav` files
- sample submission files with columns `full_secret_name,prediction`

Extract the image detection files:

```powershell
python scripts/prepare_data.py `
  --zip data/ImageCLEF2026-DeepFakeDetection-Tes.zip `
  --output data/raw/imageclef_detection `
  --prefix Data/Images_Detection/
```

Run the lightweight smoke-test baseline:

```powershell
python scripts/run_inference.py --input data/raw/test --output outputs/baseline_predictions.csv
```

Run the Lightning/Hugging Face image detector:

```powershell
python scripts/run_inference.py `
  --config configs/image_detection_commfor.yaml `
  --input data/raw/imageclef_detection/Data/Images_Detection `
  --output outputs/images_detection_submission.csv
```

The official-style output CSV includes:

- `full_secret_name`: filename with extension
- `prediction`: `1` for fake/deepfake, `0` for real

Use `configs/image_detection_commfor_scores.yaml` when you also want a `score`
column for analysis and threshold calibration.

## Next Steps

1. Confirm the exact competition submission columns once the portal reveals them.
2. Confirm whether `prediction` expects `0/1` or text labels on AI4MediaBench.
3. Add external validation data and keep a local validation manifest.
4. Add audio detection if you want to submit the audio subtask too.
5. Ensemble model scores and calibrate the final threshold.

See `docs/finetuning.md` for extracting frames from video deepfake datasets and
fine-tuning the Community Forensics checkpoint.

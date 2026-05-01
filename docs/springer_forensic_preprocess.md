# Springer Review Inspired Forensic Preprocessing

The Springer article is a review, not a single released model. The image-only idea we can test from it is
forensic preprocessing before CNN classification:

- YCbCr/color cues
- Canny-style edge artifacts
- ELA-style compression residuals

This repo supports those through `--input-mode` in the timm trainer:

- `rgb`: normal image
- `ycbcr`: color-space input
- `ela`: JPEG error-level residual
- `edges`: Canny/FIND_EDGES map
- `forensic`: 3-channel composite: `Y`, edge map, ELA residual

Recommended first run:

```bash
python scripts/train_timm_image_classifier.py \
  --train-root data/finetune/frames \
  --output models/timm_resnet50_forensic.pt \
  --model-name resnet50 \
  --input-mode forensic \
  --image-size 224 \
  --epochs 8 \
  --batch-size 64 \
  --lr 3e-5 \
  --use-amp \
  --limit-per-class 15000
```

Predict:

```bash
python scripts/predict_timm_image_classifier.py \
  --checkpoint models/timm_resnet50_forensic.pt \
  --input data/raw/imageclef_detection/Data/Images_Detection \
  --output outputs/images_detection_scores_resnet50_forensic.csv \
  --batch-size 96 \
  --use-amp \
  --tta-hflip
```

Check distribution and compare:

```bash
python scripts/prepare_submission_upload.py \
  --input outputs/images_detection_scores_resnet50_forensic.csv \
  --output-csv outputs/upload/images_detection_submission_resnet50_forensic.csv \
  --output-zip outputs/upload/images_detection_submission_resnet50_forensic.zip

python scripts/compare_submissions.py \
  outputs/upload/images_detection_submission_e5_e8_e20_avg.csv \
  outputs/upload/images_detection_submission_resnet50_forensic.csv
```

If the threshold is shifted again, use rank or high-confidence overrides instead of submitting the raw output.

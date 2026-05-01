# Yermandy CLIP Predictor

This path uses the official TorchScript checkpoint from `yermandy/deepfake-detection`.
It follows their minimal inference setup:

- model: `yermandy/deepfake-detection/model.torchscript`
- processor: `openai/clip-vit-large-patch14`
- score: softmax fake probability, column index `1`

Run on Lightning:

```bash
python scripts/predict_yermandy_clip.py \
  --input data/raw/imageclef_detection/Data/Images_Detection \
  --output outputs/images_detection_scores_yermandy_clip.csv \
  --batch-size 32 \
  --num-workers 4
```

Use your labeled datasets to calibrate the decision threshold:

```bash
python scripts/calibrate_yermandy_clip_threshold.py \
  --train-root data/finetune/frames \
  --output models/yermandy_clip_calibration.json \
  --scores-output outputs/yermandy_clip_calibration_scores.csv \
  --batch-size 64 \
  --num-workers 4 \
  --limit-per-class 15000
```

Then use the printed `best_threshold` for ImageCLEF prediction:

```bash
python scripts/predict_yermandy_clip.py \
  --input data/raw/imageclef_detection/Data/Images_Detection \
  --output outputs/images_detection_scores_yermandy_clip_calibrated.csv \
  --batch-size 64 \
  --num-workers 4 \
  --threshold BEST_THRESHOLD_HERE
```

Prepare a clean submission:

```bash
python scripts/prepare_submission_upload.py \
  --input outputs/images_detection_scores_yermandy_clip.csv \
  --output-csv outputs/upload/images_detection_submission_yermandy_clip.csv \
  --output-zip outputs/upload/images_detection_submission_yermandy_clip.zip
```

Compare with an older submission:

```bash
python scripts/compare_submissions.py \
  outputs/upload/images_detection_submission_e5_e8_e20_avg.csv \
  outputs/upload/images_detection_submission_yermandy_clip.csv
```

# Frozen Embedding Classifier

This approach uses a frozen vision foundation model as an embedding extractor
and trains a shallow classifier on top. It is intentionally different from
Community Forensics fine-tuning.

Default model:

```text
google/siglip-base-patch16-384
```

Train:

```bash
python scripts/train_embedding_classifier.py \
  --train-root data/finetune/frames \
  --output models/siglip_embedding_classifier.joblib \
  --cache models/siglip_embedding_features.npz \
  --batch-size 64 \
  --limit-per-class 15000
```

Predict ImageCLEF:

```bash
python scripts/predict_embedding_classifier.py \
  --model models/siglip_embedding_classifier.joblib \
  --input data/raw/imageclef_detection/Data/Images_Detection \
  --output outputs/images_detection_scores_siglip_embedding.csv \
  --batch-size 64
```

Ensemble:

```bash
python scripts/merge_multiple_scores.py \
  --mode avg \
  --threshold 0.5 \
  --inputs \
    outputs/images_detection_scores_siglip_embedding.csv \
    outputs/images_detection_scores_mixed_big_e20.csv \
    outputs/images_detection_scores_finetuned_full_e8.csv \
  --output outputs/images_detection_submission_siglip_e20_e8_avg.csv \
  --debug-output outputs/images_detection_siglip_e20_e8_avg_debug.csv
```


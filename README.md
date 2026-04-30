# ImageCLEF Deepfake Detection - Fresh Embedding Approach

This branch is a clean restart focused on a different strategy:

```text
frozen vision foundation model embeddings + shallow classifier
```

The default model is `google/siglip-base-patch16-384`. We extract embeddings
from labeled external real/fake frames, train a lightweight classifier, then run
ImageCLEF test images through the same embedding pipeline.

## What To Keep Outside Git

Do not commit datasets, extracted frames, checkpoints, cookies, or submissions.
They belong under ignored folders such as:

```text
data/
models/
outputs/
result/
```

## Setup On Lightning

```bash
git clone -b fresh-embedding-approach https://github.com/Mohamedbasem1/DeepFake.git
cd DeepFake
pip install -r requirements-lightning.txt
```

## Prepare Data

Download ImageCLEF and Celeb-DF:

```bash
bash scripts/bootstrap_lightning_data.sh
```

Optional external video datasets:

```bash
bash scripts/bootstrap_faceforensics_v4.sh
```

The training frames should end up under:

```text
data/finetune/frames/**/real/*.jpg
data/finetune/frames/**/fake/*.jpg
```

## Train Embedding Classifier

```bash
python scripts/train_embedding_classifier.py \
  --train-root data/finetune/frames \
  --output models/siglip_embedding_classifier.joblib \
  --cache models/siglip_embedding_features.npz \
  --batch-size 64 \
  --limit-per-class 15000
```

For a stronger run on a large machine:

```bash
python scripts/train_embedding_classifier.py \
  --train-root data/finetune/frames \
  --output models/siglip_embedding_classifier_big.joblib \
  --cache models/siglip_embedding_features_big.npz \
  --batch-size 128 \
  --limit-per-class 30000 \
  --c 0.5
```

## Predict ImageCLEF

```bash
python scripts/predict_embedding_classifier.py \
  --model models/siglip_embedding_classifier.joblib \
  --input data/raw/imageclef_detection/Data/Images_Detection \
  --output outputs/images_detection_scores_siglip_embedding.csv \
  --batch-size 64
```

## Package Submission

If you want to submit the embedding classifier directly:

```bash
python scripts/prepare_submission_upload.py \
  --input outputs/images_detection_scores_siglip_embedding.csv \
  --output-csv outputs/upload/images_detection_submission_siglip_embedding.csv \
  --output-zip outputs/upload/images_detection_submission_siglip_embedding.zip
```

## Useful Utilities

Compare two submissions:

```bash
python scripts/compare_submissions.py file_a.csv file_b.csv
```

Average or max-merge multiple scored CSVs:

```bash
python scripts/merge_multiple_scores.py \
  --mode avg \
  --inputs score_a.csv score_b.csv score_c.csv \
  --output outputs/merged_submission.csv
```


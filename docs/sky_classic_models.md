# Sky Classic Models

The referenced repository uses VGG16, VGG19, InceptionV3, ResNet50, and a small custom CNN.
For our pipeline, the fastest usable version is to train the same classic pretrained backbones through
`timm` and keep our existing ImageCLEF CSV format.

Run all four models:

```bash
bash scripts/run_sky_classic_models.sh all
```

Or run in stages:

```bash
bash scripts/run_sky_classic_models.sh train
bash scripts/run_sky_classic_models.sh predict
bash scripts/run_sky_classic_models.sh merge
```

The merged output is:

```text
outputs/upload/images_detection_submission_sky_classic_avg.csv
```

Compare against the current best:

```bash
python scripts/compare_submissions.py \
  outputs/upload/images_detection_submission_e5_e8_e20_avg.csv \
  outputs/upload/images_detection_submission_sky_classic_avg.csv
```

To run only one model manually:

```bash
python scripts/train_timm_image_classifier.py \
  --train-root data/finetune/frames \
  --output models/sky_resnet50.pt \
  --model-name resnet50 \
  --image-size 224 \
  --epochs 8 \
  --batch-size 64 \
  --lr 3e-5 \
  --use-amp \
  --limit-per-class 15000

python scripts/predict_timm_image_classifier.py \
  --checkpoint models/sky_resnet50.pt \
  --input data/raw/imageclef_detection/Data/Images_Detection \
  --output outputs/images_detection_scores_sky_resnet50.csv \
  --batch-size 96 \
  --use-amp \
  --tta-hflip
```

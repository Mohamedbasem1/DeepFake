# Timm Image Ensemble

This is a stronger version of the VGG/Inception/ResNet ensemble idea. It trains modern ImageNet-pretrained
backbones directly on `data/finetune/frames`.

Recommended first model:

```bash
python scripts/train_timm_image_classifier.py \
  --train-root data/finetune/frames \
  --output models/timm_convnext_base.pt \
  --model-name convnext_base \
  --image-size 224 \
  --epochs 8 \
  --batch-size 32 \
  --lr 2e-5 \
  --use-amp \
  --limit-per-class 15000
```

More models for a diverse ensemble:

```bash
python scripts/train_timm_image_classifier.py \
  --train-root data/finetune/frames \
  --output models/timm_efficientnet_b4.pt \
  --model-name tf_efficientnet_b4 \
  --image-size 380 \
  --epochs 8 \
  --batch-size 16 \
  --lr 1e-5 \
  --use-amp \
  --limit-per-class 15000

python scripts/train_timm_image_classifier.py \
  --train-root data/finetune/frames \
  --output models/timm_swin_base.pt \
  --model-name swin_base_patch4_window7_224 \
  --image-size 224 \
  --epochs 8 \
  --batch-size 24 \
  --lr 1e-5 \
  --use-amp \
  --limit-per-class 15000

python scripts/train_timm_image_classifier.py \
  --train-root data/finetune/frames \
  --output models/timm_resnet50.pt \
  --model-name resnet50 \
  --image-size 224 \
  --epochs 8 \
  --batch-size 64 \
  --lr 3e-5 \
  --use-amp \
  --limit-per-class 15000
```

Predict ImageCLEF:

```bash
python scripts/predict_timm_image_classifier.py \
  --checkpoint models/timm_convnext_base.pt \
  --input data/raw/imageclef_detection/Data/Images_Detection \
  --output outputs/images_detection_scores_timm_convnext_base.csv \
  --batch-size 64 \
  --use-amp \
  --tta-hflip
```

Repeat prediction for each checkpoint, then merge:

```bash
python scripts/merge_multiple_scores.py \
  --mode avg \
  --threshold 0.5 \
  --inputs \
    outputs/images_detection_scores_timm_convnext_base.csv \
    outputs/images_detection_scores_timm_efficientnet_b4.csv \
    outputs/images_detection_scores_timm_swin_base.csv \
    outputs/images_detection_scores_timm_resnet50.csv \
  --output outputs/images_detection_submission_timm_ensemble_avg.csv \
  --debug-output outputs/images_detection_timm_ensemble_avg_debug.csv
```

Package:

```bash
python scripts/prepare_submission_upload.py \
  --input outputs/images_detection_submission_timm_ensemble_avg.csv \
  --output-csv outputs/upload/images_detection_submission_timm_ensemble_avg.csv \
  --output-zip outputs/upload/images_detection_submission_timm_ensemble_avg.zip
```

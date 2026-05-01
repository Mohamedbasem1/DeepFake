# ResNeXt Frame Classifier

This is the ImageCLEF-compatible adaptation of the `modelTraining.ipynb`
approach from `adityapathakk/Deepfake-Detection-System`.

The notebook uses:

- ResNeXt for spatial frame features
- LSTM for temporal video modeling

ImageCLEF provides individual PNG images, not video sequences, so this branch
uses the ResNeXt spatial component as a frame classifier.

Train:

```bash
python scripts/train_resnext_frame_classifier.py \
  --train-root data/finetune/frames \
  --output models/resnext_frame_classifier.pt \
  --epochs 8 \
  --batch-size 64 \
  --image-size 224 \
  --lr 2e-5 \
  --head-lr 2e-4 \
  --use-amp \
  --limit-per-class 7000
```

Predict:

```bash
python scripts/predict_resnext_frame_classifier.py \
  --model models/resnext_frame_classifier.pt \
  --input data/raw/imageclef_detection/Data/Images_Detection \
  --output outputs/images_detection_scores_resnext_frame.csv
```

Package direct submission:

```bash
python scripts/prepare_submission_upload.py \
  --input outputs/images_detection_scores_resnext_frame.csv \
  --output-csv outputs/upload/images_detection_submission_resnext_frame.csv \
  --output-zip outputs/upload/images_detection_submission_resnext_frame.zip
```


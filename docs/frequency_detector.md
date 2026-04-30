# Frequency Detector

This is a lightweight frequency-domain detector meant to complement Community
Forensics. It extracts FFT radial spectra, residual spectra, blockiness, and
noise statistics, then trains a scikit-learn classifier.

Train on extracted real/fake frames:

```bash
python scripts/train_frequency_detector.py \
  --train-root data/finetune/frames \
  --output models/frequency_detector_mixed.joblib \
  --limit-per-class 15000
```

Predict ImageCLEF scores:

```bash
python scripts/predict_frequency_detector.py \
  --model models/frequency_detector_mixed.joblib \
  --input data/raw/imageclef_detection/Data/Images_Detection \
  --output outputs/images_detection_scores_frequency_mixed.csv
```

If raw probabilities are all below `0.5`, calibrate the file by rank before
ensembling:

```bash
python scripts/calibrate_score_file.py \
  --input outputs/images_detection_scores_frequency_mixed.csv \
  --output outputs/images_detection_scores_frequency_mixed_rank.csv \
  --method rank
```

Ensemble with Community Forensics:

```bash
python scripts/merge_multiple_scores.py \
  --mode avg \
  --threshold 0.5 \
  --inputs \
    outputs/images_detection_scores_frequency_mixed_rank.csv \
    outputs/images_detection_scores_mixed_big_e20.csv \
    outputs/images_detection_scores_finetuned_full_e8.csv \
  --output outputs/images_detection_submission_frequency_commfor_avg.csv \
  --debug-output outputs/images_detection_frequency_commfor_avg_debug.csv
```

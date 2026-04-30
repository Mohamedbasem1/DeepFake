# Fine-Tuning Plan

Community Forensics can be fine-tuned, but ImageCLEF's detection test files do
not include labels. Fine-tune only on external labeled data, then use the
ImageCLEF files strictly for inference.

## Useful Face/Video Datasets

- FaceForensics++
- Celeb-DF
- DeeperForensics
- DFDC

These datasets are video-based. For our image detector, extract frames and save
them in the local format expected by Community Forensics:

```text
data/finetune/frames/
  FaceForensics/
    real/
      video_a_f000012.jpg
    fake/
      video_b_f000012.jpg
```

## Frame Extraction

For folders where all videos share one label:

```bash
python scripts/extract_video_frames.py \
  --input data/external/FaceForensics/original_sequences \
  --output data/finetune/frames \
  --dataset-name FaceForensics \
  --label real \
  --frames-per-video 8

python scripts/extract_video_frames.py \
  --input data/external/FaceForensics/manipulated_sequences \
  --output data/finetune/frames \
  --dataset-name FaceForensics \
  --label fake \
  --frames-per-video 8
```

For a dataset already arranged with meaningful parent folders:

```bash
python scripts/extract_video_frames.py \
  --input data/external/CelebDF \
  --output data/finetune/frames \
  --dataset-name CelebDF \
  --label-from-parent \
  --frames-per-video 8
```

## Recommended Training Style

Start conservative:

- freeze the backbone or use a very small learning rate,
- train for 1-3 epochs,
- keep a separate validation set by dataset/source,
- compare against the pretrained checkpoint before submitting.

The risk is domain overfitting: improving on FaceForensics or DFDC can hurt on
unseen ImageCLEF generators. Keep the pretrained Community Forensics model as a
baseline and ensemble candidate.

## Fine-Tune Community Forensics

First pull the latest repo on Lightning:

```bash
git pull
pip install -r requirements-lightning.txt
```

Start with a quick head-only run:

```bash
python scripts/finetune_commfor.py \
  --train-root data/finetune/frames \
  --output models/commfor_finetuned.pt \
  --epochs 2 \
  --batch-size 32 \
  --freeze-backbone \
  --use-amp
```

If validation F1 improves and the model does not collapse to one class, try a
careful full-model run:

```bash
python scripts/finetune_commfor.py \
  --train-root data/finetune/frames \
  --output models/commfor_finetuned.pt \
  --epochs 2 \
  --batch-size 16 \
  --lr 1e-6 \
  --head-lr 2e-5 \
  --use-amp
```

Run ImageCLEF inference with the fine-tuned checkpoint:

```bash
python scripts/run_inference.py \
  --config configs/image_detection_commfor_finetuned.yaml \
  --input data/raw/imageclef_detection/Data/Images_Detection \
  --output outputs/images_detection_submission_finetuned.csv
```

For threshold analysis, export scores too:

```bash
python scripts/run_inference.py \
  --config configs/image_detection_commfor_finetuned_scores.yaml \
  --input data/raw/imageclef_detection/Data/Images_Detection \
  --output outputs/images_detection_scores_finetuned.csv
```

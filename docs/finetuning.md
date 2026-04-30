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


# Fine-Tuning Data Preparation

This branch does not fine-tune the backbone. It uses external datasets only to
train a shallow classifier on frozen embeddings.

Expected frame layout:

```text
data/finetune/frames/
  SomeDataset/
    real/
    fake/
```

Use `scripts/extract_video_frames.py` to convert video datasets into this
layout.


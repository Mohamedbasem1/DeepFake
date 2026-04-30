# External Face Deepfake Datasets

Use these datasets as labeled external data for fine-tuning. Keep them out of
git under `data/external/`, then extract frames into `data/finetune/frames/`.

## Access

### FaceForensics++

Repository: <https://github.com/ondyari/FaceForensics>

Access requires filling out the FaceForensics++ Google Form. After approval, the
authors send a download script link. Prefer compressed videos, not raw images.
The repository recommends downloading compressed videos and extracting frames
locally; `c23` is the balanced choice.

Useful parts:

- `original_sequences/youtube/c23/videos`: real
- `manipulated_sequences/Deepfakes/c23/videos`: fake
- `manipulated_sequences/Face2Face/c23/videos`: fake
- `manipulated_sequences/FaceSwap/c23/videos`: fake
- `manipulated_sequences/NeuralTextures/c23/videos`: fake
- `manipulated_sequences/FaceShifter/c23/videos`: fake, if available

### Celeb-DF

Repository: <https://github.com/yuezunli/celeb-deepfakeforensics>

Access requires submitting the linked Google or Tencent form. The expected
dataset structure is:

- `Celeb-real`: real
- `YouTube-real`: real
- `Celeb-synthesis`: fake

### DeeperForensics-1.0

Repository: <https://github.com/EndlessSora/DeeperForensics-1.0>

Access requires reading the terms of use and submitting the dataset form. The
dataset page reports about 284 GB total and recommends 300 GB or more.

Important: for detection training, the DeeperForensics authors say their
`source_videos` are not the real class. The real target videos are the
FaceForensics++ C23 target videos. Use DeeperForensics `manipulated_videos` as
fake and FaceForensics++ C23 originals as real.

## Frame Extraction

After downloading/unzipping, extract a balanced number of frames. Start small:
4-8 frames per video is enough for the first fine-tune experiment.

FaceForensics++:

```bash
python scripts/extract_video_frames.py \
  --input data/external/FaceForensics/original_sequences/youtube/c23/videos \
  --output data/finetune/frames \
  --dataset-name FaceForensics \
  --label real \
  --frames-per-video 8

python scripts/extract_video_frames.py \
  --input data/external/FaceForensics/manipulated_sequences/Deepfakes/c23/videos \
  --output data/finetune/frames \
  --dataset-name FaceForensics_Deepfakes \
  --label fake \
  --frames-per-video 8
```

Repeat for `Face2Face`, `FaceSwap`, `NeuralTextures`, and `FaceShifter` if you
download them.

Celeb-DF:

```bash
python scripts/extract_video_frames.py \
  --input data/external/Celeb-DF/Celeb-real \
  --output data/finetune/frames \
  --dataset-name CelebDF \
  --label real \
  --frames-per-video 8

python scripts/extract_video_frames.py \
  --input data/external/Celeb-DF/YouTube-real \
  --output data/finetune/frames \
  --dataset-name CelebDF_YouTube \
  --label real \
  --frames-per-video 8

python scripts/extract_video_frames.py \
  --input data/external/Celeb-DF/Celeb-synthesis \
  --output data/finetune/frames \
  --dataset-name CelebDF \
  --label fake \
  --frames-per-video 8
```

DeeperForensics fake videos:

```bash
python scripts/extract_video_frames.py \
  --input data/external/DeeperForensics-1.0/manipulated_videos/end_to_end \
  --output data/finetune/frames \
  --dataset-name DeeperForensics \
  --label fake \
  --frames-per-video 8
```

For DeeperForensics real videos, use FaceForensics++ C23 originals as described
above.

## Approved Download Shortcuts

After you receive the approved FaceForensics++ email, save the linked download
script as:

```text
data/external/FaceForensics_download.py
```

Then run:

```bash
bash scripts/bootstrap_faceforensics_deeper.sh
```

If you only have the approved script somewhere else:

```bash
FF_DOWNLOAD_SCRIPT=/path/to/download.py bash scripts/bootstrap_faceforensics_deeper.sh
```

If your email exposes the actual script URL:

```bash
FF_DOWNLOAD_SCRIPT_URL="APPROVED_FACEFORENSICS_SCRIPT_URL" \
  bash scripts/bootstrap_faceforensics_deeper.sh
```

By default this uses:

- FaceForensics++ server `EU2`
- compression `c23`
- datasets `original Deepfakes Face2Face FaceSwap NeuralTextures`
- DeeperForensics Google Drive id `1s3KwYyTIXT78VzkRazn9QDPuNh18TWe-`

For smaller/faster FaceForensics++ downloads, use `c40`:

```bash
FF_COMPRESSION=c40 bash scripts/bootstrap_faceforensics_deeper.sh
```

### Older FaceForensics v4 Script

If your email gives `faceforensics_download_v4.py`, use:

```bash
bash scripts/bootstrap_faceforensics_v4.sh
```

This downloads the `compressed` source-to-target release by default, skips mask
videos, accepts the script prompts automatically, and extracts frames from
`original` as real and `altered` as fake.

For a tiny smoke test:

```bash
SAMPLE_ONLY=1 bash scripts/bootstrap_faceforensics_v4.sh
```

## Fine-Tune

Once frames are ready:

```bash
python scripts/finetune_commfor.py \
  --train-root data/finetune/frames \
  --output models/commfor_finetuned.pt \
  --epochs 2 \
  --batch-size 32 \
  --freeze-backbone \
  --use-amp
```

Then run ImageCLEF inference with:

```bash
python scripts/run_inference.py \
  --config configs/image_detection_commfor_finetuned.yaml \
  --input data/raw/imageclef_detection/Data/Images_Detection \
  --output outputs/images_detection_submission_finetuned.csv
```

## Practical Notes

- Keep the first training run small and fast.
- Avoid using only one fake method; mix datasets if possible.
- Do not fine-tune on ImageCLEF test images because they have no labels.
- Keep pretrained Community Forensics predictions as a baseline for comparison.

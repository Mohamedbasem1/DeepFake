from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor

from predict_yermandy_clip import IMAGE_EXTENSIONS, predict_fake_scores, resolve_device, resolve_dtype


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int


class LabeledImageDataset(Dataset):
    def __init__(self, samples: list[Sample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
        return sample.path, image, sample.label


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate the Yermandy CLIP fake threshold on labeled real/fake frames."
    )
    parser.add_argument("--train-root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("models/yermandy_clip_calibration.json"))
    parser.add_argument("--scores-output", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--limit-per-class", type=int)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=1001)
    parser.add_argument("--repo-id", default="yermandy/deepfake-detection")
    parser.add_argument("--filename", default="model.torchscript")
    parser.add_argument("--model-cache-dir", type=Path, default=Path("weights/yermandy_clip"))
    parser.add_argument("--processor", default="openai/clip-vit-large-patch14")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    samples = discover_samples(args.train_root, args.limit_per_class)
    _, val_samples = split_samples(samples, args.val_frac, args.seed)
    print(f"calibration samples: {len(val_samples)}")
    print_counts("calibration", val_samples)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    model_path = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.filename,
        local_dir=str(args.model_cache_dir),
    )
    model = torch.jit.load(model_path, map_location=device).eval().to(device)
    if dtype is not torch.float32:
        model = model.to(dtype)
    processor = CLIPProcessor.from_pretrained(args.processor)

    scores, labels, names = score_samples(
        val_samples,
        model,
        processor,
        device,
        dtype,
        args.batch_size,
        args.num_workers,
    )
    best = find_best_threshold(scores, labels, args.steps)
    fixed = metrics_at_threshold(scores, labels, 0.5)

    payload = {
        "threshold": best["threshold"],
        "metrics": best,
        "metrics_at_0_5": fixed,
        "samples": len(labels),
        "real": int(np.sum(labels == 0)),
        "fake": int(np.sum(labels == 1)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.scores_output:
        write_scores(args.scores_output, names, scores, labels)

    print(f"best_threshold: {best['threshold']:.4f}")
    print_metrics("best", best)
    print_metrics("threshold_0.5", fixed)
    print(f"saved: {args.output}")
    if args.scores_output:
        print(f"scores: {args.scores_output}")


def score_samples(
    samples: list[Sample],
    model,
    processor,
    device: str,
    dtype: torch.dtype,
    batch_size: int,
    num_workers: int,
):
    loader = DataLoader(
        LabeledImageDataset(samples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: batch,
    )
    all_scores: list[float] = []
    all_labels: list[int] = []
    all_names: list[str] = []
    for batch_index, batch in enumerate(loader, start=1):
        paths = [item[0] for item in batch]
        images = [item[1] for item in batch]
        labels = [item[2] for item in batch]
        scores = predict_fake_scores(model, processor, images, device, dtype)
        all_names.extend(str(path) for path in paths)
        all_scores.extend(scores)
        all_labels.extend(labels)
        if batch_index % 25 == 0:
            print(f"batches {batch_index}/{len(loader)}")
    return (
        np.asarray(all_scores, dtype=np.float32),
        np.asarray(all_labels, dtype=np.int32),
        all_names,
    )


def find_best_threshold(scores: np.ndarray, labels: np.ndarray, steps: int) -> dict[str, float]:
    best = metrics_at_threshold(scores, labels, 0.5)
    for threshold in np.linspace(0.0, 1.0, steps):
        current = metrics_at_threshold(scores, labels, float(threshold))
        if (current["f1"], current["accuracy"]) > (best["f1"], best["accuracy"]):
            best = current
    return best


def metrics_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (scores >= threshold).astype(np.int32)
    tp = int(np.sum((predictions == 1) & (labels == 1)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))
    total = max(1, len(labels))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def discover_samples(root: Path, limit_per_class: int | None) -> list[Sample]:
    by_label = {0: [], 1: []}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = label_from_path(path)
        if label is not None:
            by_label[label].append(Sample(path=path, label=label))

    samples = []
    for label, label_samples in by_label.items():
        if limit_per_class:
            label_samples = label_samples[:limit_per_class]
        samples.extend(label_samples)
    if not samples:
        raise SystemExit(f"No labeled images found under {root}")
    return samples


def label_from_path(path: Path) -> int | None:
    parts = [part.lower() for part in path.parts]
    if "fake" in parts or "deepfake" in parts:
        return 1
    if "real" in parts or "original" in parts or "authentic" in parts:
        return 0
    return None


def split_samples(samples: list[Sample], val_frac: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    rng = random.Random(seed)
    train = []
    val = []
    for label in [0, 1]:
        label_samples = [sample for sample in samples if sample.label == label]
        rng.shuffle(label_samples)
        val_count = max(1, round(len(label_samples) * val_frac))
        val.extend(label_samples[:val_count])
        train.extend(label_samples[val_count:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def print_counts(name: str, samples: list[Sample]) -> None:
    real = sum(sample.label == 0 for sample in samples)
    fake = sum(sample.label == 1 for sample in samples)
    print(f"{name}: real={real} fake={fake}")


def print_metrics(name: str, metrics: dict[str, float]) -> None:
    print(
        f"{name}: "
        f"acc={metrics['accuracy']:.4f} "
        f"p={metrics['precision']:.4f} "
        f"r={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f} "
        f"tp={metrics['tp']} fp={metrics['fp']} tn={metrics['tn']} fn={metrics['fn']}"
    )


def write_scores(path: Path, names: list[str], scores: np.ndarray, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "score", "label"])
        writer.writeheader()
        for name, score, label in zip(names, scores, labels):
            writer.writerow({"path": name, "score": f"{float(score):.6f}", "label": int(label)})


if __name__ == "__main__":
    main()

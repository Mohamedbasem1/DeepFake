from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deepfake_detector.frequency_features import extract_frequency_features


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a frequency-domain image deepfake detector.")
    parser.add_argument("--train-root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("models/frequency_detector.joblib"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--radial-bins", type=int, default=32)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-per-class", type=int)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--num-workers", type=int, default=-1, help="Parallel feature workers. -1 uses all cores.")
    parser.add_argument("--cache", type=Path, help="Optional .npz cache for extracted features.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    samples = discover_samples(args.train_root, args.limit_per_class)
    train_samples, val_samples = split_samples(samples, args.val_frac, args.seed)

    print(f"train={len(train_samples)} val={len(val_samples)}")
    print_counts("train", train_samples)
    print_counts("val", val_samples)

    if args.cache and args.cache.exists():
        print(f"loading features from cache: {args.cache}")
        cached = np.load(args.cache, allow_pickle=False)
        x_train = cached["x_train"]
        y_train = cached["y_train"]
        x_val = cached["x_val"]
        y_val = cached["y_val"]
    else:
        x_train, y_train = featurize(train_samples, args.image_size, args.radial_bins, args.num_workers)
        x_val, y_val = featurize(val_samples, args.image_size, args.radial_bins, args.num_workers)
        if args.cache:
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.cache, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
            print(f"saved features cache: {args.cache}")

    model = HistGradientBoostingClassifier(
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        l2_regularization=0.05,
        random_state=args.seed,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)

    val_score = model.predict_proba(x_val)[:, 1]
    val_pred = (val_score >= 0.5).astype(np.int32)
    metrics = {
        "accuracy": accuracy_score(y_val, val_pred),
        "precision": precision_score(y_val, val_pred, zero_division=0),
        "recall": recall_score(y_val, val_pred, zero_division=0),
        "f1": f1_score(y_val, val_pred, zero_division=0),
    }
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "image_size": args.image_size,
            "radial_bins": args.radial_bins,
            "metrics": metrics,
        },
        args.output,
    )
    write_metrics(args.output.with_suffix(".metrics.csv"), metrics)
    print(f"saved: {args.output}")


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


def featurize(
    samples: list[Sample],
    image_size: int,
    radial_bins: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    print(f"extracting {len(samples)} feature vectors with num_workers={num_workers}")
    features = Parallel(n_jobs=num_workers, backend="loky", verbose=10)(
        delayed(extract_frequency_features)(sample.path, image_size=image_size, radial_bins=radial_bins)
        for sample in samples
    )
    labels = [sample.label for sample in samples]
    return np.vstack(features), np.asarray(labels, dtype=np.int32)


def print_counts(name: str, samples: list[Sample]) -> None:
    real = sum(sample.label == 0 for sample in samples)
    fake = sum(sample.label == 1 for sample in samples)
    print(f"{name}: real={real} fake={fake}")


def write_metrics(path: Path, metrics: dict[str, float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)


if __name__ == "__main__":
    main()

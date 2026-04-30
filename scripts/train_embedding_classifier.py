from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int


class ImageDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        processor,
        augment: bool = False,
        views_per_sample: int = 1,
        seed: int = 42,
    ) -> None:
        self.samples = samples
        self.processor = processor
        self.augment = augment
        self.views_per_sample = max(1, views_per_sample)
        self.seed = seed

    def __len__(self) -> int:
        return len(self.samples) * self.views_per_sample

    def __getitem__(self, index: int):
        sample_index = index % len(self.samples)
        view_index = index // len(self.samples)
        sample = self.samples[sample_index]
        with Image.open(sample.path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            if self.augment and view_index > 0:
                rng = random.Random(self.seed + index)
                image = augment_image(image, rng)
        return image, sample.label


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a frozen embedding classifier for deepfake detection.")
    parser.add_argument("--train-root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("models/siglip_embedding_classifier.joblib"))
    parser.add_argument("--cache", type=Path, default=Path("models/siglip_embedding_features.npz"))
    parser.add_argument("--model-id", default="google/siglip-base-patch16-384")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-per-class", type=int)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--c", type=float, default=1.0, help="Logistic regression inverse regularization.")
    parser.add_argument(
        "--train-views",
        type=int,
        default=1,
        help="Number of embedding views per training image. Values >1 add augmented views.",
    )
    parser.add_argument("--augment", action="store_true", help="Apply PIL augmentations to extra training views.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    samples = discover_samples(args.train_root, args.limit_per_class)
    train_samples, val_samples = split_samples(samples, args.val_frac, args.seed)
    print(f"train={len(train_samples)} val={len(val_samples)}")
    print_counts("train", train_samples)
    print_counts("val", val_samples)

    if args.cache.exists():
        print(f"loading cached embeddings: {args.cache}")
        cached = np.load(args.cache, allow_pickle=False)
        x_train, y_train = cached["x_train"], cached["y_train"]
        x_val, y_val = cached["x_val"], cached["y_val"]
    else:
        processor = AutoImageProcessor.from_pretrained(args.model_id)
        model = AutoModel.from_pretrained(args.model_id).to(device)
        model.eval()
        x_train, y_train = extract_embeddings(
            train_samples,
            processor,
            model,
            device,
            args.batch_size,
            args.num_workers,
            augment=args.augment,
            views_per_sample=args.train_views,
            seed=args.seed,
            split_name="train",
        )
        x_val, y_val = extract_embeddings(
            val_samples,
            processor,
            model,
            device,
            args.batch_size,
            args.num_workers,
            augment=False,
            views_per_sample=1,
            seed=args.seed,
            split_name="val",
        )
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.cache, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
        print(f"saved cache: {args.cache}")

    classifier = LogisticRegression(
        C=args.c,
        max_iter=args.max_iter,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=-1,
        verbose=1,
    )
    classifier.fit(x_train, y_train)

    val_score = classifier.predict_proba(x_val)[:, 1]
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
            "classifier": classifier,
            "model_id": args.model_id,
            "metrics": metrics,
        },
        args.output,
    )
    write_metrics(args.output.with_suffix(".metrics.csv"), metrics)
    print(f"saved: {args.output}")


@torch.no_grad()
def extract_embeddings(
    samples: list[Sample],
    processor,
    model,
    device: str,
    batch_size: int,
    num_workers: int,
    augment: bool,
    views_per_sample: int,
    seed: int,
    split_name: str,
):
    loader = DataLoader(
        ImageDataset(samples, processor, augment=augment, views_per_sample=views_per_sample, seed=seed),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: batch,
    )
    print(
        f"extracting {split_name} embeddings: "
        f"samples={len(samples)} views={views_per_sample} total={len(loader.dataset)} augment={augment}"
    )
    features = []
    labels = []
    for index, batch in enumerate(loader, start=1):
        images = [item[0] for item in batch]
        labels.extend(item[1] for item in batch)
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model(**inputs)
        embedding = get_image_embedding(outputs)
        embedding = torch.nn.functional.normalize(embedding.float(), dim=1)
        features.append(embedding.cpu().numpy())
        if index % 25 == 0:
            print(f"batches {index}/{len(loader)}")
    return np.vstack(features), np.asarray(labels, dtype=np.int32)


def augment_image(image: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.5:
        image = ImageOps.mirror(image)

    width, height = image.size
    crop_scale = rng.uniform(0.88, 1.0)
    crop_width = max(1, int(width * crop_scale))
    crop_height = max(1, int(height * crop_scale))
    left = rng.randint(0, max(width - crop_width, 0))
    top = rng.randint(0, max(height - crop_height, 0))
    image = image.crop((left, top, left + crop_width, top + crop_height)).resize((width, height), Image.Resampling.BICUBIC)

    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.85, 1.15))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.85, 1.15))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.85, 1.15))
    image = ImageEnhance.Sharpness(image).enhance(rng.uniform(0.85, 1.25))

    if rng.random() < 0.25:
        image = image.rotate(rng.uniform(-5.0, 5.0), resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0))
    return image


def get_image_embedding(outputs) -> torch.Tensor:
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state.mean(dim=1)
    raise ValueError("Could not find image embedding in model output.")


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


def write_metrics(path: Path, metrics: dict[str, float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    main()

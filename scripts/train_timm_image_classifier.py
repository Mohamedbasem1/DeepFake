from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int


class ImageDataset(Dataset):
    def __init__(self, samples: list[Sample], transform) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
        return self.transform(image), torch.tensor(sample.label, dtype=torch.float32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune a timm image classifier for real/fake images.")
    parser.add_argument("--train-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-name", default="convnext_base")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--limit-per-class", type=int)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--no-balanced-sampler", action="store_true")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--metrics-output", type=Path)
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

    train_loader = build_loader(
        train_samples,
        build_train_transform(args.image_size),
        args.batch_size,
        args.num_workers,
        balanced=not args.no_balanced_sampler,
    )
    val_loader = build_loader(
        val_samples,
        build_eval_transform(args.image_size),
        args.batch_size,
        args.num_workers,
        balanced=False,
        shuffle=False,
    )

    model = timm.create_model(args.model_name, pretrained=args.pretrained, num_classes=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    amp_enabled = args.use_amp and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    metrics_path = args.metrics_output or args.output.with_suffix(".metrics.csv")
    best_f1 = -1.0
    best_payload = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, amp_enabled)
        val_loss, val_scores, val_labels = evaluate(model, val_loader, criterion, device, amp_enabled)
        metrics = find_best_threshold(val_scores, val_labels)
        metrics["epoch"] = epoch
        metrics["train_loss"] = train_loss
        metrics["val_loss"] = val_loss
        write_metrics(metrics_path, metrics, append=epoch > 1)

        print(
            f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"acc={metrics['accuracy']:.4f} p={metrics['precision']:.4f} "
            f"r={metrics['recall']:.4f} f1={metrics['f1']:.4f} threshold={metrics['threshold']:.4f}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_payload = {
                "model_name": args.model_name,
                "image_size": args.image_size,
                "state_dict": model.state_dict(),
                "threshold": float(metrics["threshold"]),
                "metrics": metrics,
                "mean": IMAGENET_MEAN,
                "std": IMAGENET_STD,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_payload, args.output)
            print(f"saved best checkpoint: {args.output}")

    if best_payload is None:
        raise RuntimeError("Training finished without a checkpoint.")
    print(f"best_f1={best_f1:.4f}")
    print(f"metrics: {metrics_path}")


def train_one_epoch(model, loader, optimizer, criterion, scaler, device: str, amp_enabled: bool) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).view(-1, 1)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type_from_device(device), enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = images.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_items += batch_size
    return total_loss / max(1, total_items)


@torch.no_grad()
def evaluate(model, loader, criterion, device: str, amp_enabled: bool):
    model.eval()
    total_loss = 0.0
    total_items = 0
    scores = []
    labels_out = []
    for images, labels in tqdm(loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).view(-1, 1)
        with torch.amp.autocast(device_type_from_device(device), enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)
        batch_size = images.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_items += batch_size
        scores.extend(torch.sigmoid(logits.float()).view(-1).cpu().tolist())
        labels_out.extend(labels.view(-1).cpu().int().tolist())
    return total_loss / max(1, total_items), np.asarray(scores, dtype=np.float32), np.asarray(labels_out, dtype=np.int32)


def build_loader(
    samples: list[Sample],
    transform,
    batch_size: int,
    num_workers: int,
    balanced: bool,
    shuffle: bool = True,
) -> DataLoader:
    sampler = None
    if balanced:
        labels = [sample.label for sample in samples]
        counts = {label: max(1, labels.count(label)) for label in {0, 1}}
        weights = [1.0 / counts[label] for label in labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(samples), replacement=True)
        shuffle = False
    return DataLoader(
        ImageDataset(samples, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )


def build_train_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.72, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.18, hue=0.03),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.15),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(image_size: int):
    resize_size = int(round(image_size * 1.15))
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def find_best_threshold(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    best = metrics_at_threshold(scores, labels, 0.5)
    for threshold in np.linspace(0.05, 0.95, 181):
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
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    accuracy = (tp + tn) / max(1, len(labels))
    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
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


def write_metrics(path: Path, metrics: dict[str, float], append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "tp",
        "tn",
        "fp",
        "fn",
    ]
    with path.open("a" if append else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not append:
            writer.writeheader()
        writer.writerow({key: metrics[key] for key in fieldnames})


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


def device_type_from_device(device: str) -> str:
    return "cuda" if device.startswith("cuda") else "cpu"


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int


class ImageFolderDataset(Dataset):
    def __init__(self, samples: list[Sample], transform) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = self.transform(image)
        return image, torch.tensor(float(sample.label), dtype=torch.float32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a ResNeXt frame classifier inspired by modelTraining.ipynb.")
    parser.add_argument("--train-root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("models/resnext_frame_classifier.pt"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--limit-per-class", type=int)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
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

    train_loader = DataLoader(
        ImageFolderDataset(train_samples, build_train_transform(args.image_size)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
    )
    val_loader = DataLoader(
        ImageFolderDataset(val_samples, build_eval_transform(args.image_size)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
    )

    model = build_model(args.freeze_backbone).to(device)
    optimizer = build_optimizer(model, args.lr, args.head_lr, args.weight_decay, args.freeze_backbone)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=args.use_amp and device.startswith("cuda"))

    best_f1 = -1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output.with_suffix(".metrics.csv")
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "val_loss", "accuracy", "precision", "recall", "f1"],
        )
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, scaler, device, args.use_amp)
            metrics = evaluate(model, val_loader, loss_fn, device)
            writer.writerow({"epoch": epoch, "train_loss": train_loss, **metrics})
            handle.flush()
            print(
                f"epoch={epoch} train_loss={train_loss:.4f} val_loss={metrics['val_loss']:.4f} "
                f"acc={metrics['accuracy']:.4f} p={metrics['precision']:.4f} "
                f"r={metrics['recall']:.4f} f1={metrics['f1']:.4f}"
            )
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                save_checkpoint(args.output, model, args, epoch, metrics)
                print(f"saved best checkpoint: {args.output}")
            save_checkpoint(args.output.with_name(args.output.stem + "_last.pt"), model, args, epoch, metrics)
    print(f"metrics: {metrics_path}")


def build_model(freeze_backbone: bool) -> nn.Module:
    weights = models.ResNeXt50_32X4D_Weights.DEFAULT
    model = models.resnext50_32x4d(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    if freeze_backbone:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("fc.")
    return model


def build_optimizer(model: nn.Module, lr: float, head_lr: float, weight_decay: float, freeze_backbone: bool):
    if freeze_backbone:
        return torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=head_lr, weight_decay=weight_decay)

    head_params = []
    backbone_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("fc."):
            head_params.append(parameter)
        else:
            backbone_params.append(parameter)
    return torch.optim.AdamW(
        [{"params": backbone_params, "lr": lr}, {"params": head_params, "lr": head_lr}],
        weight_decay=weight_decay,
    )


def build_train_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.82, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def build_eval_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize(round(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


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


def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device: str, use_amp: bool) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp and device.startswith("cuda")):
            logits = model(images).flatten()
            loss = loss_fn(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach().cpu()) * images.size(0)
        total += images.size(0)
    return total_loss / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device: str) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    y_true = []
    y_pred = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images).flatten()
        loss = loss_fn(logits, labels)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        y_true.extend(labels.long().cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        total_loss += float(loss.detach().cpu()) * images.size(0)
        total += images.size(0)
    return {
        "val_loss": total_loss / max(total, 1),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def save_checkpoint(path: Path, model, args, epoch: int, metrics: dict[str, float]) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "image_size": args.image_size,
            "architecture": "resnext50_32x4d_frame",
        },
        path,
    )


def print_counts(name: str, samples: list[Sample]) -> None:
    real = sum(sample.label == 0 for sample in samples)
    fake = sum(sample.label == 1 for sample in samples)
    print(f"{name}: real={real} fake={fake}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    main()


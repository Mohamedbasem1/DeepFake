from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deepfake_detector.models import build_commfor_transform, get_commfor_model_class


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageSample:
    path: Path
    label: int


class FolderImageDataset(Dataset):
    def __init__(self, samples: list[ImageSample], transform) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            tensor = self.transform(image)
        return tensor, torch.tensor(float(sample.label), dtype=torch.float32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune Community Forensics on labeled image folders.")
    parser.add_argument(
        "--train-root",
        required=True,
        type=Path,
        help="Root containing real/fake folders anywhere below it.",
    )
    parser.add_argument("--output", type=Path, default=Path("models/commfor_finetuned.pt"))
    parser.add_argument("--model-repo", default="OwensLab/commfor-model-384")
    parser.add_argument("--model-size", default="small", choices=["small", "tiny"])
    parser.add_argument("--input-size", type=int, default=384, choices=[224, 384])
    parser.add_argument("--patch-size", type=int, default=16, choices=[16, 32])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--limit-per-class", type=int, help="Optional cap for quick experiments.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    train_samples, val_samples = split_samples(
        discover_samples(args.train_root, args.limit_per_class),
        val_frac=args.val_frac,
        seed=args.seed,
    )
    if not train_samples or not val_samples:
        raise SystemExit("Need non-empty train and validation splits.")

    print(f"train={len(train_samples)} val={len(val_samples)}")
    print_class_counts("train", train_samples)
    print_class_counts("val", val_samples)

    try:
        import timm
        from huggingface_hub import PyTorchModelHubMixin
        from torchvision import transforms
    except ImportError as exc:
        raise RuntimeError("Install requirements-lightning.txt before fine-tuning.") from exc

    transform = build_commfor_transform(transforms, args.input_size)
    train_loader = DataLoader(
        FolderImageDataset(train_samples, transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        FolderImageDataset(val_samples, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=False,
    )

    model_class = get_commfor_model_class(torch, timm, PyTorchModelHubMixin)
    model = model_class.from_pretrained(
        args.model_repo,
        model_size=args.model_size,
        input_size=args.input_size,
        patch_size=args.patch_size,
        freeze_backbone=args.freeze_backbone,
        device=device,
        dtype=torch.float32,
    ).to(device)

    optimizer = build_optimizer(model, args.lr, args.head_lr, args.weight_decay, args.freeze_backbone)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp and device.startswith("cuda"))

    best_f1 = -1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    history_path = args.output.with_suffix(".metrics.csv")
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "val_loss", "accuracy", "precision", "recall", "f1"],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, scaler, device, args.use_amp)
            metrics = evaluate(model, val_loader, loss_fn, device)
            row = {"epoch": epoch, "train_loss": train_loss, **metrics}
            writer.writerow(row)
            handle.flush()
            print(
                f"epoch={epoch} train_loss={train_loss:.4f} val_loss={metrics['val_loss']:.4f} "
                f"acc={metrics['accuracy']:.4f} p={metrics['precision']:.4f} "
                f"r={metrics['recall']:.4f} f1={metrics['f1']:.4f}"
            )

            last_path = args.output.with_name(args.output.stem + "_last.pt")
            save_checkpoint(last_path, model, args, epoch, metrics)
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                save_checkpoint(args.output, model, args, epoch, metrics)
                print(f"saved best checkpoint: {args.output}")

    print(f"metrics: {history_path}")


def discover_samples(root: Path, limit_per_class: int | None) -> list[ImageSample]:
    by_label = {0: [], 1: []}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = label_from_path(path)
        if label is None:
            continue
        by_label[label].append(ImageSample(path=path, label=label))

    samples: list[ImageSample] = []
    for label, label_samples in by_label.items():
        if limit_per_class:
            label_samples = label_samples[:limit_per_class]
        samples.extend(label_samples)

    if not samples:
        raise SystemExit(f"No labeled images found under {root}. Expected parent folders named real or fake.")
    return samples


def label_from_path(path: Path) -> int | None:
    parts = [part.lower() for part in path.parts]
    if "fake" in parts or "deepfake" in parts:
        return 1
    if "real" in parts or "original" in parts or "authentic" in parts:
        return 0
    return None


def split_samples(samples: list[ImageSample], val_frac: float, seed: int) -> tuple[list[ImageSample], list[ImageSample]]:
    rng = random.Random(seed)
    train: list[ImageSample] = []
    val: list[ImageSample] = []
    for label in [0, 1]:
        label_samples = [sample for sample in samples if sample.label == label]
        rng.shuffle(label_samples)
        val_count = max(1, round(len(label_samples) * val_frac))
        val.extend(label_samples[:val_count])
        train.extend(label_samples[val_count:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def build_optimizer(model, lr: float, head_lr: float, weight_decay: float, freeze_backbone: bool):
    if freeze_backbone:
        return torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=head_lr, weight_decay=weight_decay)

    head_params = []
    backbone_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("vit.head"):
            head_params.append(parameter)
        else:
            backbone_params.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )


def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device: str, use_amp: bool) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp and device.startswith("cuda")):
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
    tp = fp = tn = fn = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images).flatten()
        loss = loss_fn(logits, labels)
        probs = torch.sigmoid(logits)
        preds = probs >= 0.5
        truth = labels >= 0.5
        tp += int((preds & truth).sum().cpu())
        fp += int((preds & ~truth).sum().cpu())
        tn += int((~preds & ~truth).sum().cpu())
        fn += int((~preds & truth).sum().cpu())
        total_loss += float(loss.detach().cpu()) * images.size(0)
        total += images.size(0)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "val_loss": total_loss / max(total, 1),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def save_checkpoint(path: Path, model, args, epoch: int, metrics: dict[str, float]) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "model_repo": args.model_repo,
            "model_size": args.model_size,
            "input_size": args.input_size,
            "patch_size": args.patch_size,
        },
        path,
    )


def print_class_counts(name: str, samples: list[ImageSample]) -> None:
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


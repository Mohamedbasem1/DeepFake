from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


class ImagePathDataset(Dataset):
    def __init__(self, paths: list[Path], transform) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = self.transform(image)
        return path.name, image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict ImageCLEF scores with ResNeXt frame classifier.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/resnext_frame_classifier.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/images_detection_scores_resnext_frame.csv"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    checkpoint = torch.load(args.model, map_location="cpu")
    image_size = int(checkpoint.get("image_size", 224))
    device = resolve_device(args.device)
    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    paths = discover_images(args.input)
    loader = DataLoader(
        ImagePathDataset(paths, build_eval_transform(image_size)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["full_secret_name", "score", "prediction"])
        writer.writeheader()
        with torch.no_grad():
            for index, (names, images) in enumerate(loader, start=1):
                images = images.to(device, non_blocking=True)
                scores = torch.sigmoid(model(images).flatten()).cpu().tolist()
                for name, score in zip(names, scores):
                    writer.writerow(
                        {
                            "full_secret_name": name,
                            "score": f"{score:.6f}",
                            "prediction": "1" if score >= args.threshold else "0",
                        }
                    )
                if index % 25 == 0:
                    print(f"batches {index}/{len(loader)}")
    print(f"wrote {len(paths)} predictions to {args.output}")


def build_model() -> nn.Module:
    model = models.resnext50_32x4d(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    return model


def build_eval_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize(round(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def discover_images(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in IMAGE_EXTENSIONS else []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

import timm
import torch
import numpy as np
from PIL import Image, ImageChops, ImageFilter, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


class PathDataset(Dataset):
    def __init__(self, paths: list[Path], transform, tta_hflip: bool) -> None:
        self.paths = paths
        self.transform = transform
        self.tta_hflip = tta_hflip

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
        image_tensor = self.transform(image)
        if self.tta_hflip:
            flipped_tensor = self.transform(ImageOps.mirror(image))
            return path.name, image_tensor, flipped_tensor
        return path.name, image_tensor, image_tensor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict ImageCLEF scores with a timm real/fake checkpoint.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--tta-hflip", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_name = checkpoint["model_name"]
    image_size = int(checkpoint["image_size"])
    input_mode = checkpoint.get("input_mode", "rgb")
    threshold = float(args.threshold if args.threshold is not None else checkpoint.get("threshold", 0.5))

    model = timm.create_model(model_name, pretrained=False, num_classes=1)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device).eval()

    paths = discover_images(args.input)
    if not paths:
        raise FileNotFoundError(f"No images found under {args.input}")

    loader = DataLoader(
        PathDataset(paths, build_eval_transform(image_size, checkpoint, input_mode), args.tta_hflip),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    amp_enabled = args.use_amp and device.startswith("cuda")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["full_secret_name", "score", "prediction"])
        writer.writeheader()
        for names, images, flipped_images in tqdm(loader, desc="predict"):
            images = images.to(device, non_blocking=True)
            flipped_images = flipped_images.to(device, non_blocking=True)
            with torch.no_grad(), torch.amp.autocast(device_type_from_device(device), enabled=amp_enabled):
                logits = model(images)
                scores = torch.sigmoid(logits.float()).view(-1)
                if args.tta_hflip:
                    flipped_logits = model(flipped_images)
                    flipped_scores = torch.sigmoid(flipped_logits.float()).view(-1)
                    scores = (scores + flipped_scores) / 2
            for name, score in zip(names, scores.detach().cpu().tolist()):
                writer.writerow(
                    {
                        "full_secret_name": name,
                        "score": f"{float(score):.6f}",
                        "prediction": "1" if score >= threshold else "0",
                    }
                )

    print(f"wrote {len(paths)} predictions to {args.output}")
    print(f"model: {model_name}")
    print(f"input_mode: {input_mode}")
    print(f"threshold: {threshold:.6f}")


def build_eval_transform(image_size: int, checkpoint, input_mode: str):
    mean = tuple(checkpoint.get("mean", (0.485, 0.456, 0.406)))
    std = tuple(checkpoint.get("std", (0.229, 0.224, 0.225)))
    resize_size = int(round(image_size * 1.15))
    return transforms.Compose(
        [
            transforms.Lambda(lambda image: preprocess_image(image, input_mode)),
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def preprocess_image(image: Image.Image, input_mode: str) -> Image.Image:
    image = image.convert("RGB")
    if input_mode == "rgb":
        return image
    if input_mode == "ycbcr":
        return image.convert("YCbCr").convert("RGB")
    if input_mode == "ela":
        return ela_image(image)
    if input_mode == "edges":
        return edge_image(image)
    if input_mode == "forensic":
        y, _, _ = image.convert("YCbCr").split()
        edge = edge_image(image).convert("L")
        ela = ela_image(image).convert("L")
        return Image.merge("RGB", (y, edge, ela))
    raise ValueError(f"Unsupported input mode: {input_mode}")


def ela_image(image: Image.Image, quality: int = 90, scale: float = 12.0) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer).convert("RGB")
    diff = ImageChops.difference(image, compressed)
    arr = np.asarray(diff, dtype=np.float32) * scale
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def edge_image(image: Image.Image) -> Image.Image:
    gray = image.convert("L")
    try:
        import cv2

        arr = np.asarray(gray)
        edges = cv2.Canny(arr, 80, 160)
        return Image.fromarray(edges, mode="L").convert("RGB")
    except Exception:
        return gray.filter(ImageFilter.FIND_EDGES).convert("RGB")


def discover_images(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in IMAGE_EXTENSIONS else []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def device_type_from_device(device: str) -> str:
    return "cuda" if device.startswith("cuda") else "cpu"


if __name__ == "__main__":
    main()

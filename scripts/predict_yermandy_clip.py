from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


class ImagePathDataset(Dataset):
    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
        return path, image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict ImageCLEF scores with yermandy/deepfake-detection CLIP TorchScript model."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Image file or directory containing ImageCLEF detection images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/images_detection_scores_yermandy_clip.csv"),
        help="CSV with full_secret_name,score,prediction.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--repo-id", default="yermandy/deepfake-detection")
    parser.add_argument("--filename", default="model.torchscript")
    parser.add_argument("--model-cache-dir", type=Path, default=Path("weights/yermandy_clip"))
    parser.add_argument(
        "--processor",
        default="openai/clip-vit-large-patch14",
        help="CLIP image processor used by the official TorchScript example.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="Inference dtype. auto uses bfloat16 on CUDA and float32 on CPU.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
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
    paths = discover_images(args.input)
    if not paths:
        raise FileNotFoundError(f"No images found under {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(
        ImagePathDataset(paths),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: batch,
    )

    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["full_secret_name", "score", "prediction"])
        writer.writeheader()
        for batch_index, batch in enumerate(loader, start=1):
            batch_paths = [item[0] for item in batch]
            images = [item[1] for item in batch]
            scores = predict_fake_scores(model, processor, images, device, dtype)
            for path, score in zip(batch_paths, scores):
                writer.writerow(
                    {
                        "full_secret_name": path.name,
                        "score": f"{score:.6f}",
                        "prediction": "1" if score >= args.threshold else "0",
                    }
                )
            if batch_index % 25 == 0:
                print(f"batches {batch_index}/{len(loader)}")

    print(f"wrote {len(paths)} predictions to {args.output}")
    print(f"threshold: {args.threshold}")


@torch.no_grad()
def predict_fake_scores(model, processor, images, device: str, dtype: torch.dtype) -> list[float]:
    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    if dtype is not torch.float32:
        pixel_values = pixel_values.to(dtype)

    if device.startswith("cuda"):
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype is not torch.float32):
            logits = model(pixel_values)
    else:
        logits = model(pixel_values)

    probabilities = logits.float().softmax(dim=1)
    return probabilities[:, 1].detach().cpu().tolist()


def discover_images(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in IMAGE_EXTENSIONS else []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(requested: str, device: str) -> torch.dtype:
    if requested == "float32" or not device.startswith("cuda"):
        return torch.float32
    if requested == "float16":
        return torch.float16
    return torch.bfloat16


if __name__ == "__main__":
    main()

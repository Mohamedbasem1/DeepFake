from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import joblib
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


class PathDataset(Dataset):
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
    parser = argparse.ArgumentParser(description="Predict ImageCLEF scores with a frozen embedding classifier.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/siglip_embedding_classifier.joblib"))
    parser.add_argument("--output", type=Path, default=Path("outputs/images_detection_scores_siglip_embedding.csv"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = joblib.load(args.model)
    classifier = payload["classifier"]
    model_id = payload["model_id"]
    device = resolve_device(args.device)

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()

    paths = discover_images(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(
        PathDataset(paths),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: batch,
    )

    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["full_secret_name", "score", "prediction"])
        writer.writeheader()
        for index, batch in enumerate(loader, start=1):
            batch_paths = [item[0] for item in batch]
            images = [item[1] for item in batch]
            embeddings = extract_embeddings(images, processor, model, device)
            scores = classifier.predict_proba(embeddings)[:, 1]
            for path, score in zip(batch_paths, scores):
                writer.writerow(
                    {
                        "full_secret_name": path.name,
                        "score": f"{float(score):.6f}",
                        "prediction": "1" if score >= args.threshold else "0",
                    }
                )
            if index % 25 == 0:
                print(f"batches {index}/{len(loader)}")
    print(f"wrote {len(paths)} predictions to {args.output}")


@torch.no_grad()
def extract_embeddings(images, processor, model, device: str):
    inputs = processor(images=images, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = model(**inputs)
    embedding = get_image_embedding(outputs)
    embedding = torch.nn.functional.normalize(embedding.float(), dim=1)
    return embedding.cpu().numpy()


def get_image_embedding(outputs) -> torch.Tensor:
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state.mean(dim=1)
    raise ValueError("Could not find image embedding in model output.")


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


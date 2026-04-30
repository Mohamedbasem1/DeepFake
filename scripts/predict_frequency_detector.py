from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deepfake_detector.frequency_features import extract_frequency_features


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict ImageCLEF scores with a trained frequency detector.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/frequency_detector.joblib"))
    parser.add_argument("--output", type=Path, default=Path("outputs/images_detection_scores_frequency.csv"))
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = joblib.load(args.model)
    model = payload["model"]
    image_size = int(payload["image_size"])
    radial_bins = int(payload["radial_bins"])

    paths = discover_images(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["full_secret_name", "score", "prediction"])
        writer.writeheader()
        for index, path in enumerate(paths, start=1):
            if index % 1000 == 0:
                print(f"predict {index}/{len(paths)}")
            features = extract_frequency_features(path, image_size=image_size, radial_bins=radial_bins)
            score = float(model.predict_proba([features])[0, 1])
            writer.writerow(
                {
                    "full_secret_name": path.name,
                    "score": f"{score:.6f}",
                    "prediction": "1" if score >= args.threshold else "0",
                }
            )
    print(f"wrote {len(paths)} predictions to {args.output}")


def discover_images(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in IMAGE_EXTENSIONS else []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


if __name__ == "__main__":
    main()


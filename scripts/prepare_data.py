from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract ImageCLEF detection data safely.")
    parser.add_argument(
        "--zip",
        type=Path,
        default=Path("data/ImageCLEF2026-DeepFakeDetection-Tes.zip"),
        help="Path to the detection zip file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/imageclef_detection"),
        help="Directory to extract into.",
    )
    parser.add_argument(
        "--prefix",
        default="Data/Images_Detection/",
        help="Zip member prefix to extract. Use Data/Audio_Detection/ for audio.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    extract_prefix(args.zip, args.output, args.prefix)


def extract_prefix(zip_path: Path, output_dir: Path, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_root = output_dir.resolve()
    extracted = 0
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            if member.is_dir() or not member.filename.startswith(prefix):
                continue
            target = output_dir / member.filename
            resolved = target.resolve()
            if not resolved.is_relative_to(output_root):
                raise RuntimeError(f"Unsafe zip member path: {member.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as destination:
                destination.write(source.read())
            extracted += 1
    print(f"Extracted {extracted} files from {zip_path} to {output_dir}")


if __name__ == "__main__":
    main()


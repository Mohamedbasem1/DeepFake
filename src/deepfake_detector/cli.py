from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .predict import run_inference
from .submission import write_predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deepfake detection inference.")
    parser.add_argument("--input", required=True, type=Path, help="Input media file or directory.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/predictions.csv"),
        help="Output CSV path.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    predictions = run_inference(args.input, config)
    write_predictions(args.output, predictions, config.submission)
    print(f"Wrote {len(predictions)} predictions to {args.output}")


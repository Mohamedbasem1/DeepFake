from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print score distribution for a scored CSV.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--score-column", default="score")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scores = []
    with args.input.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            scores.append(float(row[args.score_column]))
    values = np.asarray(scores, dtype=np.float64)
    print(f"rows: {len(values)}")
    print(f"min: {values.min():.6f}")
    print(f"max: {values.max():.6f}")
    print(f"mean: {values.mean():.6f}")
    for quantile in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
        print(f"p{int(quantile * 100):02d}: {np.quantile(values, quantile):.6f}")
    print(f">=0.5: {int((values >= 0.5).sum())}")


if __name__ == "__main__":
    main()


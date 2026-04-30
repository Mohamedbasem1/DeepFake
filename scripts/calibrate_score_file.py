from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate a scored submission CSV.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--method", choices=["rank", "minmax"], default="rank")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--id-column", default="full_secret_name")
    parser.add_argument("--score-column", default="score")
    parser.add_argument("--label-column", default="prediction")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = read_rows(args.input, args.id_column, args.score_column)
    raw_scores = np.asarray([row["score"] for row in rows], dtype=np.float64)

    if args.method == "rank":
        calibrated = rank_percentiles(raw_scores)
    else:
        calibrated = minmax(raw_scores)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[args.id_column, args.score_column, args.label_column])
        writer.writeheader()
        for row, score in zip(rows, calibrated):
            writer.writerow(
                {
                    args.id_column: row["id"],
                    args.score_column: f"{score:.6f}",
                    args.label_column: "1" if score >= args.threshold else "0",
                }
            )

    print(f"input: {args.input}")
    print(f"output: {args.output}")
    print(f"method: {args.method}")
    print(f"raw_min: {raw_scores.min():.6f}")
    print(f"raw_max: {raw_scores.max():.6f}")
    print(f"raw_mean: {raw_scores.mean():.6f}")
    print(f"raw_p50: {np.quantile(raw_scores, 0.50):.6f}")
    print(f"raw_p90: {np.quantile(raw_scores, 0.90):.6f}")
    print(f"raw_p95: {np.quantile(raw_scores, 0.95):.6f}")
    print(f"raw_p99: {np.quantile(raw_scores, 0.99):.6f}")
    print(f"predicted_fake_after_calibration: {int((calibrated >= args.threshold).sum())}")


def read_rows(path: Path, id_column: str, score_column: str) -> list[dict[str, str | float]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {id_column, score_column}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        rows = []
        for row_number, row in enumerate(reader, start=2):
            item_id = (row.get(id_column) or "").strip()
            score_text = (row.get(score_column) or "").strip()
            if not item_id:
                raise ValueError(f"{path}:{row_number} has an empty id")
            rows.append({"id": item_id, "score": float(score_text)})
        return rows


def rank_percentiles(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(scores), dtype=np.float64)
    if len(scores) <= 1:
        return np.zeros_like(scores, dtype=np.float64)
    return ranks / float(len(scores) - 1)


def minmax(scores: np.ndarray) -> np.ndarray:
    low = float(scores.min())
    high = float(scores.max())
    if high <= low:
        return np.zeros_like(scores, dtype=np.float64)
    return (scores - low) / (high - low)


if __name__ == "__main__":
    main()


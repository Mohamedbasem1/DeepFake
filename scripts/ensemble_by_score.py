from __future__ import annotations

import argparse
import csv
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create an ensemble submission by trusting a scored file above a threshold."
    )
    parser.add_argument("--base", required=True, type=Path, help="Fallback submission CSV.")
    parser.add_argument("--scored", required=True, type=Path, help="CSV with score and prediction columns.")
    parser.add_argument("--output", required=True, type=Path, help="Output submission CSV.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--id-column", default="full_secret_name")
    parser.add_argument("--label-column", default="prediction")
    parser.add_argument("--score-column", default="score")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base_rows = read_rows(args.base, args.id_column, args.label_column)
    scored_rows = read_rows(args.scored, args.id_column, args.label_column, args.score_column)

    output_rows = []
    used_scored = 0
    used_base = 0
    for item_id, base_row in base_rows.items():
        if item_id not in scored_rows:
            raise ValueError(f"{item_id} exists in base file but not scored file")

        scored_row = scored_rows[item_id]
        score = float(scored_row[args.score_column])
        if score > args.threshold:
            prediction = scored_row[args.label_column]
            used_scored += 1
        else:
            prediction = base_row[args.label_column]
            used_base += 1

        output_rows.append({args.id_column: item_id, args.label_column: prediction})

    extra = set(scored_rows) - set(base_rows)
    if extra:
        raise ValueError(f"Scored file contains {len(extra)} ids not present in base file")

    write_submission(args.output, output_rows, args.id_column, args.label_column)
    print(f"wrote: {args.output}")
    print(f"rows: {len(output_rows)}")
    print(f"used_scored: {used_scored}")
    print(f"used_base: {used_base}")


def read_rows(
    path: Path,
    id_column: str,
    label_column: str,
    score_column: str | None = None,
) -> dict[str, dict[str, str]]:
    required = {id_column, label_column}
    if score_column:
        required.add(score_column)

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        rows = {}
        for row_number, row in enumerate(reader, start=2):
            item_id = (row.get(id_column) or "").strip()
            if not item_id:
                raise ValueError(f"{path}:{row_number} has an empty id")
            if item_id in rows:
                raise ValueError(f"{path}:{row_number} duplicates id {item_id}")
            rows[item_id] = {key: (value or "").strip() for key, value in row.items()}
        return rows


def write_submission(path: Path, rows: list[dict[str, str]], id_column: str, label_column: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[id_column, label_column])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()


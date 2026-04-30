from __future__ import annotations

import argparse
import csv
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare agreement between two submission CSV files.")
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--id-column", default="full_secret_name")
    parser.add_argument("--label-column", default="prediction")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    first = read_submission(args.first, args.id_column, args.label_column)
    second = read_submission(args.second, args.id_column, args.label_column)

    common_ids = set(first) & set(second)
    agree = sum(first[item_id] == second[item_id] for item_id in common_ids)
    disagree = len(common_ids) - agree
    disagree_percent = 100.0 * disagree / len(common_ids) if common_ids else 0.0

    print(f"common: {len(common_ids)}")
    print(f"agree: {agree}")
    print(f"disagree: {disagree}")
    print(f"disagree_percent: {disagree_percent:.2f}")


def read_submission(path: Path, id_column: str, label_column: str) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = {id_column, label_column} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        rows = {}
        for row_number, row in enumerate(reader, start=2):
            item_id = (row.get(id_column) or "").strip()
            label = (row.get(label_column) or "").strip()
            if not item_id:
                raise ValueError(f"{path}:{row_number} has an empty id")
            if item_id in rows:
                raise ValueError(f"{path}:{row_number} duplicates id {item_id}")
            rows[item_id] = label
        return rows


if __name__ == "__main__":
    main()


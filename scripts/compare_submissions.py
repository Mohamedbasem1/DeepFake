from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two ImageCLEF submission CSV files.")
    parser.add_argument("first", type=Path, help="First CSV, e.g. pretrained submission.")
    parser.add_argument("second", type=Path, help="Second CSV, e.g. fine-tuned submission.")
    parser.add_argument("--id-column", default="full_secret_name")
    parser.add_argument("--label-column", default="prediction")
    parser.add_argument("--changed-output", type=Path, help="Optional CSV path for rows where predictions differ.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    first = read_submission(args.first, args.id_column, args.label_column)
    second = read_submission(args.second, args.id_column, args.label_column)

    first_ids = set(first)
    second_ids = set(second)
    common_ids = sorted(first_ids & second_ids)
    missing_from_second = sorted(first_ids - second_ids)
    extra_in_second = sorted(second_ids - first_ids)

    changes = []
    agreement = Counter()
    for item_id in common_ids:
        first_label = first[item_id]
        second_label = second[item_id]
        agreement[(first_label, second_label)] += 1
        if first_label != second_label:
            changes.append((item_id, first_label, second_label))

    print(f"first rows: {len(first)}")
    print(f"second rows: {len(second)}")
    print(f"common rows: {len(common_ids)}")
    print(f"missing from second: {len(missing_from_second)}")
    print(f"extra in second: {len(extra_in_second)}")
    print(f"changed predictions: {len(changes)}")
    if common_ids:
        print(f"changed percent: {100 * len(changes) / len(common_ids):.2f}%")

    print("\nFirst label counts:")
    print_counts(Counter(first.values()))
    print("\nSecond label counts:")
    print_counts(Counter(second.values()))

    print("\nAgreement table: first -> second")
    for (first_label, second_label), count in sorted(agreement.items()):
        print(f"{first_label} -> {second_label}: {count}")

    if missing_from_second:
        print("\nFirst 10 missing from second:")
        for item_id in missing_from_second[:10]:
            print(item_id)

    if extra_in_second:
        print("\nFirst 10 extra in second:")
        for item_id in extra_in_second[:10]:
            print(item_id)

    if args.changed_output:
        write_changes(args.changed_output, changes, args.id_column)
        print(f"\nWrote changed rows to {args.changed_output}")


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


def print_counts(counts: Counter[str]) -> None:
    for label, count in sorted(counts.items()):
        print(f"{label}: {count}")


def write_changes(path: Path, changes: list[tuple[str, str, str]], id_column: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[id_column, "first_prediction", "second_prediction"])
        writer.writeheader()
        for item_id, first_label, second_label in changes:
            writer.writerow(
                {
                    id_column: item_id,
                    "first_prediction": first_label,
                    "second_prediction": second_label,
                }
            )


if __name__ == "__main__":
    main()


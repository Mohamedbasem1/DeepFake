from __future__ import annotations

import argparse
import csv
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge two scored submissions by choosing the higher score per file.")
    parser.add_argument("--first", required=True, type=Path)
    parser.add_argument("--second", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Official two-column submission output.")
    parser.add_argument("--debug-output", type=Path, help="Optional debug CSV with both scores and selected source.")
    parser.add_argument("--id-column", default="full_secret_name")
    parser.add_argument("--score-column", default="score")
    parser.add_argument("--label-column", default="prediction")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    first = read_scored(args.first, args.id_column, args.score_column, args.label_column)
    second = read_scored(args.second, args.id_column, args.score_column, args.label_column)

    first_ids = set(first)
    second_ids = set(second)
    if first_ids != second_ids:
        missing_second = sorted(first_ids - second_ids)
        missing_first = sorted(second_ids - first_ids)
        raise ValueError(
            "Files do not contain the same ids.\n"
            f"missing from second: {len(missing_second)} {missing_second[:5]}\n"
            f"missing from first: {len(missing_first)} {missing_first[:5]}"
        )

    output_rows = []
    debug_rows = []
    first_wins = 0
    second_wins = 0
    ties = 0

    for item_id in first:
        first_row = first[item_id]
        second_row = second[item_id]
        first_score = float(first_row[args.score_column])
        second_score = float(second_row[args.score_column])

        if second_score > first_score:
            selected = second_row
            selected_source = "second"
            second_wins += 1
        else:
            selected = first_row
            selected_source = "first"
            first_wins += 1
            if second_score == first_score:
                ties += 1

        output_rows.append(
            {
                args.id_column: item_id,
                args.label_column: selected[args.label_column],
            }
        )
        debug_rows.append(
            {
                args.id_column: item_id,
                "first_score": f"{first_score:.6f}",
                "first_prediction": first_row[args.label_column],
                "second_score": f"{second_score:.6f}",
                "second_prediction": second_row[args.label_column],
                "selected_source": selected_source,
                args.label_column: selected[args.label_column],
            }
        )

    write_csv(args.output, output_rows, [args.id_column, args.label_column])
    if args.debug_output:
        write_csv(
            args.debug_output,
            debug_rows,
            [
                args.id_column,
                "first_score",
                "first_prediction",
                "second_score",
                "second_prediction",
                "selected_source",
                args.label_column,
            ],
        )

    print(f"wrote: {args.output}")
    print(f"rows: {len(output_rows)}")
    print(f"first_wins: {first_wins}")
    print(f"second_wins: {second_wins}")
    print(f"ties: {ties}")
    if args.debug_output:
        print(f"debug: {args.debug_output}")


def read_scored(path: Path, id_column: str, score_column: str, label_column: str) -> dict[str, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {id_column, score_column, label_column}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        rows = {}
        for row_number, row in enumerate(reader, start=2):
            item_id = (row.get(id_column) or "").strip()
            score = (row.get(score_column) or "").strip()
            label = (row.get(label_column) or "").strip()
            if not item_id:
                raise ValueError(f"{path}:{row_number} has an empty id")
            if item_id in rows:
                raise ValueError(f"{path}:{row_number} duplicates id {item_id}")
            float(score)
            rows[item_id] = {
                id_column: item_id,
                score_column: score,
                label_column: label,
            }
        return rows


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()


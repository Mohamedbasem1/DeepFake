from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_INPUTS = [
    Path("outputs/upload/images_detection_submission_sky_classic_avg.csv"),
    Path("outputs/upload/images_detection_submission_efficientnet_b4_forensic.csv"),
    Path("outputs/upload/images_detection_submission_convnext_tiny_forensic.csv"),
    Path("outputs/upload/images_detection_submission_resnet50_forensic.csv"),
    Path("outputs/images_detection_scores_timm_convnext_base.csv"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge conservative submissions by voting on prediction=1."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=DEFAULT_INPUTS,
        help="CSV files with full_secret_name,prediction columns. Scored CSVs are also accepted.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/upload/images_detection_submission_conservative_union.csv"),
    )
    parser.add_argument(
        "--debug-output",
        type=Path,
        default=Path("outputs/images_detection_conservative_union_debug.csv"),
    )
    parser.add_argument(
        "--min-votes",
        type=int,
        default=1,
        help="Minimum number of models predicting 1. Default 1 means union.",
    )
    parser.add_argument("--id-column", default="full_secret_name")
    parser.add_argument("--label-column", default="prediction")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    submissions = [read_submission(path, args.id_column, args.label_column) for path in args.inputs]
    ids = validate_same_ids(submissions)

    output_rows = []
    debug_rows = []
    for item_id in ids:
        labels = [submission[item_id] for submission in submissions]
        votes = sum(label == "1" for label in labels)
        prediction = "1" if votes >= args.min_votes else "0"
        output_rows.append({args.id_column: item_id, args.label_column: prediction})

        debug_row = {
            args.id_column: item_id,
            "votes": str(votes),
            args.label_column: prediction,
        }
        for path, label in zip(args.inputs, labels):
            debug_row[Path(path).stem] = label
        debug_rows.append(debug_row)

    write_csv(args.output, output_rows, [args.id_column, args.label_column])
    write_csv(args.debug_output, debug_rows, list(debug_rows[0]))

    ones = sum(row[args.label_column] == "1" for row in output_rows)
    zeros = len(output_rows) - ones
    print(f"wrote: {args.output}")
    print(f"rows: {len(output_rows)}")
    print(f"min_votes: {args.min_votes}")
    print(f"labels: 1={ones} 0={zeros}")
    print(f"debug: {args.debug_output}")


def read_submission(path: Path, id_column: str, label_column: str) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = {id_column, label_column} - fieldnames
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        rows = {}
        for row_number, row in enumerate(reader, start=2):
            item_id = (row.get(id_column) or "").strip()
            label = (row.get(label_column) or "").strip()
            if not item_id:
                raise ValueError(f"{path}:{row_number} has empty id")
            if label not in {"0", "1"}:
                raise ValueError(f"{path}:{row_number} has invalid prediction {label!r}")
            if item_id in rows:
                raise ValueError(f"{path}:{row_number} duplicates id {item_id}")
            rows[item_id] = label
    return rows


def validate_same_ids(submissions: list[dict[str, str]]) -> list[str]:
    reference = set(submissions[0])
    for index, submission in enumerate(submissions[1:], start=2):
        ids = set(submission)
        if ids != reference:
            raise ValueError(
                f"Input {index} has different ids: "
                f"missing={len(reference - ids)} extra={len(ids - reference)}"
            )
    return list(submissions[0])


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

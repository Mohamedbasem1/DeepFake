from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and package an ImageCLEF image-detection submission for upload."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("result/images_detection_submission_ensemble.csv"),
        help="Submission CSV to validate/package.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("result/upload/images_detection_submission.csv"),
        help="Clean CSV written with official columns and sample order.",
    )
    parser.add_argument(
        "--output-zip",
        type=Path,
        default=Path("result/upload/images_detection_submission.zip"),
        help="Zip file to upload if the portal asks for zip.",
    )
    parser.add_argument(
        "--dataset-zip",
        type=Path,
        default=Path("data/ImageCLEF2026-DeepFakeDetection-Tes.zip"),
        help="Optional ImageCLEF detection zip used to read the official sample submission.",
    )
    parser.add_argument(
        "--sample-member",
        default="Data/Images_Detection_submission.csv",
        help="Sample submission path inside the dataset zip.",
    )
    parser.add_argument("--id-column", default="full_secret_name")
    parser.add_argument("--label-column", default="prediction")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = read_submission(args.input, args.id_column, args.label_column)
    sample_order = read_sample_order(args.dataset_zip, args.sample_member, args.id_column)

    if sample_order:
        validate_against_sample(rows, sample_order)
        ordered_ids = sample_order
    else:
        ordered_ids = sorted(rows)
        print("Sample submission not found; writing rows sorted by filename.")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_clean_csv(args.output_csv, rows, ordered_ids, args.id_column, args.label_column)
    write_zip(args.output_zip, args.output_csv)

    labels = count_labels(rows.values())
    print(f"input: {args.input}")
    print(f"rows: {len(rows)}")
    print(f"labels: {labels}")
    print(f"csv_ready: {args.output_csv}")
    print(f"zip_ready: {args.output_zip}")
    print("Upload one of those files on the AI4MediaBench submission page.")


def read_submission(path: Path, id_column: str, label_column: str) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != [id_column, label_column]:
            raise ValueError(
                f"{path} must have exactly these columns: {id_column},{label_column}. "
                f"Found: {reader.fieldnames}"
            )

        rows: dict[str, str] = {}
        for row_number, row in enumerate(reader, start=2):
            item_id = (row.get(id_column) or "").strip()
            label = (row.get(label_column) or "").strip()
            if not item_id:
                raise ValueError(f"{path}:{row_number} has an empty filename")
            if label not in {"0", "1"}:
                raise ValueError(f"{path}:{row_number} has invalid prediction {label!r}; expected 0 or 1")
            if item_id in rows:
                raise ValueError(f"{path}:{row_number} duplicates filename {item_id}")
            rows[item_id] = label
    return rows


def read_sample_order(dataset_zip: Path, sample_member: str, id_column: str) -> list[str] | None:
    if not dataset_zip.exists():
        return None

    with zipfile.ZipFile(dataset_zip) as archive:
        if sample_member not in archive.namelist():
            return None
        text = archive.read(sample_member).decode("utf-8-sig")

    reader = csv.DictReader(text.splitlines())
    if id_column not in (reader.fieldnames or []):
        raise ValueError(f"{sample_member} does not contain {id_column}")
    return [(row.get(id_column) or "").strip() for row in reader if (row.get(id_column) or "").strip()]


def validate_against_sample(rows: dict[str, str], sample_order: list[str]) -> None:
    sample_set = set(sample_order)
    row_set = set(rows)
    missing = sorted(sample_set - row_set)
    extra = sorted(row_set - sample_set)
    if missing or extra:
        message = [
            "Submission filenames do not match the official image sample.",
            f"missing: {len(missing)}",
            f"extra: {len(extra)}",
        ]
        if missing:
            message.append(f"first missing: {missing[:5]}")
        if extra:
            message.append(f"first extra: {extra[:5]}")
        raise ValueError("\n".join(message))


def write_clean_csv(
    path: Path,
    rows: dict[str, str],
    ordered_ids: list[str],
    id_column: str,
    label_column: str,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[id_column, label_column])
        writer.writeheader()
        for item_id in ordered_ids:
            writer.writerow({id_column: item_id, label_column: rows[item_id]})


def write_zip(path: Path, csv_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(csv_path, arcname=csv_path.name)


def count_labels(labels) -> dict[str, int]:
    counts = {"0": 0, "1": 0}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return counts


if __name__ == "__main__":
    main()


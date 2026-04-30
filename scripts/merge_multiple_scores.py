from __future__ import annotations

import argparse
import csv
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge multiple scored submissions.")
    parser.add_argument("--inputs", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--debug-output", type=Path)
    parser.add_argument("--mode", choices=["max", "avg"], default="avg")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--id-column", default="full_secret_name")
    parser.add_argument("--score-column", default="score")
    parser.add_argument("--label-column", default="prediction")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    submissions = [read_scored(path, args.id_column, args.score_column, args.label_column) for path in args.inputs]
    ids = validate_same_ids(submissions)

    output_rows = []
    debug_rows = []
    source_wins = {str(path): 0 for path in args.inputs}

    for item_id in ids:
        entries = [submission[item_id] for submission in submissions]
        scores = [float(entry[args.score_column]) for entry in entries]
        labels = [entry[args.label_column] for entry in entries]

        if args.mode == "max":
            best_index = max(range(len(scores)), key=lambda index: scores[index])
            prediction = labels[best_index]
            merged_score = scores[best_index]
            selected_source = str(args.inputs[best_index])
            source_wins[selected_source] += 1
        else:
            merged_score = sum(scores) / len(scores)
            prediction = "1" if merged_score >= args.threshold else "0"
            selected_source = "average"

        output_rows.append({args.id_column: item_id, args.label_column: prediction})
        debug_row = {
            args.id_column: item_id,
            "merged_score": f"{merged_score:.6f}",
            "selected_source": selected_source,
            args.label_column: prediction,
        }
        for index, path in enumerate(args.inputs, start=1):
            name = Path(path).stem
            debug_row[f"score_{index}_{name}"] = f"{scores[index - 1]:.6f}"
            debug_row[f"prediction_{index}_{name}"] = labels[index - 1]
        debug_rows.append(debug_row)

    write_csv(args.output, output_rows, [args.id_column, args.label_column])

    if args.debug_output:
        debug_fields = list(debug_rows[0]) if debug_rows else []
        write_csv(args.debug_output, debug_rows, debug_fields)

    print(f"wrote: {args.output}")
    print(f"rows: {len(output_rows)}")
    print(f"mode: {args.mode}")
    if args.mode == "avg":
        print(f"threshold: {args.threshold}")
    else:
        for source, count in source_wins.items():
            print(f"wins {source}: {count}")
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
            if label not in {"0", "1"}:
                raise ValueError(f"{path}:{row_number} has invalid prediction {label!r}")
            rows[item_id] = {
                id_column: item_id,
                score_column: score,
                label_column: label,
            }
        return rows


def validate_same_ids(submissions: list[dict[str, dict[str, str]]]) -> list[str]:
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


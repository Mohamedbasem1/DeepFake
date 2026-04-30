from __future__ import annotations

import csv
from pathlib import Path

from .config import SubmissionConfig
from .predict import Prediction


def write_predictions(path: Path, predictions: list[Prediction], config: SubmissionConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [config.id_column]
    if config.score_column:
        fieldnames.append(config.score_column)
    fieldnames.append(config.label_column)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for prediction in predictions:
            row = {
                config.id_column: prediction.item_id,
                config.label_column: prediction.label,
            }
            if config.score_column:
                row[config.score_column] = f"{prediction.score:.6f}"
            writer.writerow(row)

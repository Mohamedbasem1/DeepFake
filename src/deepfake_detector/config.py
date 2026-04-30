from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class InputConfig:
    recursive: bool
    image_extensions: tuple[str, ...]
    video_extensions: tuple[str, ...]
    max_video_frames: int


@dataclass(frozen=True)
class ModelConfig:
    name: str
    threshold: float
    options: dict[str, Any]


@dataclass(frozen=True)
class SubmissionConfig:
    id_from: str
    id_column: str
    score_column: str | None
    label_column: str
    real_label: str
    fake_label: str


@dataclass(frozen=True)
class AppConfig:
    input: InputConfig
    model: ModelConfig
    submission: SubmissionConfig


def load_config(path: Path) -> AppConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return parse_config(raw)


def parse_config(raw: dict[str, Any]) -> AppConfig:
    input_raw = raw.get("input", {})
    model_raw = raw.get("model", {})
    submission_raw = raw.get("submission", {})
    columns_raw = submission_raw.get("columns", {})
    labels_raw = submission_raw.get("labels", {})

    return AppConfig(
        input=InputConfig(
            recursive=bool(input_raw.get("recursive", True)),
            image_extensions=normalize_extensions(input_raw.get("image_extensions", [])),
            video_extensions=normalize_extensions(input_raw.get("video_extensions", [])),
            max_video_frames=int(input_raw.get("max_video_frames", 16)),
        ),
        model=ModelConfig(
            name=str(model_raw.get("name", "artifact_baseline")),
            threshold=float(model_raw.get("threshold", 0.5)),
            options={key: value for key, value in model_raw.items() if key not in {"name", "threshold"}},
        ),
        submission=SubmissionConfig(
            id_from=str(submission_raw.get("id_from", "stem")),
            id_column=str(columns_raw.get("id", "id")),
            score_column=parse_optional_column(columns_raw.get("score", "score")),
            label_column=str(columns_raw.get("label", "label")),
            real_label=str(labels_raw.get("real", "real")),
            fake_label=str(labels_raw.get("fake", "deepfake")),
        ),
    )


def normalize_extensions(values: list[str]) -> tuple[str, ...]:
    return tuple(value.lower() if value.startswith(".") else f".{value.lower()}" for value in values)


def parse_optional_column(value: Any) -> str | None:
    if value is None or value is False:
        return None
    text = str(value).strip()
    return text or None

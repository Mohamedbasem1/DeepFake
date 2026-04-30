from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .config import AppConfig
from .media import MediaItem, discover_media, read_image, read_video_frames
from .models import Detector, build_detector


@dataclass(frozen=True)
class Prediction:
    item_id: str
    path: Path
    score: float
    label: str


def run_inference(input_path: Path, config: AppConfig) -> list[Prediction]:
    detector = build_detector(config.model.name, config.model.options)
    items = discover_media(input_path, config.input)
    predictions: list[Prediction] = []

    for item in tqdm(items, desc="Predicting", unit="file"):
        score = predict_item(detector, item, config)
        label = config.submission.fake_label if score >= config.model.threshold else config.submission.real_label
        predictions.append(
            Prediction(
                item_id=make_item_id(item.path, input_path, config.submission.id_from),
                path=item.path,
                score=score,
                label=label,
            )
        )
    return predictions


def predict_item(detector: Detector, item: MediaItem, config: AppConfig) -> float:
    if item.media_type == "image":
        return clamp_score(detector.predict_image(read_image(item.path)))
    if item.media_type == "video":
        frames = read_video_frames(item.path, config.input.max_video_frames)
        scores = [detector.predict_image(frame) for frame in frames]
        return clamp_score(float(np.mean(scores)))
    raise ValueError(f"Unsupported media type: {item.media_type}")


def make_item_id(path: Path, root: Path, id_from: str) -> str:
    if id_from == "name":
        return path.name
    if id_from == "relative":
        base = root if root.is_dir() else root.parent
        return path.relative_to(base).as_posix()
    if id_from == "stem":
        return path.stem
    raise ValueError(f"Unsupported id_from value: {id_from}")


def clamp_score(score: float) -> float:
    return max(0.0, min(1.0, float(score)))

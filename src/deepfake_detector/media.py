from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .config import InputConfig


@dataclass(frozen=True)
class MediaItem:
    path: Path
    media_type: str


def discover_media(root: Path, config: InputConfig) -> list[MediaItem]:
    if root.is_file():
        item = classify_media(root, config)
        return [item] if item else []

    pattern = "**/*" if config.recursive else "*"
    items: list[MediaItem] = []
    for path in sorted(root.glob(pattern)):
        if not path.is_file():
            continue
        item = classify_media(path, config)
        if item:
            items.append(item)
    return items


def classify_media(path: Path, config: InputConfig) -> MediaItem | None:
    suffix = path.suffix.lower()
    if suffix in config.image_extensions:
        return MediaItem(path=path, media_type="image")
    if suffix in config.video_extensions:
        return MediaItem(path=path, media_type="video")
    return None


def read_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(image, dtype=np.uint8)


def read_video_frames(path: Path, max_frames: int) -> list[np.ndarray]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Video inference requires opencv-python.") from exc

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        indices = list(range(max_frames))
    else:
        indices = np.linspace(0, max(frame_count - 1, 0), num=max_frames, dtype=int).tolist()

    frames: list[np.ndarray] = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame_bgr = capture.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {path}")
    return frames


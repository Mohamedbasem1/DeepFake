from __future__ import annotations

import argparse
import csv
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract labeled frames from deepfake video datasets for image-detector fine-tuning."
    )
    parser.add_argument("--input", required=True, type=Path, help="Dataset root or folder of videos.")
    parser.add_argument("--output", required=True, type=Path, help="Output root for extracted frames.")
    parser.add_argument("--dataset-name", required=True, help="Name used under output root, e.g. FaceForensics.")
    parser.add_argument(
        "--label",
        choices=["real", "fake"],
        help="Assign one label to all videos under --input.",
    )
    parser.add_argument(
        "--label-from-parent",
        action="store_true",
        help="Infer label from parent folder names containing real/original/authentic or fake/manipulated/deepfake.",
    )
    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=8,
        help="Number of evenly spaced frames to extract per video.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=384,
        help="Resize the shorter side to this size before saving. Use 0 to keep original size.",
    )
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality.")
    parser.add_argument("--limit", type=int, help="Optional max videos to process.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.label and not args.label_from_parent:
        raise SystemExit("Provide either --label or --label-from-parent.")

    videos = discover_videos(args.input)
    if args.limit:
        videos = videos[: args.limit]

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / f"{args.dataset_name}_frames.csv"
    rows: list[dict[str, str | int]] = []

    for index, video in enumerate(videos, start=1):
        label = args.label or infer_label_from_path(video)
        if label is None:
            print(f"Skipping {video}: could not infer label")
            continue
        saved = extract_frames(
            video_path=video,
            output_dir=args.output / args.dataset_name / label,
            frames_per_video=args.frames_per_video,
            image_size=args.image_size,
            quality=args.quality,
        )
        for frame_path in saved:
            rows.append(
                {
                    "dataset": args.dataset_name,
                    "label": label,
                    "video": str(video),
                    "frame": str(frame_path),
                }
            )
        print(f"[{index}/{len(videos)}] {video.name}: saved {len(saved)} {label} frames")

    write_manifest(manifest_path, rows)
    print(f"Saved manifest: {manifest_path}")


def discover_videos(root: Path) -> list[Path]:
    if root.is_file() and root.suffix.lower() in VIDEO_EXTENSIONS:
        return [root]
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)


def infer_label_from_path(path: Path) -> str | None:
    parts = [part.lower() for part in path.parts]
    fake_terms = {
        "fake",
        "fakes",
        "deepfake",
        "deepfakes",
        "manipulated",
        "manipulation",
        "altered",
        "synthesis",
        "celeb-synthesis",
        "deepfakes",
        "faceswap",
        "face2face",
        "faceshifter",
        "neuraltextures",
        "end_to_end",
        "reenact_postprocess",
    }
    real_terms = {
        "real",
        "reals",
        "original",
        "originals",
        "authentic",
        "pristine",
        "youtube",
        "celeb-real",
        "youtube-real",
        "original_sequences",
        "source_videos",
    }
    if any(any(term in part for term in fake_terms) for part in parts):
        return "fake"
    if any(any(term in part for term in real_terms) for part in parts):
        return "real"
    return None


def extract_frames(
    video_path: Path,
    output_dir: Path,
    frames_per_video: int,
    image_size: int,
    quality: int,
) -> list[Path]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Frame extraction requires opencv-python.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"Could not open {video_path}")
        return []

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        capture.release()
        print(f"No frame count for {video_path}")
        return []

    if frames_per_video <= 1:
        indices = [frame_count // 2]
    else:
        start = max(0, int(frame_count * 0.05))
        end = max(start, int(frame_count * 0.95) - 1)
        indices = [round(start + (end - start) * i / (frames_per_video - 1)) for i in range(frames_per_video)]

    saved_paths: list[Path] = []
    for frame_number in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = capture.read()
        if not ok:
            continue
        if image_size > 0:
            frame = resize_short_side(frame, image_size, cv2)
        output_path = output_dir / f"{video_path.stem}_f{frame_number:06d}.jpg"
        cv2.imwrite(str(output_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        saved_paths.append(output_path)

    capture.release()
    return saved_paths


def resize_short_side(frame, short_side: int, cv2):
    height, width = frame.shape[:2]
    if min(height, width) == short_side:
        return frame
    if height < width:
        new_height = short_side
        new_width = round(width * short_side / height)
    else:
        new_width = short_side
        new_height = round(height * short_side / width)
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def write_manifest(path: Path, rows: list[dict[str, str | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "label", "video", "frame"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

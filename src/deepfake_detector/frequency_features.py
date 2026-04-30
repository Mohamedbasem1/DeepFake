from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps


def extract_frequency_features(path: Path, image_size: int = 256, radial_bins: int = 32) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = ImageOps.fit(image, (image_size, image_size), method=Image.Resampling.BICUBIC)
        rgb = np.asarray(image, dtype=np.float32) / 255.0

        gray_image = image.convert("L")
        gray = np.asarray(gray_image, dtype=np.float32) / 255.0
        blur = np.asarray(gray_image.filter(ImageFilter.GaussianBlur(radius=1.2)), dtype=np.float32) / 255.0
        residual = gray - blur

    gray_bins = radial_fft_bins(gray, radial_bins)
    residual_bins = radial_fft_bins(residual, radial_bins)
    channel_bins = np.concatenate([radial_fft_bins(rgb[..., channel], radial_bins // 2) for channel in range(3)])

    stats = np.array(
        [
            high_frequency_ratio(gray),
            high_frequency_ratio(residual),
            jpeg_blockiness(gray),
            local_variance_std(gray),
            float(np.mean(gray)),
            float(np.std(gray)),
            float(np.mean(np.max(rgb, axis=2) - np.min(rgb, axis=2))),
            float(np.std(np.max(rgb, axis=2) - np.min(rgb, axis=2))),
        ],
        dtype=np.float32,
    )
    return np.concatenate([gray_bins, residual_bins, channel_bins, stats]).astype(np.float32)


def radial_fft_bins(image: np.ndarray, bins: int) -> np.ndarray:
    centered = image - float(np.mean(image))
    window = np.outer(np.hanning(image.shape[0]), np.hanning(image.shape[1])).astype(np.float32)
    spectrum = np.fft.fftshift(np.fft.fft2(centered * window))
    magnitude = np.log1p(np.abs(spectrum)).astype(np.float32)

    height, width = image.shape
    yy, xx = np.indices((height, width))
    radius = np.sqrt((yy - height / 2.0) ** 2 + (xx - width / 2.0) ** 2)
    radius = radius / max(radius.max(), 1e-6)
    edges = np.linspace(0.0, 1.0, bins + 1)

    features = []
    global_mean = float(np.mean(magnitude)) + 1e-6
    for start, end in zip(edges[:-1], edges[1:]):
        mask = (radius >= start) & (radius < end)
        value = float(np.mean(magnitude[mask])) / global_mean if np.any(mask) else 0.0
        features.append(value)
    return np.asarray(features, dtype=np.float32)


def high_frequency_ratio(image: np.ndarray) -> float:
    centered = image - float(np.mean(image))
    spectrum = np.fft.fftshift(np.fft.fft2(centered))
    power = np.abs(spectrum) ** 2
    height, width = image.shape
    yy, xx = np.indices((height, width))
    radius = np.sqrt((yy - height / 2.0) ** 2 + (xx - width / 2.0) ** 2)
    high = power[radius >= 0.35 * min(height, width)].sum()
    total = power.sum() + 1e-6
    return float(high / total)


def jpeg_blockiness(gray: np.ndarray) -> float:
    if min(gray.shape) < 16:
        return 0.0
    vertical_edges = np.abs(np.diff(gray, axis=1))
    horizontal_edges = np.abs(np.diff(gray, axis=0))
    block_vertical = float(np.mean(vertical_edges[:, 7::8]))
    block_horizontal = float(np.mean(horizontal_edges[7::8, :]))
    all_edges = float(np.mean(vertical_edges) + np.mean(horizontal_edges)) + 1e-6
    return (block_vertical + block_horizontal) / all_edges


def local_variance_std(gray: np.ndarray, window: int = 9) -> float:
    pad = window // 2
    padded = np.pad(gray, pad, mode="reflect")
    integral = padded.cumsum(axis=0).cumsum(axis=1)
    integral_sq = (padded * padded).cumsum(axis=0).cumsum(axis=1)

    sums = integral[window:, window:] - integral[:-window, window:] - integral[window:, :-window] + integral[:-window, :-window]
    sums_sq = (
        integral_sq[window:, window:]
        - integral_sq[:-window, window:]
        - integral_sq[window:, :-window]
        + integral_sq[:-window, :-window]
    )
    area = float(window * window)
    mean = sums / area
    variance = np.maximum(sums_sq / area - mean * mean, 0.0)
    return float(np.std(np.sqrt(variance)))


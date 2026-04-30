from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Detector(Protocol):
    def predict_image(self, image: np.ndarray) -> float:
        """Return a probability-like fake score in [0, 1]."""


@dataclass
class ArtifactBaselineDetector:
    """Small no-weights baseline based on compression and frequency artifacts.

    This is not intended to be competitive by itself. Its job is to keep the
    whole inference path executable while stronger pretrained detectors are
    added.
    """

    def predict_image(self, image: np.ndarray) -> float:
        image_float = image.astype(np.float32) / 255.0
        gray = rgb_to_gray(image_float)

        high_frequency = mean_high_frequency_energy(gray)
        saturation_spread = float(np.std(image_float.max(axis=2) - image_float.min(axis=2)))
        blockiness = jpeg_blockiness(gray)
        noise_inconsistency = local_noise_inconsistency(gray)

        raw_score = (
            2.0 * high_frequency
            + 1.5 * blockiness
            + 1.2 * noise_inconsistency
            + 0.8 * saturation_spread
            - 1.15
        )
        return float(1.0 / (1.0 + np.exp(-4.0 * raw_score)))


def build_detector(name: str, options: dict | None = None) -> Detector:
    normalized = name.lower().strip()
    if normalized == "artifact_baseline":
        return ArtifactBaselineDetector()
    if normalized in {"hf_image_classifier", "hf_siglip"}:
        return HFImageClassifierDetector.from_options(options or {})
    if normalized in {"commfor", "community_forensics"}:
        return CommunityForensicsDetector.from_options(options or {})
    raise ValueError(f"Unknown detector: {name}")


@dataclass
class HFImageClassifierDetector:
    model_id: str
    fake_label_index: int | None = None
    fake_label_keywords: tuple[str, ...] = ("fake", "deepfake", "synthetic", "generated", "ai")
    real_label_keywords: tuple[str, ...] = ("real", "authentic", "natural", "photograph")
    device: str = "auto"
    tta: tuple[str, ...] = ("none",)
    half_precision: bool = True

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification
        except ImportError as exc:
            raise RuntimeError(
                "HF image classifier inference requires torch and transformers. "
                "Install requirements-lightning.txt on Lightning AI."
            ) from exc

        self._torch = torch
        self._device = self._resolve_device(self.device)
        self._processor = AutoImageProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForImageClassification.from_pretrained(self.model_id)
        self._model.to(self._device)
        if self.half_precision and self._device.startswith("cuda"):
            self._model.half()
        self._model.eval()
        self._fake_index = self.fake_label_index
        if self._fake_index is None:
            self._fake_index = self._infer_fake_label_index()

    @classmethod
    def from_options(cls, options: dict) -> "HFImageClassifierDetector":
        model_id = options.get("model_id")
        if not model_id:
            raise ValueError("HF detector config requires model.model_id.")
        return cls(
            model_id=str(model_id),
            fake_label_index=options.get("fake_label_index"),
            device=str(options.get("device", "auto")),
            tta=tuple(options.get("tta", ["none"])),
            half_precision=bool(options.get("half_precision", True)),
        )

    def predict_image(self, image: np.ndarray) -> float:
        from PIL import Image, ImageOps

        pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
        images = []
        for mode in self.tta:
            if mode == "none":
                images.append(pil_image)
            elif mode in {"hflip", "horizontal_flip"}:
                images.append(ImageOps.mirror(pil_image))
            else:
                raise ValueError(f"Unsupported TTA mode: {mode}")

        inputs = self._processor(images=images, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        if self.half_precision and self._device.startswith("cuda") and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].half()

        with self._torch.no_grad():
            logits = self._model(**inputs).logits
            probabilities = self._torch.softmax(logits.float(), dim=-1)
            fake_scores = probabilities[:, int(self._fake_index)]
        return float(fake_scores.mean().detach().cpu().item())

    def _resolve_device(self, requested: str) -> str:
        if requested != "auto":
            return requested
        return "cuda" if self._torch.cuda.is_available() else "cpu"

    def _infer_fake_label_index(self) -> int:
        id2label = getattr(self._model.config, "id2label", None) or {}
        normalized = {int(index): str(label).lower() for index, label in id2label.items()}
        for index, label in normalized.items():
            if any(keyword in label for keyword in self.fake_label_keywords):
                return index

        known_real = {
            index
            for index, label in normalized.items()
            if any(keyword in label for keyword in self.real_label_keywords)
        }
        unknown = [index for index in normalized if index not in known_real]
        if len(unknown) == 1:
            return unknown[0]

        raise ValueError(
            "Could not infer the fake class index from model labels. "
            "Set model.fake_label_index in the YAML config."
        )


@dataclass
class CommunityForensicsDetector:
    """Community Forensics detector using the authors' Hugging Face checkpoints."""

    model_repo: str = "OwensLab/commfor-model-384"
    input_size: int = 384
    patch_size: int = 16
    model_size: str = "small"
    threshold: float = 0.5
    device: str = "auto"
    tta: tuple[str, ...] = ("none",)
    half_precision: bool = True
    ckpt_path: str | None = None

    def __post_init__(self) -> None:
        try:
            import timm
            import torch
            from huggingface_hub import PyTorchModelHubMixin
            from torchvision import transforms
        except ImportError as exc:
            raise RuntimeError(
                "Community Forensics inference requires torch, torchvision, timm, "
                "and huggingface-hub. Install requirements-lightning.txt on Lightning AI."
            ) from exc

        self._torch = torch
        self._device = self._resolve_device(self.device)
        self._transform = build_commfor_transform(transforms, self.input_size)
        model_class = get_commfor_model_class(torch, timm, PyTorchModelHubMixin)
        self._model = model_class.from_pretrained(
            self.model_repo,
            model_size=self.model_size,
            input_size=self.input_size,
            patch_size=self.patch_size,
            freeze_backbone=False,
            device=self._device,
            dtype=torch.float32,
        )
        if self.ckpt_path:
            checkpoint = torch.load(self.ckpt_path, map_location="cpu")
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            self._model.load_state_dict(state_dict)
        self._model.to(self._device)
        if self.half_precision and self._device.startswith("cuda"):
            self._model.half()
        self._model.eval()

    @classmethod
    def from_options(cls, options: dict) -> "CommunityForensicsDetector":
        return cls(
            model_repo=str(options.get("model_repo", "OwensLab/commfor-model-384")),
            input_size=int(options.get("input_size", 384)),
            patch_size=int(options.get("patch_size", 16)),
            model_size=str(options.get("model_size", "small")),
            device=str(options.get("device", "auto")),
            tta=tuple(options.get("tta", ["none"])),
            half_precision=bool(options.get("half_precision", True)),
            ckpt_path=str(options["ckpt_path"]) if options.get("ckpt_path") else None,
        )

    def predict_image(self, image: np.ndarray) -> float:
        from PIL import Image, ImageOps

        pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
        tensors = []
        for mode in self.tta:
            if mode == "none":
                tensors.append(self._transform(pil_image))
            elif mode in {"hflip", "horizontal_flip"}:
                tensors.append(self._transform(ImageOps.mirror(pil_image)))
            else:
                raise ValueError(f"Unsupported TTA mode: {mode}")

        batch = self._torch.stack(tensors).to(self._device)
        if self.half_precision and self._device.startswith("cuda"):
            batch = batch.half()

        with self._torch.no_grad():
            logits = self._model(batch).float().flatten()
            scores = self._torch.sigmoid(logits)
        return float(scores.mean().detach().cpu().item())

    def _resolve_device(self, requested: str) -> str:
        if requested != "auto":
            return requested
        return "cuda" if self._torch.cuda.is_available() else "cpu"


def get_commfor_model_class(torch, timm, hub_mixin):
    class ViTClassifier(torch.nn.Module, hub_mixin):
        def __init__(
            self,
            model_size: str = "small",
            input_size: int = 384,
            patch_size: int = 16,
            freeze_backbone: bool = False,
            device: str = "cuda",
            dtype: torch.dtype = torch.float32,
        ) -> None:
            super().__init__()
            model_name, head_features = resolve_commfor_vit(model_size, input_size, patch_size)
            self.vit = timm.create_model(model_name, pretrained=True).to(device)
            if freeze_backbone:
                for parameter in self.vit.parameters():
                    parameter.requires_grad = False
            self.vit.head = torch.nn.Linear(
                in_features=head_features,
                out_features=1,
                bias=True,
                device=device,
                dtype=dtype,
            )

        def forward(self, x):
            return self.vit(x)

    return ViTClassifier


def resolve_commfor_vit(model_size: str, input_size: int, patch_size: int) -> tuple[str, int]:
    if model_size == "small":
        if input_size == 224 and patch_size == 32:
            return "vit_small_patch32_224.augreg_in21k_ft_in1k", 384
        if input_size == 224 and patch_size == 16:
            return "vit_small_patch16_224.augreg_in21k_ft_in1k", 384
        if input_size == 384 and patch_size == 32:
            return "vit_small_patch32_384.augreg_in21k_ft_in1k", 384
        if input_size == 384 and patch_size == 16:
            return "vit_small_patch16_384.augreg_in21k_ft_in1k", 384
        raise ValueError(f"Unsupported small ViT shape: {input_size}, patch {patch_size}")

    if model_size == "tiny":
        if patch_size != 16:
            raise ValueError("Only patch size 16 is available for ViT-Ti.")
        if input_size == 224:
            return "vit_tiny_patch16_224.augreg_in21k_ft_in1k", 192
        if input_size == 384:
            return "vit_tiny_patch16_384.augreg_in21k_ft_in1k", 192
        raise ValueError(f"Unsupported tiny ViT input size: {input_size}")

    raise ValueError(f"Unsupported Community Forensics model size: {model_size}")


def build_commfor_transform(transforms, input_size: int):
    if input_size == 224:
        resize_size = 256
    elif input_size == 384:
        resize_size = 440
    else:
        raise ValueError(f"Unsupported Community Forensics input size: {input_size}")

    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    return 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]


def mean_high_frequency_energy(gray: np.ndarray) -> float:
    centered = gray - float(np.mean(gray))
    spectrum = np.fft.fftshift(np.fft.fft2(centered))
    magnitude = np.log1p(np.abs(spectrum))
    height, width = gray.shape
    yy, xx = np.ogrid[:height, :width]
    cy, cx = height / 2.0, width / 2.0
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    cutoff = 0.35 * min(height, width)
    mask = radius >= cutoff
    if not np.any(mask):
        return 0.0
    return float(np.mean(magnitude[mask]) / (np.mean(magnitude) + 1e-6))


def jpeg_blockiness(gray: np.ndarray) -> float:
    if min(gray.shape) < 16:
        return 0.0
    vertical_edges = np.abs(np.diff(gray, axis=1))
    horizontal_edges = np.abs(np.diff(gray, axis=0))
    block_vertical = np.mean(vertical_edges[:, 7::8])
    block_horizontal = np.mean(horizontal_edges[7::8, :])
    all_edges = np.mean(vertical_edges) + np.mean(horizontal_edges) + 1e-6
    return float((block_vertical + block_horizontal) / all_edges)


def local_noise_inconsistency(gray: np.ndarray) -> float:
    try:
        from scipy.ndimage import uniform_filter
    except ImportError:
        return float(np.std(gray))

    local_mean = uniform_filter(gray, size=9)
    local_sq_mean = uniform_filter(gray * gray, size=9)
    local_var = np.maximum(local_sq_mean - local_mean * local_mean, 0.0)
    return float(np.std(np.sqrt(local_var)))

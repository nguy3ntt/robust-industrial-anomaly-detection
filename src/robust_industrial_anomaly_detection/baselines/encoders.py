"""Local, hash-pinned spatial encoders for controlled representation comparisons."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from torchvision.models import convnext_tiny  # type: ignore[import-untyped]

from robust_industrial_anomaly_detection.baselines.handcrafted import BaselineError
from robust_industrial_anomaly_detection.baselines.multiscale import (
    FrozenPyramidResNet18,
    PyramidFeatures,
    _rgb_tensor,
)
from robust_industrial_anomaly_detection.data.visa import sha256_file


@dataclass(frozen=True)
class EncoderSpec:
    weights_id: str
    url: str
    sha256: str
    channels: tuple[int, int, int]


# Full digests observed after the official filename-prefix verification, not
# independently published full provider checksums. No runtime remote code loads.
ENCODERS = {
    "resnet18": EncoderSpec(
        "torchvision.ResNet18_Weights.IMAGENET1K_V1",
        "https://download.pytorch.org/models/resnet18-f37072fd.pth",
        "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
        (64, 128, 256),
    ),
    "convnext_tiny": EncoderSpec(
        "torchvision.ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
        "https://download.pytorch.org/models/convnext_tiny-983f1562.pth",
        "983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d",
        (96, 192, 384),
    ),
}


class SpatialEncoder(Protocol):
    @property
    def model(self) -> nn.Module: ...

    def extract_rgb(self, images: Sequence[np.ndarray]) -> tuple[PyramidFeatures, ...]: ...


def checkpoint_path(name: str) -> Path:
    if name not in ENCODERS:
        raise BaselineError(f"unsupported encoder: {name}")
    cache = os.environ.get("TORCH_HOME")
    if not cache:
        raise BaselineError("TORCH_HOME must identify the local model cache")
    return Path(cache) / "hub" / "checkpoints" / ENCODERS[name].url.rsplit("/", 1)[-1]


def verify_checkpoint(name: str) -> Path:
    path = checkpoint_path(name)
    if not path.is_file() or sha256_file(path) != ENCODERS[name].sha256:
        raise BaselineError(
            f"missing or modified {name} checkpoint; run scripts/prepare_m11_encoders.py"
        )
    return path


class _ConvNeXtPyramid(nn.Module):
    """First three stages, aligned to the ResNet reference's 28x28 patch grid."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        stages = list(backbone.get_submodule("features").children())
        self.first = nn.Sequential(stages[0], stages[1])
        self.second = nn.Sequential(stages[2], stages[3])
        self.third = nn.Sequential(stages[4], stages[5])

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        first = self.first(inputs)
        second = self.second(first)
        third = self.third(second)
        return functional.adaptive_avg_pool2d(first, second.shape[-2:]), second, third


class FrozenPyramidConvNeXtTiny:
    def __init__(self, *, device: str, image_size: int, batch_size: int) -> None:
        if image_size < 64 or image_size % 32 or batch_size < 1:
            raise BaselineError("invalid encoder input size or batch size")
        if device not in {"cpu", "cuda"} or (device == "cuda" and not torch.cuda.is_available()):
            raise BaselineError("requested encoder device is unavailable")
        path = verify_checkpoint("convnext_tiny")
        backbone = convnext_tiny(weights=None)
        backbone.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True), strict=True
        )
        self.model = _ConvNeXtPyramid(backbone).eval().to(device)
        self.model.requires_grad_(False)
        self.device = torch.device(device)
        self.image_size = image_size
        self.batch_size = batch_size

    def extract_rgb(self, images: Sequence[np.ndarray]) -> tuple[PyramidFeatures, ...]:
        if not images:
            raise BaselineError("at least one RGB image is required")
        result: list[PyramidFeatures] = []
        with torch.inference_mode():
            for offset in range(0, len(images), self.batch_size):
                inputs = torch.stack(
                    [
                        _rgb_tensor(image, self.image_size)
                        for image in images[offset : offset + self.batch_size]
                    ]
                ).to(self.device)
                layers = self.model(inputs)
                arrays = [
                    layer.permute(0, 2, 3, 1).contiguous().cpu().numpy().astype(np.float32)
                    for layer in layers
                ]
                result.extend(PyramidFeatures(*parts) for parts in zip(*arrays, strict=True))
        return tuple(result)


def build_encoder(name: str, *, device: str, image_size: int, batch_size: int) -> SpatialEncoder:
    verify_checkpoint(name)
    if name == "resnet18":
        return FrozenPyramidResNet18(device=device, image_size=image_size, batch_size=batch_size)
    return FrozenPyramidConvNeXtTiny(device=device, image_size=image_size, batch_size=batch_size)

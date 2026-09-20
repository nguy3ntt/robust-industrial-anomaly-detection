"""Frozen multi-scale features and bounded patch-memory indexes."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image, ImageOps
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18  # type: ignore[import-untyped]

from robust_industrial_anomaly_detection.baselines.handcrafted import BaselineError

FeatureMode = Literal["mid", "multiscale"]
SelectionMode = Literal["random", "kcenter"]


@dataclass(frozen=True)
class PyramidFeatures:
    """Frozen spatial features aligned to one input image."""

    layer1: np.ndarray
    layer2: np.ndarray
    layer3: np.ndarray


@dataclass(frozen=True)
class MemoryBuildStats:
    """Exact logical sizes and selection diagnostics for one memory bank."""

    candidate_count: int
    selected_count: int
    feature_dimension: int
    projected_dimension: int
    raw_candidate_bytes: int
    bank_bytes: int
    projection_bytes: int
    coverage_radius: float
    selection: SelectionMode


class _PyramidModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        layer1 = self.layer1(self.stem(inputs))
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer1 = functional.adaptive_avg_pool2d(layer1, output_size=layer2.shape[-2:])
        return layer1, layer2, layer3


def load_fitted_rgb(path: Path, image_size: int) -> np.ndarray:
    """Load one image as a centered square RGB uint8 array."""

    try:
        with Image.open(path) as source:
            fitted = ImageOps.fit(
                source.convert("RGB"),
                (image_size, image_size),
                method=Image.Resampling.BILINEAR,
                centering=(0.5, 0.5),
            )
            return np.asarray(fitted, dtype=np.uint8)
    except (OSError, ValueError) as exc:
        raise BaselineError(f"cannot read image {path}: {exc}") from exc


def _rgb_tensor(rgb: np.ndarray, image_size: int) -> Tensor:
    pixels = np.asarray(rgb)
    if pixels.shape != (image_size, image_size, 3) or pixels.dtype != np.uint8:
        raise BaselineError(
            f"RGB inputs must be uint8 arrays shaped ({image_size}, {image_size}, 3)"
        )
    normalized = pixels.astype(np.float32) / 255.0
    tensor = torch.from_numpy(normalized).permute(2, 0, 1)
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
    deviation = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / deviation


def _image_tensor(path: Path, image_size: int) -> Tensor:
    return _rgb_tensor(load_fitted_rgb(path, image_size), image_size)


class FrozenPyramidResNet18:
    """Extract pooled layer-1 plus native layer-2/layer-3 ResNet-18 features."""

    weights_id = "torchvision.ResNet18_Weights.IMAGENET1K_V1"
    weights_url = ResNet18_Weights.DEFAULT.url

    def __init__(self, *, device: str, image_size: int = 224, batch_size: int = 16) -> None:
        if image_size < 64 or image_size % 32:
            raise BaselineError("image_size must be at least 64 and divisible by 32")
        if batch_size < 1:
            raise BaselineError("batch_size must be positive")
        if device == "cuda" and not torch.cuda.is_available():
            raise BaselineError("CUDA was requested but is unavailable")
        self.device = torch.device(device)
        self.image_size = image_size
        self.batch_size = batch_size
        self.model = _PyramidModule().eval().to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def extract(self, paths: Sequence[Path]) -> tuple[PyramidFeatures, ...]:
        """Extract deterministic spatial feature pyramids in input order."""

        if not paths:
            raise BaselineError("at least one image path is required")
        return self._extract_tensors([_image_tensor(path, self.image_size) for path in paths])

    def extract_rgb(self, images: Sequence[np.ndarray]) -> tuple[PyramidFeatures, ...]:
        """Extract pyramids from already fitted RGB arrays without writing images."""

        if not images:
            raise BaselineError("at least one RGB image is required")
        return self._extract_tensors([_rgb_tensor(image, self.image_size) for image in images])

    def _extract_tensors(self, tensors: Sequence[Tensor]) -> tuple[PyramidFeatures, ...]:
        output: list[PyramidFeatures] = []
        with torch.inference_mode():
            for start in range(0, len(tensors), self.batch_size):
                inputs = torch.stack(list(tensors[start : start + self.batch_size])).to(
                    self.device, non_blocking=True
                )
                layers = self.model(inputs)
                arrays = [
                    layer.permute(0, 2, 3, 1).contiguous().cpu().numpy().astype(np.float32)
                    for layer in layers
                ]
                output.extend(
                    PyramidFeatures(layer1, layer2, layer3)
                    for layer1, layer2, layer3 in zip(*arrays, strict=True)
                )
        return tuple(output)


def compose_feature_map(features: PyramidFeatures, mode: FeatureMode) -> np.ndarray:
    """Compose a normalized mid-level or layer-1-to-3 multi-scale patch grid."""

    layer1 = np.asarray(features.layer1, dtype=np.float32)
    layer2 = np.asarray(features.layer2, dtype=np.float32)
    layer3 = np.asarray(features.layer3, dtype=np.float32)
    if mode == "mid":
        target_height, target_width = layer3.shape[:2]
        pooled_layer2 = layer2.reshape(
            target_height,
            layer2.shape[0] // target_height,
            target_width,
            layer2.shape[1] // target_width,
            -1,
        ).mean(axis=(1, 3))
        combined = np.concatenate((pooled_layer2, layer3), axis=2)
    elif mode == "multiscale":
        target_height, target_width = layer2.shape[:2]
        resized_layer3 = cv2.resize(
            layer3,
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR,
        )
        combined = np.concatenate((layer1, layer2, resized_layer3), axis=2)
    else:
        raise BaselineError(f"unsupported feature mode: {mode}")
    norms = np.linalg.norm(combined, axis=2, keepdims=True)
    normalized = combined / np.maximum(norms, float(np.finfo(np.float32).eps))
    return cast(np.ndarray, normalized.astype(np.float32, copy=False))


def _squared_distances(values: np.ndarray, center: np.ndarray) -> np.ndarray:
    distances = np.sum(values * values, axis=1) + float(np.dot(center, center))
    distances -= 2.0 * (values @ center)
    return np.maximum(distances, 0.0)


def _kcenter_indexes(values: np.ndarray, count: int) -> tuple[np.ndarray, float]:
    """Select a deterministic greedy k-center subset and its covering radius."""

    if count >= values.shape[0]:
        indexes = np.arange(values.shape[0], dtype=np.int64)
        return indexes, 0.0
    mean = np.mean(values, axis=0)
    first = int(np.argmin(_squared_distances(values, mean)))
    indexes = np.empty(count, dtype=np.int64)
    indexes[0] = first
    minimum = _squared_distances(values, values[first])
    for position in range(1, count):
        selected = int(np.argmax(minimum))
        indexes[position] = selected
        minimum = np.minimum(minimum, _squared_distances(values, values[selected]))
    return indexes, float(np.sqrt(np.max(minimum)))


class ProjectedPatchMemory:
    """Bounded projected patch memory with random or k-center selection."""

    def __init__(
        self,
        *,
        seed: int,
        bank_size: int,
        projected_dimension: int,
        patches_per_image: int,
        candidate_limit: int,
        image_score_fraction: float,
        selection: SelectionMode,
    ) -> None:
        if min(bank_size, projected_dimension, patches_per_image, candidate_limit) < 1:
            raise BaselineError("memory sizes must be positive")
        if bank_size > candidate_limit:
            raise BaselineError("bank_size cannot exceed candidate_limit")
        if not 0.0 < image_score_fraction <= 1.0:
            raise BaselineError("image_score_fraction must be in (0, 1]")
        self.seed = seed
        self.bank_size = bank_size
        self.projected_dimension = projected_dimension
        self.patches_per_image = patches_per_image
        self.candidate_limit = candidate_limit
        self.image_score_fraction = image_score_fraction
        self.selection = selection
        self._projection: np.ndarray | None = None
        self._bank: np.ndarray | None = None
        self._stats: MemoryBuildStats | None = None

    @property
    def projection(self) -> np.ndarray:
        if self._projection is None:
            raise BaselineError("memory has not been fitted")
        return self._projection

    @property
    def bank(self) -> np.ndarray:
        if self._bank is None:
            raise BaselineError("memory has not been fitted")
        return self._bank

    @property
    def stats(self) -> MemoryBuildStats:
        if self._stats is None:
            raise BaselineError("memory has not been fitted")
        return self._stats

    def _project(self, patches: np.ndarray) -> np.ndarray:
        projected = np.asarray(patches, dtype=np.float32) @ self.projection
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        normalized = projected / np.maximum(norms, float(np.finfo(np.float32).eps))
        return cast(np.ndarray, normalized.astype(np.float32, copy=False))

    def fit(self, feature_maps: Sequence[np.ndarray]) -> MemoryBuildStats:
        """Fit from normal-only feature maps using a bounded candidate pool."""

        if len(feature_maps) < 2:
            raise BaselineError("at least two normal feature maps are required")
        rng = np.random.default_rng(self.seed)
        sampled: list[np.ndarray] = []
        feature_dimension: int | None = None
        for feature_map in feature_maps:
            array = np.asarray(feature_map, dtype=np.float32)
            if array.ndim != 3:
                raise BaselineError("feature maps must be height x width x channels")
            flattened = array.reshape(-1, array.shape[-1])
            feature_dimension = flattened.shape[1]
            count = min(self.patches_per_image, flattened.shape[0])
            sampled.append(flattened[rng.choice(flattened.shape[0], size=count, replace=False)])
        assert feature_dimension is not None
        candidates = np.concatenate(sampled)
        if candidates.shape[0] > self.candidate_limit:
            candidates = candidates[
                rng.choice(candidates.shape[0], size=self.candidate_limit, replace=False)
            ]
        projection_rng = np.random.default_rng(self.seed + 1)
        self._projection = projection_rng.standard_normal(
            (feature_dimension, self.projected_dimension)
        ).astype(np.float32) / np.float32(np.sqrt(self.projected_dimension))
        projected = self._project(candidates)
        if self.selection == "kcenter":
            indexes, radius = _kcenter_indexes(projected, min(self.bank_size, len(projected)))
        else:
            count = min(self.bank_size, len(projected))
            indexes = rng.choice(len(projected), size=count, replace=False)
            selected = projected[indexes]
            distances = (
                np.sum(projected * projected, axis=1)[:, None]
                + np.sum(selected * selected, axis=1)[None, :]
            )
            distances -= 2.0 * (projected @ selected.T)
            minimum = np.min(np.maximum(distances, 0.0), axis=1)
            radius = float(np.sqrt(np.max(minimum)))
        self._bank = projected[indexes].astype(np.float32, copy=False)
        self._stats = MemoryBuildStats(
            candidate_count=int(projected.shape[0]),
            selected_count=int(self._bank.shape[0]),
            feature_dimension=feature_dimension,
            projected_dimension=self.projected_dimension,
            raw_candidate_bytes=int(candidates.nbytes),
            bank_bytes=int(self._bank.nbytes),
            projection_bytes=int(self._projection.nbytes),
            coverage_radius=radius,
            selection=self.selection,
        )
        return self._stats

    def fit_projected(self, bank: np.ndarray, projection: np.ndarray) -> MemoryBuildStats:
        """Load an already selected projected bank, used by the unified index."""

        selected = np.asarray(bank, dtype=np.float32)
        matrix = np.asarray(projection, dtype=np.float32)
        if selected.ndim != 2 or matrix.ndim != 2 or selected.shape[1] != matrix.shape[1]:
            raise BaselineError("projected bank and projection have incompatible shapes")
        self._projection = matrix
        self._bank = selected
        self._stats = MemoryBuildStats(
            candidate_count=int(selected.shape[0]),
            selected_count=int(selected.shape[0]),
            feature_dimension=int(matrix.shape[0]),
            projected_dimension=int(matrix.shape[1]),
            raw_candidate_bytes=int(selected.nbytes),
            bank_bytes=int(selected.nbytes),
            projection_bytes=int(matrix.nbytes),
            coverage_radius=0.0,
            selection=self.selection,
        )
        return self._stats

    def score(
        self, feature_maps: Sequence[np.ndarray]
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        """Return top-tail image scores and nearest-memory patch maps."""

        bank = self.bank
        bank_norm = np.sum(bank * bank, axis=1)[None, :]
        image_scores: list[float] = []
        score_maps: list[np.ndarray] = []
        for feature_map in feature_maps:
            array = np.asarray(feature_map, dtype=np.float32)
            height, width, channels = array.shape
            projected = self._project(array.reshape(-1, channels))
            distances = np.sum(projected * projected, axis=1)[:, None] + bank_norm
            distances -= 2.0 * (projected @ bank.T)
            minimum = np.sqrt(np.maximum(np.min(distances, axis=1), 0.0))
            score_map = minimum.reshape(height, width).astype(np.float32, copy=False)
            count = max(1, int(np.ceil(score_map.size * self.image_score_fraction)))
            top_scores = np.partition(score_map.reshape(-1), -count)[-count:]
            image_scores.append(float(np.mean(top_scores)))
            score_maps.append(score_map)
        return np.asarray(image_scores, dtype=np.float64), tuple(score_maps)

    def save(self, path: Path, *, metadata: dict[str, object]) -> int:
        """Atomically save the inference index and return its compressed size."""

        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with temporary.open("xb") as stream:
                np.savez_compressed(
                    stream,
                    bank=self.bank,
                    projection=self.projection,
                    stats=np.asarray(str({**asdict(self.stats), **metadata})),
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path.stat().st_size


def load_projected_index(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load only numeric arrays from a locally generated M3 index."""

    try:
        with np.load(path, allow_pickle=False) as payload:
            bank = np.asarray(payload["bank"], dtype=np.float32)
            projection = np.asarray(payload["projection"], dtype=np.float32)
    except (OSError, KeyError, ValueError) as exc:
        raise BaselineError(f"cannot load patch index {path}: {exc}") from exc
    return bank, projection

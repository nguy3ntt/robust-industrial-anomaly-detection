"""Frozen ResNet feature and patch-memory baselines."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image, ImageOps
from sklearn.neighbors import NearestNeighbors  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18  # type: ignore[import-untyped]

from robust_industrial_anomaly_detection.baselines.handcrafted import BaselineError


@dataclass(frozen=True)
class FrozenFeatures:
    """Global and spatial features in input order."""

    global_features: np.ndarray
    patch_features: tuple[np.ndarray, ...]


class _ResNetFeatureModule(nn.Module):
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
        self.layer4 = backbone.layer4

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        values = self.layer1(self.stem(inputs))
        layer2 = self.layer2(values)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        global_features = functional.adaptive_avg_pool2d(layer4, output_size=1).flatten(1)
        pooled_layer2 = functional.adaptive_avg_pool2d(layer2, output_size=layer3.shape[-2:])
        patches = torch.cat((pooled_layer2, layer3), dim=1)
        patches = patches.permute(0, 2, 3, 1).contiguous()
        return functional.normalize(global_features, dim=1), functional.normalize(patches, dim=3)


def _image_tensor(path: Path, image_size: int) -> Tensor:
    try:
        with Image.open(path) as source:
            rgb = source.convert("RGB")
            fitted = ImageOps.fit(
                rgb,
                (image_size, image_size),
                method=Image.Resampling.BILINEAR,
                centering=(0.5, 0.5),
            )
            pixels = np.asarray(fitted, dtype=np.float32) / 255.0
    except (OSError, ValueError) as exc:
        raise BaselineError(f"cannot read image {path}: {exc}") from exc
    tensor = torch.from_numpy(pixels).permute(2, 0, 1)
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
    standard_deviation = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / standard_deviation


class FrozenResNet18:
    """Extract frozen ImageNet ResNet-18 global and intermediate patch features."""

    weights_id = "torchvision.ResNet18_Weights.IMAGENET1K_V1"
    weights_url = ResNet18_Weights.DEFAULT.url

    def __init__(self, *, device: str, image_size: int = 224, batch_size: int = 32) -> None:
        if image_size < 64 or image_size % 32:
            raise BaselineError("image_size must be at least 64 and divisible by 32")
        if batch_size < 1:
            raise BaselineError("batch_size must be positive")
        if device == "cuda" and not torch.cuda.is_available():
            raise BaselineError("CUDA was requested but is unavailable")
        self.device = torch.device(device)
        self.image_size = image_size
        self.batch_size = batch_size
        self.model = _ResNetFeatureModule().eval().to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def extract(self, paths: Sequence[Path]) -> FrozenFeatures:
        """Extract deterministic features without gradients or augmentation."""

        if not paths:
            raise BaselineError("at least one image path is required")
        global_batches: list[np.ndarray] = []
        patch_batches: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(paths), self.batch_size):
                batch_paths = paths[start : start + self.batch_size]
                inputs = torch.stack(
                    [_image_tensor(path, self.image_size) for path in batch_paths]
                ).to(self.device, non_blocking=True)
                global_features, patch_features = self.model(inputs)
                global_batches.append(global_features.cpu().numpy().astype(np.float32))
                patch_batches.extend(patch_features.cpu().numpy().astype(np.float32, copy=False))
        return FrozenFeatures(
            global_features=np.concatenate(global_batches),
            patch_features=tuple(patch_batches),
        )


class GlobalFeatureNearestNeighbour:
    """Nearest normal image in standardized frozen global-feature space."""

    def __init__(self) -> None:
        self._scaler = StandardScaler()
        self._neighbours = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1)
        self._fitted = False

    def fit(self, features: np.ndarray) -> None:
        array = np.asarray(features, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] < 2:
            raise BaselineError("at least two global training features are required")
        self._neighbours.fit(self._scaler.fit_transform(array))
        self._fitted = True

    def score(self, features: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise BaselineError("global baseline must be fitted before scoring")
        array = np.asarray(features, dtype=np.float32)
        if array.ndim != 2:
            raise BaselineError("global features must be two-dimensional")
        distances, _ = self._neighbours.kneighbors(self._scaler.transform(array))
        return cast(np.ndarray, distances[:, 0].astype(np.float64, copy=False))


class PatchMemoryBank:
    """Deterministically sampled normal-patch memory with random projection."""

    def __init__(
        self,
        *,
        seed: int,
        bank_size: int,
        projected_dimension: int,
        patches_per_image: int,
        image_score_fraction: float,
    ) -> None:
        if min(bank_size, projected_dimension, patches_per_image) < 1:
            raise BaselineError("patch-memory sizes must be positive")
        if not 0.0 < image_score_fraction <= 1.0:
            raise BaselineError("image_score_fraction must be in (0, 1]")
        self.seed = seed
        self.bank_size = bank_size
        self.projected_dimension = projected_dimension
        self.patches_per_image = patches_per_image
        self.image_score_fraction = image_score_fraction
        self._projection: np.ndarray | None = None
        self._neighbours: NearestNeighbors | None = None

    def _project(self, patches: np.ndarray) -> np.ndarray:
        if self._projection is None:
            raise BaselineError("patch memory must be fitted before projection")
        projected = np.asarray(patches, dtype=np.float32) @ self._projection
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        return cast(np.ndarray, projected / np.maximum(norms, np.finfo(np.float32).eps))

    def fit(self, feature_maps: Sequence[np.ndarray]) -> None:
        if len(feature_maps) < 2:
            raise BaselineError("at least two normal feature maps are required")
        rng = np.random.default_rng(self.seed)
        sampled: list[np.ndarray] = []
        feature_dimension: int | None = None
        for feature_map in feature_maps:
            array = np.asarray(feature_map, dtype=np.float32)
            if array.ndim != 3:
                raise BaselineError("patch feature maps must be height x width x channels")
            flattened = array.reshape(-1, array.shape[-1])
            feature_dimension = flattened.shape[1]
            count = min(self.patches_per_image, flattened.shape[0])
            indexes = rng.choice(flattened.shape[0], size=count, replace=False)
            sampled.append(flattened[indexes])
        assert feature_dimension is not None
        projection_rng = np.random.default_rng(self.seed + 1)
        self._projection = projection_rng.standard_normal(
            (feature_dimension, self.projected_dimension)
        ).astype(np.float32) / np.sqrt(self.projected_dimension)
        candidates = self._project(np.concatenate(sampled))
        if candidates.shape[0] > self.bank_size:
            indexes = rng.choice(candidates.shape[0], size=self.bank_size, replace=False)
            candidates = candidates[indexes]
        neighbours = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1)
        neighbours.fit(candidates)
        self._neighbours = neighbours

    def score(
        self, feature_maps: Sequence[np.ndarray]
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        if self._neighbours is None:
            raise BaselineError("patch memory must be fitted before scoring")
        image_scores: list[float] = []
        score_maps: list[np.ndarray] = []
        for feature_map in feature_maps:
            array = np.asarray(feature_map, dtype=np.float32)
            if array.ndim != 3:
                raise BaselineError("patch feature maps must be height x width x channels")
            height, width, channels = array.shape
            projected = self._project(array.reshape(-1, channels))
            distances, _ = self._neighbours.kneighbors(projected)
            score_map = distances[:, 0].reshape(height, width).astype(np.float32)
            count = max(1, int(np.ceil(score_map.size * self.image_score_fraction)))
            top_scores = np.partition(score_map.reshape(-1), -count)[-count:]
            image_scores.append(float(np.mean(top_scores)))
            score_maps.append(score_map)
        return np.asarray(image_scores, dtype=np.float64), tuple(score_maps)

"""Deterministic overlapping capture tiles with normalized, edge-safe stitching."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageOps

from robust_industrial_anomaly_detection.baselines.encoders import SpatialEncoder
from robust_industrial_anomaly_detection.baselines.multiscale import (
    ProjectedPatchMemory,
    compose_feature_map,
)

ViewMode = Literal["global", "tiled"]


@dataclass(frozen=True)
class Tile:
    top: int
    left: int
    size: int


def tile_layout(height: int, width: int, tile_size: int, stride: int) -> tuple[Tile, ...]:
    """Cover a rectangular frame, anchoring final windows to the right/bottom."""

    if not 0 < stride <= tile_size <= min(height, width):
        raise ValueError("tiles require positive stride <= tile size <= frame dimensions")

    def origins(length: int) -> list[int]:
        values = list(range(0, length - tile_size + 1, stride))
        if values[-1] != length - tile_size:
            values.append(length - tile_size)
        return values

    return tuple(Tile(top, left, tile_size) for top in origins(height) for left in origins(width))


def blend_maps(
    maps: Sequence[np.ndarray],
    tiles: Sequence[Tile],
    shape: tuple[int, int],
    *,
    floor: float = 0.05,
) -> np.ndarray:
    """Blend all tile scores, retaining finite, positive coverage at frame edges."""

    if not maps or len(maps) != len(tiles) or not 0 < floor <= 1 or min(shape) < 1:
        raise ValueError("invalid tile maps, frame shape, or blend floor")
    total = np.zeros(shape, dtype=np.float64)
    coverage = np.zeros(shape, dtype=np.float64)
    for values, tile in zip(maps, tiles, strict=True):
        source = np.asarray(values, dtype=np.float32)
        if source.ndim != 2 or not np.all(np.isfinite(source)):
            raise ValueError("tile maps must be finite two-dimensional arrays")
        if (
            min(tile.top, tile.left) < 0
            or tile.size < 1
            or (tile.top + tile.size > shape[0] or tile.left + tile.size > shape[1])
        ):
            raise ValueError("tile extends outside the frame")
        axis = floor + (1 - floor) * np.sin(np.pi * (np.arange(tile.size) + 0.5) / tile.size) ** 2
        weight = axis[:, None] * axis[None, :]
        region = np.s_[tile.top : tile.top + tile.size, tile.left : tile.left + tile.size]
        resized = cv2.resize(source, (tile.size, tile.size), interpolation=cv2.INTER_LINEAR)
        total[region] += resized * weight
        coverage[region] += weight
    if np.any(coverage <= 0):
        raise ValueError("tile layout leaves uncovered pixels")
    return np.asarray(total / coverage, dtype=np.float32)


def aligned_canvas(
    image: Image.Image, mask: Image.Image | None, *, dataset: str, size: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Align RGB and conservative positive-coverage masks on one declared field of view."""

    if dataset not in {"visa", "ksdd2"} or size < 16:
        raise ValueError("unknown canvas geometry or invalid size")
    if mask is not None and mask.size != image.size:
        raise ValueError("image and source mask dimensions differ")
    rgb = image.convert("RGB")
    if mask is None:
        source_positive = 0
        floating = Image.fromarray(np.zeros((image.height, image.width), dtype=np.float32))
    else:
        positive = np.asarray(mask.convert("L")) > 0
        source_positive = int(np.count_nonzero(positive))
        floating = Image.fromarray(positive.astype(np.float32))
    if dataset == "visa":
        rgb = ImageOps.fit(rgb, (size, size), method=Image.Resampling.BILINEAR)
        floating = ImageOps.fit(floating, (size, size), method=Image.Resampling.BOX)
    else:
        rgb = rgb.resize((size, size), Image.Resampling.BILINEAR)
        floating = floating.resize((size, size), Image.Resampling.BOX)
    return (
        np.asarray(rgb, dtype=np.uint8),
        np.asarray(floating, dtype=np.float32) > 0,
        source_positive,
    )


def view_features(
    encoder: SpatialEncoder,
    images: Sequence[np.ndarray],
    *,
    mode: ViewMode,
    input_size: int,
    tile_size: int,
    stride: int,
) -> list[list[np.ndarray]]:
    """Extract global or tile features without multiplying the per-capture fit budget."""

    if mode not in {"global", "tiled"} or not images or input_size != tile_size:
        raise ValueError("unsupported view mode or incompatible tile/input sizes")
    views: list[np.ndarray] = []
    counts: list[int] = []
    for image in images:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("view inputs must be RGB uint8")
        if mode == "global":
            views.append(cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_AREA))
            counts.append(1)
        else:
            layout = tile_layout(image.shape[0], image.shape[1], tile_size, stride)
            views.extend(image[t.top : t.top + t.size, t.left : t.left + t.size] for t in layout)
            counts.append(len(layout))
    features = [compose_feature_map(value, "multiscale") for value in encoder.extract_rgb(views)]
    if len(features) != len(views):
        raise ValueError("encoder returned a different number of spatial views")
    grouped = []
    offset = 0
    for count in counts:
        grouped.append(features[offset : offset + count])
        offset += count
    return grouped


def score_views(
    memory: ProjectedPatchMemory,
    features: Sequence[np.ndarray],
    *,
    mode: ViewMode,
    canvas_size: int,
    tile_size: int,
    stride: int,
    floor: float,
    sigma: float,
    score_fraction: float,
) -> tuple[float, np.ndarray]:
    """Aggregate image evidence on a common field and smooth only after blending."""

    if mode not in {"global", "tiled"} or sigma < 0 or not 0 < score_fraction <= 1:
        raise ValueError("invalid view aggregation settings")
    _, maps = memory.score(features)
    if mode == "global":
        if len(maps) != 1:
            raise ValueError("global readout requires exactly one view")
        fine = cv2.resize(maps[0], (canvas_size, canvas_size), interpolation=cv2.INTER_LINEAR)
    else:
        fine = blend_maps(
            maps,
            tile_layout(canvas_size, canvas_size, tile_size, stride),
            (canvas_size, canvas_size),
            floor=floor,
        )
    count = max(1, int(np.ceil(fine.size * score_fraction)))
    score = float(np.mean(np.partition(fine.reshape(-1), -count)[-count:]))
    evidence = cv2.GaussianBlur(fine, (0, 0), sigmaX=sigma) if sigma else fine
    return score, np.asarray(evidence, dtype=np.float32)


def evaluation_support(values: np.ndarray, threshold: float, *, grid: int) -> np.ndarray:
    """Threshold at the calibrated readout resolution, then lift support to the common grid."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or not np.all(np.isfinite(array)) or not np.isfinite(threshold) or grid < 1:
        raise ValueError("invalid evidence or readout threshold")
    coarse = cv2.resize(array, (grid, grid), interpolation=cv2.INTER_LINEAR)
    support = (coarse > threshold).astype(np.uint8)
    return cv2.resize(
        support, (array.shape[1], array.shape[0]), interpolation=cv2.INTER_NEAREST
    ).astype(bool)

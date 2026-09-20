"""Deterministic colour and texture sanity baseline."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]


class BaselineError(RuntimeError):
    """Raised when a baseline cannot read or score its inputs."""


def _read_rgb(path: Path, size: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise BaselineError(f"cannot read image: {path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


def _channel_statistics(image: np.ndarray) -> np.ndarray:
    values: list[float] = []
    for channel in cv2.split(image):
        normalized = channel.astype(np.float32) / 255.0
        values.extend(
            (
                float(np.mean(normalized)),
                float(np.std(normalized)),
                float(np.quantile(normalized, 0.10)),
                float(np.quantile(normalized, 0.50)),
                float(np.quantile(normalized, 0.90)),
            )
        )
        histogram = cv2.calcHist([channel], [0], None, [16], [0, 256]).reshape(-1)
        values.extend((histogram / max(float(histogram.sum()), 1.0)).tolist())
    return np.asarray(values, dtype=np.float32)


def _lbp_histogram(gray: np.ndarray) -> np.ndarray:
    center = gray[1:-1, 1:-1]
    neighbours = (
        gray[:-2, :-2],
        gray[:-2, 1:-1],
        gray[:-2, 2:],
        gray[1:-1, 2:],
        gray[2:, 2:],
        gray[2:, 1:-1],
        gray[2:, :-2],
        gray[1:-1, :-2],
    )
    codes = np.zeros(center.shape, dtype=np.uint8)
    for bit, neighbour in enumerate(neighbours):
        codes |= (neighbour >= center).astype(np.uint8) << bit
    histogram = np.bincount(codes.reshape(-1), minlength=256).astype(np.float32)
    return histogram / max(float(histogram.sum()), 1.0)


def _gradient_histogram(gray: np.ndarray) -> np.ndarray:
    floating = gray.astype(np.float32) / 255.0
    dx = cv2.Sobel(floating, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(floating, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(dx, dy, angleInDegrees=True)
    histogram, _ = np.histogram(angle, bins=18, range=(0.0, 360.0), weights=magnitude)
    histogram = histogram.astype(np.float32)
    return histogram / max(float(histogram.sum()), 1.0)


def extract_handcrafted(path: Path, *, size: int = 128) -> np.ndarray:
    """Extract interpretable global colour, LBP, and gradient descriptors."""

    if size < 16:
        raise BaselineError("descriptor image size must be at least 16")
    rgb = _read_rgb(path, size)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return np.concatenate(
        (
            _channel_statistics(rgb),
            _channel_statistics(hsv),
            _lbp_histogram(gray),
            _gradient_histogram(gray),
        )
    ).astype(np.float32, copy=False)


class HandcraftedNearestNeighbour:
    """One-nearest-neighbour detector in standardized handcrafted-feature space."""

    def __init__(self) -> None:
        self._scaler = StandardScaler()
        self._neighbours = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1)
        self._fitted = False

    def fit(self, features: np.ndarray) -> None:
        """Fit only on normal training descriptors."""

        array = np.asarray(features, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] < 2:
            raise BaselineError("at least two two-dimensional training descriptors are required")
        standardized = self._scaler.fit_transform(array)
        self._neighbours.fit(standardized)
        self._fitted = True

    def score(self, features: np.ndarray) -> np.ndarray:
        """Return distance to the closest normal training descriptor."""

        if not self._fitted:
            raise BaselineError("baseline must be fitted before scoring")
        array = np.asarray(features, dtype=np.float32)
        if array.ndim != 2:
            raise BaselineError("features must be two-dimensional")
        distances, _ = self._neighbours.kneighbors(self._scaler.transform(array))
        return cast(np.ndarray, distances[:, 0].astype(np.float64, copy=False))


def extract_many(paths: Sequence[Path], *, size: int) -> np.ndarray:
    """Extract descriptors in stable input order."""

    if not paths:
        raise BaselineError("at least one image path is required")
    return np.stack([extract_handcrafted(path, size=size) for path in paths])

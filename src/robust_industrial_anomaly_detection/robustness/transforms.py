"""Deterministic image/mask transformations for controlled robustness tests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal, cast

import cv2
import numpy as np

ShiftFamily = Literal[
    "clean",
    "jpeg",
    "resize",
    "blur",
    "noise",
    "photometric",
    "geometric",
]


class RobustnessTransformError(ValueError):
    """Raised when a shift definition or input violates the protocol."""


@dataclass(frozen=True)
class ShiftSpec:
    """One frozen shift condition from a severity ladder."""

    identifier: str
    family: ShiftFamily
    severity: int
    parameters: dict[str, float | int]

    def __post_init__(self) -> None:
        if not self.identifier or self.severity < 0:
            raise RobustnessTransformError("shift identifier and severity are required")
        required: dict[ShiftFamily, set[str]] = {
            "clean": set(),
            "jpeg": {"quality"},
            "resize": {"scale"},
            "blur": {"sigma"},
            "noise": {"standard_deviation"},
            "photometric": {"brightness", "contrast"},
            "geometric": {"angle_degrees", "translate_x", "translate_y"},
        }
        if set(self.parameters) != required[self.family]:
            raise RobustnessTransformError(
                f"{self.identifier} parameters must be {sorted(required[self.family])}"
            )
        if self.family == "clean" and self.severity != 0:
            raise RobustnessTransformError("clean severity must be zero")


@dataclass(frozen=True)
class ShiftedPair:
    """Shifted RGB/mask pair plus an inverse map for explanation alignment."""

    image: np.ndarray
    mask: np.ndarray
    inverse_affine: np.ndarray | None


def _validate_inputs(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(image)
    binary = np.asarray(mask, dtype=bool)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise RobustnessTransformError("image must be an RGB uint8 array")
    if binary.shape != rgb.shape[:2]:
        raise RobustnessTransformError("mask must match the image height and width")
    return rgb, binary


def _rng(seed: int, key: str, identifier: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{key}:{identifier}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def apply_shift(
    image: np.ndarray,
    mask: np.ndarray,
    spec: ShiftSpec,
    *,
    seed: int,
    sample_key: str,
) -> ShiftedPair:
    """Apply a deterministic condition, preserving image/mask correspondence."""

    rgb, binary = _validate_inputs(image, mask)
    height, width = binary.shape
    inverse: np.ndarray | None = None
    if spec.family == "clean":
        shifted = rgb.copy()
    elif spec.family == "jpeg":
        quality = int(spec.parameters["quality"])
        if not 1 <= quality <= 100:
            raise RobustnessTransformError("JPEG quality must be in [1, 100]")
        ok, encoded = cv2.imencode(
            ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        if not ok:
            raise RobustnessTransformError("JPEG encoding failed")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise RobustnessTransformError("JPEG decoding failed")
        shifted = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    elif spec.family == "resize":
        scale = float(spec.parameters["scale"])
        if not 0.1 <= scale < 1.0:
            raise RobustnessTransformError("resize scale must be in [0.1, 1.0)")
        small = cv2.resize(
            rgb,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        shifted = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    elif spec.family == "blur":
        sigma = float(spec.parameters["sigma"])
        if sigma <= 0.0:
            raise RobustnessTransformError("blur sigma must be positive")
        shifted = cv2.GaussianBlur(rgb, (0, 0), sigmaX=sigma, sigmaY=sigma)
    elif spec.family == "noise":
        standard_deviation = float(spec.parameters["standard_deviation"])
        if not 0.0 < standard_deviation < 1.0:
            raise RobustnessTransformError("noise standard deviation must be in (0, 1)")
        noise = _rng(seed, sample_key, spec.identifier).normal(
            0.0, 255.0 * standard_deviation, size=rgb.shape
        )
        shifted = np.clip(rgb.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)
    elif spec.family == "photometric":
        brightness = float(spec.parameters["brightness"])
        contrast = float(spec.parameters["contrast"])
        if brightness <= 0.0 or contrast <= 0.0:
            raise RobustnessTransformError("brightness and contrast must be positive")
        values = rgb.astype(np.float32) / 255.0
        channel_mean = np.mean(values, axis=(0, 1), keepdims=True)
        values = (values - channel_mean) * contrast + channel_mean
        shifted = np.clip(values * brightness * 255.0, 0.0, 255.0).astype(np.uint8)
    else:
        angle = float(spec.parameters["angle_degrees"])
        translate_x = float(spec.parameters["translate_x"])
        translate_y = float(spec.parameters["translate_y"])
        matrix = cv2.getRotationMatrix2D(
            ((width - 1) / 2.0, (height - 1) / 2.0), angle, 1.0
        ).astype(np.float32)
        matrix[0, 2] += translate_x
        matrix[1, 2] += translate_y
        shifted = cv2.warpAffine(
            rgb,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        binary = cv2.warpAffine(
            binary.astype(np.uint8),
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)
        inverse = cast(np.ndarray, cv2.invertAffineTransform(matrix)).astype(np.float32)
    return ShiftedPair(shifted.astype(np.uint8, copy=False), binary, inverse)


def align_score_map(score_map: np.ndarray, inverse_affine: np.ndarray | None) -> np.ndarray:
    """Align a shifted explanation map back to clean image coordinates."""

    values = np.asarray(score_map, dtype=np.float32)
    if values.ndim != 2:
        raise RobustnessTransformError("score maps must be two-dimensional")
    if inverse_affine is None:
        return values.copy()
    return cv2.warpAffine(
        values,
        inverse_affine,
        (values.shape[1], values.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    ).astype(np.float32, copy=False)


def image_descriptors(image: np.ndarray) -> tuple[float, float]:
    """Return normalized luminance and edge energy for shortcut diagnostics."""

    rgb = np.asarray(image, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise RobustnessTransformError("image must be RGB")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    horizontal = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    vertical = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(horizontal * horizontal + vertical * vertical)
    return float(np.mean(gray)), float(np.mean(edge))

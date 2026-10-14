"""Normal-only spatial calibration and seeded thresholded localization."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

LocalizationPolicy = Literal[
    "baseline_global", "raw_cleaned", "raw_hysteresis", "spatial_global", "spatial_hysteresis"
]
POLICIES: tuple[LocalizationPolicy, ...] = (
    "baseline_global",
    "raw_cleaned",
    "raw_hysteresis",
    "spatial_global",
    "spatial_hysteresis",
)


class LocalizationError(ValueError):
    """Localization input or artifact violates the calibrated-map contract."""


@dataclass(frozen=True)
class LocalizationModel:
    center: np.ndarray
    scale: np.ndarray
    raw_high: float
    raw_low: float
    spatial_high: float
    spatial_low: float
    spatial_global_high: float
    minimum_region_pixels: int
    growth_radius: int
    fit_samples: int
    calibration_samples: int
    normal_pixel_budget: float
    calibration_raw_support_rate: float
    calibration_spatial_support_rate: float
    provenance: Mapping[str, Any]


def _map(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.ndim != 2 or min(result.shape) < 2 or not np.all(np.isfinite(result)):
        raise LocalizationError("score map must be a finite two-dimensional grid")
    return result


def seeded_regions(
    score_map: np.ndarray,
    *,
    high: float,
    low: float,
    minimum_pixels: int,
    growth_radius: int | None = None,
) -> np.ndarray:
    """Retain low-threshold components only when seeded by high-threshold evidence."""

    values = _map(score_map)
    if not np.isfinite(high) or not np.isfinite(low) or high < low or minimum_pixels < 1:
        raise LocalizationError("region thresholds or minimum area are invalid")
    support = (values > low).astype(np.uint8)
    if growth_radius is not None:
        if growth_radius < 0:
            raise LocalizationError("growth radius cannot be negative")
        seeds = (values > high).astype(np.uint8)
        kernel = np.ones((2 * growth_radius + 1, 2 * growth_radius + 1), np.uint8)
        support &= cv2.dilate(seeds, kernel).astype(np.uint8)
    count, components, stats, _ = cv2.connectedComponentsWithStats(support, connectivity=8)
    seeded = np.zeros(count, dtype=bool)
    seeded[np.unique(components[values > high])] = True
    seeded[0] = False
    seeded &= stats[:, cv2.CC_STAT_AREA] >= minimum_pixels
    return np.asarray(seeded[components], dtype=bool)


def spatial_scores(score_map: np.ndarray, model: LocalizationModel) -> np.ndarray:
    """Compare each pixel against its fitted normal location/variation."""

    values = _map(score_map)
    if values.shape != model.center.shape:
        raise LocalizationError("score map geometry does not match spatial calibration")
    return np.asarray((values - model.center) / model.scale, dtype=np.float32)


def localization_support(
    score_map: np.ndarray,
    model: LocalizationModel,
    policy: LocalizationPolicy,
    *,
    baseline_threshold: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return evidence map, exact binary support, and its strong-evidence threshold."""

    if policy not in POLICIES:
        raise LocalizationError("unsupported localization policy")
    values = spatial_scores(score_map, model) if policy.startswith("spatial") else _map(score_map)
    if policy == "baseline_global":
        return values, values > baseline_threshold, baseline_threshold
    if policy == "raw_cleaned":
        support = seeded_regions(
            values,
            high=baseline_threshold,
            low=baseline_threshold,
            minimum_pixels=model.minimum_region_pixels,
            growth_radius=0,
        )
        return values, support, baseline_threshold
    high = model.spatial_high if policy.startswith("spatial") else model.raw_high
    if policy == "spatial_global":
        high = model.spatial_global_high
    low = model.spatial_low if policy.startswith("spatial") else model.raw_low
    if policy.endswith("hysteresis"):
        support = seeded_regions(
            values,
            high=high,
            low=low,
            minimum_pixels=model.minimum_region_pixels,
            growth_radius=model.growth_radius,
        )
    else:
        support = seeded_regions(
            values, high=high, low=high, minimum_pixels=model.minimum_region_pixels, growth_radius=0
        )
    return values, support, high


def _calibrate_regions(
    maps: np.ndarray,
    *,
    high_quantile: float,
    low_quantile: float,
    minimum_pixels: int,
    budget: float,
    steps: int,
    growth_radius: int,
) -> tuple[float, float, float]:
    low = float(np.quantile(maps, low_quantile, method="higher"))
    initial = float(np.quantile(maps, high_quantile, method="higher"))

    def rate(high: float) -> float:
        return (
            sum(
                int(
                    np.count_nonzero(
                        seeded_regions(
                            item,
                            high=high,
                            low=low,
                            minimum_pixels=minimum_pixels,
                            growth_radius=growth_radius,
                        )
                    )
                )
                for item in maps
            )
            / maps.size
        )

    if rate(initial) <= budget:
        return initial, low, rate(initial)
    left, right = initial, float(np.max(maps))
    for _ in range(steps):
        midpoint = (left + right) / 2.0
        if rate(midpoint) <= budget:
            right = midpoint
        else:
            left = midpoint
    return right, low, rate(right)


def fit_localization_model(
    fit_maps: Sequence[np.ndarray],
    calibration_maps: Sequence[np.ndarray],
    *,
    high_quantile: float,
    low_quantile: float,
    minimum_region_fraction: float,
    scale_floor_fraction: float,
    normal_pixel_budget: float,
    search_steps: int,
    provenance: Mapping[str, Any],
) -> LocalizationModel:
    """Fit spatial references on normals and calibrate support on separate normals."""

    if len(fit_maps) < 16 or len(calibration_maps) < 16:
        raise LocalizationError("localization requires at least 16 fit and calibration maps")
    if not 0 < low_quantile <= high_quantile < 1:
        raise LocalizationError("localization quantiles must satisfy 0 < low <= high < 1")
    if not 0 < minimum_region_fraction < 1 or not 0 < normal_pixel_budget < 1:
        raise LocalizationError("region area and normal pixel budget must be probabilities")
    if not 0 < scale_floor_fraction <= 1 or search_steps < 1:
        raise LocalizationError("scale floor and search steps must be positive")
    fit = np.stack([_map(item) for item in fit_maps])
    calibration = np.stack([_map(item) for item in calibration_maps])
    if fit.shape[1:] != calibration.shape[1:]:
        raise LocalizationError("fit and calibration map geometry differs")
    center = np.median(fit, axis=0).astype(np.float32)
    mad = (1.4826 * np.median(np.abs(fit - center), axis=0)).astype(np.float32)
    floor = max(float(np.median(mad)) * scale_floor_fraction, 1e-5)
    scale = np.maximum(mad, floor).astype(np.float32)
    minimum_pixels = max(1, math.ceil(center.size * minimum_region_fraction))
    # Half a cell of the frozen 14x14 feature lattice, expressed on the evidence grid.
    growth_radius = max(1, math.ceil(center.shape[0] / 28))
    raw_high, raw_low, raw_rate = _calibrate_regions(
        calibration,
        high_quantile=high_quantile,
        low_quantile=low_quantile,
        minimum_pixels=minimum_pixels,
        budget=normal_pixel_budget,
        steps=search_steps,
        growth_radius=growth_radius,
    )
    spatial_high, spatial_low, spatial_rate = _calibrate_regions(
        (calibration - center) / scale,
        high_quantile=high_quantile,
        low_quantile=low_quantile,
        minimum_pixels=minimum_pixels,
        budget=normal_pixel_budget,
        steps=search_steps,
        growth_radius=growth_radius,
    )
    spatial_global_high = float(
        np.quantile((calibration - center) / scale, high_quantile, method="higher")
    )
    return LocalizationModel(
        center,
        scale,
        raw_high,
        raw_low,
        spatial_high,
        spatial_low,
        spatial_global_high,
        minimum_pixels,
        growth_radius,
        len(fit_maps),
        len(calibration_maps),
        normal_pixel_budget,
        raw_rate,
        spatial_rate,
        provenance,
    )


def save_localization_model(path: Path, model: LocalizationModel) -> None:
    """Atomically serialize numeric maps and a JSON-only contract without pickle."""

    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 3,
        "raw_high": model.raw_high,
        "raw_low": model.raw_low,
        "spatial_high": model.spatial_high,
        "spatial_low": model.spatial_low,
        "spatial_global_high": model.spatial_global_high,
        "minimum_region_pixels": model.minimum_region_pixels,
        "growth_radius": model.growth_radius,
        "fit_samples": model.fit_samples,
        "calibration_samples": model.calibration_samples,
        "normal_pixel_budget": model.normal_pixel_budget,
        "calibration_raw_support_rate": model.calibration_raw_support_rate,
        "calibration_spatial_support_rate": model.calibration_spatial_support_rate,
        "provenance": dict(model.provenance),
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                center=model.center,
                scale=model.scale,
                metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_localization_model(path: Path) -> LocalizationModel:
    """Load and validate the normal-only spatial calibration contract."""

    try:
        with np.load(path, allow_pickle=False) as payload:
            center = _map(payload["center"])
            scale = _map(payload["scale"])
            metadata = json.loads(str(payload["metadata"].item()))
        if metadata["schema_version"] != 3 or center.shape != scale.shape or np.any(scale <= 0):
            raise LocalizationError("spatial artifact schema, shape, or scale is invalid")
        high_low = [
            float(metadata[key]) for key in ("raw_high", "raw_low", "spatial_high", "spatial_low")
        ]
        if (
            not np.all(np.isfinite(high_low))
            or high_low[0] < high_low[1]
            or high_low[2] < high_low[3]
        ):
            raise LocalizationError("spatial artifact thresholds are invalid")
        global_high = float(metadata["spatial_global_high"])
        if not np.isfinite(global_high) or global_high < high_low[3]:
            raise LocalizationError("spatial global threshold is invalid")
        if min(metadata["fit_samples"], metadata["calibration_samples"]) < 16:
            raise LocalizationError("spatial artifact normal counts are invalid")
        if not 1 <= metadata["minimum_region_pixels"] <= center.size:
            raise LocalizationError("spatial artifact region area is invalid")
        if not 1 <= metadata["growth_radius"] <= min(center.shape):
            raise LocalizationError("spatial artifact growth radius is invalid")
        budget = float(metadata["normal_pixel_budget"])
        rates = [
            float(metadata[key])
            for key in ("calibration_raw_support_rate", "calibration_spatial_support_rate")
        ]
        if not 0 < budget < 1 or any(not 0 <= value <= budget + 1e-8 for value in rates):
            raise LocalizationError("spatial artifact calibration exceeds its normal budget")
        if not isinstance(metadata["provenance"], dict):
            raise LocalizationError("spatial artifact provenance is invalid")
        return LocalizationModel(
            center,
            scale,
            high_low[0],
            high_low[1],
            high_low[2],
            high_low[3],
            global_high,
            metadata["minimum_region_pixels"],
            metadata["growth_radius"],
            metadata["fit_samples"],
            metadata["calibration_samples"],
            budget,
            rates[0],
            rates[1],
            metadata["provenance"],
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        if isinstance(exc, LocalizationError):
            raise
        raise LocalizationError(f"cannot load localization artifact: {path.name}") from exc

"""M4 controlled robustness, transfer, and failure-analysis experiment."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from urllib.parse import urlparse

import cv2
import numpy as np
import yaml
from dotenv import load_dotenv
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
    roc_auc_score,
)

from robust_industrial_anomaly_detection.baselines.multiscale import (
    FrozenPyramidResNet18,
    ProjectedPatchMemory,
    compose_feature_map,
    load_fitted_rgb,
    load_projected_index,
)
from robust_industrial_anomaly_detection.data.visa import load_trusted_manifest, sha256_file
from robust_industrial_anomaly_detection.evaluation.metrics import (
    calibrate_threshold,
    image_metrics,
    pixel_metrics,
    pro_auc,
)
from robust_industrial_anomaly_detection.evaluation.robustness import (
    bootstrap_mean_interval,
    heatmap_stability,
    paired_image_delta_intervals,
    spearman_stability,
)
from robust_industrial_anomaly_detection.experiments.m2 import (
    Sample,
    _atomic_csv,
    _atomic_json,
    _environment,
    _git_metadata,
    _load_masks,
    _read_csv,
    _resize_maps,
    _samples,
)
from robust_industrial_anomaly_detection.robustness.transforms import (
    ShiftFamily,
    ShiftSpec,
    align_score_map,
    apply_shift,
    image_descriptors,
)


class M4Error(RuntimeError):
    """Raised when M4 cannot satisfy its frozen experiment contract."""


@dataclass(frozen=True)
class InterventionSettings:
    identifier: str
    bank_size: int
    projected_dimension: int
    candidate_limit: int
    patches_per_view: int
    augmentations: tuple[ShiftSpec, ...]


@dataclass(frozen=True)
class M4Settings:
    name: str
    seed: int
    bootstrap_samples: int
    bootstrap_confidence: float
    image_threshold_quantile: float
    pixel_threshold_quantile: float
    pro_max_fpr: float
    pro_thresholds: int
    source_experiment: str
    reference_variant: str
    candidate_variant: str
    image_size: int
    batch_size: int
    gaussian_sigma: float
    image_score_fraction: float
    conditions: tuple[ShiftSpec, ...]
    heatmap_top_fraction: float
    transfer_bank_per_category: int
    intervention: InterventionSettings
    quick_train: int
    quick_validation: int
    quick_test_normal: int
    quick_test_anomaly: int
    quick_bootstrap_samples: int
    quick_pro_thresholds: int
    quick_intervention_bank: int
    quick_intervention_candidates: int
    quick_transfer_bank_per_category: int


@dataclass(frozen=True)
class M4RunResult:
    run_dir: Path
    summary: dict[str, object]


@dataclass(frozen=True)
class ModelState:
    identifier: str
    feature_mode: str
    memory: ProjectedPatchMemory
    image_threshold: float
    pixel_threshold: float


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise M4Error(f"{name} must be a mapping")
    return cast(Mapping[str, object], value)


def _sequence(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise M4Error(f"{name} must be a sequence")
    return value


def _string(section: Mapping[str, object], name: str) -> str:
    value = section.get(name)
    if not isinstance(value, str) or not value:
        raise M4Error(f"{name} must be a non-empty string")
    return value


def _integer(section: Mapping[str, object], name: str) -> int:
    value = section.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise M4Error(f"{name} must be a positive integer")
    return value


def _number(section: Mapping[str, object], name: str) -> float:
    value = section.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise M4Error(f"{name} must be numeric")
    return float(value)


def _shift(raw: object, name: str) -> ShiftSpec:
    section = _mapping(raw, name)
    family = _string(section, "family")
    if family not in {"clean", "jpeg", "resize", "blur", "noise", "photometric", "geometric"}:
        raise M4Error(f"unsupported shift family: {family}")
    severity = _integer(section, "severity")
    parameters = _mapping(section.get("parameters"), f"{name}.parameters")
    parsed: dict[str, float | int] = {}
    for key, value in parameters.items():
        if (
            not isinstance(key, str)
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
        ):
            raise M4Error(f"{name} parameters must be numeric")
        parsed[key] = value
    return ShiftSpec(
        _string(section, "id"),
        cast(ShiftFamily, family),
        severity,
        parsed,
    )


def load_m4_settings(path: Path) -> M4Settings:
    """Load and strictly validate the frozen M4 configuration."""

    try:
        root = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "configuration")
    except (OSError, yaml.YAMLError) as exc:
        raise M4Error(f"cannot load M4 configuration {path}: {exc}") from exc
    experiment = _mapping(root.get("experiment"), "experiment")
    source = _mapping(root.get("source_m3"), "source_m3")
    stability = _mapping(root.get("stability"), "stability")
    transfer = _mapping(root.get("transfer"), "transfer")
    intervention_raw = _mapping(root.get("intervention"), "intervention")
    quick = _mapping(root.get("quick"), "quick")
    conditions = tuple(
        _shift(item, f"conditions[{index}]")
        for index, item in enumerate(_sequence(root.get("conditions"), "conditions"))
    )
    augmentations = tuple(
        _shift(item, f"mild_augmentation_pool[{index}]")
        for index, item in enumerate(
            _sequence(
                intervention_raw.get("mild_augmentation_pool"),
                "mild_augmentation_pool",
            )
        )
    )
    families = {condition.family for condition in conditions}
    expected = {"jpeg", "resize", "blur", "noise", "photometric", "geometric"}
    if families != expected:
        raise M4Error(f"conditions must cover exactly {sorted(expected)}")
    for family in expected:
        severities = sorted(
            condition.severity for condition in conditions if condition.family == family
        )
        if severities != [1, 2]:
            raise M4Error(f"{family} must define severity levels 1 and 2")
    if {augmentation.family for augmentation in augmentations} != expected:
        raise M4Error("intervention augmentation pool must cover every shift family once")
    intervention = InterventionSettings(
        identifier=_string(intervention_raw, "id"),
        bank_size=_integer(intervention_raw, "bank_size"),
        projected_dimension=_integer(intervention_raw, "projected_dimension"),
        candidate_limit=_integer(intervention_raw, "candidate_limit"),
        patches_per_view=_integer(intervention_raw, "patches_per_view"),
        augmentations=augmentations,
    )
    settings = M4Settings(
        name=_string(experiment, "name"),
        seed=_integer(experiment, "seed"),
        bootstrap_samples=_integer(experiment, "bootstrap_samples"),
        bootstrap_confidence=_number(experiment, "bootstrap_confidence"),
        image_threshold_quantile=_number(experiment, "image_threshold_quantile"),
        pixel_threshold_quantile=_number(experiment, "pixel_threshold_quantile"),
        pro_max_fpr=_number(experiment, "pro_max_fpr"),
        pro_thresholds=_integer(experiment, "pro_thresholds"),
        source_experiment=_string(source, "experiment_name"),
        reference_variant=_string(source, "reference_variant"),
        candidate_variant=_string(source, "candidate_variant"),
        image_size=_integer(source, "image_size"),
        batch_size=_integer(source, "batch_size"),
        gaussian_sigma=_number(source, "gaussian_sigma"),
        image_score_fraction=_number(source, "image_score_fraction"),
        conditions=conditions,
        heatmap_top_fraction=_number(stability, "heatmap_top_fraction"),
        transfer_bank_per_category=_integer(transfer, "bank_per_source_category"),
        intervention=intervention,
        quick_train=_integer(quick, "train_per_category"),
        quick_validation=_integer(quick, "validation_per_category"),
        quick_test_normal=_integer(quick, "test_normal_per_category"),
        quick_test_anomaly=_integer(quick, "test_anomaly_per_category"),
        quick_bootstrap_samples=_integer(quick, "bootstrap_samples"),
        quick_pro_thresholds=_integer(quick, "pro_thresholds"),
        quick_intervention_bank=_integer(quick, "intervention_bank_size_cap"),
        quick_intervention_candidates=_integer(quick, "intervention_candidate_limit_cap"),
        quick_transfer_bank_per_category=_integer(quick, "transfer_bank_per_source_category"),
    )
    for name in (
        "bootstrap_confidence",
        "image_threshold_quantile",
        "pixel_threshold_quantile",
        "pro_max_fpr",
        "image_score_fraction",
        "heatmap_top_fraction",
    ):
        if not 0.0 < cast(float, getattr(settings, name)) < 1.0:
            raise M4Error(f"{name} must be strictly between zero and one")
    return settings


def _partition(
    samples: Sequence[Sample], settings: M4Settings, quick: bool
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    train = [sample for sample in samples if sample.split == "train"]
    validation = [sample for sample in samples if sample.split == "validation"]
    test = [sample for sample in samples if sample.split == "test"]
    if quick:
        train = train[: settings.quick_train]
        validation = validation[: settings.quick_validation]
        normal = [sample for sample in test if sample.label == "normal"][
            : settings.quick_test_normal
        ]
        anomaly = [sample for sample in test if sample.label == "anomaly"][
            : settings.quick_test_anomaly
        ]
        test = normal + anomaly
    if (
        len(train) < 2
        or not validation
        or {sample.label for sample in test}
        != {
            "normal",
            "anomaly",
        }
    ):
        raise M4Error("every selected category requires train, validation, and both test labels")
    return train, validation, test


def _find_source_run(artifact_root: Path, name: str, explicit: Path | None) -> Path:
    runs_root = (artifact_root / "runs").resolve(strict=True)
    if explicit is not None:
        candidate = explicit.resolve(strict=True)
        try:
            candidate.relative_to(runs_root)
        except ValueError as exc:
            raise M4Error("source M3 run must be below the configured run root") from exc
        candidates = [candidate]
    else:
        candidates = sorted(runs_root.glob(f"{name}-full-*"), reverse=True)
    for candidate in candidates:
        try:
            metadata = json.loads((candidate / "metadata.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if metadata.get("status") == "complete" and metadata.get("mode") == "full":
            return candidate
    raise M4Error("no complete full M3 source run is available")


def _verify_source_run(
    source_run: Path,
    *,
    manifest_hash: str,
    audit_hash: str,
    categories: Sequence[str],
    variants: Sequence[str],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    try:
        metadata = cast(
            dict[str, object], json.loads((source_run / "metadata.json").read_text("utf-8"))
        )
        index_manifest = cast(
            dict[str, dict[str, object]],
            json.loads((source_run / "index_manifest.json").read_text("utf-8")),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise M4Error(f"cannot read M3 source artifacts: {exc}") from exc
    if metadata.get("status") != "complete" or metadata.get("mode") != "full":
        raise M4Error("M3 source run must be complete and full")
    if (
        metadata.get("manifest_sha256") != manifest_hash
        or metadata.get("audit_sha256") != audit_hash
    ):
        raise M4Error("M3 source run used different trusted data inputs")
    for variant in variants:
        for category in categories:
            relative = f"indexes/category_specific/{variant}/{category}.npz"
            entry = index_manifest.get(relative)
            path = source_run / relative
            if entry is None or not path.is_file():
                raise M4Error(f"M3 source index is missing: {relative}")
            if path.stat().st_size != entry.get("bytes") or sha256_file(path) != entry.get(
                "sha256"
            ):
                raise M4Error(f"M3 source index failed integrity verification: {relative}")
    metrics = _read_csv(source_run / "category_metrics.csv")
    return metrics, metadata


def _source_thresholds(
    rows: Sequence[Mapping[str, object]], variant: str, category: str
) -> tuple[float, float]:
    matches = [
        row
        for row in rows
        if row.get("scope") == "category_specific"
        and row.get("variant") == variant
        and row.get("category") == category
    ]
    if len(matches) != 1:
        raise M4Error(f"source metrics missing unique row for {variant}/{category}")
    return float(cast(int | float, matches[0]["threshold"])), float(
        cast(int | float, matches[0]["pixel_threshold"])
    )


def _memory_from_index(path: Path, settings: M4Settings) -> ProjectedPatchMemory:
    bank, projection = load_projected_index(path)
    memory = ProjectedPatchMemory(
        seed=settings.seed,
        bank_size=len(bank),
        projected_dimension=projection.shape[1],
        patches_per_image=1,
        candidate_limit=len(bank),
        image_score_fraction=settings.image_score_fraction,
        selection="kcenter",
    )
    memory.fit_projected(bank, projection)
    return memory


def _stable_index(key: str, count: int) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) % count


def _build_intervention(
    *,
    run_dir: Path,
    category: str,
    train: Sequence[Sample],
    validation: Sequence[Sample],
    extractor: FrozenPyramidResNet18,
    settings: M4Settings,
    quick: bool,
) -> ModelState:
    clean_train = [load_fitted_rgb(sample.image_path, settings.image_size) for sample in train]
    zero_mask = np.zeros((settings.image_size, settings.image_size), dtype=bool)
    augmented: list[np.ndarray] = []
    for sample, image in zip(train, clean_train, strict=True):
        spec = settings.intervention.augmentations[
            _stable_index(sample.sample_id, len(settings.intervention.augmentations))
        ]
        augmented.append(
            apply_shift(
                image,
                zero_mask,
                spec,
                seed=settings.seed,
                sample_key=sample.sample_id,
            ).image
        )
    validation_images = [
        load_fitted_rgb(sample.image_path, settings.image_size) for sample in validation
    ]
    pyramids = extractor.extract_rgb([*clean_train, *augmented, *validation_images])
    maps = tuple(compose_feature_map(pyramid, "multiscale") for pyramid in pyramids)
    training_count = 2 * len(train)
    bank_size = (
        min(settings.intervention.bank_size, settings.quick_intervention_bank)
        if quick
        else settings.intervention.bank_size
    )
    candidate_limit = (
        min(settings.intervention.candidate_limit, settings.quick_intervention_candidates)
        if quick
        else settings.intervention.candidate_limit
    )
    memory = ProjectedPatchMemory(
        seed=settings.seed,
        bank_size=bank_size,
        projected_dimension=settings.intervention.projected_dimension,
        patches_per_image=settings.intervention.patches_per_view,
        candidate_limit=candidate_limit,
        image_score_fraction=settings.image_score_fraction,
        selection="kcenter",
    )
    memory.fit(maps[:training_count])
    validation_scores, validation_raw = memory.score(maps[training_count:])
    validation_maps = _resize_maps(
        validation_raw,
        (settings.image_size, settings.image_size),
        settings.gaussian_sigma,
    )
    image_threshold = calibrate_threshold(validation_scores, settings.image_threshold_quantile)
    pixel_threshold = calibrate_threshold(
        np.concatenate([score_map.reshape(-1) for score_map in validation_maps]),
        settings.pixel_threshold_quantile,
    )
    index_path = run_dir / "indexes" / "intervention" / f"{category}.npz"
    memory.save(
        index_path,
        metadata={
            "category": category,
            "variant": settings.intervention.identifier,
            "train_policy": "normal-only-clean-plus-deterministic-mild-augmentation",
        },
    )
    return ModelState(
        settings.intervention.identifier,
        "multiscale",
        memory,
        image_threshold,
        pixel_threshold,
    )


def _load_category_models(
    *,
    source_run: Path,
    source_metrics: Sequence[Mapping[str, object]],
    run_dir: Path,
    category: str,
    train: Sequence[Sample],
    validation: Sequence[Sample],
    extractor: FrozenPyramidResNet18,
    settings: M4Settings,
    quick: bool,
) -> tuple[ModelState, ...]:
    result: list[ModelState] = []
    for identifier, feature_mode in (
        (settings.reference_variant, "mid"),
        (settings.candidate_variant, "multiscale"),
    ):
        thresholds = _source_thresholds(source_metrics, identifier, category)
        result.append(
            ModelState(
                identifier,
                feature_mode,
                _memory_from_index(
                    source_run / "indexes" / "category_specific" / identifier / f"{category}.npz",
                    settings,
                ),
                thresholds[0],
                thresholds[1],
            )
        )
    result.append(
        _build_intervention(
            run_dir=run_dir,
            category=category,
            train=train,
            validation=validation,
            extractor=extractor,
            settings=settings,
            quick=quick,
        )
    )
    return tuple(result)


def _build_transfer_memory(
    *,
    source_run: Path,
    target: str,
    all_categories: Sequence[str],
    settings: M4Settings,
    quick: bool,
) -> ProjectedPatchMemory:
    per_category = (
        settings.quick_transfer_bank_per_category if quick else settings.transfer_bank_per_category
    )
    banks: list[np.ndarray] = []
    projection: np.ndarray | None = None
    for category in all_categories:
        if category == target:
            continue
        bank, current_projection = load_projected_index(
            source_run
            / "indexes"
            / "category_specific"
            / settings.candidate_variant
            / f"{category}.npz"
        )
        if projection is None:
            projection = current_projection
        elif not np.array_equal(projection, current_projection):
            raise M4Error("source category indexes do not share one projection")
        banks.append(bank[:per_category])
    if projection is None or not banks:
        raise M4Error("leave-one-category-out memory requires source categories")
    combined = np.concatenate(banks)
    memory = ProjectedPatchMemory(
        seed=settings.seed,
        bank_size=len(combined),
        projected_dimension=projection.shape[1],
        patches_per_image=1,
        candidate_limit=len(combined),
        image_score_fraction=settings.image_score_fraction,
        selection="kcenter",
    )
    memory.fit_projected(combined, projection)
    return memory


def _pixel_ranking(masks: Sequence[np.ndarray], maps: Sequence[np.ndarray]) -> tuple[float, float]:
    target = np.concatenate([np.asarray(mask, dtype=np.uint8).reshape(-1) for mask in masks])
    scores = np.concatenate(
        [np.asarray(score_map, dtype=np.float32).reshape(-1) for score_map in maps]
    )
    return float(roc_auc_score(target, scores)), float(average_precision_score(target, scores))


def _condition_inputs(
    images: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    test: Sequence[Sample],
    condition: ShiftSpec,
    settings: M4Settings,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray | None]]:
    shifted_images: list[np.ndarray] = []
    shifted_masks: list[np.ndarray] = []
    inverses: list[np.ndarray | None] = []
    for image, mask, sample in zip(images, masks, test, strict=True):
        pair = apply_shift(
            image,
            mask,
            condition,
            seed=settings.seed,
            sample_key=sample.sample_id,
        )
        shifted_images.append(pair.image)
        shifted_masks.append(pair.mask)
        inverses.append(pair.inverse_affine)
    return shifted_images, shifted_masks, inverses


def _save_failure_panel(
    *,
    run_dir: Path,
    model: str,
    category: str,
    condition: str,
    kind: str,
    sample: Sample,
    clean_image: np.ndarray,
    shifted_image: np.ndarray,
    shifted_mask: np.ndarray,
    shifted_map: np.ndarray,
) -> dict[str, object]:
    minimum = float(np.min(shifted_map))
    scale = max(
        float(np.quantile(shifted_map, 0.995)) - minimum,
        float(np.finfo(np.float32).eps),
    )
    normalized = np.clip((shifted_map - minimum) / scale, 0.0, 1.0)
    heat = cv2.cvtColor(
        cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO),
        cv2.COLOR_BGR2RGB,
    )
    overlay = cv2.addWeighted(shifted_image, 0.55, heat, 0.45, 0.0)
    ground_truth = np.zeros_like(shifted_image)
    ground_truth[shifted_mask] = 255
    panel = np.concatenate((clean_image, shifted_image, overlay, ground_truth), axis=1)
    label = f"{model} | {condition} | {kind}"
    cv2.putText(panel, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    relative = (
        Path("failure_examples") / model / condition / category / f"{sample.sample_id}-{kind}.png"
    )
    destination = run_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)):
        raise M4Error(f"cannot write failure example: {destination}")
    return {
        "model": model,
        "category": category,
        "condition": condition,
        "failure_kind": kind,
        "sample_id": sample.sample_id,
        "path": relative.as_posix(),
    }


def _failure_clusters(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, int, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["model"]),
                str(row["condition"]),
                str(row["family"]),
                int(cast(int | float, row["severity"])),
                str(row["category"]),
            )
        ].append(row)
    result: list[dict[str, object]] = []
    for (model, condition, family, severity, category), group in sorted(grouped.items()):
        labels = np.asarray([int(cast(int | float, row["target"])) for row in group])
        score_delta = np.asarray([float(cast(int | float, row["score_delta"])) for row in group])
        brightness = np.asarray(
            [float(cast(int | float, row["brightness_delta"])) for row in group]
        )
        edge = np.asarray([float(cast(int | float, row["edge_delta"])) for row in group])
        result.append(
            {
                "model": model,
                "condition": condition,
                "family": family,
                "severity": severity,
                "category": category,
                "samples": len(group),
                "new_false_positives": sum(
                    row["transition"] == "new_false_positive" for row in group
                ),
                "new_false_negatives": sum(
                    row["transition"] == "new_false_negative" for row in group
                ),
                "recovered": sum(str(row["transition"]).startswith("recovered") for row in group),
                "failure_rate": float(
                    np.mean([int(cast(int | float, row["failed_under_shift"])) for row in group])
                ),
                "mean_heatmap_correlation": float(
                    np.mean([float(cast(int | float, row["heatmap_correlation"])) for row in group])
                ),
                "mean_heatmap_top_iou": float(
                    np.mean([float(cast(int | float, row["heatmap_top_iou"])) for row in group])
                ),
                "normal_score_delta": float(np.mean(score_delta[labels == 0])),
                "anomaly_score_delta": float(np.mean(score_delta[labels == 1])),
                "score_brightness_spearman": spearman_stability(score_delta, brightness),
                "score_edge_spearman": spearman_stability(score_delta, edge),
            }
        )
    return result


def _aggregate_rows(
    rows: Sequence[Mapping[str, object]], settings: M4Settings, bootstrap_samples: int
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["model"]),
                str(row["condition"]),
                str(row["family"]),
                int(cast(int | float, row["severity"])),
            )
        ].append(row)
    metrics = (
        "image_auroc",
        "image_ap",
        "test_normal_fpr",
        "balanced_accuracy",
        "pixel_auroc",
        "pixel_ap",
        "pixel_iou_mean",
        "aupro_30",
        "score_rank_spearman",
        "heatmap_correlation",
        "heatmap_top_iou",
        "latency_ms_per_image",
    )
    result: list[dict[str, object]] = []
    for (model, condition, family, severity), group in sorted(grouped.items()):
        record: dict[str, object] = {
            "model": model,
            "condition": condition,
            "family": family,
            "severity": severity,
            "categories": len(group),
        }
        for metric in metrics:
            record[f"macro_{metric}"] = float(
                np.mean([float(cast(int | float, row[metric])) for row in group])
            )
        for metric in (
            "image_auroc",
            "image_ap",
            "test_normal_fpr",
            "pixel_auroc",
            "pixel_ap",
            "pixel_iou_mean",
            "aupro_30",
        ):
            values = [float(cast(int | float, row[f"delta_{metric}"])) for row in group]
            low, high = bootstrap_mean_interval(
                values,
                seed=settings.seed + _stable_index(f"{model}:{condition}:{metric}", 1_000_000),
                samples=bootstrap_samples,
                confidence=settings.bootstrap_confidence,
            )
            record[f"macro_delta_{metric}"] = float(np.mean(values))
            record[f"macro_delta_{metric}_ci_low"] = low
            record[f"macro_delta_{metric}_ci_high"] = high
        result.append(record)
    return result


def _family_rows(
    robustness_rows: Sequence[Mapping[str, object]],
    settings: M4Settings,
    bootstrap_samples: int,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in robustness_rows:
        if row["family"] != "clean":
            grouped[(str(row["model"]), str(row["family"]))].append(row)
    result: list[dict[str, object]] = []
    for (model, family), group in sorted(grouped.items()):
        record: dict[str, object] = {
            "model": model,
            "family": family,
            "conditions": len(group),
        }
        for metric in (
            "image_auroc",
            "test_normal_fpr",
            "pixel_auroc",
            "pixel_iou_mean",
            "aupro_30",
        ):
            values = [float(cast(int | float, row[f"macro_delta_{metric}"])) for row in group]
            low, high = bootstrap_mean_interval(
                values,
                seed=settings.seed + _stable_index(f"family:{model}:{family}:{metric}", 1_000_000),
                samples=bootstrap_samples,
                confidence=settings.bootstrap_confidence,
            )
            record[f"mean_delta_{metric}"] = float(np.mean(values))
            record[f"mean_delta_{metric}_ci_low"] = low
            record[f"mean_delta_{metric}_ci_high"] = high
        record["mean_score_rank_spearman"] = float(
            np.mean([float(cast(int | float, row["macro_score_rank_spearman"])) for row in group])
        )
        record["mean_heatmap_correlation"] = float(
            np.mean([float(cast(int | float, row["macro_heatmap_correlation"])) for row in group])
        )
        result.append(record)
    return result


def _difference_test(
    values: Sequence[float], *, name: str, settings: M4Settings, bootstrap_samples: int
) -> dict[str, object]:
    low, high = bootstrap_mean_interval(
        values,
        seed=settings.seed + _stable_index(name, 1_000_000),
        samples=bootstrap_samples,
        confidence=settings.bootstrap_confidence,
    )
    estimate = float(np.mean(values))
    conclusion = "supported" if low > 0.0 else "rejected" if high < 0.0 else "inconclusive"
    return {
        "test": name,
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "confidence": settings.bootstrap_confidence,
        "conclusion": conclusion,
        "positive_direction": "supports the stated hypothesis/intervention",
        "units": len(values),
    }


def _hypothesis_rows(
    metrics: Sequence[Mapping[str, object]],
    stability: Sequence[Mapping[str, object]],
    settings: M4Settings,
    bootstrap_samples: int,
) -> list[dict[str, object]]:
    lookup = {
        (str(row["model"]), str(row["category"]), str(row["condition"])): row for row in metrics
    }
    h2: list[float] = []
    intervention: list[float] = []
    for row in metrics:
        if row["family"] == "clean":
            continue
        key = (str(row["category"]), str(row["condition"]))
        if row["model"] == settings.candidate_variant and row["family"] in {
            "resize",
            "blur",
            "geometric",
        }:
            reference = lookup[(settings.reference_variant, *key)]
            h2.append(
                float(cast(int | float, row["delta_image_auroc"]))
                - float(cast(int | float, reference["delta_image_auroc"]))
            )
        if row["model"] == settings.intervention.identifier:
            candidate = lookup[(settings.candidate_variant, *key)]
            intervention.append(
                float(cast(int | float, row["delta_image_auroc"]))
                - float(cast(int | float, candidate["delta_image_auroc"]))
            )
    clean_correct_rows = [
        row
        for row in stability
        if row["model"] == settings.candidate_variant
        and int(cast(int | float, row["clean_correct"])) == 1
    ]
    failed = [
        float(cast(int | float, row["heatmap_correlation"]))
        for row in clean_correct_rows
        if int(cast(int | float, row["failed_under_shift"])) == 1
    ]
    retained = [
        float(cast(int | float, row["heatmap_correlation"]))
        for row in clean_correct_rows
        if int(cast(int | float, row["failed_under_shift"])) == 0
    ]
    rows = [
        _difference_test(
            h2,
            name="H2_multiscale_image_robustness_advantage",
            settings=settings,
            bootstrap_samples=bootstrap_samples,
        ),
        _difference_test(
            intervention,
            name="augmentation_intervention_image_robustness_advantage",
            settings=settings,
            bootstrap_samples=bootstrap_samples,
        ),
    ]
    if failed and retained:
        rng = np.random.default_rng(settings.seed + 404)
        estimates = []
        for _ in range(bootstrap_samples):
            failed_sample = rng.choice(failed, size=len(failed), replace=True)
            retained_sample = rng.choice(retained, size=len(retained), replace=True)
            estimates.append(float(np.mean(retained_sample) - np.mean(failed_sample)))
        tail = (1.0 - settings.bootstrap_confidence) / 2.0
        low = float(np.quantile(estimates, tail))
        high = float(np.quantile(estimates, 1.0 - tail))
        rows.append(
            {
                "test": "H5_explanation_stability_failure_separation",
                "estimate": float(np.mean(retained) - np.mean(failed)),
                "ci_low": low,
                "ci_high": high,
                "confidence": settings.bootstrap_confidence,
                "conclusion": "supported"
                if low > 0.0
                else "rejected"
                if high < 0.0
                else "inconclusive",
                "positive_direction": "retained-correct heatmaps are more stable than failures",
                "units": len(failed) + len(retained),
                "failed_units": len(failed),
                "retained_units": len(retained),
            }
        )
    else:
        rows.append(
            {
                "test": "H5_explanation_stability_failure_separation",
                "conclusion": "not_estimable",
                "failed_units": len(failed),
                "retained_units": len(retained),
            }
        )
    return rows


def run_m4(
    *,
    project_root: Path,
    experiment_config: Path,
    data_config: Path,
    categories: Sequence[str] | None,
    device: str,
    quick: bool,
    source_m3_run: Path | None = None,
    resume_run: Path | None = None,
) -> M4RunResult:
    """Run the frozen M4 shift, transfer, stability, and intervention protocol."""

    load_dotenv(project_root / ".env")
    settings = load_m4_settings(experiment_config)
    data_root_raw = os.environ.get("ANOMALY_DATA_ROOT")
    artifact_root_raw = os.environ.get("ANOMALY_ARTIFACT_ROOT")
    if not data_root_raw or not artifact_root_raw:
        raise M4Error("ANOMALY_DATA_ROOT and ANOMALY_ARTIFACT_ROOT must be configured")
    data_root = Path(data_root_raw)
    artifact_root = Path(artifact_root_raw)
    manifest_dir = data_root / "manifests" / "visa-20220922"
    image_root = data_root / "sources" / "visa-20220922"
    manifest_path = manifest_dir / "manifest.jsonl"
    audit_path = manifest_dir / "audit.json"
    manifest_hash = sha256_file(manifest_path)
    audit_hash = sha256_file(audit_path)
    records = load_trusted_manifest(manifest_path, audit_path, data_config, image_root)
    all_samples = _samples(records, image_root)
    all_categories = sorted({sample.category for sample in all_samples})
    selected_categories = list(categories) if categories else all_categories
    unknown = sorted(set(selected_categories) - set(all_categories))
    if unknown:
        raise M4Error(f"unknown categories: {unknown}")
    source_run = _find_source_run(artifact_root, settings.source_experiment, source_m3_run)
    source_metrics, source_metadata = _verify_source_run(
        source_run,
        manifest_hash=manifest_hash,
        audit_hash=audit_hash,
        categories=all_categories,
        variants=(settings.reference_variant, settings.candidate_variant),
    )
    mode = "smoke" if quick else "full"
    conditions = (ShiftSpec("clean", "clean", 0, {}), *settings.conditions)
    bootstrap_samples = settings.quick_bootstrap_samples if quick else settings.bootstrap_samples
    pro_thresholds = settings.quick_pro_thresholds if quick else settings.pro_thresholds
    resolved: dict[str, object] = {
        **asdict(settings),
        "conditions": [asdict(condition) for condition in settings.conditions],
        "intervention": {
            **asdict(settings.intervention),
            "augmentations": [asdict(item) for item in settings.intervention.augmentations],
        },
        "mode": mode,
        "categories": selected_categories,
        "device": device,
        "source_m3_run": str(source_run),
        "source_m3_metadata_sha256": sha256_file(source_run / "metadata.json"),
    }
    hashes = {
        "manifest_sha256": manifest_hash,
        "audit_sha256": audit_hash,
        "experiment_config_sha256": sha256_file(experiment_config),
        "data_config_sha256": sha256_file(data_config),
        "source_m3_metadata_sha256": sha256_file(source_run / "metadata.json"),
        "source_m3_index_manifest_sha256": sha256_file(source_run / "index_manifest.json"),
    }
    if resume_run is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_dir = artifact_root / "runs" / f"{settings.name}-{mode}-{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8"
        )
        metadata: dict[str, object] = {
            "schema_version": 1,
            "status": "running",
            "started_at_utc": datetime.now(UTC).isoformat(),
            "mode": mode,
            "seed": settings.seed,
            **hashes,
            "git": _git_metadata(project_root),
            "source_m3": {
                "run": str(source_run),
                "revision": cast(Mapping[str, object], source_metadata.get("git", {})).get(
                    "revision"
                ),
            },
            "model": {
                "backbone": "resnet18",
                "weights": FrozenPyramidResNet18.weights_id,
                "weights_url": FrozenPyramidResNet18.weights_url,
                "weights_frozen": True,
            },
        }
        metric_rows: list[dict[str, object]] = []
        prediction_rows: list[dict[str, object]] = []
        stability_rows: list[dict[str, object]] = []
        transfer_rows: list[dict[str, object]] = []
        example_rows: list[dict[str, object]] = []
        _atomic_json(run_dir / "metadata.json", metadata)
        _atomic_json(run_dir / "environment.json", _environment(device))
    else:
        run_dir = resume_run.resolve(strict=True)
        if yaml.safe_load((run_dir / "resolved_config.yaml").read_text("utf-8")) != resolved:
            raise M4Error("resume settings differ from the interrupted run")
        metadata = cast(
            dict[str, object], json.loads((run_dir / "metadata.json").read_text("utf-8"))
        )
        if metadata.get("status") != "running" or any(
            metadata.get(key) != value for key, value in hashes.items()
        ):
            raise M4Error("M4 resume inputs changed or run is not incomplete")
        metric_rows = _read_csv(run_dir / "category_metrics.csv")
        prediction_rows = _read_csv(run_dir / "predictions.csv")
        stability_rows = _read_csv(run_dir / "sample_stability.csv")
        transfer_rows = _read_csv(run_dir / "category_transfer.csv")
        example_rows = _read_csv(run_dir / "failure_examples.csv")
        resumed = metadata.get("resumed_at_utc", [])
        if not isinstance(resumed, list):
            resumed = []
        resumed.append(datetime.now(UTC).isoformat())
        metadata["resumed_at_utc"] = resumed
        _atomic_json(run_dir / "metadata.json", metadata)

    random.seed(settings.seed)
    np.random.seed(settings.seed)
    extractor = FrozenPyramidResNet18(
        device=device, image_size=settings.image_size, batch_size=settings.batch_size
    )
    checkpoint = (
        Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))
        / "hub"
        / "checkpoints"
        / Path(urlparse(FrozenPyramidResNet18.weights_url).path).name
    )
    if not checkpoint.is_file():
        raise M4Error(f"model checkpoint is unavailable: {checkpoint}")
    metadata["checkpoint_sha256"] = sha256_file(checkpoint)
    metadata["checkpoint_bytes"] = checkpoint.stat().st_size
    _atomic_json(run_dir / "metadata.json", metadata)
    expected_rows_per_category = len(conditions) * 3
    completed_categories = {
        category
        for category in selected_categories
        if sum(row.get("category") == category for row in metric_rows) == expected_rows_per_category
        and sum(row.get("category") == category for row in transfer_rows) == 1
    }

    for category_number, category in enumerate(selected_categories, start=1):
        if category in completed_categories:
            continue
        metric_rows = [row for row in metric_rows if row.get("category") != category]
        prediction_rows = [row for row in prediction_rows if row.get("category") != category]
        stability_rows = [row for row in stability_rows if row.get("category") != category]
        transfer_rows = [row for row in transfer_rows if row.get("category") != category]
        example_rows = [row for row in example_rows if row.get("category") != category]
        category_samples = [sample for sample in all_samples if sample.category == category]
        train, validation, test = _partition(category_samples, settings, quick)
        models = _load_category_models(
            source_run=source_run,
            source_metrics=source_metrics,
            run_dir=run_dir,
            category=category,
            train=train,
            validation=validation,
            extractor=extractor,
            settings=settings,
            quick=quick,
        )
        transfer_memory = _build_transfer_memory(
            source_run=source_run,
            target=category,
            all_categories=all_categories,
            settings=settings,
            quick=quick,
        )
        clean_images = [load_fitted_rgb(sample.image_path, settings.image_size) for sample in test]
        clean_masks = _load_masks(test, (settings.image_size, settings.image_size))
        clean_descriptors = [image_descriptors(image) for image in clean_images]
        labels = np.asarray([sample.target for sample in test], dtype=np.int8)
        clean_scores: dict[str, np.ndarray] = {}
        clean_maps: dict[str, tuple[np.ndarray, ...]] = {}
        clean_metrics: dict[str, dict[str, float | int]] = {}
        clean_predictions: dict[str, np.ndarray] = {}
        for condition_number, condition in enumerate(conditions, start=1):
            shifted_images, shifted_masks, inverses = _condition_inputs(
                clean_images, clean_masks, test, condition, settings
            )
            shifted_descriptors = [image_descriptors(image) for image in shifted_images]
            started = time.perf_counter()
            pyramids = extractor.extract_rgb(shifted_images)
            extraction_seconds = time.perf_counter() - started
            mid_maps = tuple(compose_feature_map(pyramid, "mid") for pyramid in pyramids)
            multiscale_maps = tuple(
                compose_feature_map(pyramid, "multiscale") for pyramid in pyramids
            )
            for model in models:
                features = mid_maps if model.feature_mode == "mid" else multiscale_maps
                score_started = time.perf_counter()
                scores, raw_maps = model.memory.score(features)
                score_seconds = time.perf_counter() - score_started
                score_maps = tuple(
                    _resize_maps(
                        raw_maps,
                        (settings.image_size, settings.image_size),
                        settings.gaussian_sigma,
                    )
                )
                metrics: dict[str, object] = {
                    "model": model.identifier,
                    "category": category,
                    "condition": condition.identifier,
                    "family": condition.family,
                    "severity": condition.severity,
                    "threshold_source": "clean-validation-normal-only",
                    **image_metrics(labels, scores, model.image_threshold),
                    **pixel_metrics(shifted_masks, score_maps, model.pixel_threshold),
                    "aupro_30": pro_auc(
                        shifted_masks,
                        score_maps,
                        max_fpr=settings.pro_max_fpr,
                        thresholds=pro_thresholds,
                    ),
                    "feature_extraction_seconds": extraction_seconds,
                    "score_seconds": score_seconds,
                    "latency_ms_per_image": 1000.0 * score_seconds / len(test),
                }
                predicted = scores > model.image_threshold
                if condition.family == "clean":
                    clean_scores[model.identifier] = scores
                    clean_maps[model.identifier] = score_maps
                    clean_metrics[model.identifier] = cast(dict[str, float | int], metrics.copy())
                    clean_predictions[model.identifier] = predicted
                    metrics.update(
                        {
                            "score_rank_spearman": 1.0,
                            "heatmap_correlation": 1.0,
                            "heatmap_correlation_ci_low": 1.0,
                            "heatmap_correlation_ci_high": 1.0,
                            "heatmap_top_iou": 1.0,
                        }
                    )
                    for metric in (
                        "image_auroc",
                        "image_ap",
                        "test_normal_fpr",
                        "pixel_auroc",
                        "pixel_ap",
                        "pixel_iou_mean",
                        "aupro_30",
                    ):
                        metrics[f"delta_{metric}"] = 0.0
                        if metric in {"image_auroc", "image_ap", "test_normal_fpr"}:
                            metrics[f"delta_{metric}_ci_low"] = 0.0
                            metrics[f"delta_{metric}_ci_high"] = 0.0
                else:
                    baseline_scores = clean_scores[model.identifier]
                    baseline_maps = clean_maps[model.identifier]
                    baseline_metrics = clean_metrics[model.identifier]
                    aligned = tuple(
                        align_score_map(score_map, inverse)
                        for score_map, inverse in zip(score_maps, inverses, strict=True)
                    )
                    correlations, overlaps = heatmap_stability(
                        baseline_maps, aligned, top_fraction=settings.heatmap_top_fraction
                    )
                    stability_low, stability_high = bootstrap_mean_interval(
                        correlations,
                        seed=settings.seed
                        + _stable_index(
                            f"stability:{model.identifier}:{category}:{condition.identifier}",
                            1_000_000,
                        ),
                        samples=bootstrap_samples,
                        confidence=settings.bootstrap_confidence,
                    )
                    metrics.update(
                        {
                            "score_rank_spearman": spearman_stability(baseline_scores, scores),
                            "heatmap_correlation": float(np.mean(correlations)),
                            "heatmap_correlation_ci_low": stability_low,
                            "heatmap_correlation_ci_high": stability_high,
                            "heatmap_top_iou": float(np.mean(overlaps)),
                        }
                    )
                    intervals = paired_image_delta_intervals(
                        labels,
                        baseline_scores,
                        scores,
                        threshold=model.image_threshold,
                        seed=settings.seed
                        + _stable_index(
                            f"delta:{model.identifier}:{category}:{condition.identifier}",
                            1_000_000,
                        ),
                        samples=bootstrap_samples,
                        confidence=settings.bootstrap_confidence,
                    )
                    for metric in (
                        "image_auroc",
                        "image_ap",
                        "test_normal_fpr",
                        "pixel_auroc",
                        "pixel_ap",
                        "pixel_iou_mean",
                        "aupro_30",
                    ):
                        metrics[f"delta_{metric}"] = float(
                            cast(int | float, metrics[metric])
                        ) - float(baseline_metrics[metric])
                        if metric in intervals:
                            metrics[f"delta_{metric}_ci_low"] = intervals[metric]["low"]
                            metrics[f"delta_{metric}_ci_high"] = intervals[metric]["high"]
                    clean_predicted = clean_predictions[model.identifier]
                    correct_clean = clean_predicted == labels.astype(bool)
                    correct_shift = predicted == labels.astype(bool)
                    condition_stability_rows: list[dict[str, object]] = []
                    for index, sample in enumerate(test):
                        if clean_predicted[index] == predicted[index]:
                            transition = (
                                "unchanged_correct" if correct_shift[index] else "unchanged_error"
                            )
                        elif labels[index] == 0 and predicted[index]:
                            transition = "new_false_positive"
                        elif labels[index] == 1 and not predicted[index]:
                            transition = "new_false_negative"
                        elif correct_shift[index]:
                            transition = (
                                "recovered_normal" if labels[index] == 0 else "recovered_anomaly"
                            )
                        else:
                            transition = "changed_error"
                        row = {
                            "model": model.identifier,
                            "category": category,
                            "condition": condition.identifier,
                            "family": condition.family,
                            "severity": condition.severity,
                            "sample_id": sample.sample_id,
                            "label": sample.label,
                            "target": sample.target,
                            "clean_score": float(baseline_scores[index]),
                            "shifted_score": float(scores[index]),
                            "score_delta": float(scores[index] - baseline_scores[index]),
                            "clean_predicted_anomaly": int(clean_predicted[index]),
                            "shifted_predicted_anomaly": int(predicted[index]),
                            "clean_correct": int(correct_clean[index]),
                            "shifted_correct": int(correct_shift[index]),
                            "failed_under_shift": int(
                                correct_clean[index] and not correct_shift[index]
                            ),
                            "transition": transition,
                            "heatmap_correlation": float(correlations[index]),
                            "heatmap_top_iou": float(overlaps[index]),
                            "brightness_delta": shifted_descriptors[index][0]
                            - clean_descriptors[index][0],
                            "edge_delta": shifted_descriptors[index][1]
                            - clean_descriptors[index][1],
                        }
                        condition_stability_rows.append(row)
                    stability_rows.extend(condition_stability_rows)
                    if condition.severity == 2:
                        for kind in ("new_false_positive", "new_false_negative"):
                            candidates = [
                                index
                                for index, row in enumerate(condition_stability_rows)
                                if row["transition"] == kind
                            ]
                            if candidates:
                                index = min(
                                    candidates,
                                    key=lambda item: float(
                                        cast(
                                            int | float,
                                            condition_stability_rows[item]["heatmap_correlation"],
                                        )
                                    ),
                                )
                                example_rows.append(
                                    _save_failure_panel(
                                        run_dir=run_dir,
                                        model=model.identifier,
                                        category=category,
                                        condition=condition.identifier,
                                        kind=kind,
                                        sample=test[index],
                                        clean_image=clean_images[index],
                                        shifted_image=shifted_images[index],
                                        shifted_mask=shifted_masks[index],
                                        shifted_map=score_maps[index],
                                    )
                                )
                metric_rows.append(metrics)
                prediction_rows.extend(
                    {
                        "model": model.identifier,
                        "category": category,
                        "condition": condition.identifier,
                        "family": condition.family,
                        "severity": condition.severity,
                        "sample_id": sample.sample_id,
                        "label": sample.label,
                        "target": sample.target,
                        "score": float(score),
                        "threshold": model.image_threshold,
                        "predicted_anomaly": int(score > model.image_threshold),
                    }
                    for sample, score in zip(test, scores, strict=True)
                )
            if condition.family == "clean":
                transfer_scores, transfer_raw = transfer_memory.score(multiscale_maps)
                transfer_maps = _resize_maps(
                    transfer_raw,
                    (settings.image_size, settings.image_size),
                    settings.gaussian_sigma,
                )
                transfer_pixel_auroc, transfer_pixel_ap = _pixel_ranking(clean_masks, transfer_maps)
                candidate_scores = clean_scores[settings.candidate_variant]
                candidate_metrics = clean_metrics[settings.candidate_variant]
                intervals = paired_image_delta_intervals(
                    labels,
                    candidate_scores,
                    transfer_scores,
                    threshold=models[1].image_threshold,
                    seed=settings.seed + _stable_index(f"transfer:{category}", 1_000_000),
                    samples=bootstrap_samples,
                    confidence=settings.bootstrap_confidence,
                )
                transfer_rows.append(
                    {
                        "category": category,
                        "protocol": "leave-one-category-out-balanced-memory",
                        "target_category_training_samples": 0,
                        "source_categories": len(all_categories) - 1,
                        "bank_per_source_category": settings.quick_transfer_bank_per_category
                        if quick
                        else settings.transfer_bank_per_category,
                        "image_auroc": float(roc_auc_score(labels, transfer_scores)),
                        "image_ap": float(average_precision_score(labels, transfer_scores)),
                        "pixel_auroc": transfer_pixel_auroc,
                        "pixel_ap": transfer_pixel_ap,
                        "delta_image_auroc_vs_category_specific": float(
                            roc_auc_score(labels, transfer_scores)
                        )
                        - float(candidate_metrics["image_auroc"]),
                        "delta_image_auroc_ci_low": intervals["image_auroc"]["low"],
                        "delta_image_auroc_ci_high": intervals["image_auroc"]["high"],
                        "delta_image_ap_vs_category_specific": float(
                            average_precision_score(labels, transfer_scores)
                        )
                        - float(candidate_metrics["image_ap"]),
                        "delta_image_ap_ci_low": intervals["image_ap"]["low"],
                        "delta_image_ap_ci_high": intervals["image_ap"]["high"],
                    }
                )
            print(
                f"[{category_number}/{len(selected_categories)}] {category} "
                f"condition {condition_number}/{len(conditions)}: {condition.identifier}",
                flush=True,
            )
        _atomic_csv(run_dir / "category_metrics.csv", metric_rows)
        _atomic_csv(run_dir / "predictions.csv", prediction_rows)
        _atomic_csv(run_dir / "sample_stability.csv", stability_rows)
        _atomic_csv(run_dir / "category_transfer.csv", transfer_rows)
        _atomic_csv(run_dir / "failure_examples.csv", example_rows)
        _atomic_json(
            run_dir / "progress.json",
            {
                "completed_categories": category_number,
                "total_categories": len(selected_categories),
            },
        )

    robustness_rows = _aggregate_rows(metric_rows, settings, bootstrap_samples)
    family_rows = _family_rows(robustness_rows, settings, bootstrap_samples)
    cluster_rows = _failure_clusters(stability_rows)
    hypothesis_rows = _hypothesis_rows(metric_rows, stability_rows, settings, bootstrap_samples)
    _atomic_csv(run_dir / "robustness_summary.csv", robustness_rows)
    _atomic_csv(run_dir / "family_summary.csv", family_rows)
    _atomic_csv(run_dir / "failure_clusters.csv", cluster_rows)
    _atomic_csv(run_dir / "hypothesis_tests.csv", hypothesis_rows)
    index_manifest = {
        path.relative_to(run_dir).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted((run_dir / "indexes").rglob("*.npz"))
    }
    _atomic_json(run_dir / "index_manifest.json", index_manifest)
    summary: dict[str, object] = {
        "mode": mode,
        "seed": settings.seed,
        "categories": len(selected_categories),
        "conditions": len(conditions),
        "models": [
            settings.reference_variant,
            settings.candidate_variant,
            settings.intervention.identifier,
        ],
        "hypotheses": hypothesis_rows,
        "source_m3_run": str(source_run),
        "mask_preprocessing": "center-crop-nearest-matched-to-model-input",
    }
    _atomic_json(run_dir / "summary.json", summary)
    metadata["status"] = "complete"
    metadata["completed_at_utc"] = datetime.now(UTC).isoformat()
    for filename in (
        "category_metrics.csv",
        "predictions.csv",
        "sample_stability.csv",
        "category_transfer.csv",
        "failure_examples.csv",
        "robustness_summary.csv",
        "family_summary.csv",
        "failure_clusters.csv",
        "hypothesis_tests.csv",
        "index_manifest.json",
        "summary.json",
    ):
        path = run_dir / filename
        if path.is_file():
            metadata[f"{path.stem}_sha256"] = sha256_file(path)
    metadata["failure_example_sha256"] = {
        path.relative_to(run_dir).as_posix(): sha256_file(path)
        for path in sorted((run_dir / "failure_examples").rglob("*.png"))
    }
    _atomic_json(run_dir / "metadata.json", metadata)
    return M4RunResult(run_dir, summary)

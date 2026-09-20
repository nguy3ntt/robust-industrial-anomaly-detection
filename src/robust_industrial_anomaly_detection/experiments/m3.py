"""M3 multi-scale, coreset, and unified-memory experiment."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from urllib.parse import urlparse

import numpy as np
import torch
import yaml
from dotenv import load_dotenv

from robust_industrial_anomaly_detection.baselines.multiscale import (
    FeatureMode,
    FrozenPyramidResNet18,
    ProjectedPatchMemory,
    PyramidFeatures,
    SelectionMode,
    compose_feature_map,
    load_projected_index,
)
from robust_industrial_anomaly_detection.data.visa import load_trusted_manifest, sha256_file
from robust_industrial_anomaly_detection.evaluation.metrics import (
    bootstrap_image_metrics,
    calibrate_threshold,
    image_metrics,
    pixel_metrics,
    pro_auc,
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
    _save_qualitative_panels,
)


class M3Error(RuntimeError):
    """Raised when an M3 run cannot satisfy its frozen protocol."""


@dataclass(frozen=True)
class Variant:
    identifier: str
    feature_mode: FeatureMode
    selection: SelectionMode
    bank_size: int
    projected_dimension: int
    patches_per_image: int
    candidate_limit: int
    image_score_fraction: float


@dataclass(frozen=True)
class M3Settings:
    name: str
    seed: int
    image_threshold_quantile: float
    pixel_threshold_quantile: float
    bootstrap_samples: int
    bootstrap_confidence: float
    pro_max_fpr: float
    pro_thresholds: int
    image_size: int
    batch_size: int
    gaussian_sigma: float
    variants: tuple[Variant, ...]
    final_variant: str
    unified_bank_per_category: int
    quick_train: int
    quick_validation: int
    quick_test_normal: int
    quick_test_anomaly: int
    quick_bootstrap_samples: int
    quick_bank_size_cap: int
    quick_candidate_limit_cap: int
    quick_patches_per_image: int
    quick_unified_bank_per_category: int


@dataclass(frozen=True)
class M3RunResult:
    run_dir: Path
    summary: dict[str, object]


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise M3Error(f"{name} must be a mapping")
    return cast(Mapping[str, object], value)


def _integer(section: Mapping[str, object], name: str) -> int:
    value = section.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise M3Error(f"{name} must be a positive integer")
    return value


def _number(section: Mapping[str, object], name: str) -> float:
    value = section.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise M3Error(f"{name} must be numeric")
    return float(value)


def _string(section: Mapping[str, object], name: str) -> str:
    value = section.get(name)
    if not isinstance(value, str) or not value:
        raise M3Error(f"{name} must be a non-empty string")
    return value


def load_m3_settings(path: Path) -> M3Settings:
    """Load and validate the frozen M3 ablation configuration."""

    try:
        root = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "configuration")
    except (OSError, yaml.YAMLError) as exc:
        raise M3Error(f"cannot load M3 configuration {path}: {exc}") from exc
    experiment = _mapping(root.get("experiment"), "experiment")
    backbone = _mapping(root.get("backbone"), "backbone")
    common = _mapping(root.get("common_memory"), "common_memory")
    final = _mapping(root.get("final_model"), "final_model")
    quick = _mapping(root.get("quick"), "quick")
    raw_variants = root.get("ablations")
    if not isinstance(raw_variants, list) or not raw_variants:
        raise M3Error("ablations must be a non-empty list")
    projected_dimension = _integer(common, "projected_dimension")
    patches_per_image = _integer(common, "patches_per_image")
    candidate_limit = _integer(common, "candidate_limit")
    score_fraction = _number(common, "image_score_fraction")
    variants: list[Variant] = []
    for index, value in enumerate(raw_variants):
        section = _mapping(value, f"ablations[{index}]")
        identifier = _string(section, "id")
        if not re.fullmatch(r"[a-z0-9_]+", identifier):
            raise M3Error(f"invalid variant identifier: {identifier}")
        mode = _string(section, "feature_mode")
        selection = _string(section, "selection")
        if mode not in {"mid", "multiscale"} or selection not in {"random", "kcenter"}:
            raise M3Error(f"invalid feature/selection mode in {identifier}")
        variant = Variant(
            identifier=identifier,
            feature_mode=cast(FeatureMode, mode),
            selection=cast(SelectionMode, selection),
            bank_size=_integer(section, "bank_size"),
            projected_dimension=projected_dimension,
            patches_per_image=patches_per_image,
            candidate_limit=(
                _integer(section, "candidate_limit")
                if "candidate_limit" in section
                else candidate_limit
            ),
            image_score_fraction=score_fraction,
        )
        if variant.bank_size > variant.candidate_limit:
            raise M3Error(f"variant {identifier} bank exceeds its candidate bound")
        variants.append(variant)
    if len({variant.identifier for variant in variants}) != len(variants):
        raise M3Error("ablation identifiers must be unique")
    final_variant = _string(final, "variant")
    if final_variant not in {variant.identifier for variant in variants}:
        raise M3Error("final_model.variant is not a configured ablation")
    settings = M3Settings(
        name=_string(experiment, "name"),
        seed=_integer(experiment, "seed"),
        image_threshold_quantile=_number(experiment, "image_threshold_quantile"),
        pixel_threshold_quantile=_number(experiment, "pixel_threshold_quantile"),
        bootstrap_samples=_integer(experiment, "bootstrap_samples"),
        bootstrap_confidence=_number(experiment, "bootstrap_confidence"),
        pro_max_fpr=_number(experiment, "pro_max_fpr"),
        pro_thresholds=_integer(experiment, "pro_thresholds"),
        image_size=_integer(backbone, "image_size"),
        batch_size=_integer(backbone, "batch_size"),
        gaussian_sigma=_number(common, "gaussian_sigma"),
        variants=tuple(variants),
        final_variant=final_variant,
        unified_bank_per_category=_integer(final, "unified_bank_per_category"),
        quick_train=_integer(quick, "train_per_category"),
        quick_validation=_integer(quick, "validation_per_category"),
        quick_test_normal=_integer(quick, "test_normal_per_category"),
        quick_test_anomaly=_integer(quick, "test_anomaly_per_category"),
        quick_bootstrap_samples=_integer(quick, "bootstrap_samples"),
        quick_bank_size_cap=_integer(quick, "bank_size_cap"),
        quick_candidate_limit_cap=_integer(quick, "candidate_limit_cap"),
        quick_patches_per_image=_integer(quick, "patches_per_image"),
        quick_unified_bank_per_category=_integer(quick, "unified_bank_per_category"),
    )
    for field in (
        "image_threshold_quantile",
        "pixel_threshold_quantile",
        "bootstrap_confidence",
        "pro_max_fpr",
    ):
        if not 0.0 < cast(float, getattr(settings, field)) < 1.0:
            raise M3Error(f"{field} must be strictly between zero and one")
    return settings


def _effective_variant(variant: Variant, settings: M3Settings, quick: bool) -> Variant:
    if not quick:
        return variant
    bank_size = min(variant.bank_size, settings.quick_bank_size_cap)
    candidate_limit = max(
        bank_size, min(variant.candidate_limit, settings.quick_candidate_limit_cap)
    )
    return replace(
        variant,
        bank_size=bank_size,
        candidate_limit=candidate_limit,
        patches_per_image=min(variant.patches_per_image, settings.quick_patches_per_image),
    )


def _compose_all(pyramids: Sequence[PyramidFeatures], mode: FeatureMode) -> tuple[np.ndarray, ...]:
    return tuple(compose_feature_map(features, mode) for features in pyramids)


def _partition_samples(
    samples: Sequence[Sample], settings: M3Settings, quick: bool
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
    if len(train) < 2 or not validation or not test:
        raise M3Error("every category requires train, validation, and test samples")
    if {sample.label for sample in test} != {"normal", "anomaly"}:
        raise M3Error("every category test partition must contain both labels")
    return train, validation, test


def _metric_record(
    *,
    variant: str,
    scope: str,
    category: str,
    validation_scores: np.ndarray,
    test_scores: np.ndarray,
    test: Sequence[Sample],
    settings: M3Settings,
    bootstrap_samples: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    threshold = calibrate_threshold(validation_scores, settings.image_threshold_quantile)
    labels = np.asarray([sample.target for sample in test], dtype=np.int8)
    metrics: dict[str, object] = {
        "variant": variant,
        "scope": scope,
        "category": category,
        "train_policy": "normal-only",
        "threshold_source": "validation-normal-only",
        **image_metrics(labels, test_scores, threshold),
    }
    interval_seed = settings.seed + int(
        hashlib.sha256(f"{variant}\0{scope}\0{category}".encode()).hexdigest()[:8], 16
    )
    intervals = bootstrap_image_metrics(
        labels,
        test_scores,
        seed=interval_seed,
        samples=bootstrap_samples,
        confidence=settings.bootstrap_confidence,
    )
    for name, interval in intervals.items():
        metrics[f"{name}_ci_low"] = interval["low"]
        metrics[f"{name}_ci_high"] = interval["high"]
    predictions = [
        {
            "variant": variant,
            "scope": scope,
            "category": category,
            "sample_id": sample.sample_id,
            "label": sample.label,
            "target": sample.target,
            "score": float(score),
            "threshold": threshold,
            "predicted_anomaly": int(score > threshold),
        }
        for sample, score in zip(test, test_scores, strict=True)
    ]
    return metrics, predictions


def _macro_rows(metric_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in metric_rows:
        groups[(cast(str, row["scope"]), cast(str, row["variant"]))].append(row)
    metric_names = (
        "image_auroc",
        "image_ap",
        "image_fpr_at_95_tpr",
        "balanced_accuracy",
        "test_normal_fpr",
        "pixel_auroc",
        "pixel_ap",
        "pixel_iou_mean",
        "aupro_30",
        "feature_extraction_seconds",
        "composition_seconds",
        "fit_seconds",
        "score_seconds",
        "end_to_end_seconds",
        "latency_ms_per_image",
        "bank_bytes",
        "compressed_index_bytes",
        "feature_cache_bytes",
        "estimated_search_madds",
    )
    output: list[dict[str, object]] = []
    for (scope, variant), rows in sorted(groups.items()):
        summary: dict[str, object] = {
            "scope": scope,
            "variant": variant,
            "categories": len(rows),
        }
        for name in metric_names:
            present = [
                float(cast(int | float, row[name]))
                for row in rows
                if name in row and isinstance(row[name], (int, float))
            ]
            if present:
                summary[f"macro_{name}"] = float(np.mean(present))
        output.append(summary)
    return output


def _add_reference_deltas(rows: list[dict[str, object]], reference: str) -> None:
    reference_row = next(
        row for row in rows if row["scope"] == "category_specific" and row["variant"] == reference
    )
    for row in rows:
        for metric in (
            "image_auroc",
            "image_ap",
            "test_normal_fpr",
            "pixel_auroc",
            "pixel_ap",
            "pixel_iou_mean",
            "aupro_30",
            "end_to_end_seconds",
            "compressed_index_bytes",
        ):
            key = f"macro_{metric}"
            if key in row and key in reference_row:
                row[f"delta_{metric}_vs_reference"] = float(
                    cast(float, row[key]) - cast(float, reference_row[key])
                )


def _publish_progress(
    run_dir: Path,
    metric_rows: Sequence[Mapping[str, object]],
    prediction_rows: Sequence[Mapping[str, object]],
    qualitative_rows: Sequence[Mapping[str, object]],
    *,
    total_jobs: int,
) -> None:
    _atomic_csv(run_dir / "category_metrics.csv", metric_rows)
    _atomic_csv(run_dir / "predictions.csv", prediction_rows)
    if qualitative_rows:
        _atomic_csv(run_dir / "qualitative_index.csv", qualitative_rows)
    _atomic_json(
        run_dir / "progress.json",
        {"completed_jobs": len(metric_rows), "total_jobs": total_jobs},
    )


def _run_scoring_job(
    *,
    memory: ProjectedPatchMemory,
    variant: str,
    scope: str,
    category: str,
    validation_maps: Sequence[np.ndarray],
    test_maps: Sequence[np.ndarray],
    validation: Sequence[Sample],
    test: Sequence[Sample],
    settings: M3Settings,
    bootstrap_samples: int,
    feature_seconds: float,
    composition_seconds: float,
    fit_seconds: float,
    feature_cache_bytes: int,
    compressed_index_bytes: int,
) -> tuple[dict[str, object], list[dict[str, object]], list[np.ndarray], float]:
    scoring_started = time.perf_counter()
    validation_scores, validation_raw = memory.score(validation_maps)
    test_scores, test_raw = memory.score(test_maps)
    evaluation_shape = (settings.image_size, settings.image_size)
    validation_score_maps = _resize_maps(validation_raw, evaluation_shape, settings.gaussian_sigma)
    test_score_maps = _resize_maps(test_raw, evaluation_shape, settings.gaussian_sigma)
    pixel_threshold = calibrate_threshold(
        np.concatenate([score_map.reshape(-1) for score_map in validation_score_maps]),
        settings.pixel_threshold_quantile,
    )
    score_seconds = time.perf_counter() - scoring_started
    metrics, predictions = _metric_record(
        variant=variant,
        scope=scope,
        category=category,
        validation_scores=validation_scores,
        test_scores=test_scores,
        test=test,
        settings=settings,
        bootstrap_samples=bootstrap_samples,
    )
    masks = _load_masks(test, test_score_maps[0].shape)
    metrics.update(pixel_metrics(masks, test_score_maps, pixel_threshold))
    metrics["aupro_30"] = pro_auc(
        masks,
        test_score_maps,
        max_fpr=settings.pro_max_fpr,
        thresholds=settings.pro_thresholds,
    )
    total_images = len(validation) + len(test)
    end_to_end = feature_seconds + composition_seconds + fit_seconds + score_seconds
    stats = memory.stats
    metrics.update(
        {
            "train_count": 0,
            "validation_normal_count": len(validation),
            "feature_mode": "multiscale" if test_maps[0].shape[0] > 14 else "mid",
            "selection": stats.selection,
            "candidate_count": stats.candidate_count,
            "bank_count": stats.selected_count,
            "feature_dimension": stats.feature_dimension,
            "projected_dimension": stats.projected_dimension,
            "coverage_radius": stats.coverage_radius,
            "raw_candidate_bytes": stats.raw_candidate_bytes,
            "bank_bytes": stats.bank_bytes,
            "projection_bytes": stats.projection_bytes,
            "compressed_index_bytes": compressed_index_bytes,
            "feature_cache_bytes": feature_cache_bytes,
            "feature_extraction_seconds": feature_seconds,
            "composition_seconds": composition_seconds,
            "fit_seconds": fit_seconds,
            "score_seconds": score_seconds,
            "end_to_end_seconds": end_to_end,
            "latency_ms_per_image": 1000.0 * score_seconds / total_images,
            "estimated_search_madds": int(
                sum(
                    feature_map.shape[0] * feature_map.shape[1]
                    for feature_map in (*validation_maps, *test_maps)
                )
                * stats.selected_count
                * stats.projected_dimension
            ),
        }
    )
    return metrics, predictions, test_score_maps, pixel_threshold


def run_m3(
    *,
    project_root: Path,
    experiment_config: Path,
    data_config: Path,
    categories: Sequence[str] | None,
    device: str,
    quick: bool,
    resume_run: Path | None = None,
) -> M3RunResult:
    """Run frozen M3 ablations plus category-specific/unified comparison."""

    load_dotenv(project_root / ".env")
    settings = load_m3_settings(experiment_config)
    data_root_raw = os.environ.get("ANOMALY_DATA_ROOT")
    artifact_root_raw = os.environ.get("ANOMALY_ARTIFACT_ROOT")
    if not data_root_raw or not artifact_root_raw:
        raise M3Error("ANOMALY_DATA_ROOT and ANOMALY_ARTIFACT_ROOT must be configured")
    data_root = Path(data_root_raw)
    artifact_root = Path(artifact_root_raw)
    manifest_dir = data_root / "manifests" / "visa-20220922"
    image_root = data_root / "sources" / "visa-20220922"
    manifest_path = manifest_dir / "manifest.jsonl"
    audit_path = manifest_dir / "audit.json"
    records = load_trusted_manifest(manifest_path, audit_path, data_config, image_root)
    all_samples = _samples(records, image_root)
    available_categories = sorted({sample.category for sample in all_samples})
    selected_categories = list(categories) if categories else available_categories
    unknown = sorted(set(selected_categories) - set(available_categories))
    if unknown:
        raise M3Error(f"unknown categories: {unknown}")
    effective_variants = tuple(
        _effective_variant(variant, settings, quick) for variant in settings.variants
    )
    final_variant = next(
        variant for variant in effective_variants if variant.identifier == settings.final_variant
    )
    unified_per_category = (
        settings.quick_unified_bank_per_category if quick else settings.unified_bank_per_category
    )
    if unified_per_category > final_variant.bank_size:
        raise M3Error("unified per-category memory exceeds the final category bank")

    random.seed(settings.seed)
    np.random.seed(settings.seed)
    mode = "smoke" if quick else "full"
    resolved: dict[str, object] = {
        **asdict(settings),
        "variants": [asdict(variant) for variant in settings.variants],
        "effective_variants": [asdict(variant) for variant in effective_variants],
        "effective_unified_bank_per_category": unified_per_category,
        "mode": mode,
        "categories": selected_categories,
        "device": device,
        "data_root": str(data_root),
        "artifact_root": str(artifact_root),
    }
    current_hashes = {
        "manifest_sha256": sha256_file(manifest_path),
        "audit_sha256": sha256_file(audit_path),
        "experiment_config_sha256": sha256_file(experiment_config),
        "data_config_sha256": sha256_file(data_config),
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
            **current_hashes,
            "git": _git_metadata(project_root),
            "model": {
                "backbone": "resnet18",
                "weights": FrozenPyramidResNet18.weights_id,
                "weights_url": FrozenPyramidResNet18.weights_url,
                "weights_frozen": True,
                "layers": ["layer1", "layer2", "layer3"],
            },
        }
        metric_rows: list[dict[str, object]] = []
        prediction_rows: list[dict[str, object]] = []
        qualitative_rows: list[dict[str, object]] = []
        _atomic_json(run_dir / "metadata.json", metadata)
        _atomic_json(run_dir / "environment.json", _environment(device))
    else:
        run_dir = resume_run.resolve(strict=True)
        try:
            run_dir.relative_to((artifact_root / "runs").resolve(strict=True))
        except ValueError as exc:
            raise M3Error("resume directory must be below the configured run root") from exc
        if yaml.safe_load((run_dir / "resolved_config.yaml").read_text("utf-8")) != resolved:
            raise M3Error("resume settings differ from the interrupted run")
        metadata_raw = json.loads((run_dir / "metadata.json").read_text("utf-8"))
        if not isinstance(metadata_raw, dict) or metadata_raw.get("status") != "running":
            raise M3Error("only an incomplete M3 run can be resumed")
        metadata = cast(dict[str, object], metadata_raw)
        if any(metadata.get(key) != value for key, value in current_hashes.items()):
            raise M3Error("M3 input or configuration hashes changed")
        resumed = metadata.get("resumed_at_utc", [])
        if not isinstance(resumed, list):
            resumed = []
        resumed.append(datetime.now(UTC).isoformat())
        metadata["resumed_at_utc"] = resumed
        metric_rows = _read_csv(run_dir / "category_metrics.csv")
        prediction_rows = _read_csv(run_dir / "predictions.csv")
        qualitative_rows = _read_csv(run_dir / "qualitative_index.csv")
        _atomic_json(run_dir / "metadata.json", metadata)

    extractor = FrozenPyramidResNet18(
        device=device,
        image_size=settings.image_size,
        batch_size=settings.batch_size,
    )
    torch_home = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))
    checkpoint = (
        torch_home
        / "hub"
        / "checkpoints"
        / Path(urlparse(FrozenPyramidResNet18.weights_url).path).name
    )
    if not checkpoint.is_file():
        raise M3Error(f"model checkpoint is unavailable: {checkpoint}")
    model_metadata = metadata.get("model")
    if isinstance(model_metadata, dict):
        model_metadata["checkpoint_sha256"] = sha256_file(checkpoint)
        model_metadata["checkpoint_bytes"] = checkpoint.stat().st_size
        _atomic_json(run_dir / "metadata.json", metadata)

    completed = {
        (cast(str, row["scope"]), cast(str, row["variant"]), cast(str, row["category"]))
        for row in metric_rows
    }
    bootstrap_samples = settings.quick_bootstrap_samples if quick else settings.bootstrap_samples
    total_jobs = len(selected_categories) * (len(effective_variants) + 1)
    final_index_paths: dict[str, Path] = {}
    for category in selected_categories:
        final_index_paths[category] = (
            run_dir / "indexes" / "category_specific" / settings.final_variant / f"{category}.npz"
        )
        pending = [
            variant
            for variant in effective_variants
            if ("category_specific", variant.identifier, category) not in completed
        ]
        if not pending:
            if not final_index_paths[category].is_file():
                raise M3Error(f"completed category is missing final index: {category}")
            continue
        category_samples = [sample for sample in all_samples if sample.category == category]
        train, validation, test = _partition_samples(category_samples, settings, quick)
        extraction_started = time.perf_counter()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        pyramids = extractor.extract([sample.image_path for sample in (*train, *validation, *test)])
        extraction_seconds = time.perf_counter() - extraction_started
        peak_gpu_bytes = int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
        train_count = len(train)
        validation_count = len(validation)
        for feature_mode in ("mid", "multiscale"):
            mode_variants = [variant for variant in pending if variant.feature_mode == feature_mode]
            if not mode_variants:
                continue
            composition_started = time.perf_counter()
            maps = _compose_all(pyramids, feature_mode)
            composition_seconds = time.perf_counter() - composition_started
            feature_cache_bytes = sum(feature_map.nbytes for feature_map in maps)
            train_maps = maps[:train_count]
            validation_maps = maps[train_count : train_count + validation_count]
            test_maps = maps[train_count + validation_count :]
            for variant in mode_variants:
                memory = ProjectedPatchMemory(
                    seed=settings.seed,
                    bank_size=variant.bank_size,
                    projected_dimension=variant.projected_dimension,
                    patches_per_image=variant.patches_per_image,
                    candidate_limit=variant.candidate_limit,
                    image_score_fraction=variant.image_score_fraction,
                    selection=variant.selection,
                )
                fit_started = time.perf_counter()
                memory.fit(train_maps)
                fit_seconds = time.perf_counter() - fit_started
                index_path = (
                    run_dir
                    / "indexes"
                    / "category_specific"
                    / variant.identifier
                    / f"{category}.npz"
                )
                compressed_bytes = memory.save(
                    index_path,
                    metadata={
                        "scope": "category_specific",
                        "variant": variant.identifier,
                        "category": category,
                    },
                )
                metrics, predictions, score_maps, pixel_threshold = _run_scoring_job(
                    memory=memory,
                    variant=variant.identifier,
                    scope="category_specific",
                    category=category,
                    validation_maps=validation_maps,
                    test_maps=test_maps,
                    validation=validation,
                    test=test,
                    settings=settings,
                    bootstrap_samples=bootstrap_samples,
                    feature_seconds=extraction_seconds,
                    composition_seconds=composition_seconds,
                    fit_seconds=fit_seconds,
                    feature_cache_bytes=feature_cache_bytes,
                    compressed_index_bytes=compressed_bytes,
                )
                metrics["train_count"] = len(train)
                metrics["peak_gpu_memory_bytes"] = peak_gpu_bytes
                metric_rows.append(metrics)
                prediction_rows.extend(predictions)
                if variant.identifier == settings.final_variant:
                    qualitative = _save_qualitative_panels(
                        run_dir,
                        f"category_specific/{category}",
                        test,
                        np.asarray([row["score"] for row in predictions], dtype=np.float64),
                        score_maps,
                        image_threshold=float(cast(int | float, metrics["threshold"])),
                        pixel_threshold=pixel_threshold,
                        image_size=settings.image_size,
                    )
                    for row in qualitative:
                        row["category"] = category
                        row["scope"] = "category_specific"
                        row["variant"] = variant.identifier
                    qualitative_rows.extend(qualitative)
                _publish_progress(
                    run_dir,
                    metric_rows,
                    prediction_rows,
                    qualitative_rows,
                    total_jobs=total_jobs,
                )
                print(
                    f"[{len(metric_rows)}/{total_jobs}] category_specific/"
                    f"{variant.identifier}/{category}: "
                    f"AUROC={float(cast(int | float, metrics['image_auroc'])):.4f} "
                    f"pixel={float(cast(int | float, metrics['pixel_auroc'])):.4f}",
                    flush=True,
                )
            del maps
        del pyramids

    combined_banks: list[np.ndarray] = []
    shared_projection: np.ndarray | None = None
    unified_train_count = 0
    for category in selected_categories:
        category_train, _, _ = _partition_samples(
            [sample for sample in all_samples if sample.category == category], settings, quick
        )
        unified_train_count += len(category_train)

    for category in selected_categories:
        bank, projection = load_projected_index(final_index_paths[category])
        if shared_projection is None:
            shared_projection = projection
        elif not np.array_equal(shared_projection, projection):
            raise M3Error("category indexes do not share one deterministic projection")
        combined_banks.append(bank[:unified_per_category])
    assert shared_projection is not None
    unified_bank = np.concatenate(combined_banks)
    unified_memory = ProjectedPatchMemory(
        seed=settings.seed,
        bank_size=len(unified_bank),
        projected_dimension=final_variant.projected_dimension,
        patches_per_image=final_variant.patches_per_image,
        candidate_limit=len(unified_bank),
        image_score_fraction=final_variant.image_score_fraction,
        selection="kcenter",
    )
    unified_build_started = time.perf_counter()
    unified_memory.fit_projected(unified_bank, shared_projection)
    unified_build_seconds = time.perf_counter() - unified_build_started
    unified_index_path = run_dir / "indexes" / "unified" / "multiscale_coreset_balanced.npz"
    unified_compressed_bytes = unified_memory.save(
        unified_index_path,
        metadata={
            "scope": "unified",
            "variant": "multiscale_coreset_balanced",
            "categories": selected_categories,
            "per_category": unified_per_category,
        },
    )
    for category in selected_categories:
        unified_key = ("unified", "multiscale_coreset_balanced", category)
        if unified_key in completed:
            continue
        category_samples = [sample for sample in all_samples if sample.category == category]
        _, validation, test = _partition_samples(category_samples, settings, quick)
        extraction_started = time.perf_counter()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        pyramids = extractor.extract([sample.image_path for sample in (*validation, *test)])
        extraction_seconds = time.perf_counter() - extraction_started
        peak_gpu_bytes = int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
        composition_started = time.perf_counter()
        maps = _compose_all(pyramids, "multiscale")
        composition_seconds = time.perf_counter() - composition_started
        validation_maps = maps[: len(validation)]
        test_maps = maps[len(validation) :]
        metrics, predictions, score_maps, pixel_threshold = _run_scoring_job(
            memory=unified_memory,
            variant="multiscale_coreset_balanced",
            scope="unified",
            category=category,
            validation_maps=validation_maps,
            test_maps=test_maps,
            validation=validation,
            test=test,
            settings=settings,
            bootstrap_samples=bootstrap_samples,
            feature_seconds=extraction_seconds,
            composition_seconds=composition_seconds,
            fit_seconds=unified_build_seconds,
            feature_cache_bytes=sum(feature_map.nbytes for feature_map in maps),
            compressed_index_bytes=unified_compressed_bytes,
        )
        metrics["train_count"] = unified_train_count
        metrics["peak_gpu_memory_bytes"] = peak_gpu_bytes
        metrics["unified_bank_per_category"] = unified_per_category
        metric_rows.append(metrics)
        prediction_rows.extend(predictions)
        qualitative = _save_qualitative_panels(
            run_dir,
            f"unified/{category}",
            test,
            np.asarray([row["score"] for row in predictions], dtype=np.float64),
            score_maps,
            image_threshold=float(cast(int | float, metrics["threshold"])),
            pixel_threshold=pixel_threshold,
            image_size=settings.image_size,
        )
        for row in qualitative:
            row["category"] = category
            row["scope"] = "unified"
            row["variant"] = "multiscale_coreset_balanced"
        qualitative_rows.extend(qualitative)
        _publish_progress(
            run_dir,
            metric_rows,
            prediction_rows,
            qualitative_rows,
            total_jobs=total_jobs,
        )
        print(
            f"[{len(metric_rows)}/{total_jobs}] unified/multiscale_coreset_balanced/"
            f"{category}: AUROC={float(cast(int | float, metrics['image_auroc'])):.4f} "
            f"pixel={float(cast(int | float, metrics['pixel_auroc'])):.4f}",
            flush=True,
        )

    macro_rows = _macro_rows(metric_rows)
    _add_reference_deltas(macro_rows, effective_variants[0].identifier)
    _atomic_csv(run_dir / "ablation_summary.csv", macro_rows)
    scope_rows = [
        row
        for row in macro_rows
        if row["variant"] in {settings.final_variant, "multiscale_coreset_balanced"}
    ]
    _atomic_csv(run_dir / "scope_comparison.csv", scope_rows)
    summary: dict[str, object] = {
        "mode": mode,
        "seed": settings.seed,
        "categories": len(selected_categories),
        "models": macro_rows,
    }
    _atomic_json(run_dir / "summary.json", summary)
    index_manifest = {
        path.relative_to(run_dir).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted((run_dir / "indexes").rglob("*.npz"))
    }
    _atomic_json(run_dir / "index_manifest.json", index_manifest)
    metadata["status"] = "complete"
    metadata["completed_at_utc"] = datetime.now(UTC).isoformat()
    for filename in (
        "category_metrics.csv",
        "predictions.csv",
        "qualitative_index.csv",
        "ablation_summary.csv",
        "scope_comparison.csv",
        "summary.json",
        "index_manifest.json",
    ):
        path = run_dir / filename
        if path.is_file():
            metadata[f"{path.stem}_sha256"] = sha256_file(path)
    metadata["qualitative_panel_sha256"] = {
        path.relative_to(run_dir).as_posix(): sha256_file(path)
        for path in sorted((run_dir / "qualitative").rglob("*.png"))
    }
    _atomic_json(run_dir / "metadata.json", metadata)
    return M3RunResult(run_dir, summary)

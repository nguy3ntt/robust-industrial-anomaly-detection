"""M2 baseline experiment with immutable inputs and complete run provenance."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlparse

import cv2
import numpy as np
import yaml
from dotenv import load_dotenv
from PIL import Image, ImageOps

from robust_industrial_anomaly_detection.baselines.handcrafted import (
    HandcraftedNearestNeighbour,
    extract_many,
)
from robust_industrial_anomaly_detection.baselines.resnet import (
    FrozenResNet18,
    GlobalFeatureNearestNeighbour,
    PatchMemoryBank,
)
from robust_industrial_anomaly_detection.data.visa import (
    load_trusted_manifest,
    sha256_file,
)
from robust_industrial_anomaly_detection.evaluation.metrics import (
    bootstrap_image_metrics,
    calibrate_threshold,
    image_metrics,
    json_ready,
    pixel_metrics,
    pro_auc,
)

BaselineName = Literal["handcrafted", "global_nn", "patch_memory"]
BASELINES: tuple[BaselineName, ...] = ("handcrafted", "global_nn", "patch_memory")


class ExperimentError(RuntimeError):
    """Raised when M2 cannot satisfy its reproducibility contract."""


@dataclass(frozen=True)
class Sample:
    sample_id: str
    category: str
    split: str
    label: str
    image_path: Path
    mask_path: Path | None

    @property
    def target(self) -> int:
        return int(self.label == "anomaly")


@dataclass(frozen=True)
class ExperimentSettings:
    name: str
    seed: int
    image_threshold_quantile: float
    pixel_threshold_quantile: float
    bootstrap_samples: int
    bootstrap_confidence: float
    handcrafted_image_size: int
    frozen_image_size: int
    frozen_batch_size: int
    patch_bank_size: int
    patch_projected_dimension: int
    patch_patches_per_image: int
    patch_image_score_fraction: float
    patch_gaussian_sigma: float
    pro_max_fpr: float
    pro_thresholds: int
    quick_train: int
    quick_validation: int
    quick_test_normal: int
    quick_test_anomaly: int
    quick_bootstrap_samples: int


@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    summary: dict[str, Any]


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ExperimentError(f"{name} must be a mapping")
    return cast(Mapping[str, object], value)


def _number(section: Mapping[str, object], name: str) -> float:
    value = section.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ExperimentError(f"{name} must be numeric")
    return float(value)


def _integer(section: Mapping[str, object], name: str) -> int:
    value = section.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ExperimentError(f"{name} must be a positive integer")
    return value


def load_settings(path: Path) -> ExperimentSettings:
    """Load and validate the checked-in M2 experiment configuration."""

    try:
        root = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "configuration")
    except (OSError, yaml.YAMLError) as exc:
        raise ExperimentError(f"cannot load experiment configuration {path}: {exc}") from exc
    experiment = _mapping(root.get("experiment"), "experiment")
    handcrafted = _mapping(root.get("handcrafted"), "handcrafted")
    frozen = _mapping(root.get("frozen_resnet18"), "frozen_resnet18")
    patch = _mapping(root.get("patch_memory"), "patch_memory")
    quick = _mapping(root.get("quick"), "quick")
    name = experiment.get("name")
    if not isinstance(name, str) or not name:
        raise ExperimentError("experiment.name must be a non-empty string")
    settings = ExperimentSettings(
        name=name,
        seed=_integer(experiment, "seed"),
        image_threshold_quantile=_number(experiment, "image_threshold_quantile"),
        pixel_threshold_quantile=_number(experiment, "pixel_threshold_quantile"),
        bootstrap_samples=_integer(experiment, "bootstrap_samples"),
        bootstrap_confidence=_number(experiment, "bootstrap_confidence"),
        handcrafted_image_size=_integer(handcrafted, "image_size"),
        frozen_image_size=_integer(frozen, "image_size"),
        frozen_batch_size=_integer(frozen, "batch_size"),
        patch_bank_size=_integer(patch, "bank_size"),
        patch_projected_dimension=_integer(patch, "projected_dimension"),
        patch_patches_per_image=_integer(patch, "patches_per_image"),
        patch_image_score_fraction=_number(patch, "image_score_fraction"),
        patch_gaussian_sigma=_number(patch, "gaussian_sigma"),
        pro_max_fpr=_number(patch, "pro_max_fpr"),
        pro_thresholds=_integer(patch, "pro_thresholds"),
        quick_train=_integer(quick, "train_per_category"),
        quick_validation=_integer(quick, "validation_per_category"),
        quick_test_normal=_integer(quick, "test_normal_per_category"),
        quick_test_anomaly=_integer(quick, "test_anomaly_per_category"),
        quick_bootstrap_samples=_integer(quick, "bootstrap_samples"),
    )
    for field_name in (
        "image_threshold_quantile",
        "pixel_threshold_quantile",
        "bootstrap_confidence",
        "patch_image_score_fraction",
        "pro_max_fpr",
    ):
        if not 0.0 < cast(float, getattr(settings, field_name)) < 1.0:
            raise ExperimentError(f"{field_name} must be strictly between zero and one")
    return settings


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_csv(path: Path) -> list[dict[str, object]]:
    """Read a prior artifact CSV and restore numeric values for aggregation."""

    if not path.is_file():
        return []
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            parsed: dict[str, object] = {}
            for key, value in row.items():
                if value is None or value == "":
                    continue
                if key in {
                    "baseline",
                    "category",
                    "label",
                    "sample_id",
                    "split",
                    "threshold_source",
                    "train_policy",
                }:
                    parsed[key] = value
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def _git_metadata(project_root: Path) -> dict[str, object]:
    def run(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={project_root.as_posix()}", *arguments],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        revision = run("rev-parse", "HEAD")
        status = run("status", "--short")
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"available": False, "error": str(exc)}
    return {
        "available": True,
        "revision": revision,
        "dirty": bool(status),
        "status": status.splitlines(),
    }


def _environment(device: str) -> dict[str, object]:
    packages: dict[str, str | None] = {}
    for name in (
        "numpy",
        "opencv-python-headless",
        "pillow",
        "pyyaml",
        "scikit-learn",
        "scipy",
        "torch",
        "torchvision",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result: dict[str, object] = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "device_requested": device,
    }
    if packages["torch"] is not None:
        import torch

        result["cuda_available"] = torch.cuda.is_available()
        result["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            result["gpu"] = torch.cuda.get_device_name(0)
    return result


def _samples(records: Sequence[dict[str, Any]], image_root: Path) -> tuple[Sample, ...]:
    result: list[Sample] = []
    root = image_root.resolve(strict=True)
    for record in records:
        image_path = root.joinpath(*cast(str, record["image_path"]).split("/"))
        mask_raw = record["mask_path"]
        mask_path = root.joinpath(*cast(str, mask_raw).split("/")) if mask_raw else None
        result.append(
            Sample(
                sample_id=cast(str, record["sample_id"]),
                category=cast(str, record["category"]),
                split=cast(str, record["split"]),
                label=cast(str, record["label"]),
                image_path=image_path,
                mask_path=mask_path,
            )
        )
    return tuple(result)


def _partition(
    samples: Sequence[Sample], settings: ExperimentSettings, quick: bool
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
        raise ExperimentError("every category requires train, validation, and test samples")
    if {sample.label for sample in test} != {"normal", "anomaly"}:
        raise ExperimentError("every category test partition must contain both labels")
    return train, validation, test


def _record_metrics(
    baseline: BaselineName,
    category: str,
    validation_scores: np.ndarray,
    test_scores: np.ndarray,
    test: Sequence[Sample],
    settings: ExperimentSettings,
    bootstrap_samples: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    threshold = calibrate_threshold(validation_scores, settings.image_threshold_quantile)
    labels = np.asarray([sample.target for sample in test], dtype=np.int8)
    metrics: dict[str, object] = {
        "baseline": baseline,
        "category": category,
        "train_policy": "normal-only",
        "threshold_source": "validation-normal-only",
        **image_metrics(labels, test_scores, threshold),
    }
    intervals = bootstrap_image_metrics(
        labels,
        test_scores,
        seed=settings.seed + int(hashlib.sha256(category.encode()).hexdigest()[:8], 16),
        samples=bootstrap_samples,
        confidence=settings.bootstrap_confidence,
    )
    for metric, interval in intervals.items():
        metrics[f"{metric}_ci_low"] = interval["low"]
        metrics[f"{metric}_ci_high"] = interval["high"]
    predictions = [
        {
            "baseline": baseline,
            "category": category,
            "sample_id": sample.sample_id,
            "split": "test",
            "label": sample.label,
            "target": sample.target,
            "score": float(score),
            "threshold": threshold,
            "predicted_anomaly": int(score > threshold),
        }
        for sample, score in zip(test, test_scores, strict=True)
    ]
    return metrics, predictions


def _load_masks(test: Sequence[Sample], shape: tuple[int, int]) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for sample in test:
        if sample.mask_path is None:
            masks.append(np.zeros(shape, dtype=bool))
            continue
        try:
            with Image.open(sample.mask_path) as source:
                fitted = ImageOps.fit(
                    source.convert("L"),
                    (shape[1], shape[0]),
                    method=Image.Resampling.NEAREST,
                    centering=(0.5, 0.5),
                )
                masks.append(np.asarray(fitted, dtype=np.uint8) > 0)
        except (OSError, ValueError) as exc:
            raise ExperimentError(f"cannot read mask: {sample.mask_path}") from exc
    return masks


def _resize_maps(
    score_maps: Sequence[np.ndarray], shape: tuple[int, int], sigma: float
) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for score_map in score_maps:
        resized = cv2.resize(
            np.asarray(score_map, dtype=np.float32),
            (shape[1], shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        if sigma > 0.0:
            resized = cv2.GaussianBlur(resized, (0, 0), sigmaX=sigma, sigmaY=sigma)
        result.append(resized.astype(np.float32, copy=False))
    return result


def _save_qualitative_panels(
    run_dir: Path,
    category: str,
    test: Sequence[Sample],
    scores: np.ndarray,
    score_maps: Sequence[np.ndarray],
    *,
    image_threshold: float,
    pixel_threshold: float,
    image_size: int,
) -> list[dict[str, object]]:
    """Save one high-scoring normal and anomaly panel for local inspection."""

    selected: list[int] = []
    for label in ("normal", "anomaly"):
        candidates = [index for index, sample in enumerate(test) if sample.label == label]
        if candidates:
            selected.append(max(candidates, key=lambda index: float(scores[index])))
    output_dir = run_dir / "qualitative" / category
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for index in selected:
        sample = test[index]
        score_map = np.asarray(score_maps[index], dtype=np.float32)
        with Image.open(sample.image_path) as source:
            fitted = ImageOps.fit(
                source.convert("RGB"),
                (image_size, image_size),
                method=Image.Resampling.BILINEAR,
                centering=(0.5, 0.5),
            )
            rgb = np.asarray(fitted, dtype=np.uint8)
        upper = max(float(np.quantile(score_map, 0.995)), pixel_threshold)
        scale = max(upper - float(np.min(score_map)), float(np.finfo(np.float32).eps))
        normalized = np.clip((score_map - float(np.min(score_map))) / scale, 0.0, 1.0)
        heat_bgr = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(rgb, 0.55, heat_rgb, 0.45, 0.0)
        ground_truth = np.zeros_like(rgb)
        if sample.mask_path is not None:
            try:
                with Image.open(sample.mask_path) as source:
                    fitted_mask = ImageOps.fit(
                        source.convert("L"),
                        (image_size, image_size),
                        method=Image.Resampling.NEAREST,
                        centering=(0.5, 0.5),
                    )
                    ground_truth[np.asarray(fitted_mask, dtype=np.uint8) > 0] = (
                        255,
                        255,
                        255,
                    )
            except (OSError, ValueError) as exc:
                raise ExperimentError(f"cannot read mask: {sample.mask_path}") from exc
        panel = np.concatenate((rgb, overlay, ground_truth), axis=1)
        label = f"{sample.label} score={scores[index]:.4f} threshold={image_threshold:.4f}"
        cv2.putText(
            panel,
            label,
            (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        relative = Path("qualitative") / category / f"{sample.sample_id}-{sample.label}.png"
        destination = run_dir / relative
        if not cv2.imwrite(str(destination), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)):
            raise ExperimentError(f"cannot write qualitative panel: {destination}")
        rows.append(
            {
                "category": category,
                "sample_id": sample.sample_id,
                "label": sample.label,
                "score": float(scores[index]),
                "image_threshold": image_threshold,
                "pixel_threshold": pixel_threshold,
                "panel_path": relative.as_posix(),
                "panel_columns": "input|anomaly-overlay|ground-truth-mask",
            }
        )
    return rows


def _macro_summary(metrics: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_baseline: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in metrics:
        by_baseline[cast(str, row["baseline"])].append(row)
    summary: dict[str, object] = {}
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
        "detector_seconds",
        "feature_extraction_seconds",
        "end_to_end_seconds",
        "images_per_second",
    )
    for baseline, rows in sorted(by_baseline.items()):
        values: dict[str, object] = {"categories": len(rows)}
        for name in metric_names:
            present = [
                float(cast(int | float, row[name]))
                for row in rows
                if name in row and isinstance(row[name], (int, float))
            ]
            if present:
                values[f"macro_{name}"] = float(np.mean(present))
        summary[baseline] = values
    return summary


def run_m2(
    *,
    project_root: Path,
    experiment_config: Path,
    data_config: Path,
    baselines: Sequence[BaselineName],
    categories: Sequence[str] | None,
    device: str,
    quick: bool,
    resume_run: Path | None = None,
) -> RunResult:
    """Execute M2 and publish self-describing, incrementally durable artifacts."""

    load_dotenv(project_root / ".env")
    settings = load_settings(experiment_config)
    data_root_raw = os.environ.get("ANOMALY_DATA_ROOT")
    artifact_root_raw = os.environ.get("ANOMALY_ARTIFACT_ROOT")
    if not data_root_raw or not artifact_root_raw:
        raise ExperimentError("ANOMALY_DATA_ROOT and ANOMALY_ARTIFACT_ROOT must be configured")
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
        raise ExperimentError(f"unknown categories: {unknown}")
    invalid_baselines = sorted(set(baselines) - set(BASELINES))
    if invalid_baselines:
        raise ExperimentError(f"unknown baselines: {invalid_baselines}")

    random.seed(settings.seed)
    np.random.seed(settings.seed)
    mode = "smoke" if quick else "full"
    resolved = {
        **asdict(settings),
        "mode": mode,
        "baselines": list(baselines),
        "categories": selected_categories,
        "device": device,
        "data_root": str(data_root),
        "artifact_root": str(artifact_root),
    }
    uses_frozen_backbone = bool(set(baselines) & {"global_nn", "patch_memory"})
    model_metadata: dict[str, object] = {
        "backbone": "resnet18" if uses_frozen_backbone else None,
        "weights": FrozenResNet18.weights_id if uses_frozen_backbone else None,
        "weights_url": FrozenResNet18.weights_url if uses_frozen_backbone else None,
        "weights_frozen": True,
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
            "model": model_metadata,
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
            raise ExperimentError("resume directory must be below the configured run root") from exc
        prior_resolved = yaml.safe_load((run_dir / "resolved_config.yaml").read_text("utf-8"))
        if prior_resolved != resolved:
            raise ExperimentError("resume settings differ from the interrupted run")
        metadata_raw = json.loads((run_dir / "metadata.json").read_text("utf-8"))
        if not isinstance(metadata_raw, dict) or metadata_raw.get("status") != "running":
            raise ExperimentError("only an incomplete run with status=running can be resumed")
        metadata = cast(dict[str, object], metadata_raw)
        if any(metadata.get(key) != value for key, value in current_hashes.items()):
            raise ExperimentError("dataset or configuration hashes changed since the run started")
        prior_model = metadata.get("model")
        if not isinstance(prior_model, dict):
            raise ExperimentError("resume metadata has no valid model record")
        model_metadata = cast(dict[str, object], prior_model)
        resume_times = metadata.get("resumed_at_utc", [])
        if not isinstance(resume_times, list):
            resume_times = []
        resume_times.append(datetime.now(UTC).isoformat())
        metadata["resumed_at_utc"] = resume_times
        metric_rows = _read_csv(run_dir / "category_metrics.csv")
        prediction_rows = _read_csv(run_dir / "predictions.csv")
        qualitative_rows = _read_csv(run_dir / "qualitative_index.csv")
        _atomic_json(run_dir / "metadata.json", metadata)

    extractor = (
        FrozenResNet18(
            device=device,
            image_size=settings.frozen_image_size,
            batch_size=settings.frozen_batch_size,
        )
        if set(baselines) & {"global_nn", "patch_memory"}
        else None
    )
    if extractor is not None:
        torch_home = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))
        checkpoint = (
            torch_home
            / "hub"
            / "checkpoints"
            / Path(urlparse(FrozenResNet18.weights_url).path).name
        )
        if not checkpoint.is_file():
            raise ExperimentError(f"downloaded model checkpoint is unavailable: {checkpoint}")
        model_metadata["checkpoint_sha256"] = sha256_file(checkpoint)
        model_metadata["checkpoint_bytes"] = checkpoint.stat().st_size
        _atomic_json(run_dir / "metadata.json", metadata)
    bootstrap_samples = settings.quick_bootstrap_samples if quick else settings.bootstrap_samples
    total_jobs = len(baselines) * len(selected_categories)
    completed = {(cast(str, row["baseline"]), cast(str, row["category"])) for row in metric_rows}
    completed_jobs = len(completed)
    for category in selected_categories:
        pending_baselines = [
            baseline for baseline in baselines if (baseline, category) not in completed
        ]
        if not pending_baselines:
            continue
        category_samples = [sample for sample in all_samples if sample.category == category]
        train, validation, test = _partition(category_samples, settings, quick)
        frozen = None
        frozen_seconds = 0.0
        if extractor is not None and set(pending_baselines) & {"global_nn", "patch_memory"}:
            frozen_started = time.perf_counter()
            frozen = extractor.extract(
                [sample.image_path for sample in (*train, *validation, *test)]
            )
            frozen_seconds = time.perf_counter() - frozen_started
        train_count = len(train)
        validation_count = len(validation)
        for baseline in pending_baselines:
            started = time.perf_counter()
            if baseline == "handcrafted":
                model = HandcraftedNearestNeighbour()
                train_features = extract_many(
                    [sample.image_path for sample in train], size=settings.handcrafted_image_size
                )
                validation_features = extract_many(
                    [sample.image_path for sample in validation],
                    size=settings.handcrafted_image_size,
                )
                test_features = extract_many(
                    [sample.image_path for sample in test], size=settings.handcrafted_image_size
                )
                model.fit(train_features)
                validation_scores = model.score(validation_features)
                test_scores = model.score(test_features)
                score_maps: list[np.ndarray] | None = None
            else:
                assert frozen is not None
                train_slice = slice(0, train_count)
                validation_slice = slice(train_count, train_count + validation_count)
                test_slice = slice(train_count + validation_count, None)
                if baseline == "global_nn":
                    global_model = GlobalFeatureNearestNeighbour()
                    global_model.fit(frozen.global_features[train_slice])
                    validation_scores = global_model.score(frozen.global_features[validation_slice])
                    test_scores = global_model.score(frozen.global_features[test_slice])
                    score_maps = None
                else:
                    patch_model = PatchMemoryBank(
                        seed=settings.seed,
                        bank_size=settings.patch_bank_size,
                        projected_dimension=settings.patch_projected_dimension,
                        patches_per_image=settings.patch_patches_per_image,
                        image_score_fraction=settings.patch_image_score_fraction,
                    )
                    patch_model.fit(frozen.patch_features[train_slice])
                    validation_scores, validation_maps_raw = patch_model.score(
                        frozen.patch_features[validation_slice]
                    )
                    test_scores, test_maps_raw = patch_model.score(
                        frozen.patch_features[test_slice]
                    )
                    evaluation_shape = (settings.frozen_image_size, settings.frozen_image_size)
                    validation_maps = _resize_maps(
                        validation_maps_raw, evaluation_shape, settings.patch_gaussian_sigma
                    )
                    score_maps = _resize_maps(
                        test_maps_raw, evaluation_shape, settings.patch_gaussian_sigma
                    )
                    pixel_threshold = calibrate_threshold(
                        np.concatenate([score_map.reshape(-1) for score_map in validation_maps]),
                        settings.pixel_threshold_quantile,
                    )
            metrics, predictions = _record_metrics(
                baseline,
                category,
                validation_scores,
                test_scores,
                test,
                settings,
                bootstrap_samples,
            )
            metrics["train_count"] = len(train)
            metrics["validation_normal_count"] = len(validation)
            if score_maps is not None:
                masks = _load_masks(test, score_maps[0].shape)
                metrics.update(pixel_metrics(masks, score_maps, pixel_threshold))
                metrics["aupro_30"] = pro_auc(
                    masks,
                    score_maps,
                    max_fpr=settings.pro_max_fpr,
                    thresholds=settings.pro_thresholds,
                )
                qualitative_rows.extend(
                    _save_qualitative_panels(
                        run_dir,
                        category,
                        test,
                        test_scores,
                        score_maps,
                        image_threshold=float(cast(int | float, metrics["threshold"])),
                        pixel_threshold=pixel_threshold,
                        image_size=settings.frozen_image_size,
                    )
                )
                _atomic_csv(run_dir / "qualitative_index.csv", qualitative_rows)
            detector_seconds = time.perf_counter() - started
            feature_seconds = 0.0 if baseline == "handcrafted" else frozen_seconds
            end_to_end_seconds = detector_seconds + feature_seconds
            metrics["detector_seconds"] = detector_seconds
            metrics["feature_extraction_seconds"] = feature_seconds
            metrics["end_to_end_seconds"] = end_to_end_seconds
            metrics["images_per_second"] = (
                len(train) + len(validation) + len(test)
            ) / end_to_end_seconds
            metric_rows.append(metrics)
            prediction_rows.extend(predictions)
            completed_jobs += 1
            _atomic_csv(run_dir / "category_metrics.csv", metric_rows)
            _atomic_csv(run_dir / "predictions.csv", prediction_rows)
            _atomic_json(
                run_dir / "progress.json",
                {"completed_jobs": completed_jobs, "total_jobs": total_jobs},
            )
            print(
                f"[{completed_jobs}/{total_jobs}] {baseline}/{category}: "
                f"AUROC={float(cast(int | float, metrics['image_auroc'])):.4f} "
                f"AP={float(cast(int | float, metrics['image_ap'])):.4f}",
                flush=True,
            )

    summary = {
        "mode": mode,
        "seed": settings.seed,
        "completed_categories": len(selected_categories),
        "results": _macro_summary(metric_rows),
    }
    _atomic_json(run_dir / "summary.json", summary)
    metadata["status"] = "complete"
    metadata["completed_at_utc"] = datetime.now(UTC).isoformat()
    metadata["category_metrics_sha256"] = sha256_file(run_dir / "category_metrics.csv")
    metadata["predictions_sha256"] = sha256_file(run_dir / "predictions.csv")
    qualitative_index = run_dir / "qualitative_index.csv"
    if qualitative_index.is_file():
        metadata["qualitative_index_sha256"] = sha256_file(qualitative_index)
        metadata["qualitative_panel_sha256"] = {
            path.relative_to(run_dir).as_posix(): sha256_file(path)
            for path in sorted((run_dir / "qualitative").rglob("*.png"))
        }
    _atomic_json(run_dir / "metadata.json", metadata)
    return RunResult(run_dir, summary)

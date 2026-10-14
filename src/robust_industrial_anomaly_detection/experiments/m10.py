"""Fixed normal-only localization ablations on verified VisA and KolektorSDD2 models."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from dotenv import load_dotenv

from robust_industrial_anomaly_detection.baselines.multiscale import (
    ProjectedPatchMemory,
    compose_feature_map,
    load_fitted_rgb,
    load_projected_index,
)
from robust_industrial_anomaly_detection.data.ksdd2 import load_ksdd2_manifest, read_config
from robust_industrial_anomaly_detection.data.visa import load_trusted_manifest, sha256_file
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
from robust_industrial_anomaly_detection.experiments.m4 import load_m4_settings
from robust_industrial_anomaly_detection.experiments.m9 import full_frame_pair
from robust_industrial_anomaly_detection.prototype.engine import (
    InspectionEngine,
    load_prototype_settings,
)
from robust_industrial_anomaly_detection.prototype.localization import (
    POLICIES,
    fit_localization_model,
    load_localization_model,
    localization_support,
    save_localization_model,
)
from robust_industrial_anomaly_detection.robustness.transforms import ShiftSpec, apply_shift


class M10Error(ValueError):
    """The localization experiment violates its frozen evaluation contract."""


@dataclass(frozen=True)
class LocalizationUnit:
    dataset: str
    category: str
    samples: tuple[Sample, ...]
    memory: ProjectedPatchMemory
    baseline_threshold: float
    grid: int
    geometry: str
    provenance: dict[str, object]


def mask_measurements(mask: np.ndarray, support: np.ndarray, label: str) -> dict[str, object]:
    """Measure actual support; visible-defect metrics never reward empty normal masks."""

    target = np.asarray(mask, dtype=bool)
    predicted = np.asarray(support, dtype=bool)
    if target.ndim != 2 or target.shape != predicted.shape or label not in {"normal", "anomaly"}:
        raise M10Error("binary localization masks/label have incompatible contracts")
    intersection = int(np.count_nonzero(target & predicted))
    positive = int(np.count_nonzero(target))
    flagged = int(np.count_nonzero(predicted))
    union = positive + flagged - intersection
    visible = positive > 0 and label == "anomaly"
    result: dict[str, object] = {
        "mask_pixels": positive,
        "flagged_pixels": flagged,
        "intersection_pixels": intersection,
        "pixel_count": target.size,
        "false_positive_pixels": flagged - intersection,
        "background_pixels": target.size - positive,
        "all_image_iou": intersection / union if union else 1.0,
        "foreground_iou": intersection / union if visible else None,
        "foreground_dice": 2 * intersection / (positive + flagged) if visible else None,
        "foreground_recall": intersection / positive if visible else None,
        "foreground_precision": intersection / flagged
        if visible and flagged
        else (0.0 if visible else None),
        "has_support": int(flagged > 0),
    }
    return result


def _metrics(rows: Sequence[dict[str, Any]]) -> dict[str, object]:
    normals = [row for row in rows if row["label"] == "normal"]
    visible = [row for row in rows if row["foreground_iou"] is not None]

    def average(selected: Sequence[dict[str, Any]], key: str) -> float | None:
        return float(np.mean([row[key] for row in selected])) if selected else None

    return {
        "samples": len(rows),
        "normal_samples": len(normals),
        "anomaly_samples": sum(row["label"] == "anomaly" for row in rows),
        "visible_anomaly_samples": len(visible),
        "empty_anomaly_masks": sum(
            row["label"] == "anomaly" and row["mask_pixels"] == 0 for row in rows
        ),
        "all_image_iou": average(rows, "all_image_iou"),
        "foreground_iou": average(visible, "foreground_iou"),
        "foreground_dice": average(visible, "foreground_dice"),
        "foreground_recall": average(visible, "foreground_recall"),
        "foreground_precision": average(visible, "foreground_precision"),
        "normal_pixel_fpr": sum(row["flagged_pixels"] for row in normals)
        / sum(row["pixel_count"] for row in normals),
        "normal_image_support_rate": average(normals, "has_support"),
        "background_pixel_fpr": sum(row["false_positive_pixels"] for row in rows)
        / sum(row["background_pixels"] for row in rows),
    }


def _bootstrap_deltas(
    rows: Sequence[dict[str, Any]],
    *,
    seed: int,
    samples: int,
    policy: str,
) -> dict[str, object]:
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row["foreground_iou"] is not None:
            grouped[row["sample_id"]][row["policy"]] = row["foreground_iou"]
    deltas = np.asarray([group[policy] - group["baseline_global"] for group in grouped.values()])
    if not len(deltas):
        return {"paired_foreground_iou_delta": None}
    rng = np.random.default_rng(seed)
    estimates = np.mean(rng.choice(deltas, (samples, len(deltas)), replace=True), axis=1)
    return {
        "paired_foreground_iou_delta": float(np.mean(deltas)),
        "paired_delta_ci_low": float(np.quantile(estimates, 0.025)),
        "paired_delta_ci_high": float(np.quantile(estimates, 0.975)),
        "paired_samples": len(deltas),
    }


def _verify_checkpoint(root: Path, checkpoint: Path) -> bool:
    if not checkpoint.is_file():
        return False
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    for filename, expected in payload["files"].items():
        path = (root / filename).resolve(strict=True)
        path.relative_to(root.resolve(strict=True))
        if sha256_file(path) != expected:
            raise M10Error(f"checkpoint artifact changed: {filename}")
    return True


def _load_units(
    *,
    project_root: Path,
    data_root: Path,
    artifact_root: Path,
    config: dict[str, Any],
    engine: InspectionEngine,
    datasets: Sequence[str],
) -> tuple[list[LocalizationUnit], dict[str, str]]:
    units: list[LocalizationUnit] = []
    hashes: dict[str, str] = {}
    if "visa" in datasets:
        data_config = project_root / config["sources"]["visa_config"]
        manifest_dir = data_root / "manifests" / "visa-20220922"
        root = data_root / "sources" / "visa-20220922"
        records = load_trusted_manifest(
            manifest_dir / "manifest.jsonl", manifest_dir / "audit.json", data_config, root
        )
        samples = _samples(records, root)
        hashes.update(
            {
                "visa_manifest_sha256": sha256_file(manifest_dir / "manifest.jsonl"),
                "visa_audit_sha256": sha256_file(manifest_dir / "audit.json"),
                "visa_m5_metadata_sha256": sha256_file(engine.m5_run / "metadata.json"),
                "visa_m4_metadata_sha256": sha256_file(engine.m4_run / "metadata.json"),
            }
        )
        for category in engine.categories:
            units.append(
                LocalizationUnit(
                    "visa",
                    category,
                    tuple(sample for sample in samples if sample.category == category),
                    engine._memory(category),
                    engine._calibration[category].pixel_threshold,
                    engine.image_size,
                    "center-crop",
                    {
                        "manifest_sha256": hashes["visa_manifest_sha256"],
                        "source_index_sha256": sha256_file(engine._index_paths[category]),
                        "source_m5_metadata_sha256": hashes["visa_m5_metadata_sha256"],
                    },
                )
            )
    if "ksdd2" in datasets:
        data_config = project_root / config["sources"]["ksdd2_config"]
        ksdd_records, root = load_ksdd2_manifest(data_config, data_root)
        dataset_id = read_config(data_config)["dataset"]["id"]
        manifest = data_root / "manifests" / dataset_id / "manifest.jsonl"
        source = artifact_root / "runs" / config["sources"]["ksdd2_m9_run"]
        source.resolve(strict=True).relative_to((artifact_root / "runs").resolve(strict=True))
        metadata = json.loads((source / "metadata.json").read_text("utf-8"))
        if (
            metadata.get("status") != "complete"
            or metadata.get("mode") != "full"
            or metadata.get("manifest_sha256") != sha256_file(manifest)
        ):
            raise M10Error("KolektorSDD2 source run is incomplete or used different data")
        if metadata.get("checkpoint_sha256") != sha256_file(engine.checkpoint_path):
            raise M10Error("KolektorSDD2 backbone differs from VisA")
        for filename in (
            "calibration.json",
            "resolved_config.json",
            "indexes/augmented_coreset_512.npz",
        ):
            if metadata["output_hashes"].get(filename) != sha256_file(source / filename):
                raise M10Error(f"KolektorSDD2 source artifact changed: {filename}")
        resolved = json.loads((source / "resolved_config.json").read_text("utf-8"))
        calibration = json.loads((source / "calibration.json").read_text("utf-8"))
        index = source / "indexes" / "augmented_coreset_512.npz"
        bank, projection = load_projected_index(index)
        memory = ProjectedPatchMemory(
            seed=config["experiment"]["seed"],
            bank_size=len(bank),
            projected_dimension=projection.shape[1],
            patches_per_image=1,
            candidate_limit=len(bank),
            image_score_fraction=resolved["model"]["image_score_fraction"],
            selection="kcenter",
        )
        memory.fit_projected(bank, projection)
        hashes["ksdd2_manifest_sha256"] = sha256_file(manifest)
        hashes["ksdd2_m9_metadata_sha256"] = sha256_file(source / "metadata.json")
        units.append(
            LocalizationUnit(
                "ksdd2",
                "surface",
                _samples(ksdd_records, root),
                memory,
                calibration["parameters"]["augmented_coreset_512"]["pixel_threshold"],
                resolved["model"]["localization_size"],
                resolved["model"]["geometry"],
                {
                    "manifest_sha256": hashes["ksdd2_manifest_sha256"],
                    "source_index_sha256": sha256_file(index),
                    "source_m9_metadata_sha256": hashes["ksdd2_m9_metadata_sha256"],
                },
            )
        )
    return units, hashes


def run_m10(
    *,
    project_root: Path,
    config_path: Path,
    datasets: Sequence[str] | None,
    categories: Sequence[str] | None,
    device: str,
    quick: bool,
    resume: Path | None,
    reference_run: Path | None = None,
) -> dict[str, object]:
    """Calibrate on separate normal partitions and evaluate frozen binary-support ablations."""

    load_dotenv(project_root / ".env")
    config = read_config(config_path)
    localization = config["localization"]
    if localization["primary_policy"] != "raw_cleaned":
        raise M10Error("primary policy must be frozen before test evaluation")
    selected_datasets = list(datasets) if datasets else ["visa", "ksdd2"]
    if len(set(selected_datasets)) != len(selected_datasets):
        raise M10Error("duplicate datasets selected")
    data_root = Path(os.environ["ANOMALY_DATA_ROOT"])
    artifact_root = Path(os.environ["ANOMALY_ARTIFACT_ROOT"])
    engine = InspectionEngine(
        project_root=project_root,
        settings=load_prototype_settings(project_root / "configs/prototype/m6.yaml"),
        device=device,
        m5_run=artifact_root / "runs" / config["sources"]["visa_m5_run"],
    )
    engine.extractor.batch_size = localization["batch_size"]
    units, source_hashes = _load_units(
        project_root=project_root,
        data_root=data_root,
        artifact_root=artifact_root,
        config=config,
        engine=engine,
        datasets=selected_datasets,
    )
    if categories:
        unknown = set(categories) - {unit.category for unit in units}
        if unknown:
            raise M10Error(f"unknown categories: {sorted(unknown)}")
        units = [unit for unit in units if unit.category in categories]
    robustness_path = project_root / config["sources"]["robustness_config"]
    frozen = load_m4_settings(robustness_path)
    condition_index = {condition.identifier: condition for condition in frozen.conditions}
    condition_index["clean"] = ShiftSpec("clean", "clean", 0, {})
    conditions = [condition_index[name] for name in config["conditions"]]
    mode = "smoke" if quick else "full"
    resolved = {
        **config,
        "mode": mode,
        "device": device,
        "units": [[unit.dataset, unit.category] for unit in units],
        "reference_run": str(reference_run) if reference_run else None,
    }
    code_paths = [
        Path(__file__),
        project_root / "src/robust_industrial_anomaly_detection/prototype/localization.py",
    ]
    hashes = {
        **source_hashes,
        "experiment_config_sha256": sha256_file(config_path),
        "robustness_config_sha256": sha256_file(robustness_path),
        "checkpoint_sha256": sha256_file(engine.checkpoint_path),
        "code_sha256": hashlib.sha256(
            "".join(sha256_file(path) for path in code_paths).encode()
        ).hexdigest(),
    }
    if resume is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_dir = artifact_root / "runs" / f"{config['experiment']['name']}-{mode}-{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=False)
        metadata: dict[str, Any] = {
            "schema_version": 1,
            "status": "running",
            "mode": mode,
            "started_at_utc": datetime.now(UTC).isoformat(),
            "git": _git_metadata(project_root),
            **hashes,
        }
        _atomic_json(run_dir / "metadata.json", metadata)
        _atomic_json(run_dir / "resolved_config.json", resolved)
        _atomic_json(run_dir / "environment.json", _environment(device))
    else:
        run_dir = resume.resolve(strict=True)
        run_dir.relative_to((artifact_root / "runs").resolve(strict=True))
        metadata = json.loads((run_dir / "metadata.json").read_text("utf-8"))
        if metadata.get("status") != "running" or any(
            metadata.get(key) != value for key, value in hashes.items()
        ):
            raise M10Error("resume source/config/code changed or run is not incomplete")
        if json.loads((run_dir / "resolved_config.json").read_text("utf-8")) != resolved:
            raise M10Error("resume settings changed")
    print(f"Run artifacts: {run_dir}", flush=True)
    completed: list[str] = []
    minimum_area = float(localization["minimum_region_fraction"])
    for unit_position, unit in enumerate(units, start=1):
        unit_dir = run_dir / "units" / unit.dataset / unit.category
        unit_dir.mkdir(parents=True, exist_ok=True)
        train = sorted(
            [sample for sample in unit.samples if sample.split == "train"],
            key=lambda sample: hashlib.sha256(
                f"{config['experiment']['seed']}:{sample.sample_id}".encode()
            ).digest(),
        )
        train = train[
            : config["quick"]["fit_normals"] if quick else localization["fit_normals_cap"]
        ]
        calibration = [sample for sample in unit.samples if sample.split == "validation"]
        test = [sample for sample in unit.samples if sample.split == "test"]
        if any(sample.label != "normal" for sample in [*train, *calibration]):
            raise M10Error("localization reference/calibration must be normal-only")
        if quick:
            calibration = calibration[: config["quick"]["calibration_normals"]]
            test = [sample for sample in test if sample.label == "normal"][
                : config["quick"]["test_normals"]
            ] + [sample for sample in test if sample.label == "anomaly"][
                : config["quick"]["test_anomalies"]
            ]

        def pair(sample: Sample, current: LocalizationUnit = unit) -> tuple[np.ndarray, np.ndarray]:
            if current.dataset == "ksdd2":
                if sample.mask_path is None:
                    raise M10Error("KolektorSDD2 mask is missing")
                return full_frame_pair(sample.image_path, sample.mask_path, engine.image_size)
            return load_fitted_rgb(sample.image_path, engine.image_size), _load_masks(
                [sample], (engine.image_size, engine.image_size)
            )[0]

        def score_maps(
            images: Sequence[np.ndarray], current: LocalizationUnit = unit
        ) -> list[np.ndarray]:
            maps: list[np.ndarray] = []
            for offset in range(0, len(images), localization["batch_size"]):
                pyramids = engine.extractor.extract_rgb(
                    images[offset : offset + localization["batch_size"]]
                )
                _, raw = current.memory.score(
                    [compose_feature_map(item, "multiscale") for item in pyramids]
                )
                fitted = _resize_maps(
                    raw, (engine.image_size, engine.image_size), engine.gaussian_sigma
                )
                if current.grid != engine.image_size:
                    fitted = [
                        cv2.resize(item, (current.grid, current.grid), interpolation=cv2.INTER_AREA)
                        for item in fitted
                    ]
                maps.extend(fitted)
            return maps

        fit_checkpoint = unit_dir / "fit_checkpoint.json"
        model_path = unit_dir / "localization.npz"
        if not _verify_checkpoint(unit_dir, fit_checkpoint) and reference_run is not None:
            reference_path = reference_run / "units" / unit.dataset / unit.category
            if _verify_checkpoint(reference_path, reference_path / "fit_checkpoint.json"):
                reference_path.resolve(strict=True).relative_to(
                    (artifact_root / "runs").resolve(strict=True)
                )
                source_ids = json.loads((reference_path / "fit_checkpoint.json").read_text("utf-8"))
                reference_model = load_localization_model(reference_path / "localization.npz")
                if (
                    source_ids["fit_ids"] != [sample.sample_id for sample in train]
                    or source_ids["calibration_ids"] != [sample.sample_id for sample in calibration]
                    or reference_model.provenance.get("source_index_sha256")
                    != unit.provenance["source_index_sha256"]
                    or reference_model.provenance.get("manifest_sha256")
                    != unit.provenance["manifest_sha256"]
                ):
                    raise M10Error("reused reference has different normal inputs or source model")
                save_localization_model(model_path, reference_model)
                _atomic_json(
                    fit_checkpoint,
                    {
                        "files": {"localization.npz": sha256_file(model_path)},
                        "fit_ids": source_ids["fit_ids"],
                        "calibration_ids": source_ids["calibration_ids"],
                        "reference_sha256": sha256_file(reference_path / "localization.npz"),
                    },
                )
        if not _verify_checkpoint(unit_dir, fit_checkpoint):
            fit_maps = score_maps([pair(sample)[0] for sample in train])
            calibration_maps = score_maps([pair(sample)[0] for sample in calibration])
            model = fit_localization_model(
                fit_maps,
                calibration_maps,
                high_quantile=localization["high_quantile"],
                low_quantile=localization["low_quantile"],
                minimum_region_fraction=minimum_area,
                scale_floor_fraction=localization["scale_floor_fraction"],
                normal_pixel_budget=localization["normal_pixel_budget"],
                search_steps=localization["threshold_search_steps"],
                provenance={
                    **unit.provenance,
                    "dataset": unit.dataset,
                    "category": unit.category,
                    "geometry": unit.geometry,
                    "image_size": engine.image_size,
                    "evaluation_grid": unit.grid,
                },
            )
            save_localization_model(model_path, model)
            _atomic_json(
                fit_checkpoint,
                {
                    "files": {"localization.npz": sha256_file(model_path)},
                    "fit_ids": [sample.sample_id for sample in train],
                    "calibration_ids": [sample.sample_id for sample in calibration],
                },
            )
            del fit_maps, calibration_maps
        model = load_localization_model(model_path)
        clean_pairs: list[tuple[np.ndarray, np.ndarray]] | None = None
        for condition in conditions:
            condition_dir = unit_dir / condition.identifier
            checkpoint = condition_dir / "checkpoint.json"
            key = f"{unit.dataset}/{unit.category}/{condition.identifier}"
            if _verify_checkpoint(condition_dir, checkpoint):
                completed.append(key)
                continue
            if clean_pairs is None:
                clean_pairs = [pair(sample) for sample in test]
            condition_dir.mkdir(parents=True, exist_ok=True)
            start = time.perf_counter()
            shifted = [
                apply_shift(
                    image,
                    mask,
                    condition,
                    seed=config["experiment"]["seed"],
                    sample_key=sample.sample_id,
                )
                for sample, (image, mask) in zip(test, clean_pairs, strict=True)
            ]
            maps = score_maps([item.image for item in shifted])
            masks = [
                cv2.resize(
                    item.mask.astype(np.uint8),
                    (unit.grid, unit.grid),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                for item in shifted
            ]
            rows: list[dict[str, Any]] = []
            for sample, score_map, mask in zip(test, maps, masks, strict=True):
                for policy in POLICIES:
                    _, support, _ = localization_support(
                        score_map, model, policy, baseline_threshold=unit.baseline_threshold
                    )
                    rows.append(
                        {
                            "dataset": unit.dataset,
                            "category": unit.category,
                            "condition": condition.identifier,
                            "policy": policy,
                            "sample_id": sample.sample_id,
                            "label": sample.label,
                            **mask_measurements(mask, support, sample.label),
                        }
                    )
            metrics = [
                {
                    "dataset": unit.dataset,
                    "category": unit.category,
                    "condition": condition.identifier,
                    "policy": policy,
                    "evaluation_grid": unit.grid,
                    "baseline_threshold": unit.baseline_threshold,
                    "raw_high": model.raw_high,
                    "spatial_high": model.spatial_high,
                    "spatial_global_high": model.spatial_global_high,
                    "calibration_samples": model.calibration_samples,
                    "calibration_raw_support_rate": model.calibration_raw_support_rate,
                    "calibration_spatial_support_rate": model.calibration_spatial_support_rate,
                    **_metrics([row for row in rows if row["policy"] == policy]),
                }
                for policy in POLICIES
            ]
            for row in metrics:
                if row["policy"] != "baseline_global":
                    row.update(
                        _bootstrap_deltas(
                            rows,
                            seed=config["experiment"]["seed"],
                            samples=config["quick"]["bootstrap_samples"]
                            if quick
                            else config["experiment"]["bootstrap_samples"],
                            policy=str(row["policy"]),
                        )
                    )
                row["condition_seconds"] = time.perf_counter() - start
            _atomic_csv(condition_dir / "metrics.csv", metrics)
            _atomic_csv(condition_dir / "measurements.csv", rows)
            _atomic_json(
                checkpoint,
                {
                    "files": {
                        name: sha256_file(condition_dir / name)
                        for name in ("metrics.csv", "measurements.csv")
                    }
                },
            )
            completed.append(key)
            _atomic_json(
                run_dir / "progress.json",
                {
                    "completed": completed,
                    "total": len(units) * len(conditions),
                    "resume_command": (
                        "python scripts/run_m10_localization.py "
                        f'--device {device} {"--quick " if quick else ""}--resume "{run_dir}"'
                    ),
                },
            )
            print(f"[{unit_position}/{len(units)}] {key}: localization evaluated", flush=True)
    rows = [
        row
        for unit in units
        for condition in conditions
        for row in _read_csv(
            run_dir / "units" / unit.dataset / unit.category / condition.identifier / "metrics.csv"
        )
    ]
    _atomic_csv(run_dir / "metrics.csv", rows)
    summary: dict[str, object] = {
        "run_dir": str(run_dir),
        "mode": mode,
        "primary_policy": localization["primary_policy"],
        "units": len(units),
        "conditions": len(conditions),
        "normal_pixel_budget": localization["normal_pixel_budget"],
        "model_detection_changed": False,
        "capture_gate_changed": False,
        "source_m5_metadata_sha256": source_hashes.get("visa_m5_metadata_sha256"),
        "clean_results": [row for row in rows if row["condition"] == "clean"],
    }
    _atomic_json(run_dir / "summary.json", summary)
    metadata.update({"status": "complete", "completed_at_utc": datetime.now(UTC).isoformat()})
    metadata["output_hashes"] = {
        path.relative_to(run_dir).as_posix(): sha256_file(path)
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name != "metadata.json"
    }
    _atomic_json(run_dir / "metadata.json", metadata)
    return summary

"""Independent second-dataset evaluation of the frozen normal-only method."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np
from dotenv import load_dotenv
from PIL import Image

from robust_industrial_anomaly_detection.baselines.multiscale import (
    FrozenPyramidResNet18,
    ProjectedPatchMemory,
    compose_feature_map,
    load_projected_index,
)
from robust_industrial_anomaly_detection.data.ksdd2 import load_ksdd2_manifest, read_config
from robust_industrial_anomaly_detection.data.visa import sha256_file
from robust_industrial_anomaly_detection.evaluation.calibration import (
    conformal_p_values,
    exact_binomial_interval,
    selective_decisions,
    selective_metrics,
)
from robust_industrial_anomaly_detection.evaluation.metrics import (
    bootstrap_image_metrics,
    calibrate_threshold,
    image_metrics,
    pixel_metrics,
    pro_auc,
)
from robust_industrial_anomaly_detection.experiments.m2 import (
    _atomic_csv,
    _atomic_json,
    _environment,
    _git_metadata,
    _read_csv,
)
from robust_industrial_anomaly_detection.experiments.m4 import load_m4_settings
from robust_industrial_anomaly_detection.experiments.m8 import load_m8_settings
from robust_industrial_anomaly_detection.prototype.quality import (
    CaptureQualityModel,
    capture_metrics,
    evaluate_capture_quality,
    fit_category_quality_model,
    load_quality_model,
    quality_model_dict,
)
from robust_industrial_anomaly_detection.robustness.transforms import ShiftSpec, apply_shift

MODELS = ("plain_coreset_512", "augmented_coreset_512")


class M9Error(ValueError):
    """External validation violates its frozen protocol or provenance."""


def full_frame_pair(image_path: Path, mask_path: Path, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Resize the entire camera frame and mask together; never discard end-of-strip defects."""

    with Image.open(image_path) as image, Image.open(mask_path) as mask:
        rgb = np.asarray(image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR))
        binary = np.asarray(mask.convert("L").resize((size, size), Image.Resampling.NEAREST)) > 0
    return rgb.astype(np.uint8), binary


def guarded_metrics(
    labels: Sequence[int], decisions: Sequence[str] | np.ndarray, rejected: Sequence[bool]
) -> dict[str, object]:
    """Count withheld inputs explicitly; report automatic errors and deferrals separately."""

    target = np.asarray(labels, dtype=np.int8)
    action = np.asarray(decisions)
    reject = np.asarray(rejected, dtype=bool)
    if target.shape != action.shape or target.shape != reject.shape or target.ndim != 1:
        raise M9Error("guarded policy vectors must have equal shapes")
    normal, anomaly = target == 0, target == 1
    automatic = (~reject) & (action != "review")
    errors = ((action == "anomalous") & normal) | ((action == "normal") & anomaly)
    false_alert = normal & (action != "normal")
    result: dict[str, object] = {
        "quality_rejection_rate": float(np.mean(reject)),
        "quality_normal_rejection_rate": float(np.mean(reject[normal])),
        "quality_anomaly_rejection_rate": float(np.mean(reject[anomaly])),
        "guarded_normal_fpr": float(np.mean(((~reject) & (action == "anomalous"))[normal])),
        "guarded_auto_coverage": float(np.mean(automatic)),
        "guarded_selective_risk": float(np.mean(errors[automatic])) if automatic.any() else None,
        "guarded_auto_anomaly_recall": float(
            np.mean(((~reject) & (action == "anomalous"))[anomaly])
        ),
        "guarded_anomaly_actionable_rate": float(np.mean((reject | (action != "normal"))[anomaly])),
        "false_alerts": int(np.sum(false_alert)),
        "false_alerts_intercepted": int(np.sum(reject & false_alert)),
        "false_alert_interception_rate": float(np.mean(reject[false_alert]))
        if false_alert.any()
        else None,
    }
    for label, selected in (("normal", normal), ("anomaly", anomaly)):
        low, high = exact_binomial_interval(int(np.sum(reject[selected])), int(np.sum(selected)))
        result[f"quality_{label}_rejection_ci_low"] = low
        result[f"quality_{label}_rejection_ci_high"] = high
    return result


def _memory(config: Mapping[str, Any], *, patches: int) -> ProjectedPatchMemory:
    return ProjectedPatchMemory(
        seed=int(config["seed"]),
        bank_size=int(config["bank_size"]),
        projected_dimension=int(config["projected_dimension"]),
        candidate_limit=int(config["candidate_limit"]),
        patches_per_image=patches,
        image_score_fraction=float(config["image_score_fraction"]),
        selection="kcenter",
    )


def _maps(
    raw_maps: Sequence[np.ndarray], input_size: int, output_size: int, sigma: float
) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for raw in raw_maps:
        interpolated = cv2.resize(raw, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
        smooth = cv2.GaussianBlur(interpolated, (0, 0), sigmaX=sigma)
        result.append(cv2.resize(smooth, (output_size, output_size), interpolation=cv2.INTER_AREA))
    return result


def load_m9_config(path: Path) -> dict[str, Any]:
    config = read_config(path)
    model = config["model"]
    calibration = config["calibration"]
    if model["geometry"] != "full-frame-bilinear-resize":
        raise M9Error("M9 requires full-frame geometry")
    if model["patch_budget_per_image"] != 24 or model["bank_size"] != 512:
        raise M9Error("M9 must retain the frozen 24-patch/512-entry budget")
    if not 0 < calibration["anomaly_alpha"] <= calibration["review_alpha"] < 1:
        raise M9Error("invalid anomaly/review calibration levels")
    if not 0 < calibration["quality_alpha"] < 1:
        raise M9Error("invalid quality calibration level")
    if config["quick"]["calibration_normals"] < np.ceil(1 / calibration["quality_alpha"] - 1):
        raise M9Error("quick calibration cannot resolve the quality alpha")
    return config


def run_m9(
    *,
    project_root: Path,
    config_path: Path,
    data_config: Path,
    device: str,
    quick: bool,
    resume: Path | None,
) -> dict[str, object]:
    """Fit local normal memories/calibration and evaluate clean plus the frozen M4 shifts."""

    load_dotenv(project_root / ".env")
    config = load_m9_config(config_path)
    data_root = Path(os.environ["ANOMALY_DATA_ROOT"])
    artifacts = Path(os.environ["ANOMALY_ARTIFACT_ROOT"]) / "runs"
    records, source_root = load_ksdd2_manifest(data_config, data_root)
    data_settings = read_config(data_config)
    manifest_dir = data_root / "manifests" / data_settings["dataset"]["id"]
    robustness_path = project_root / config["source_protocols"]["robustness_config"]
    quality_path = project_root / config["source_protocols"]["quality_config"]
    frozen = load_m4_settings(robustness_path)
    gate_protocol = load_m8_settings(quality_path)
    conditions = (ShiftSpec("clean", "clean", 0, {}), *frozen.conditions)
    seed = int(config["experiment"]["seed"])
    model_cfg = {**config["model"], "seed": seed}
    if (
        seed != frozen.seed
        or model_cfg["image_size"] != frozen.image_size
        or model_cfg["projected_dimension"] != frozen.intervention.projected_dimension
        or model_cfg["candidate_limit"] != frozen.intervention.candidate_limit
        or model_cfg["image_score_fraction"] != frozen.image_score_fraction
        or model_cfg["gaussian_sigma"] != frozen.gaussian_sigma
        or config["calibration"]["quality_alpha"] != gate_protocol.alpha
    ):
        raise M9Error("external method differs from the frozen VisA model/gate recipe")
    train = [row for row in records if row["split"] == "train"]
    validation = [row for row in records if row["split"] == "validation"]
    test = [row for row in records if row["split"] == "test"]
    if any(row["label"] != "normal" for row in [*train, *validation]):
        raise M9Error("fitting/calibration must contain only normals")
    if quick:
        limits = config["quick"]
        train = train[: limits["train_normals"]]
        validation = validation[: limits["calibration_normals"]]
        test = [row for row in test if row["label"] == "normal"][: limits["test_normals"]] + [
            row for row in test if row["label"] == "anomaly"
        ][: limits["test_anomalies"]]
    if len(validation) < np.ceil(1 / config["calibration"]["quality_alpha"] - 1):
        raise M9Error("not enough normal calibration samples")
    mode = "smoke" if quick else "full"
    resolved = {**config, "mode": mode, "device": device}
    hashes = {
        "config_sha256": sha256_file(config_path),
        "data_config_sha256": sha256_file(data_config),
        "manifest_sha256": sha256_file(manifest_dir / "manifest.jsonl"),
        "audit_sha256": sha256_file(manifest_dir / "audit.json"),
        "robustness_config_sha256": sha256_file(robustness_path),
        "quality_config_sha256": sha256_file(quality_path),
        "code_sha256": hashlib.sha256(
            "".join(
                f"{path.relative_to(project_root).as_posix()}:{sha256_file(path)}\n"
                for path in sorted((project_root / "src").rglob("*.py"))
            ).encode()
        ).hexdigest(),
    }
    artifacts.mkdir(parents=True, exist_ok=True)
    if resume is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_dir = artifacts / f"{config['experiment']['name']}-{mode}-{timestamp}"
        run_dir.mkdir(exist_ok=False)
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
        run_dir.relative_to(artifacts.resolve(strict=True))
        metadata = json.loads((run_dir / "metadata.json").read_text("utf-8"))
        if metadata.get("status") != "running" or any(
            metadata.get(key) != value for key, value in hashes.items()
        ):
            raise M9Error("resume inputs changed or run is not incomplete")
        if json.loads((run_dir / "resolved_config.json").read_text("utf-8")) != resolved:
            raise M9Error("resume settings changed")
    size = int(model_cfg["image_size"])
    grid = int(model_cfg["localization_size"])
    batch_size = int(model_cfg["batch_size"])
    extractor = FrozenPyramidResNet18(device=device, image_size=size, batch_size=batch_size)
    checkpoint = (
        Path(os.environ["TORCH_HOME"])
        / "hub"
        / "checkpoints"
        / Path(urlparse(extractor.weights_url).path).name
    )
    checkpoint_hash = sha256_file(checkpoint)
    if "checkpoint_sha256" in metadata and metadata["checkpoint_sha256"] != checkpoint_hash:
        raise M9Error("model checkpoint changed")
    metadata["checkpoint_sha256"] = checkpoint_hash
    metadata["weights_id"] = extractor.weights_id
    _atomic_json(run_dir / "metadata.json", metadata)

    def pair(row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        return full_frame_pair(
            source_root / row["image_path"], source_root / row["mask_path"], size
        )

    training_checkpoint = run_dir / "training_checkpoint.json"
    memories: dict[str, ProjectedPatchMemory] = {}
    calibration_scores: dict[str, list[float]] = {}
    parameters: dict[str, dict[str, float]] = {}
    if not training_checkpoint.is_file():
        start = time.perf_counter()
        patch_cache: dict[str, list[np.ndarray]] = {name: [] for name in MODELS}
        half_budget = int(model_cfg["patch_budget_per_image"]) // 2
        for offset in range(0, len(train), batch_size):
            rows = train[offset : offset + batch_size]
            clean_images = [pair(row)[0] for row in rows]
            augmented_images = []
            for row, image in zip(rows, clean_images, strict=True):
                index = int(hashlib.sha256(row["sample_id"].encode()).hexdigest()[:16], 16) % len(
                    frozen.intervention.augmentations
                )
                augmented_images.append(
                    apply_shift(
                        image,
                        np.zeros((size, size), bool),
                        frozen.intervention.augmentations[index],
                        seed=seed,
                        sample_key=row["sample_id"],
                    ).image
                )
            pyramids = extractor.extract_rgb([*clean_images, *augmented_images])
            for position, row in enumerate(rows):
                clean_map = compose_feature_map(pyramids[position], "multiscale").reshape(-1, 448)
                aug_map = compose_feature_map(pyramids[position + len(rows)], "multiscale").reshape(
                    -1, 448
                )
                rng = np.random.default_rng(seed + int(row["sample_id"][:8], 16))
                indexes = rng.choice(len(clean_map), size=2 * half_budget, replace=False)
                patch_cache[MODELS[0]].append(clean_map[indexes].reshape(1, 2 * half_budget, 448))
                patch_cache[MODELS[1]].append(
                    clean_map[indexes[:half_budget]].reshape(1, half_budget, 448)
                )
                patch_cache[MODELS[1]].append(
                    aug_map[rng.choice(len(aug_map), half_budget, replace=False)].reshape(
                        1, half_budget, 448
                    )
                )
            if offset % (batch_size * 20) == 0:
                print(
                    f"Training normal features: "
                    f"{min(offset + batch_size, len(train))}/{len(train)}",
                    flush=True,
                )
        for name in MODELS:
            memory = _memory(
                model_cfg, patches=2 * half_budget if name == MODELS[0] else half_budget
            )
            memory.fit(patch_cache[name])
            memory.save(
                run_dir / "indexes" / f"{name}.npz",
                metadata={
                    "dataset": data_settings["dataset"]["id"],
                    "geometry": model_cfg["geometry"],
                },
            )
            memories[name] = memory
        del patch_cache
        validation_images = [pair(row)[0] for row in validation]
        validation_features = [
            compose_feature_map(pyramid, "multiscale")
            for pyramid in extractor.extract_rgb(validation_images)
        ]
        for name, memory in memories.items():
            scores, raw = memory.score(validation_features)
            maps = _maps(raw, size, grid, model_cfg["gaussian_sigma"])
            calibration_scores[name] = scores.tolist()
            parameters[name] = {
                "image_threshold": calibrate_threshold(scores, 0.95),
                "pixel_threshold": calibrate_threshold(
                    np.concatenate([item.ravel() for item in maps]),
                    config["calibration"]["pixel_quantile"],
                ),
            }
        quality = CaptureQualityModel(
            schema_version=1,
            image_size=size,
            alpha=config["calibration"]["quality_alpha"],
            manifest_sha256=hashes["manifest_sha256"],
            categories={
                "surface": fit_category_quality_model(
                    [capture_metrics(pair(row)[0]) for row in train],
                    [capture_metrics(image) for image in validation_images],
                )
            },
        )
        _atomic_json(run_dir / "quality_model.json", quality_model_dict(quality))
        _atomic_json(
            run_dir / "calibration.json", {"scores": calibration_scores, "parameters": parameters}
        )
        runtime_files = [
            "quality_model.json",
            "calibration.json",
            *[f"indexes/{name}.npz" for name in MODELS],
        ]
        _atomic_json(
            training_checkpoint,
            {
                "training_seconds": time.perf_counter() - start,
                "files": {filename: sha256_file(run_dir / filename) for filename in runtime_files},
            },
        )
    else:
        checkpoint_data = json.loads(training_checkpoint.read_text("utf-8"))
        for filename, expected_hash in checkpoint_data["files"].items():
            if sha256_file(run_dir / filename) != expected_hash:
                raise M9Error(f"training checkpoint changed: {filename}")
        for name in MODELS:
            bank, projection = load_projected_index(run_dir / "indexes" / f"{name}.npz")
            memory = _memory(model_cfg, patches=1)
            memory.fit_projected(bank, projection)
            memories[name] = memory
        calibration = json.loads((run_dir / "calibration.json").read_text("utf-8"))
        calibration_scores, parameters = calibration["scores"], calibration["parameters"]
        quality = load_quality_model(run_dir / "quality_model.json")

    labels = [int(row["label"] == "anomaly") for row in test]
    completed: list[str] = []
    for condition in conditions:
        condition_dir = run_dir / "conditions" / condition.identifier
        commit = condition_dir / "checkpoint.json"
        if commit.is_file():
            saved = json.loads(commit.read_text("utf-8"))
            for filename, expected_hash in saved.items():
                if sha256_file(condition_dir / filename) != expected_hash:
                    raise M9Error(
                        f"condition checkpoint changed: {condition.identifier}/{filename}"
                    )
            completed.append(condition.identifier)
            continue
        condition_dir.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        predictions: list[dict[str, object]] = []
        all_scores: dict[str, list[float]] = {name: [] for name in MODELS}
        all_maps: dict[str, list[np.ndarray]] = {name: [] for name in MODELS}
        all_masks: list[np.ndarray] = []
        rejected: list[bool] = []
        for offset in range(0, len(test), batch_size):
            rows = test[offset : offset + batch_size]
            pairs = [
                apply_shift(*pair(row), condition, seed=seed, sample_key=row["sample_id"])
                for row in rows
            ]
            images = [item.image for item in pairs]
            masks = [
                cv2.resize(
                    item.mask.astype(np.uint8), (grid, grid), interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                for item in pairs
            ]
            all_masks.extend(masks)
            qualities = [
                evaluate_capture_quality(image, category="surface", model=quality)
                for image in images
            ]
            rejected.extend(item.status == "recapture" for item in qualities)
            features = [
                compose_feature_map(pyramid, "multiscale")
                for pyramid in extractor.extract_rgb(images)
            ]
            for name, memory in memories.items():
                scores, raw = memory.score(features)
                maps = _maps(raw, size, grid, model_cfg["gaussian_sigma"])
                all_scores[name].extend(scores.tolist())
                all_maps[name].extend(maps)
                p_values = conformal_p_values(calibration_scores[name], scores)
                decisions = selective_decisions(
                    p_values,
                    anomaly_alpha=config["calibration"]["anomaly_alpha"],
                    review_alpha=config["calibration"]["review_alpha"],
                )
                for row, score, p_value, decision, capture in zip(
                    rows, scores, p_values, decisions, qualities, strict=True
                ):
                    predictions.append(
                        {
                            "model": name,
                            "condition": condition.identifier,
                            "sample_id": row["sample_id"],
                            "label": row["label"],
                            "score": float(score),
                            "conformal_p": float(p_value),
                            "decision": str(decision),
                            "quality_p": capture.p_value,
                            "quality_status": capture.status,
                            "guarded_decision": "recapture"
                            if capture.status == "recapture"
                            else str(decision),
                        }
                    )
        metrics: list[dict[str, object]] = []
        bootstrap_count = (
            config["quick"]["bootstrap_samples"]
            if quick
            else config["experiment"]["bootstrap_samples"]
        )
        for name in MODELS:
            p_values = conformal_p_values(calibration_scores[name], all_scores[name])
            decisions = selective_decisions(
                p_values,
                anomaly_alpha=config["calibration"]["anomaly_alpha"],
                review_alpha=config["calibration"]["review_alpha"],
            )
            result: dict[str, object] = {
                "model": name,
                "condition": condition.identifier,
                "family": condition.family,
                "severity": condition.severity,
                "evaluation_grid": grid,
                "anomaly_masks_empty_at_evaluation_grid": sum(
                    label == 1 and not mask.any()
                    for label, mask in zip(labels, all_masks, strict=True)
                ),
                **image_metrics(labels, all_scores[name], parameters[name]["image_threshold"]),
                **selective_metrics(labels, decisions),
                **guarded_metrics(labels, decisions, rejected),
                **pixel_metrics(all_masks, all_maps[name], parameters[name]["pixel_threshold"]),
                "aupro_30": pro_auc(all_masks, all_maps[name], max_fpr=0.30, thresholds=30),
            }
            if condition.family == "clean":
                intervals = bootstrap_image_metrics(
                    labels,
                    all_scores[name],
                    seed=seed,
                    samples=bootstrap_count,
                    confidence=config["experiment"]["confidence"],
                )
                for metric, interval in intervals.items():
                    result[f"{metric}_ci_low"] = interval["low"]
                    result[f"{metric}_ci_high"] = interval["high"]
            result["condition_seconds"] = time.perf_counter() - start
            metrics.append(result)
        _atomic_csv(condition_dir / "predictions.csv", predictions)
        _atomic_csv(condition_dir / "metrics.csv", metrics)
        _atomic_json(
            commit,
            {
                filename: sha256_file(condition_dir / filename)
                for filename in ("predictions.csv", "metrics.csv")
            },
        )
        completed.append(condition.identifier)
        _atomic_json(
            run_dir / "progress.json",
            {
                "completed_conditions": completed,
                "total_conditions": len(conditions),
                "resume_command": (
                    "python scripts/run_m9_external_validation.py "
                    f'{"--quick " if quick else ""}--resume "{run_dir}"'
                ),
            },
        )
        print(
            f"[{len(completed)}/{len(conditions)}] {condition.identifier}: evaluated both models",
            flush=True,
        )
    metrics_rows = [
        row
        for condition in conditions
        for row in _read_csv(run_dir / "conditions" / condition.identifier / "metrics.csv")
    ]
    prediction_rows = [
        row
        for condition in conditions
        for row in _read_csv(run_dir / "conditions" / condition.identifier / "predictions.csv")
    ]
    _atomic_csv(run_dir / "metrics.csv", metrics_rows)
    _atomic_csv(run_dir / "predictions.csv", prediction_rows)
    summary: dict[str, object] = {
        "run_dir": str(run_dir),
        "mode": mode,
        "dataset": data_settings["dataset"],
        "normal_fit_samples": len(train),
        "normal_calibration_samples": len(validation),
        "official_test_samples": len(test),
        "conditions": len(conditions),
        "models": list(MODELS),
        "preprocessing": config["model"]["geometry"],
        "localization_grid": grid,
        "clean_results": [row for row in metrics_rows if row["condition"] == "clean"],
        "interpretation": (
            "fixed method with dataset-local normal adaptation; not zero-shot VisA transfer"
        ),
        "source_images_published": False,
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

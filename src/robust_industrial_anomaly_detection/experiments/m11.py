"""Matched-budget frozen encoder evaluation with integrity-gated resumable units."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from dotenv import load_dotenv
from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore[import-untyped]

from robust_industrial_anomaly_detection.baselines.encoders import (
    ENCODERS,
    SpatialEncoder,
    build_encoder,
    verify_checkpoint,
)
from robust_industrial_anomaly_detection.baselines.multiscale import (
    ProjectedPatchMemory,
    compose_feature_map,
    load_fitted_rgb,
    load_projected_index,
)
from robust_industrial_anomaly_detection.data.ksdd2 import load_ksdd2_manifest, read_config
from robust_industrial_anomaly_detection.data.visa import load_trusted_manifest, sha256_file
from robust_industrial_anomaly_detection.evaluation.calibration import (
    conformal_p_values,
    selective_decisions,
    selective_metrics,
)
from robust_industrial_anomaly_detection.evaluation.metrics import (
    calibrate_threshold,
    image_metrics,
    json_ready,
    pixel_metrics,
    pro_auc,
)
from robust_industrial_anomaly_detection.evaluation.robustness import spearman_stability
from robust_industrial_anomaly_detection.experiments.m2 import (
    Sample,
    _atomic_csv,
    _atomic_json,
    _environment,
    _git_metadata,
    _load_masks,
    _read_csv,
    _samples,
)
from robust_industrial_anomaly_detection.experiments.m4 import load_m4_settings
from robust_industrial_anomaly_detection.experiments.m9 import _maps, full_frame_pair
from robust_industrial_anomaly_detection.experiments.m10 import _metrics, mask_measurements
from robust_industrial_anomaly_detection.prototype.benchmark import _working_set_bytes
from robust_industrial_anomaly_detection.robustness.transforms import ShiftSpec, apply_shift


class M11Error(ValueError):
    """The representation study violates its data or reproducibility contract."""


@dataclass(frozen=True)
class EncoderUnit:
    dataset: str
    category: str
    samples: tuple[Sample, ...]


def load_config(path: Path) -> dict[str, Any]:
    config = read_config(path)
    model, calibration = config["model"], config["calibration"]
    if config["encoders"] != list(ENCODERS) or config["datasets"] != ["visa", "ksdd2"]:
        raise M11Error("M11 requires the declared two-encoder/two-dataset comparison")
    if (
        model["image_size"] != 224
        or model["localization_size"] != 56
        or model["bank_size"] != 512
        or model["patch_budget_per_image"] != 24
        or model["projected_dimension"] != 64
        or model["candidate_limit"] != 4096
        or model["fit_normals_cap"] < 2
        or model["batch_size"] < 1
        or model["torch_threads"] < 1
        or model["gaussian_sigma"] != 4.0
        or model["image_score_fraction"] != 0.01
    ):
        raise M11Error("M11 input/memory budget differs from the fixed protocol")
    if not 0 < calibration["anomaly_alpha"] <= calibration["review_alpha"] < 1:
        raise M11Error("invalid normal-only decision levels")
    if not 0 < calibration["pixel_quantile"] < 1:
        raise M11Error("invalid pixel calibration quantile")
    if (
        len(config["conditions"]) != len(set(config["conditions"]))
        or config["conditions"][0] != "clean"
    ):
        raise M11Error("conditions must be unique with clean first")
    if config["quick"]["calibration_normals"] < np.ceil(1 / calibration["anomaly_alpha"] - 1):
        raise M11Error("quick calibration cannot resolve the anomaly level")
    if (
        config["experiment"]["bootstrap_samples"] < 1
        or not 0 < config["experiment"]["confidence"] < 1
    ):
        raise M11Error("invalid uncertainty settings")
    return config


def partition(
    samples: Sequence[Sample],
    config: Mapping[str, Any],
    quick: bool,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    seed = config["experiment"]["seed"]
    grouped = {
        split: sorted(
            (sample for sample in samples if sample.split == split),
            key=lambda sample: hashlib.sha256(f"{seed}:{sample.sample_id}".encode()).hexdigest(),
        )
        for split in ("train", "validation", "test")
    }
    if any(sample.label != "normal" for sample in grouped["train"] + grouped["validation"]):
        raise M11Error("fitting and calibration must contain only normals")
    ids = [sample.sample_id for split in grouped.values() for sample in split]
    if len(ids) != len(set(ids)):
        raise M11Error("sample IDs overlap or repeat across partitions")
    train = grouped["train"][: config["model"]["fit_normals_cap"]]
    validation, test = grouped["validation"], grouped["test"]
    if quick:
        limits = config["quick"]
        train = train[: limits["train_normals"]]
        validation = validation[: limits["calibration_normals"]]
        test = [s for s in test if s.label == "normal"][: limits["test_normals"]] + [
            s for s in test if s.label == "anomaly"
        ][: limits["test_anomalies"]]
    if len(train) < 2 or len(validation) < np.ceil(1 / config["calibration"]["anomaly_alpha"] - 1):
        raise M11Error("insufficient normal training/calibration images")
    if {sample.label for sample in test} != {"normal", "anomaly"}:
        raise M11Error("test partition requires both labels")
    return train, validation, test


def sample_pair(sample: Sample, dataset: str, size: int) -> tuple[np.ndarray, np.ndarray]:
    if dataset == "visa":
        return load_fitted_rgb(sample.image_path, size), _load_masks([sample], (size, size))[0]
    if dataset != "ksdd2" or sample.mask_path is None:
        raise M11Error("unknown image geometry or missing full-frame mask")
    return full_frame_pair(sample.image_path, sample.mask_path, size)


def verify_commit(root: Path, commit: Path) -> bool:
    if not commit.is_file():
        return False
    payload = json.loads(commit.read_text("utf-8"))
    files = payload.get("files")
    if not isinstance(files, dict) or not files:
        raise M11Error("empty or invalid checkpoint")
    for filename, expected in files.items():
        path = (root / filename).resolve(strict=True)
        path.relative_to(root.resolve(strict=True))
        if sha256_file(path) != expected:
            raise M11Error(f"checkpoint artifact changed: {filename}")
    return True


def _commit(root: Path, filename: str, files: Sequence[str]) -> None:
    _atomic_json(root / filename, {"files": {name: sha256_file(root / name) for name in files}})


def _memory(config: Mapping[str, Any]) -> ProjectedPatchMemory:
    model = config["model"]
    return ProjectedPatchMemory(
        seed=config["experiment"]["seed"],
        bank_size=model["bank_size"],
        projected_dimension=model["projected_dimension"],
        patches_per_image=12,
        candidate_limit=model["candidate_limit"],
        image_score_fraction=model["image_score_fraction"],
        selection="kcenter",
    )


def _features(extractor: SpatialEncoder, images: Sequence[np.ndarray]) -> list[np.ndarray]:
    return [compose_feature_map(pyramid, "multiscale") for pyramid in extractor.extract_rgb(images)]


def _fit(
    unit: EncoderUnit,
    extractor: SpatialEncoder,
    name: str,
    config: Mapping[str, Any],
    train: Sequence[Sample],
    validation: Sequence[Sample],
    augmentations: Sequence[ShiftSpec],
    directory: Path,
) -> tuple[ProjectedPatchMemory, dict[str, Any]]:
    directory.mkdir(parents=True, exist_ok=True)
    memory = _memory(config)
    if verify_commit(directory, directory / "fit_checkpoint.json"):
        bank, projection = load_projected_index(directory / "index.npz")
        memory.fit_projected(bank, projection)
        return memory, json.loads((directory / "calibration.json").read_text("utf-8"))
    start = time.perf_counter()
    seed, model = config["experiment"]["seed"], config["model"]
    size, batch = model["image_size"], model["batch_size"]
    cache: list[np.ndarray] = []
    for offset in range(0, len(train), batch):
        rows = train[offset : offset + batch]
        images = [sample_pair(row, unit.dataset, size)[0] for row in rows]
        augmented = []
        for sample, image in zip(rows, images, strict=True):
            index = int(hashlib.sha256(sample.sample_id.encode()).hexdigest()[:16], 16) % len(
                augmentations
            )
            augmented.append(
                apply_shift(
                    image,
                    np.zeros((size, size), bool),
                    augmentations[index],
                    seed=seed,
                    sample_key=sample.sample_id,
                ).image
            )
        features = _features(extractor, [*images, *augmented])
        for position, sample in enumerate(rows):
            clean = features[position].reshape(-1, features[position].shape[-1])
            aug = features[position + len(rows)].reshape(-1, clean.shape[-1])
            rng = np.random.default_rng(seed + int(sample.sample_id[:8], 16))
            indexes = rng.choice(len(clean), 24, replace=False)
            cache.extend(
                (
                    clean[indexes[:12]].reshape(1, 12, -1),
                    aug[rng.choice(len(aug), 12, replace=False)].reshape(1, 12, -1),
                )
            )
        if offset % (batch * 4) == 0:
            print(
                f"{unit.dataset}/{unit.category}/{name}: fit "
                f"{min(offset + batch, len(train))}/{len(train)}",
                flush=True,
            )
    stats = memory.fit(cache)
    del cache
    scores: list[float] = []
    maps: list[np.ndarray] = []
    for offset in range(0, len(validation), batch):
        images = [
            sample_pair(row, unit.dataset, size)[0] for row in validation[offset : offset + batch]
        ]
        values, raw = memory.score(_features(extractor, images))
        scores.extend(values.tolist())
        maps.extend(_maps(raw, size, model["localization_size"], model["gaussian_sigma"]))
    calibration = {
        "scores": scores,
        "image_threshold": calibrate_threshold(scores, 0.95),
        "pixel_threshold": calibrate_threshold(
            np.concatenate([value.ravel() for value in maps]),
            config["calibration"]["pixel_quantile"],
        ),
    }
    memory.save(
        directory / "index.npz",
        metadata={"encoder": name, "dataset": unit.dataset, "category": unit.category},
    )
    _atomic_json(directory / "calibration.json", calibration)
    _atomic_json(
        directory / "fit.json",
        {
            "fit_ids": [sample.sample_id for sample in train],
            "calibration_ids": [sample.sample_id for sample in validation],
            "training_seconds": time.perf_counter() - start,
            "candidate_count": stats.candidate_count,
            "bank_entries": stats.selected_count,
            "bank_bytes": stats.bank_bytes,
            "projection_bytes": stats.projection_bytes,
            "feature_dimension": stats.feature_dimension,
            "weights_sha256": ENCODERS[name].sha256,
        },
    )
    _commit(directory, "fit_checkpoint.json", ["index.npz", "calibration.json", "fit.json"])
    return memory, calibration


def paired_comparison(
    baseline: Sequence[Mapping[str, Any]],
    modern: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    samples: int,
    confidence: float,
) -> dict[str, object]:
    if [row["sample_id"] for row in baseline] != [row["sample_id"] for row in modern]:
        raise M11Error("paired encoder predictions have mismatched sample order")
    if [row["label"] for row in baseline] != [row["label"] for row in modern]:
        raise M11Error("paired encoder predictions have mismatched labels")
    labels = np.asarray([row["label"] == "anomaly" for row in baseline], np.int8)
    negative, positive = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
    if not len(negative) or not len(positive) or samples < 1 or not 0 < confidence < 1:
        raise M11Error("paired bootstrap requires both labels and valid uncertainty settings")
    scores = [np.asarray([row["score"] for row in rows], float) for rows in (baseline, modern)]
    decisions = [np.asarray([row["decision"] for row in rows]) for rows in (baseline, modern)]
    ious = [
        np.asarray(
            [
                np.nan if row.get("foreground_iou") in (None, "") else float(row["foreground_iou"])
                for row in rows
            ]
        )
        for rows in (baseline, modern)
    ]

    def delta(indexes: np.ndarray) -> dict[str, float]:
        y = labels[indexes]
        normal, anomaly = y == 0, y == 1
        values = []
        for score, decision, iou in zip(scores, decisions, ious, strict=True):
            selected = decision[indexes]
            visible = iou[indexes][np.isfinite(iou[indexes])]
            values.append(
                {
                    "image_auroc": float(roc_auc_score(y, score[indexes])),
                    "image_ap": float(average_precision_score(y, score[indexes])),
                    "normal_fpr": float(np.mean(selected[normal] == "anomalous")),
                    "anomaly_confident_recall": float(np.mean(selected[anomaly] == "anomalous")),
                    "foreground_iou": float(np.mean(visible)) if len(visible) else float("nan"),
                }
            )
        return {key: values[1][key] - values[0][key] for key in values[0]}

    point = delta(np.arange(len(labels)))
    rng = np.random.default_rng(seed)
    estimates = [
        delta(
            np.concatenate(
                (rng.choice(negative, len(negative)), rng.choice(positive, len(positive)))
            )
        )
        for _ in range(samples)
    ]
    tail = (1 - confidence) / 2
    result: dict[str, object] = {"paired_samples": len(labels)}
    for metric, value in point.items():
        result[f"{metric}_delta"] = value
        result[f"{metric}_delta_ci_low"] = float(
            np.quantile([row[metric] for row in estimates], tail)
        )
        result[f"{metric}_delta_ci_high"] = float(
            np.quantile([row[metric] for row in estimates], 1 - tail)
        )
    return result


def _evaluate(
    unit: EncoderUnit,
    name: str,
    extractor: SpatialEncoder,
    memory: ProjectedPatchMemory,
    calibration: Mapping[str, Any],
    config: Mapping[str, Any],
    test: Sequence[Sample],
    condition: ShiftSpec,
    directory: Path,
    clean_scores: Sequence[float] | None,
) -> None:
    model, levels = config["model"], config["calibration"]
    size, grid, batch = model["image_size"], model["localization_size"], model["batch_size"]
    predictions: list[dict[str, Any]] = []
    maps: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    feature_seconds = memory_seconds = pipeline_seconds = 0.0
    for offset in range(0, len(test), batch):
        start = time.perf_counter()
        rows = test[offset : offset + batch]
        pairs = [
            apply_shift(
                *sample_pair(row, unit.dataset, size),
                condition,
                seed=config["experiment"]["seed"],
                sample_key=row.sample_id,
            )
            for row in rows
        ]
        targets = [
            cv2.resize(
                pair.mask.astype(np.uint8), (grid, grid), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            for pair in pairs
        ]
        feature_start = time.perf_counter()
        features = _features(extractor, [pair.image for pair in pairs])
        feature_seconds += time.perf_counter() - feature_start
        score_start = time.perf_counter()
        scores, raw = memory.score(features)
        memory_seconds += time.perf_counter() - score_start
        evidence = _maps(raw, size, grid, model["gaussian_sigma"])
        p_values = conformal_p_values(calibration["scores"], scores)
        decisions = selective_decisions(
            p_values, anomaly_alpha=levels["anomaly_alpha"], review_alpha=levels["review_alpha"]
        )
        for row, mask, score_map, score, p_value, decision in zip(
            rows, targets, evidence, scores, p_values, decisions, strict=True
        ):
            predictions.append(
                {
                    "sample_id": row.sample_id,
                    "label": row.label,
                    "score": float(score),
                    "conformal_p": float(p_value),
                    "decision": str(decision),
                    **mask_measurements(
                        mask, score_map > calibration["pixel_threshold"], row.label
                    ),
                }
            )
        masks.extend(targets)
        maps.extend(evidence)
        pipeline_seconds += time.perf_counter() - start
    labels = [int(row["label"] == "anomaly") for row in predictions]
    values = [row["score"] for row in predictions]
    metrics = {
        "dataset": unit.dataset,
        "category": unit.category,
        "encoder": name,
        "condition": condition.identifier,
        "evaluation_grid": grid,
        "pixel_threshold": calibration["pixel_threshold"],
        **image_metrics(labels, values, calibration["image_threshold"]),
        **selective_metrics(labels, [row["decision"] for row in predictions]),
        **pixel_metrics(masks, maps, calibration["pixel_threshold"]),
        **_metrics(predictions),
        "aupro_30": pro_auc(masks, maps, max_fpr=0.3, thresholds=30),
        "mean_batch_feature_ms": feature_seconds * 1000 / len(test),
        "mean_batch_memory_ms": memory_seconds * 1000 / len(test),
        "mean_batch_pipeline_ms": pipeline_seconds * 1000 / len(test),
        "clean_score_rank_correlation": spearman_stability(clean_scores, values)
        if clean_scores is not None
        else 1.0,
    }
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_csv(directory / "predictions.csv", predictions)
    _atomic_csv(directory / "metrics.csv", [metrics])
    _commit(directory, "checkpoint.json", ["predictions.csv", "metrics.csv"])


def _benchmark(
    unit: EncoderUnit,
    name: str,
    extractor: SpatialEncoder,
    memory: ProjectedPatchMemory,
    test: Sequence[Sample],
    config: Mapping[str, Any],
    directory: Path,
) -> None:
    if verify_commit(directory, directory / "benchmark_checkpoint.json"):
        return
    selected = [s for s in test if s.label == "normal"][: config["benchmark"]["normal_samples"]] + [
        s for s in test if s.label == "anomaly"
    ][: config["benchmark"]["anomaly_samples"]]
    model = config["model"]
    # Each call contains one image, regardless of the experiment's batch size.
    warm = sample_pair(selected[0], unit.dataset, model["image_size"])[0]
    memory.score(_features(extractor, [warm]))
    timings = []
    for sample in selected:
        start = time.perf_counter()
        image, _ = sample_pair(sample, unit.dataset, model["image_size"])
        feature_start = time.perf_counter()
        features = _features(extractor, [image])
        feature_ms = (time.perf_counter() - feature_start) * 1000
        score_start = time.perf_counter()
        _, raw = memory.score(features)
        memory_ms = (time.perf_counter() - score_start) * 1000
        _maps(raw, model["image_size"], model["localization_size"], model["gaussian_sigma"])
        timings.append(
            {
                "sample_id": sample.sample_id,
                "label": sample.label,
                "feature_ms": feature_ms,
                "memory_ms": memory_ms,
                "total_ms": (time.perf_counter() - start) * 1000,
            }
        )
    _atomic_csv(directory / "benchmark_timings.csv", timings)
    _atomic_json(
        directory / "benchmark.json",
        {
            "encoder": name,
            "samples": len(timings),
            "median_ms": float(np.median([row["total_ms"] for row in timings])),
            "p95_ms": float(np.quantile([row["total_ms"] for row in timings], 0.95)),
            "mean_feature_ms": float(np.mean([row["feature_ms"] for row in timings])),
            "active_parameters": sum(
                parameter.numel() for parameter in extractor.model.parameters()
            ),
            "checkpoint_bytes": verify_checkpoint(name).stat().st_size,
            "index_bytes": (directory / "index.npz").stat().st_size,
            "working_set_bytes": _working_set_bytes(),
        },
    )
    _commit(directory, "benchmark_checkpoint.json", ["benchmark_timings.csv", "benchmark.json"])


def _units(
    project: Path, data: Path, config: Mapping[str, Any], datasets: Sequence[str]
) -> tuple[list[EncoderUnit], dict[str, str]]:
    units: list[EncoderUnit] = []
    hashes: dict[str, str] = {}
    for dataset in datasets:
        data_config = project / config["sources"][f"{dataset}_config"]
        if dataset == "visa":
            dataset_id = "visa-20220922"
            root = data / "sources" / dataset_id
            manifest = data / "manifests" / dataset_id
            records = load_trusted_manifest(
                manifest / "manifest.jsonl", manifest / "audit.json", data_config, root
            )
        else:
            dataset_id = read_config(data_config)["dataset"]["id"]
            ksdd_records, root = load_ksdd2_manifest(data_config, data)
            records = tuple(ksdd_records)
            manifest = data / "manifests" / dataset_id
        samples = _samples(records, root)
        for category in sorted({sample.category for sample in samples}):
            units.append(
                EncoderUnit(dataset, category, tuple(s for s in samples if s.category == category))
            )
        for key, path in (
            ("config", data_config),
            ("manifest", manifest / "manifest.jsonl"),
            ("audit", manifest / "audit.json"),
        ):
            hashes[f"{dataset}_{key}_sha256"] = sha256_file(path)
    return units, hashes


def run_m11(
    *,
    project_root: Path,
    config_path: Path,
    device: str,
    quick: bool,
    resume: Path | None,
    datasets: Sequence[str] | None = None,
    categories: Sequence[str] | None = None,
    max_new_conditions: int | None = None,
) -> dict[str, object]:
    load_dotenv(project_root / ".env")
    config = load_config(config_path)
    torch.set_num_threads(config["model"]["torch_threads"])
    cv2.setNumThreads(1)
    seed = config["experiment"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    selected_datasets = sorted(set(datasets or config["datasets"]))
    if not set(selected_datasets).issubset(config["datasets"]):
        raise M11Error("unknown dataset selection")
    units, hashes = _units(
        project_root, Path(os.environ["ANOMALY_DATA_ROOT"]), config, selected_datasets
    )
    if categories:
        if not set(categories).issubset({unit.category for unit in units}):
            raise M11Error("unknown category selection")
        units = [unit for unit in units if unit.category in categories]
    frozen_path = project_root / config["sources"]["robustness_config"]
    frozen = load_m4_settings(frozen_path)
    conditions_by_id = {
        c.identifier: c for c in (ShiftSpec("clean", "clean", 0, {}), *frozen.conditions)
    }
    if seed != frozen.seed or not set(config["conditions"]).issubset(conditions_by_id):
        raise M11Error("robustness source/seed differs from the fixed protocol")
    conditions = [conditions_by_id[name] for name in config["conditions"]]
    for name in config["encoders"]:
        hashes[f"{name}_checkpoint_sha256"] = sha256_file(verify_checkpoint(name))
    hashes.update(
        {
            "config_sha256": sha256_file(config_path),
            "robustness_sha256": sha256_file(frozen_path),
            "code_sha256": hashlib.sha256(
                "".join(
                    f"{path.relative_to(project_root).as_posix()}:{sha256_file(path)}\n"
                    for path in sorted((project_root / "src").rglob("*.py"))
                ).encode()
            ).hexdigest(),
        }
    )
    environment = _environment(device)
    mode = "smoke" if quick else "full"
    resolved = {
        **config,
        "mode": mode,
        "device": device,
        "selected_datasets": selected_datasets,
        "selected_categories": sorted(set(categories or [])),
    }
    runs = Path(os.environ["ANOMALY_ARTIFACT_ROOT"]) / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    if resume is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run = runs / f"{config['experiment']['name']}-{mode}-{timestamp}"
        run.mkdir(exist_ok=False)
        metadata: dict[str, Any] = {
            "status": "running",
            "mode": mode,
            "started_at_utc": datetime.now(UTC).isoformat(),
            "schema_version": 1,
            "git": _git_metadata(project_root),
            **hashes,
        }
        _atomic_json(run / "metadata.json", metadata)
        _atomic_json(run / "resolved_config.json", resolved)
        _atomic_json(run / "environment.json", environment)
    else:
        run = resume.resolve(strict=True)
        run.relative_to(runs.resolve(strict=True))
        metadata = json.loads((run / "metadata.json").read_text("utf-8"))
        if metadata.get("status") != "running" or any(
            metadata.get(key) != value for key, value in hashes.items()
        ):
            raise M11Error("resume requires unchanged inputs/code and an incomplete run")
        if json.loads((run / "resolved_config.json").read_text("utf-8")) != resolved:
            raise M11Error("resume settings changed")
        old_env = json.loads((run / "environment.json").read_text("utf-8"))
        if (
            old_env["packages"] != environment["packages"]
            or old_env["python"] != environment["python"]
        ):
            raise M11Error("resume environment changed")
    print(f"Run: {run}", flush=True)
    completed = new_conditions = 0
    total = len(units) * len(config["encoders"]) * len(conditions)
    resume_command = (
        f'python scripts/run_m11_encoder_comparison.py --config "{config_path}" '
        f'--device {device} --resume "{run}"'
    )
    resume_command += " --quick" if quick else ""
    resume_command += "".join(f" --dataset {dataset}" for dataset in selected_datasets)
    resume_command += "".join(
        f' --category "{category}"' for category in sorted(set(categories or []))
    )
    for name in config["encoders"]:
        extractor = build_encoder(
            name,
            device=device,
            image_size=config["model"]["image_size"],
            batch_size=config["model"]["batch_size"],
        )
        for unit in units:
            train, validation, test = partition(unit.samples, config, quick)
            directory = run / "units" / unit.dataset / unit.category / name
            memory, calibration = _fit(
                unit,
                extractor,
                name,
                config,
                train,
                validation,
                frozen.intervention.augmentations,
                directory,
            )
            clean_scores: list[float] | None = None
            for condition in conditions:
                target = directory / "conditions" / condition.identifier
                if not verify_commit(target, target / "checkpoint.json"):
                    _evaluate(
                        unit,
                        name,
                        extractor,
                        memory,
                        calibration,
                        config,
                        test,
                        condition,
                        target,
                        clean_scores,
                    )
                    new_conditions += 1
                completed += 1
                if condition.identifier == "clean":
                    clean_scores = [
                        float(str(row["score"])) for row in _read_csv(target / "predictions.csv")
                    ]
                _atomic_json(
                    run / "progress.json",
                    {
                        "completed_conditions": completed,
                        "total_conditions": total,
                        "resume_command": resume_command,
                        "current_unit": f"{unit.dataset}/{unit.category}/{name}",
                    },
                )
                print(
                    f"[{completed}/{total}] {unit.dataset}/{unit.category}/"
                    f"{name}/{condition.identifier}",
                    flush=True,
                )
                if max_new_conditions is not None and new_conditions >= max_new_conditions:
                    return {
                        "status": "running",
                        "run_dir": str(run),
                        "resume_command": resume_command,
                    }
            _benchmark(unit, name, extractor, memory, test, config, directory)
        del extractor
        gc.collect()
    metrics: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    benchmarks: list[dict[str, Any]] = []
    bootstrap = (
        config["quick"]["bootstrap_samples"] if quick else config["experiment"]["bootstrap_samples"]
    )
    for unit in units:
        for name in config["encoders"]:
            directory = run / "units" / unit.dataset / unit.category / name
            benchmarks.append(
                {
                    "dataset": unit.dataset,
                    "category": unit.category,
                    **json.loads((directory / "benchmark.json").read_text("utf-8")),
                }
            )
            for condition in conditions:
                metrics.extend(
                    _read_csv(directory / "conditions" / condition.identifier / "metrics.csv")
                )
        # Paired IDs and geometry are mandatory, including unchanged fit/calibration identities.
        fit = [
            json.loads(
                (run / "units" / unit.dataset / unit.category / name / "fit.json").read_text(
                    "utf-8"
                )
            )
            for name in config["encoders"]
        ]
        if any(fit[0][key] != fit[1][key] for key in ("fit_ids", "calibration_ids")):
            raise M11Error("encoder normal-reference sample IDs differ")
        for condition in conditions:
            predictions = [
                _read_csv(
                    run
                    / "units"
                    / unit.dataset
                    / unit.category
                    / name
                    / "conditions"
                    / condition.identifier
                    / "predictions.csv"
                )
                for name in config["encoders"]
            ]
            comparisons.append(
                {
                    "dataset": unit.dataset,
                    "category": unit.category,
                    "condition": condition.identifier,
                    **paired_comparison(
                        *predictions,
                        seed=seed,
                        samples=bootstrap
                        if condition.identifier == "clean"
                        else min(bootstrap, 100),
                        confidence=config["experiment"]["confidence"],
                    ),
                }
            )
    _atomic_csv(run / "metrics.csv", metrics)
    _atomic_csv(run / "comparisons.csv", json_ready(comparisons))
    _atomic_csv(run / "benchmarks.csv", benchmarks)
    summary = {
        "status": "complete",
        "run_dir": str(run),
        "mode": mode,
        "units": len(units),
        "encoders": config["encoders"],
        "conditions": len(conditions),
        "metric_rows": len(metrics),
        "comparison_rows": len(comparisons),
        "live_inspector_changed": False,
        "test_images": sum(len(partition(unit.samples, config, quick)[2]) for unit in units),
    }
    _atomic_json(run / "summary.json", summary)
    metadata.update(
        {
            "status": "complete",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "output_hashes": {
                path.relative_to(run).as_posix(): sha256_file(path)
                for path in sorted(run.rglob("*"))
                if path.is_file() and path.name != "metadata.json"
            },
        }
    )
    _atomic_json(run / "metadata.json", metadata)
    return summary

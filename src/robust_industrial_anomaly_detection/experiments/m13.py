"""Matched normal-only small-defect resolution/tiling study, with resumable conditions."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from dotenv import load_dotenv
from PIL import Image

from robust_industrial_anomaly_detection.baselines.encoders import (
    SpatialEncoder,
    build_encoder,
    verify_checkpoint,
)
from robust_industrial_anomaly_detection.baselines.multiscale import (
    ProjectedPatchMemory,
    load_projected_index,
)
from robust_industrial_anomaly_detection.baselines.tiling import (
    ViewMode,
    aligned_canvas,
    evaluation_support,
    score_views,
    view_features,
)
from robust_industrial_anomaly_detection.data.ksdd2 import read_config
from robust_industrial_anomaly_detection.data.visa import sha256_file
from robust_industrial_anomaly_detection.evaluation.calibration import (
    conformal_p_values,
    selective_decisions,
    selective_metrics,
)
from robust_industrial_anomaly_detection.evaluation.metrics import (
    calibrate_threshold,
    image_metrics,
    json_ready,
)
from robust_industrial_anomaly_detection.evaluation.small_defects import (
    aggregate_captures,
    aggregate_regions,
    capture_measurements,
    paired_deltas,
    region_measurements,
)
from robust_industrial_anomaly_detection.experiments.m2 import (
    Sample,
    _atomic_csv,
    _atomic_json,
    _environment,
    _git_metadata,
    _read_csv,
)
from robust_industrial_anomaly_detection.experiments.m4 import load_m4_settings
from robust_industrial_anomaly_detection.experiments.m11 import (
    EncoderUnit,
    _commit,
    _units,
    partition,
    verify_commit,
)
from robust_industrial_anomaly_detection.experiments.m12 import verify_files
from robust_industrial_anomaly_detection.prototype.benchmark import _working_set_bytes
from robust_industrial_anomaly_detection.robustness.transforms import ShiftSpec, apply_shift

READOUTS = ("global56", "global336", "tiled336")
MODES: tuple[ViewMode, ...] = ("global", "tiled")


def load_config(path: Path) -> dict[str, Any]:
    config = read_config(path)
    expected = {
        "image_size": 224,
        "canvas_size": 336,
        "tile_size": 224,
        "tile_stride": 112,
        "blend_floor": 0.05,
        "batch_size": 4,
        "encoder_batch_size": 16,
        "torch_threads": 4,
        "fit_normals_cap": 128,
        "bank_size": 512,
        "projected_dimension": 64,
        "candidate_limit": 4096,
        "patch_budget_per_image": 24,
        "image_score_fraction": 0.01,
        "gaussian_sigma": 6.0,
    }
    if config["model"] != expected or config["datasets"] != ["visa", "ksdd2"]:
        raise ValueError("small-defect study requires the declared matched input/memory budget")
    if config["calibration"] != {
        "anomaly_alpha": 0.05,
        "review_alpha": 0.20,
        "pixel_quantile": 0.995,
    }:
        raise ValueError("small-defect study requires frozen normal-only calibration")
    if config["evaluation"] != {
        "coarse_grid": 56,
        "small_region_fraction": 0.001,
        "tiny_region_fraction": 0.0001,
        "region_recall_threshold": 0.10,
    }:
        raise ValueError("small-defect cohorts/readouts differ from the fixed protocol")
    if config["conditions"] != ["clean", "blur_severe", "noise_severe"]:
        raise ValueError("small-defect study requires the declared clean/blur/noise conditions")
    if (
        config["experiment"]["bootstrap_samples"] < 1
        or not 0 < config["experiment"]["confidence"] < 1
    ):
        raise ValueError("invalid small-defect uncertainty settings")
    return config


def scaled_shift(condition: ShiftSpec, canvas: int = 336) -> ShiftSpec:
    parameters = dict(condition.parameters)
    ratio = canvas / 224
    for name in ("sigma", "translate_x", "translate_y"):
        if name in parameters:
            parameters[name] *= ratio
    return ShiftSpec(condition.identifier, condition.family, condition.severity, parameters)


def sample_canvas(sample: Sample, dataset: str) -> tuple[np.ndarray, np.ndarray, int]:
    with Image.open(sample.image_path) as image:
        if sample.mask_path is None:
            return aligned_canvas(image, None, dataset=dataset, size=336)
        with Image.open(sample.mask_path) as mask:
            return aligned_canvas(image, mask, dataset=dataset, size=336)


def _memory(seed: int) -> ProjectedPatchMemory:
    return ProjectedPatchMemory(
        seed=seed,
        bank_size=512,
        projected_dimension=64,
        patches_per_image=12,
        candidate_limit=4096,
        image_score_fraction=0.01,
        selection="kcenter",
    )


def _features(
    encoder: SpatialEncoder, images: Sequence[np.ndarray], mode: ViewMode
) -> list[list[np.ndarray]]:
    return view_features(encoder, images, mode=mode, input_size=224, tile_size=224, stride=112)


def _score(
    memory: ProjectedPatchMemory, features: Sequence[np.ndarray], mode: ViewMode
) -> tuple[float, np.ndarray]:
    return score_views(
        memory,
        features,
        mode=mode,
        canvas_size=336,
        tile_size=224,
        stride=112,
        floor=0.05,
        sigma=6.0,
        score_fraction=0.01,
    )


def _fit(
    unit: EncoderUnit,
    mode: ViewMode,
    encoder: SpatialEncoder,
    train: Sequence[Sample],
    validation: Sequence[Sample],
    augmentations: Sequence[ShiftSpec],
    config: Mapping[str, Any],
    directory: Path,
) -> tuple[ProjectedPatchMemory, dict[str, Any]]:
    memory = _memory(config["experiment"]["seed"])
    directory.mkdir(parents=True, exist_ok=True)
    if verify_commit(directory, directory / "fit_checkpoint.json"):
        fit = json.loads((directory / "fit.json").read_text("utf-8"))
        if fit["fit_ids"] != [s.sample_id for s in train] or fit["calibration_ids"] != [
            s.sample_id for s in validation
        ]:
            raise ValueError("small-defect fit/calibration membership changed")
        memory.fit_projected(*load_projected_index(directory / "index.npz"))
        return memory, json.loads((directory / "calibration.json").read_text("utf-8"))
    start = time.perf_counter()
    seed = config["experiment"]["seed"]
    batch = config["model"]["batch_size"]
    patches = []
    for offset in range(0, len(train), batch):
        samples = train[offset : offset + batch]
        images = [sample_canvas(sample, unit.dataset)[0] for sample in samples]
        augmented = []
        for sample, image in zip(samples, images, strict=True):
            index = int(hashlib.sha256(sample.sample_id.encode()).hexdigest()[:16], 16) % len(
                augmentations
            )
            augmented.append(
                apply_shift(
                    image,
                    np.zeros((336, 336), bool),
                    scaled_shift(augmentations[index]),
                    seed=seed,
                    sample_key=sample.sample_id,
                ).image
            )
        features = _features(encoder, [*images, *augmented], mode)
        for position, sample in enumerate(samples):
            rng = np.random.default_rng(seed + int(sample.sample_id[:8], 16))
            for views in (features[position], features[position + len(samples)]):
                combined = np.concatenate([view.reshape(-1, 448) for view in views])
                patches.append(
                    combined[rng.choice(len(combined), 12, replace=False)].reshape(1, 12, 448)
                )
        if offset % (batch * 8) == 0:
            print(
                f"Fit {unit.dataset}/{unit.category}/{mode}: "
                f"{min(offset + batch, len(train))}/{len(train)}",
                flush=True,
            )
    stats = memory.fit(patches)
    del patches
    image_scores = []
    maps = []
    for offset in range(0, len(validation), batch):
        images = [
            sample_canvas(sample, unit.dataset)[0] for sample in validation[offset : offset + batch]
        ]
        for capture_features in _features(encoder, images, mode):
            score, evidence = _score(memory, capture_features, mode)
            image_scores.append(score)
            maps.append(evidence)
    thresholds = {
        str(grid): calibrate_threshold(
            np.concatenate(
                [
                    cv2.resize(values, (grid, grid), interpolation=cv2.INTER_LINEAR).reshape(-1)
                    for values in maps
                ]
            ),
            0.995,
        )
        for grid in ((56, 336) if mode == "global" else (336,))
    }
    calibration = {
        "scores": image_scores,
        "image_threshold": calibrate_threshold(image_scores, 0.95),
        "pixel_thresholds": thresholds,
        "calibration_samples": len(validation),
    }
    memory.save(
        directory / "index.npz",
        metadata={"mode": mode, "category": unit.category, "dataset": unit.dataset},
    )
    _atomic_json(directory / "calibration.json", calibration)
    _atomic_json(
        directory / "fit.json",
        {
            "fit_ids": [s.sample_id for s in train],
            "calibration_ids": [s.sample_id for s in validation],
            "candidate_count": stats.candidate_count,
            "bank_entries": stats.selected_count,
            "patch_budget_per_capture": 24,
            "fit_seconds": time.perf_counter() - start,
            "post_fit_working_set_bytes": _working_set_bytes(),
        },
    )
    _commit(directory, "fit_checkpoint.json", ["index.npz", "calibration.json", "fit.json"])
    return memory, calibration


def _save_arrays(path: Path, arrays: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _evaluate(
    unit: EncoderUnit,
    encoder: SpatialEncoder,
    memories: Mapping[str, ProjectedPatchMemory],
    calibrations: Mapping[str, Mapping[str, Any]],
    test: Sequence[Sample],
    condition: ShiftSpec,
    config: Mapping[str, Any],
    directory: Path,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    seed, batch = config["experiment"]["seed"], config["model"]["batch_size"]
    predictions: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    times = {mode: {key: 0.0 for key in ("feature_ms", "score_merge_ms")} for mode in MODES}
    examples: dict[str, tuple[int, dict[str, np.ndarray], dict[str, Any]]] = {}
    for offset in range(0, len(test), batch):
        samples = test[offset : offset + batch]
        pairs = [sample_canvas(sample, unit.dataset) for sample in samples]
        shifted = [
            apply_shift(
                image, mask, scaled_shift(condition), seed=seed, sample_key=sample.sample_id
            )
            for sample, (image, mask, _) in zip(samples, pairs, strict=True)
        ]
        outputs: dict[str, list[tuple[float, np.ndarray]]] = {}
        for mode in MODES:
            start = time.perf_counter()
            features = _features(encoder, [pair.image for pair in shifted], mode)
            times[mode]["feature_ms"] += (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            outputs[mode] = [_score(memories[mode], values, mode) for values in features]
            times[mode]["score_merge_ms"] += (time.perf_counter() - start) * 1000
        for position, (sample, pair, source) in enumerate(
            zip(samples, shifted, pairs, strict=True)
        ):
            mask = pair.mask
            coarse_mask = cv2.resize(
                mask.astype(np.uint8), (56, 56), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            for readout in READOUTS:
                mode = "tiled" if readout == "tiled336" else "global"
                grid = 56 if readout == "global56" else 336
                score, values = outputs[mode][position]
                reference = calibrations[mode]
                p_value = float(conformal_p_values(reference["scores"], [score])[0])
                decision = str(
                    selective_decisions([p_value], anomaly_alpha=0.05, review_alpha=0.20)[0]
                )
                threshold = reference["pixel_thresholds"][str(grid)]
                support = evaluation_support(values, threshold, grid=grid)
                regions = region_measurements(mask, support)
                metadata = {
                    "sample_id": sample.sample_id,
                    "label": sample.label,
                    "readout": readout,
                    "condition": condition.identifier,
                }
                row = {
                    **metadata,
                    "score": score,
                    "conformal_p": p_value,
                    "decision": decision,
                    "pixel_threshold": threshold,
                    "source_positive_pixels": source[2],
                    "coarse_mask_pixels": int(coarse_mask.sum()),
                    "coarse_mask_lost": int(mask.any() and not coarse_mask.any()),
                    **capture_measurements(
                        mask, support, sample.label, small_fraction=0.001, regions=regions
                    ),
                }
                if readout == "global56":
                    low_support = (
                        cv2.resize(values, (56, 56), interpolation=cv2.INTER_LINEAR) > threshold
                    )
                    union = np.count_nonzero(coarse_mask | low_support)
                    row["legacy_coarse_iou"] = (
                        float(np.count_nonzero(coarse_mask & low_support) / union)
                        if coarse_mask.any() and union
                        else None
                    )
                predictions.append(row)
                region_rows.extend({**metadata, **region} for region in regions)
            if condition.identifier == "clean":
                name = "normal" if sample.label == "normal" else "smallest-anomaly"
                area = int(mask.sum())
                if (name == "normal" and name not in examples) or (
                    name == "smallest-anomaly"
                    and area > 0
                    and (name not in examples or area < examples[name][0])
                ):
                    examples[name] = (
                        area,
                        {
                            "rgb": pair.image,
                            "mask": mask,
                            "global_map": outputs["global"][position][1],
                            "tiled_map": outputs["tiled"][position][1],
                        },
                        {
                            "sample_id": sample.sample_id,
                            "label": sample.label,
                            "fine_mask_pixels": area,
                        },
                    )
    metrics: list[dict[str, Any]] = []
    region_metrics: list[dict[str, Any]] = []
    for readout in READOUTS:
        rows = [row for row in predictions if row["readout"] == readout]
        mode = "tiled" if readout == "tiled336" else "global"
        identity = {
            "dataset": unit.dataset,
            "category": unit.category,
            "condition": condition.identifier,
            "readout": readout,
        }
        labels = [int(row["label"] == "anomaly") for row in rows]
        metrics.append(
            {
                **identity,
                **image_metrics(
                    labels, [row["score"] for row in rows], calibrations[mode]["image_threshold"]
                ),
                **selective_metrics(labels, [row["decision"] for row in rows]),
                **aggregate_captures(rows),
                "pixel_threshold": rows[0]["pixel_threshold"],
                "mean_batch_feature_ms": times[mode]["feature_ms"] / len(test),
                "mean_batch_score_merge_ms": times[mode]["score_merge_ms"] / len(test),
                "legacy_coarse_iou": float(
                    np.mean(
                        [
                            row["legacy_coarse_iou"]
                            for row in rows
                            if row.get("legacy_coarse_iou") is not None
                        ]
                    )
                )
                if any(row.get("legacy_coarse_iou") is not None for row in rows)
                else None,
            }
        )
        region_metrics.extend(
            {**identity, **row}
            for row in aggregate_regions([row for row in region_rows if row["readout"] == readout])
        )
    _atomic_csv(directory / "predictions.csv", predictions)
    _atomic_csv(directory / "regions.csv", region_rows)
    # _atomic_csv intentionally omits an empty table; empty regions are valid.
    _atomic_csv(directory / "metrics.csv", metrics)
    _atomic_csv(directory / "region_metrics.csv", region_metrics)
    files = ["predictions.csv", "metrics.csv", "region_metrics.csv"]
    if region_rows:
        files.append("regions.csv")
    for name, (_, arrays, metadata) in examples.items():
        _save_arrays(directory / f"{name}.npz", arrays)
        _atomic_json(directory / f"{name}.json", metadata)
        files.extend([f"{name}.npz", f"{name}.json"])
    _commit(directory, "checkpoint.json", files)


def _benchmark(
    unit: EncoderUnit,
    encoder: SpatialEncoder,
    memories: Mapping[str, ProjectedPatchMemory],
    test: Sequence[Sample],
    config: Mapping[str, Any],
    directory: Path,
) -> None:
    if verify_commit(directory, directory / "benchmark_checkpoint.json"):
        return
    samples = [s for s in test if s.label == "normal"][: config["benchmark"]["normal_samples"]] + [
        s for s in test if s.label == "anomaly"
    ][: config["benchmark"]["anomaly_samples"]]
    timings = []
    for mode in MODES:
        _score(
            memories[mode],
            _features(encoder, [sample_canvas(samples[0], unit.dataset)[0]], mode)[0],
            mode,
        )
        for sample in samples:
            start = time.perf_counter()
            image = sample_canvas(sample, unit.dataset)[0]
            feature_start = time.perf_counter()
            features = _features(encoder, [image], mode)[0]
            feature_ms = (time.perf_counter() - feature_start) * 1000
            _score(memories[mode], features, mode)
            timings.append(
                {
                    "mode": mode,
                    "sample_id": sample.sample_id,
                    "label": sample.label,
                    "feature_ms": feature_ms,
                    "total_ms": (time.perf_counter() - start) * 1000,
                    "encoder_views": len(features),
                }
            )
    _atomic_csv(directory / "benchmark_timings.csv", timings)
    _commit(directory, "benchmark_checkpoint.json", ["benchmark_timings.csv"])


def run_m13(
    *,
    project: Path,
    config_path: Path,
    quick: bool = False,
    resume: Path | None = None,
    datasets: Sequence[str] | None = None,
    categories: Sequence[str] | None = None,
    max_new_conditions: int | None = None,
) -> dict[str, Any]:
    load_dotenv(project / ".env")
    config = load_config(config_path)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    seed = config["experiment"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if max_new_conditions is not None and max_new_conditions < 1:
        raise ValueError("max-new-conditions must be positive")
    selected = sorted(set(datasets or config["datasets"]))
    if not set(selected).issubset(config["datasets"]):
        raise ValueError("unknown small-defect dataset selection")
    data = Path(os.environ["ANOMALY_DATA_ROOT"])
    print("Verifying trusted manifests and source inventory...", flush=True)
    units, hashes = _units(project, data, config, selected)
    if categories:
        if not set(categories).issubset({unit.category for unit in units}):
            raise ValueError("unknown small-defect category selection")
        units = [unit for unit in units if unit.category in categories]
    manifests = {}
    for dataset in selected:
        dataset_id = "visa-20220922" if dataset == "visa" else "ksdd2-20230928"
        manifests[dataset] = {
            row["sample_id"]: row
            for row in (
                json.loads(line)
                for line in (data / "manifests" / dataset_id / "manifest.jsonl")
                .read_text("utf-8")
                .splitlines()
            )
        }
    robustness = project / config["sources"]["robustness_config"]
    frozen = load_m4_settings(robustness)
    if frozen.seed != seed:
        raise ValueError("small-defect seed differs from frozen robustness source")
    conditions = {c.identifier: c for c in (ShiftSpec("clean", "clean", 0, {}), *frozen.conditions)}
    code = [*sorted((project / "src").rglob("*.py")), project / "scripts/run_m13_small_defects.py"]
    hashes.update(
        {
            "config_sha256": sha256_file(config_path),
            "robustness_sha256": sha256_file(robustness),
            "weights_sha256": sha256_file(verify_checkpoint("resnet18")),
            "code_sha256": hashlib.sha256(
                "".join(
                    f"{path.relative_to(project).as_posix()}:{sha256_file(path)}\n" for path in code
                ).encode()
            ).hexdigest(),
        }
    )
    mode = "smoke" if quick else "full"
    resolved = {
        **config,
        "mode": mode,
        "selected_datasets": selected,
        "selected_categories": sorted(set(categories or [])),
    }
    environment = _environment("cpu")
    runs = Path(os.environ["ANOMALY_ARTIFACT_ROOT"]) / "runs"
    if resume is None:
        run = runs / f"{config['experiment']['name']}-{mode}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
        run.mkdir(parents=True, exist_ok=False)
        metadata: dict[str, Any] = {
            "schema_version": 1,
            "status": "running",
            "mode": mode,
            "started_at_utc": datetime.now(UTC).isoformat(),
            "git": _git_metadata(project),
            **hashes,
        }
        _atomic_json(run / "metadata.json", metadata)
        _atomic_json(run / "resolved_config.json", resolved)
        _atomic_json(run / "environment.json", environment)
    else:
        run = resume.resolve(strict=True)
        run.relative_to(runs.resolve(strict=True))
        metadata = json.loads((run / "metadata.json").read_text("utf-8"))
        if metadata["status"] != "running" or any(
            metadata.get(key) != value for key, value in hashes.items()
        ):
            raise ValueError(
                "small-defect resume requires unchanged inputs/code and incomplete status"
            )
        if json.loads((run / "resolved_config.json").read_text("utf-8")) != resolved:
            raise ValueError("small-defect resume settings changed")
        old_env = json.loads((run / "environment.json").read_text("utf-8"))
        if (
            old_env["packages"] != environment["packages"]
            or old_env["python"] != environment["python"]
        ):
            raise ValueError("small-defect resume environment changed")
    print(f"Run: {run}", flush=True)
    encoder = build_encoder("resnet18", device="cpu", image_size=224, batch_size=16)
    completed = new_conditions = 0
    resume_command = f'python scripts/run_m13_small_defects.py --resume "{run}"'
    resume_command += " --quick" if quick else ""
    resume_command += "".join(f" --dataset {name}" for name in selected)
    resume_command += "".join(f" --category {name}" for name in sorted(set(categories or [])))
    for unit in units:
        train, validation, test = partition(unit.samples, config, quick)
        directory = run / "units" / unit.dataset / unit.category
        directory.mkdir(parents=True, exist_ok=True)
        dataset_id = "visa-20220922" if unit.dataset == "visa" else "ksdd2-20230928"
        source = data / "sources" / dataset_id
        inputs: dict[str, str] = {}
        for sample in [*train, *validation, *test]:
            row = manifests[unit.dataset][sample.sample_id]
            inputs[row["image_path"]] = row["image_sha256"]
            if row["mask_path"]:
                inputs[row["mask_path"]] = row["mask_sha256"]
        verify_files(source, inputs)
        memories: dict[str, ProjectedPatchMemory] = {}
        calibrations: dict[str, Mapping[str, Any]] = {}
        for name in MODES:
            memories[name], calibrations[name] = _fit(
                unit,
                name,
                encoder,
                train,
                validation,
                frozen.intervention.augmentations,
                config,
                directory / name,
            )
        if not np.array_equal(memories["global"].projection, memories["tiled"].projection):
            raise ValueError("matched global/tiled projections differ")
        for condition_id in config["conditions"]:
            target = directory / "conditions" / condition_id
            if not verify_commit(target, target / "checkpoint.json"):
                _evaluate(
                    unit,
                    encoder,
                    memories,
                    calibrations,
                    test,
                    conditions[condition_id],
                    config,
                    target,
                )
                new_conditions += 1
            completed += 1
            _atomic_json(
                run / "progress.json",
                {
                    "completed_conditions": completed,
                    "total_conditions": len(units) * len(config["conditions"]),
                    "resume_command": resume_command,
                    "current_unit": f"{unit.dataset}/{unit.category}",
                },
            )
            print(
                f"[{completed}/{len(units) * len(config['conditions'])}] "
                f"{unit.dataset}/{unit.category}/{condition_id}",
                flush=True,
            )
            if max_new_conditions is not None and new_conditions >= max_new_conditions:
                return {"status": "running", "run_dir": str(run), "resume_command": resume_command}
        _benchmark(unit, encoder, memories, test, config, directory)
    metrics: list[dict[str, Any]] = []
    region_metrics: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    resources = []
    bootstrap = (
        config["quick"]["bootstrap_samples"] if quick else config["experiment"]["bootstrap_samples"]
    )
    for unit in units:
        directory = run / "units" / unit.dataset / unit.category
        identity = {"dataset": unit.dataset, "category": unit.category}
        timings: list[dict[str, Any]] = list(_read_csv(directory / "benchmark_timings.csv"))
        for name in MODES:
            times = [row["total_ms"] for row in timings if row["mode"] == name]
            fit = json.loads((directory / name / "fit.json").read_text("utf-8"))
            resources.append(
                {
                    **identity,
                    "mode": name,
                    "samples": len(times),
                    "median_ms": float(np.median(times)),
                    "p95_ms": float(np.quantile(times, 0.95)),
                    "index_bytes": (directory / name / "index.npz").stat().st_size,
                    "encoder_views": 1 if name == "global" else 4,
                    **{
                        key: fit[key]
                        for key in ("candidate_count", "bank_entries", "post_fit_working_set_bytes")
                    },
                }
            )
        for condition_id in config["conditions"]:
            target = directory / "conditions" / condition_id
            metrics.extend(_read_csv(target / "metrics.csv"))
            region_metrics.extend(_read_csv(target / "region_metrics.csv"))
            predictions = _read_csv(target / "predictions.csv")
            selected_rows = {
                name: [row for row in predictions if row["readout"] == name] for name in READOUTS
            }
            for contrast, candidate in (("resolution", "global56"), ("tiling", "tiled336")):
                # Resolution is fine minus coarse; tiling is tiled minus fine.
                baseline_rows = (
                    selected_rows["global56"]
                    if contrast == "resolution"
                    else selected_rows["global336"]
                )
                candidate_rows = (
                    selected_rows["global336"]
                    if contrast == "resolution"
                    else selected_rows[candidate]
                )
                comparisons.append(
                    {
                        **identity,
                        "condition": condition_id,
                        "contrast": contrast,
                        **paired_deltas(
                            baseline_rows,
                            candidate_rows,
                            seed=seed,
                            samples=bootstrap,
                            confidence=config["experiment"]["confidence"],
                        ),
                    }
                )
    _atomic_csv(run / "metrics.csv", metrics)
    _atomic_csv(run / "region_metrics.csv", region_metrics)
    _atomic_csv(run / "comparisons.csv", json_ready(comparisons))
    _atomic_csv(run / "resources.csv", resources)
    summary = {
        "status": "complete",
        "mode": mode,
        "units": len(units),
        "metric_rows": len(metrics),
        "comparison_rows": len(comparisons),
        "test_images": sum(len(partition(unit.samples, config, quick)[2]) for unit in units),
        "live_inspector_changed": False,
        "run_dir": str(run),
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

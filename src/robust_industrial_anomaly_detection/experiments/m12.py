"""Versioned normal-only quality-gate comparison without neural retraining."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from dotenv import load_dotenv
from PIL import Image

from robust_industrial_anomaly_detection.baselines.multiscale import load_fitted_rgb
from robust_industrial_anomaly_detection.data.ksdd2 import load_ksdd2_manifest
from robust_industrial_anomaly_detection.data.visa import load_trusted_manifest, sha256_file
from robust_industrial_anomaly_detection.evaluation.calibration import exact_binomial_interval
from robust_industrial_anomaly_detection.experiments.m2 import (
    _atomic_csv,
    _atomic_json,
    _environment,
    _git_metadata,
    _read_csv,
)
from robust_industrial_anomaly_detection.experiments.m8 import (
    _aggregate,
    _macro_summary,
    _source_decisions,
    load_m8_settings,
)
from robust_industrial_anomaly_detection.prototype.quality import (
    CaptureQualityModel,
    capture_measurements,
    evaluate_capture_quality,
    fit_category_quality_model,
    load_quality_model,
    quality_model_dict,
)
from robust_industrial_anomaly_detection.robustness.transforms import ShiftSpec, apply_shift


def verify_files(root: Path, hashes: Mapping[str, str]) -> None:
    """Reject missing/modified committed artifacts and paths escaping the run."""

    resolved_root = root.resolve(strict=True)
    for name, expected in hashes.items():
        path = (root / name).resolve(strict=True)
        path.relative_to(resolved_root)
        if sha256_file(path) != expected:
            raise ValueError(f"modified quality artifact: {name}")


def normal_partitions(
    records: Sequence[dict[str, Any]], *, quick: Mapping[str, int] | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Allow only disjoint, audited train/calibration normals and provider tests."""

    selected = [
        [row for row in records if row["split"] == split]
        for split in ("train", "validation", "test")
    ]
    train, validation, test = selected
    if any(row["label"] != "normal" for row in [*train, *validation]):
        raise ValueError("quality fitting/calibration must contain only normals")
    ids = [row["sample_id"] for group in selected for row in group]
    if len(ids) != len(set(ids)):
        raise ValueError("quality partitions have duplicate or overlapping sample IDs")
    if quick:
        train = train[: quick["train_normals"]]
        validation = validation[: quick["calibration_normals"]]
        test = [row for row in test if row["label"] == "normal"][: quick["test_normals"]] + [
            row for row in test if row["label"] == "anomaly"
        ][: quick["test_anomalies"]]
    if len(train) < 16 or len(validation) < 49:
        raise ValueError("quality fitting requires 16 normals and calibration at least 49")
    if {row["label"] for row in test} != {"normal", "anomaly"}:
        raise ValueError("quality evaluation requires both official-test labels")
    return train, validation, test


def aggregate_comparison(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use unchanged M8 denominators and expose clean/anomaly deferral burden."""

    output: list[dict[str, Any]] = []
    for gate in ("v1", "v2"):
        selected = [row for row in rows if row["gate"] == gate]
        for aggregate in _aggregate(selected):
            low, high = exact_binomial_interval(
                int(cast(int, aggregate["quality_rejections"])),
                int(cast(int, aggregate["samples"])),
            )
            output.append(
                {**aggregate, "gate": gate, "rejection_ci_low": low, "rejection_ci_high": high}
            )
    return output


def run_m12(
    *,
    project: Path,
    config_path: Path,
    quick: bool = False,
    resume: Path | None = None,
    stop_after_conditions: int | None = None,
) -> dict[str, Any]:
    """Fit both-domain v2 references and compare against exact historical gates."""

    load_dotenv(project / ".env")
    config = yaml.safe_load(config_path.read_text("utf-8"))
    if config["quality_gate"] != {"schema_version": 2, "image_size": 224, "alpha": 0.02}:
        raise ValueError("M12 requires the declared v2 geometry and calibration")
    if stop_after_conditions is not None and stop_after_conditions < 1:
        raise ValueError("stop-after-conditions must be positive")
    seed = config["experiment"]["seed"]
    protocol = load_m8_settings(project / config["source_protocol"])
    if seed != protocol.seed or protocol.alpha != 0.02 or protocol.image_size != 224:
        raise ValueError("quality comparison differs from its frozen source protocol")
    data_root = Path(os.environ["ANOMALY_DATA_ROOT"])
    runs_root = Path(os.environ["ANOMALY_ARTIFACT_ROOT"]) / "runs"
    sources: dict[str, Path] = {}
    source_metadata: dict[str, Any] = {}
    for name, pin in config["sources"].items():
        source = (runs_root / pin["run"]).resolve(strict=True)
        source.relative_to(runs_root.resolve(strict=True))
        if sha256_file(source / "metadata.json") != pin["metadata_sha256"]:
            raise ValueError(f"quality source metadata differs from pin: {name}")
        meta = json.loads((source / "metadata.json").read_text("utf-8"))
        if meta.get("status") != "complete" or meta.get("mode") != "full":
            raise ValueError("quality comparison requires complete full sources")
        sources[name], source_metadata[name] = source, meta
    visa_dir = data_root / "manifests/visa-20220922"
    visa_root = data_root / "sources/visa-20220922"
    visa_records = load_trusted_manifest(
        visa_dir / "manifest.jsonl",
        visa_dir / "audit.json",
        project / "configs/data/visa.yaml",
        visa_root,
    )
    ksdd_records, ksdd_root = load_ksdd2_manifest(project / "configs/data/ksdd2.yaml", data_root)
    ksdd_dir = data_root / "manifests/ksdd2-20230928"
    decisions, _ = _source_decisions(
        sources["m5"],
        settings=protocol,
        manifest_hash=sha256_file(visa_dir / "manifest.jsonl"),
        audit_hash=sha256_file(visa_dir / "audit.json"),
    )
    for name in ("m8", "m9"):
        meta = source_metadata[name]
        if name == "m8":
            verify_files(
                sources[name],
                {
                    "quality_model.json": meta["quality_model_sha256"],
                    "quality_predictions.csv": meta["quality_predictions_sha256"],
                },
            )
        else:
            verify_files(sources[name], meta["output_hashes"])
        directory = visa_dir if name == "m8" else ksdd_dir
        if meta["manifest_sha256"] != sha256_file(directory / "manifest.jsonl"):
            raise ValueError("quality source used a different trusted manifest")
    originals = {
        "visa": load_quality_model(sources["m8"] / "quality_model.json"),
        "ksdd2": load_quality_model(sources["m9"] / "quality_model.json"),
    }
    if any(model.schema_version != 1 for model in originals.values()):
        raise ValueError("historical reference must use v1 quality measurements")
    historical: dict[tuple[str, str, str, str], str] = {}
    for history_row in _read_csv(sources["m8"] / "quality_predictions.csv"):
        historical[
            (
                "visa",
                str(history_row["category"]),
                str(history_row["condition"]),
                str(history_row["sample_id"]),
            )
        ] = str(history_row["quality_status"])
    for history_row in _read_csv(sources["m9"] / "predictions.csv"):
        if history_row["model"] != "augmented_coreset_512":
            continue
        history_key = ("surface", str(history_row["condition"]), str(history_row["sample_id"]))
        decisions[history_key] = str(history_row["decision"])
        historical[("ksdd2", *history_key)] = str(history_row["quality_status"])
    code_files = sorted((project / "src").rglob("*.py"))
    code_files.append(project / "scripts/run_m12_quality_generalization.py")
    hashes = {
        "config_sha256": sha256_file(config_path),
        "source_protocol_sha256": sha256_file(project / config["source_protocol"]),
        "code_sha256": hashlib.sha256(
            "".join(
                f"{path.relative_to(project)}:{sha256_file(path)}\n" for path in code_files
            ).encode()
        ).hexdigest(),
        **{
            f"{dataset}_{name}_sha256": sha256_file(directory / filename)
            for dataset, directory in (("visa", visa_dir), ("ksdd2", ksdd_dir))
            for name, filename in (("manifest", "manifest.jsonl"), ("audit", "audit.json"))
        },
    }
    resolved = {**config, "mode": "smoke" if quick else "full"}
    if resume is None:
        run = runs_root / (
            f"{config['experiment']['name']}-{resolved['mode']}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
        )
        run.mkdir(parents=True, exist_ok=False)
        metadata: dict[str, Any] = {
            "schema_version": 1,
            "status": "running",
            "mode": resolved["mode"],
            "started_at_utc": datetime.now(UTC).isoformat(),
            "source_m5_metadata_sha256": config["sources"]["m5"]["metadata_sha256"],
            "manifest_sha256": hashes["visa_manifest_sha256"],
            "git": _git_metadata(project),
            **hashes,
        }
        _atomic_json(run / "metadata.json", metadata)
        _atomic_json(run / "resolved_config.json", resolved)
        _atomic_json(run / "environment.json", _environment("cpu"))
    else:
        run = resume.resolve(strict=True)
        run.relative_to(runs_root.resolve(strict=True))
        metadata = json.loads((run / "metadata.json").read_text("utf-8"))
        if metadata["status"] != "running" or any(metadata.get(k) != v for k, v in hashes.items()):
            raise ValueError("quality resume inputs changed or run is not incomplete")
        if json.loads((run / "resolved_config.json").read_text("utf-8")) != resolved:
            raise ValueError("quality resume settings changed")
    print(f"Quality comparison run: {run}", flush=True)
    units = [("visa", category) for category in sorted({row["category"] for row in visa_records})]
    units.append(("ksdd2", "surface"))
    conditions = (ShiftSpec("clean", "clean", 0, {}), *protocol.conditions)
    metrics: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    completed: list[str] = []
    processed = 0
    for dataset, category in units:
        records = (
            [row for row in visa_records if row["category"] == category]
            if dataset == "visa"
            else ksdd_records
        )
        train, validation, test = normal_partitions(
            records, quick=config["quick"] if quick else None
        )
        root = visa_root if dataset == "visa" else ksdd_root
        verify_files(
            root, {row["image_path"]: row["image_sha256"] for row in [*train, *validation, *test]}
        )

        def image(row: Mapping[str, Any], root: Path = root, dataset: str = dataset) -> np.ndarray:
            path = (root / row["image_path"]).resolve(strict=True)
            path.relative_to(root.resolve(strict=True))
            if dataset == "visa":
                return load_fitted_rgb(path, 224)
            with Image.open(path) as source:
                return np.asarray(
                    source.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
                )

        unit = run / "units" / dataset / category
        unit.mkdir(parents=True, exist_ok=True)
        fit_commit = unit / "fit_checkpoint.json"
        if fit_commit.exists():
            saved = json.loads(fit_commit.read_text("utf-8"))
            verify_files(unit, saved["files"])
            if saved["fit_ids"] != [row["sample_id"] for row in train] or saved[
                "calibration_ids"
            ] != [row["sample_id"] for row in validation]:
                raise ValueError("quality fit/calibration membership changed")
            candidate = load_quality_model(unit / "quality_model.json")
        else:
            print(
                f"Fit {dataset}/{category}: {len(train)} normal + {len(validation)} calibration",
                flush=True,
            )
            reference = fit_category_quality_model(
                [capture_measurements(image(row), schema_version=2) for row in train],
                [capture_measurements(image(row), schema_version=2) for row in validation],
                schema_version=2,
            )
            candidate = CaptureQualityModel(
                2, 224, 0.02, hashes[f"{dataset}_manifest_sha256"], {category: reference}
            )
            _atomic_json(unit / "quality_model.json", quality_model_dict(candidate))
            _atomic_json(
                fit_commit,
                {
                    "fit_ids": [row["sample_id"] for row in train],
                    "calibration_ids": [row["sample_id"] for row in validation],
                    "files": {"quality_model.json": sha256_file(unit / "quality_model.json")},
                },
            )
        clean_images: list[np.ndarray] | None = None
        for condition in conditions:
            directory = unit / condition.identifier
            commit = directory / "checkpoint.json"
            key = f"{dataset}/{category}/{condition.identifier}"
            if commit.exists():
                verify_files(directory, json.loads(commit.read_text("utf-8")))
            else:
                directory.mkdir(parents=True, exist_ok=True)
                predictions: list[dict[str, Any]] = []
                condition_times: list[dict[str, Any]] = []
                if clean_images is None:
                    clean_images = [image(row) for row in test]
                for row, clean_image in zip(test, clean_images, strict=True):
                    shifted = apply_shift(
                        clean_image,
                        np.zeros((224, 224), bool),
                        condition,
                        seed=seed,
                        sample_key=row["sample_id"],
                    ).image
                    source_decision = decisions[(category, condition.identifier, row["sample_id"])]
                    for gate, model in (("v1", originals[dataset]), ("v2", candidate)):
                        start = time.perf_counter()
                        decision = evaluate_capture_quality(shifted, category=category, model=model)
                        elapsed = (time.perf_counter() - start) * 1000
                        if (
                            gate == "v1"
                            and decision.status
                            != historical[
                                (dataset, category, condition.identifier, row["sample_id"])
                            ]
                        ):
                            raise ValueError("historical quality dispositions did not reproduce")
                        predictions.append(
                            {
                                "gate": gate,
                                "category": category,
                                "sample_id": row["sample_id"],
                                "label": row["label"],
                                "condition": condition.identifier,
                                "family": condition.family,
                                "severity": condition.severity,
                                "quality_status": decision.status,
                                "quality_p": decision.p_value,
                                "source_decision": source_decision,
                            }
                        )
                        condition_times.append({"gate": gate, "elapsed_ms": elapsed})
                _atomic_csv(directory / "predictions.csv", predictions)
                _atomic_csv(directory / "timings.csv", condition_times)
                _atomic_json(
                    commit,
                    {
                        name: sha256_file(directory / name)
                        for name in ("predictions.csv", "timings.csv")
                    },
                )
                processed += 1
                print(f"Completed {key}", flush=True)
            rows = _read_csv(directory / "predictions.csv")
            metrics.extend({"dataset": dataset, **row} for row in aggregate_comparison(rows))
            timings.extend(
                {"dataset": dataset, **row} for row in _read_csv(directory / "timings.csv")
            )
            completed.append(key)
            _atomic_json(
                run / "progress.json",
                {
                    "status": "running",
                    "run_dir": str(run),
                    "completed_conditions": completed,
                    "expected_conditions": len(units) * len(conditions),
                },
            )
            if stop_after_conditions is not None and processed >= stop_after_conditions:
                return {
                    "run_dir": str(run),
                    "status": "running",
                    "completed_conditions": len(completed),
                }
    _atomic_csv(run / "metrics.csv", metrics)
    summaries = [
        {
            "dataset": dataset,
            "gate": gate,
            **_macro_summary(
                [row for row in metrics if row["dataset"] == dataset and row["gate"] == gate]
            ),
        }
        for dataset in ("visa", "ksdd2")
        for gate in ("v1", "v2")
    ]
    _atomic_json(
        run / "summary.json", {"mode": resolved["mode"], "units": len(units), "results": summaries}
    )
    _atomic_csv(run / "summary.csv", summaries)
    resources = [
        {
            "dataset": dataset,
            "gate": gate,
            "mean_gate_ms": float(
                np.mean(
                    [
                        float(row["elapsed_ms"])
                        for row in timings
                        if row["dataset"] == dataset and row["gate"] == gate
                    ]
                )
            ),
        }
        for dataset in ("visa", "ksdd2")
        for gate in ("v1", "v2")
    ]
    _atomic_csv(run / "resources.csv", resources)
    visa_categories = {
        category: load_quality_model(
            run / "units/visa" / category / "quality_model.json"
        ).categories[category]
        for dataset, category in units
        if dataset == "visa"
    }
    _atomic_json(
        run / "quality_model.json",
        quality_model_dict(
            CaptureQualityModel(2, 224, 0.02, hashes["visa_manifest_sha256"], visa_categories)
        ),
    )
    _atomic_json(run / "progress.json", {"status": "complete", "completed_conditions": completed})
    metadata.update(
        {
            "status": "complete",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "quality_model_sha256": sha256_file(run / "quality_model.json"),
            "output_hashes": {
                file.relative_to(run).as_posix(): sha256_file(file)
                for file in sorted(run.rglob("*"))
                if file.is_file() and file.name != "metadata.json"
            },
        }
    )
    _atomic_json(run / "metadata.json", metadata)
    return {"run_dir": str(run), "status": "complete", "results": summaries}

"""M8 normal-only pre-inference capture-quality gate experiment."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
import yaml
from dotenv import load_dotenv

from robust_industrial_anomaly_detection.baselines.multiscale import load_fitted_rgb
from robust_industrial_anomaly_detection.data.visa import load_trusted_manifest, sha256_file
from robust_industrial_anomaly_detection.experiments.m2 import (
    Sample,
    _atomic_csv,
    _atomic_json,
    _environment,
    _git_metadata,
    _read_csv,
    _samples,
)
from robust_industrial_anomaly_detection.prototype.quality import (
    CaptureQualityModel,
    CategoryQualityModel,
    capture_metrics,
    evaluate_capture_quality,
    fit_category_quality_model,
    load_quality_model,
    quality_model_dict,
)
from robust_industrial_anomaly_detection.robustness.transforms import (
    ShiftFamily,
    ShiftSpec,
    apply_shift,
)


class M8Error(RuntimeError):
    """Raised when M8 cannot satisfy its frozen quality-gate contract."""


@dataclass(frozen=True)
class M8Settings:
    name: str
    seed: int
    source_experiment: str
    primary_model: str
    decision_method: str
    decision_scope: str
    image_size: int
    alpha: float
    minimum_fit_samples: int
    minimum_calibration_samples: int
    conditions: tuple[ShiftSpec, ...]
    quick_train: int
    quick_validation: int
    quick_test_normal: int
    quick_test_anomaly: int


@dataclass(frozen=True)
class M8RunResult:
    run_dir: Path
    summary: dict[str, object]


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise M8Error(f"{name} must be a mapping")
    return cast(Mapping[str, object], value)


def _string(section: Mapping[str, object], name: str) -> str:
    value = section.get(name)
    if not isinstance(value, str) or not value:
        raise M8Error(f"{name} must be a non-empty string")
    return value


def _integer(section: Mapping[str, object], name: str) -> int:
    value = section.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise M8Error(f"{name} must be a positive integer")
    return value


def _probability(section: Mapping[str, object], name: str) -> float:
    value = section.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise M8Error(f"{name} must be numeric")
    result = float(value)
    if not 0.0 < result < 1.0:
        raise M8Error(f"{name} must be strictly between zero and one")
    return result


def _condition(raw: object, index: int) -> ShiftSpec:
    section = _mapping(raw, f"conditions[{index}]")
    parameters = _mapping(section.get("parameters"), f"conditions[{index}].parameters")
    try:
        return ShiftSpec(
            identifier=_string(section, "id"),
            family=cast(ShiftFamily, _string(section, "family")),
            severity=_integer(section, "severity"),
            parameters={str(key): cast(float | int, value) for key, value in parameters.items()},
        )
    except (TypeError, ValueError) as exc:
        raise M8Error(f"invalid condition {index}: {exc}") from exc


def load_m8_settings(path: Path) -> M8Settings:
    """Load and validate the frozen M8 quality-gate protocol."""

    try:
        root = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "configuration")
    except (OSError, yaml.YAMLError) as exc:
        raise M8Error(f"cannot load M8 configuration {path}: {exc}") from exc
    experiment = _mapping(root.get("experiment"), "experiment")
    source = _mapping(root.get("source_m5"), "source_m5")
    gate = _mapping(root.get("quality_gate"), "quality_gate")
    quick = _mapping(root.get("quick"), "quick")
    raw_conditions = root.get("conditions")
    if not isinstance(raw_conditions, Sequence) or isinstance(raw_conditions, (str, bytes)):
        raise M8Error("conditions must be a sequence")
    conditions = tuple(_condition(raw, index) for index, raw in enumerate(raw_conditions))
    if len(conditions) != 12 or len({item.identifier for item in conditions}) != 12:
        raise M8Error("M8 requires the twelve unique frozen M4 shift conditions")
    family_severities = {(item.family, item.severity) for item in conditions}
    expected = {
        (family, severity)
        for family in ("jpeg", "resize", "blur", "noise", "photometric", "geometric")
        for severity in (1, 2)
    }
    if family_severities != expected:
        raise M8Error("M8 conditions must cover every frozen family at severities one and two")
    settings = M8Settings(
        name=_string(experiment, "name"),
        seed=_integer(experiment, "seed"),
        source_experiment=_string(source, "experiment_name"),
        primary_model=_string(source, "primary_model"),
        decision_method=_string(source, "decision_method"),
        decision_scope=_string(source, "decision_scope"),
        image_size=_integer(gate, "image_size"),
        alpha=_probability(gate, "alpha"),
        minimum_fit_samples=_integer(gate, "minimum_fit_samples"),
        minimum_calibration_samples=_integer(gate, "minimum_calibration_samples"),
        conditions=conditions,
        quick_train=_integer(quick, "train_per_category"),
        quick_validation=_integer(quick, "validation_per_category"),
        quick_test_normal=_integer(quick, "test_normal_per_category"),
        quick_test_anomaly=_integer(quick, "test_anomaly_per_category"),
    )
    if settings.image_size < 64 or settings.image_size % 32:
        raise M8Error("quality_gate.image_size must be at least 64 and divisible by 32")
    if settings.minimum_fit_samples < 16 or settings.minimum_calibration_samples < 16:
        raise M8Error("quality calibration minimums cannot be below 16")
    minimum_conformal_samples = int(np.ceil(1.0 / settings.alpha - 1.0))
    if settings.minimum_calibration_samples < minimum_conformal_samples:
        raise M8Error("quality calibration minimum cannot resolve the configured alpha")
    if settings.quick_validation < minimum_conformal_samples:
        raise M8Error("quick validation count cannot resolve the configured alpha")
    if settings.decision_method != "split_conformal" or settings.decision_scope != "category":
        raise M8Error("M8 source decisions must use category split-conformal calibration")
    return settings


def _find_m5_run(artifact_root: Path, name: str, explicit: Path | None) -> Path:
    runs_root = (artifact_root / "runs").resolve(strict=True)
    candidates = (
        [explicit.resolve(strict=True)]
        if explicit
        else sorted(runs_root.glob(f"{name}-full-*"), reverse=True)
    )
    for candidate in candidates:
        try:
            candidate.relative_to(runs_root)
            metadata = json.loads((candidate / "metadata.json").read_text(encoding="utf-8"))
        except (ValueError, OSError, json.JSONDecodeError):
            continue
        if metadata.get("status") == "complete" and metadata.get("mode") == "full":
            return candidate
    raise M8Error("no complete full M5 source run is available")


def _source_decisions(
    run: Path,
    *,
    settings: M8Settings,
    manifest_hash: str,
    audit_hash: str,
) -> tuple[dict[tuple[str, str, str], str], dict[str, object]]:
    try:
        metadata = cast(
            dict[str, object], json.loads((run / "metadata.json").read_text(encoding="utf-8"))
        )
        summary = cast(
            dict[str, object], json.loads((run / "summary.json").read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise M8Error(f"cannot read M5 source artifacts: {exc}") from exc
    if (
        metadata.get("manifest_sha256") != manifest_hash
        or metadata.get("audit_sha256") != audit_hash
    ):
        raise M8Error("M5 source used different trusted data inputs")
    if summary.get("primary_model") != settings.primary_model:
        raise M8Error("M5 primary model does not match the M8 protocol")
    path = run / "decisions.csv"
    if metadata.get("decisions_sha256") != sha256_file(path):
        raise M8Error("M5 decisions failed integrity verification")
    selected: dict[tuple[str, str, str], str] = {}
    for row in _read_csv(path):
        if (
            row.get("model") != settings.primary_model
            or row.get("method") != settings.decision_method
            or row.get("scope") != settings.decision_scope
        ):
            continue
        key = (cast(str, row["category"]), cast(str, row["condition"]), cast(str, row["sample_id"]))
        if key in selected:
            raise M8Error(f"duplicate M5 decision key: {key}")
        selected[key] = cast(str, row["decision"])
    if not selected:
        raise M8Error("M5 source has no matching decisions")
    return selected, metadata


def _partition(
    samples: Sequence[Sample], settings: M8Settings, quick: bool
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    train = [sample for sample in samples if sample.split == "train" and sample.label == "normal"]
    validation = [
        sample for sample in samples if sample.split == "validation" and sample.label == "normal"
    ]
    test = [sample for sample in samples if sample.split == "test"]
    if quick:
        train = train[: settings.quick_train]
        validation = validation[: settings.quick_validation]
        test = [sample for sample in test if sample.label == "normal"][
            : settings.quick_test_normal
        ] + [sample for sample in test if sample.label == "anomaly"][: settings.quick_test_anomaly]
    if len(train) < settings.minimum_fit_samples:
        raise M8Error("quality fitting has too few training-normal samples")
    if len(validation) < settings.minimum_calibration_samples:
        raise M8Error("quality calibration has too few validation-normal samples")
    if {sample.label for sample in test} != {"normal", "anomaly"}:
        raise M8Error("quality evaluation requires both official-test labels")
    return train, validation, test


def _evaluate_category(
    *,
    category: str,
    test: Sequence[Sample],
    model: CaptureQualityModel,
    settings: M8Settings,
    source_decisions: Mapping[tuple[str, str, str], str],
) -> list[dict[str, object]]:
    clean = ShiftSpec("clean", "clean", 0, {})
    rows: list[dict[str, object]] = []
    for sample in test:
        image = load_fitted_rgb(sample.image_path, settings.image_size)
        mask = np.zeros(image.shape[:2], dtype=bool)
        for condition in (clean, *settings.conditions):
            shifted = apply_shift(
                image, mask, condition, seed=settings.seed, sample_key=sample.sample_id
            ).image
            decision = evaluate_capture_quality(shifted, category=category, model=model)
            key = (category, condition.identifier, sample.sample_id)
            source_decision = source_decisions.get(key)
            if source_decision is None:
                raise M8Error(f"M5 source decision coverage is missing {key}")
            rows.append(
                {
                    "category": category,
                    "sample_id": sample.sample_id,
                    "label": sample.label,
                    "target": sample.target,
                    "condition": condition.identifier,
                    "family": condition.family,
                    "severity": condition.severity,
                    "quality_status": decision.status,
                    "quality_score": decision.score,
                    "quality_p": decision.p_value,
                    "quality_alpha": decision.alpha,
                    "quality_contributors": "|".join(decision.contributors),
                    "source_decision": source_decision,
                    **{f"metric_{name}": value for name, value in decision.metrics.items()},
                    **{f"component_{name}": value for name, value in decision.components.items()},
                }
            )
    return rows


def _aggregate(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, int, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                cast(str, row["category"]),
                cast(str, row["condition"]),
                cast(str, row["family"]),
                int(cast(int | float, row["severity"])),
                cast(str, row["label"]),
            )
        ].append(row)
    output: list[dict[str, object]] = []
    for (category, condition, family, severity, label), group in sorted(grouped.items()):
        rejected = [row["quality_status"] == "recapture" for row in group]
        source_alert = [row["source_decision"] != "normal" for row in group]
        source_anomalous = [row["source_decision"] == "anomalous" for row in group]
        rejected_count = sum(rejected)
        false_alerts = sum(source_alert) if label == "normal" else 0
        intercepted = (
            sum(reject and alert for reject, alert in zip(rejected, source_alert, strict=True))
            if label == "normal"
            else 0
        )
        output.append(
            {
                "category": category,
                "condition": condition,
                "family": family,
                "severity": severity,
                "label": label,
                "samples": len(group),
                "quality_rejections": rejected_count,
                "quality_rejection_rate": rejected_count / len(group),
                "model_pass_rate": (len(group) - rejected_count) / len(group),
                "source_alerts": sum(source_alert),
                "source_anomalous": sum(source_anomalous),
                "false_alerts_intercepted": intercepted,
                "false_alert_interception_rate": (
                    float(intercepted / false_alerts) if false_alerts else None
                ),
                "guarded_auto_anomalous_rate": sum(
                    not reject and anomalous
                    for reject, anomalous in zip(rejected, source_anomalous, strict=True)
                )
                / len(group),
                "guarded_actionable_rate": sum(
                    reject or alert for reject, alert in zip(rejected, source_alert, strict=True)
                )
                / len(group),
            }
        )
    return output


def _macro_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    def rate(*, label: str, shifted: bool) -> float:
        selected = [
            row
            for row in rows
            if row["label"] == label and (row["condition"] != "clean") == shifted
        ]
        total = sum(int(cast(int | float, row["samples"])) for row in selected)
        rejected = sum(int(cast(int | float, row["quality_rejections"])) for row in selected)
        return rejected / total if total else 0.0

    shifted_normals = [
        row for row in rows if row["label"] == "normal" and row["condition"] != "clean"
    ]
    false_alerts = sum(int(cast(int | float, row["source_alerts"])) for row in shifted_normals)
    intercepted = sum(
        int(cast(int | float, row["false_alerts_intercepted"])) for row in shifted_normals
    )
    clean_anomalies = [
        row for row in rows if row["label"] == "anomaly" and row["condition"] == "clean"
    ]
    clean_anomaly_count = sum(int(cast(int | float, row["samples"])) for row in clean_anomalies)

    def weighted_clean_anomaly(metric: str) -> float:
        return (
            sum(
                float(cast(int | float, row[metric])) * int(cast(int | float, row["samples"]))
                for row in clean_anomalies
            )
            / clean_anomaly_count
        )

    source_anomalous = sum(
        int(cast(int | float, row["source_anomalous"])) for row in clean_anomalies
    )
    return {
        "clean_normal_rejection_rate": rate(label="normal", shifted=False),
        "shifted_normal_rejection_rate": rate(label="normal", shifted=True),
        "clean_anomaly_rejection_rate": rate(label="anomaly", shifted=False),
        "shifted_anomaly_rejection_rate": rate(label="anomaly", shifted=True),
        "shifted_normal_false_alerts": false_alerts,
        "shifted_normal_false_alerts_intercepted": intercepted,
        "shifted_normal_false_alert_interception_rate": (
            intercepted / false_alerts if false_alerts else None
        ),
        "clean_anomaly_source_anomalous_rate": source_anomalous / clean_anomaly_count,
        "clean_anomaly_guarded_auto_anomalous_rate": weighted_clean_anomaly(
            "guarded_auto_anomalous_rate"
        ),
        "clean_anomaly_guarded_actionable_rate": weighted_clean_anomaly("guarded_actionable_rate"),
    }


def run_m8(
    *,
    project_root: Path,
    experiment_config: Path,
    data_config: Path,
    categories: Sequence[str] | None,
    quick: bool,
    source_m5_run: Path | None = None,
    resume_run: Path | None = None,
) -> M8RunResult:
    """Fit, evaluate, and serialize the pre-inference quality gate."""

    load_dotenv(project_root / ".env")
    settings = load_m8_settings(experiment_config)
    data_root_raw = os.environ.get("ANOMALY_DATA_ROOT")
    artifact_root_raw = os.environ.get("ANOMALY_ARTIFACT_ROOT")
    if not data_root_raw or not artifact_root_raw:
        raise M8Error("ANOMALY_DATA_ROOT and ANOMALY_ARTIFACT_ROOT must be configured")
    data_root, artifact_root = Path(data_root_raw), Path(artifact_root_raw)
    manifest_dir = data_root / "manifests" / "visa-20220922"
    image_root = data_root / "sources" / "visa-20220922"
    manifest_path, audit_path = manifest_dir / "manifest.jsonl", manifest_dir / "audit.json"
    manifest_hash, audit_hash = sha256_file(manifest_path), sha256_file(audit_path)
    records = load_trusted_manifest(manifest_path, audit_path, data_config, image_root)
    all_samples = _samples(records, image_root)
    available = sorted({sample.category for sample in all_samples})
    selected = list(categories) if categories else available
    if len(selected) != len(set(selected)):
        raise M8Error("categories cannot contain duplicates")
    if unknown := sorted(set(selected) - set(available)):
        raise M8Error(f"unknown categories: {unknown}")
    m5_run = _find_m5_run(artifact_root, settings.source_experiment, source_m5_run)
    decisions, m5_metadata = _source_decisions(
        m5_run,
        settings=settings,
        manifest_hash=manifest_hash,
        audit_hash=audit_hash,
    )
    partitions = {
        category: _partition(
            [sample for sample in all_samples if sample.category == category], settings, quick
        )
        for category in selected
    }
    mode = "smoke" if quick else "full"
    resolved = {
        **asdict(settings),
        "conditions": [asdict(item) for item in settings.conditions],
        "mode": mode,
        "categories": selected,
        "source_m5_run": str(m5_run),
    }
    hashes = {
        "manifest_sha256": manifest_hash,
        "audit_sha256": audit_hash,
        "experiment_config_sha256": sha256_file(experiment_config),
        "data_config_sha256": sha256_file(data_config),
        "source_m5_metadata_sha256": sha256_file(m5_run / "metadata.json"),
        "source_m5_decisions_sha256": sha256_file(m5_run / "decisions.csv"),
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
            "source_m5": {
                "run": str(m5_run),
                "revision": cast(Mapping[str, object], m5_metadata.get("git", {})).get("revision"),
            },
        }
        category_models: dict[str, CategoryQualityModel] = {}
        prediction_rows: list[dict[str, object]] = []
        _atomic_json(run_dir / "metadata.json", metadata)
        _atomic_json(run_dir / "environment.json", _environment("cpu"))
    else:
        run_dir = resume_run.resolve(strict=True)
        try:
            run_dir.relative_to((artifact_root / "runs").resolve(strict=True))
        except ValueError as exc:
            raise M8Error("resume directory must be below the configured run root") from exc
        if yaml.safe_load((run_dir / "resolved_config.yaml").read_text("utf-8")) != resolved:
            raise M8Error("resume settings differ from the interrupted run")
        metadata = cast(
            dict[str, object], json.loads((run_dir / "metadata.json").read_text("utf-8"))
        )
        if metadata.get("status") != "running" or any(
            metadata.get(key) != value for key, value in hashes.items()
        ):
            raise M8Error("M8 resume inputs changed or run is not incomplete")
        model_path = run_dir / "quality_model.json"
        predictions_path = run_dir / "quality_predictions.csv"
        if model_path.is_file() != predictions_path.is_file():
            raise M8Error("M8 resume checkpoints are incomplete")
        if model_path.is_file():
            partial = load_quality_model(model_path)
            category_models = dict(partial.categories)
            prediction_rows = _read_csv(predictions_path)
            progress_path = run_dir / "progress.json"
            if not progress_path.is_file():
                raise M8Error("M8 resume progress is missing")
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if progress.get("quality_model_sha256") != sha256_file(model_path) or progress.get(
                "quality_predictions_sha256"
            ) != sha256_file(predictions_path):
                raise M8Error("M8 resume checkpoint failed integrity verification")
        else:
            category_models = {}
            prediction_rows = []
        resumed = metadata.get("resumed_at_utc", [])
        if not isinstance(resumed, list):
            resumed = []
        resumed.append(datetime.now(UTC).isoformat())
        metadata["resumed_at_utc"] = resumed
        _atomic_json(run_dir / "metadata.json", metadata)

    for position, category in enumerate(selected, start=1):
        if category in category_models:
            continue
        train, validation, test = partitions[category]
        reference = fit_category_quality_model(
            [
                capture_metrics(load_fitted_rgb(sample.image_path, settings.image_size))
                for sample in train
            ],
            [
                capture_metrics(load_fitted_rgb(sample.image_path, settings.image_size))
                for sample in validation
            ],
        )
        category_models[category] = reference
        current_model = CaptureQualityModel(
            schema_version=1,
            image_size=settings.image_size,
            alpha=settings.alpha,
            manifest_sha256=manifest_hash,
            categories=category_models,
        )
        prediction_rows.extend(
            _evaluate_category(
                category=category,
                test=test,
                model=current_model,
                settings=settings,
                source_decisions=decisions,
            )
        )
        _atomic_json(run_dir / "quality_model.json", quality_model_dict(current_model))
        _atomic_csv(run_dir / "quality_predictions.csv", prediction_rows)
        _atomic_json(
            run_dir / "progress.json",
            {
                "stage": "quality_evaluation",
                "completed_categories": sorted(category_models),
                "completed_count": len(category_models),
                "total_categories": len(selected),
                "quality_model_sha256": sha256_file(run_dir / "quality_model.json"),
                "quality_predictions_sha256": sha256_file(run_dir / "quality_predictions.csv"),
                "resume_command_hint": (
                    "python scripts/run_m8_quality_gate.py "
                    f'{"--quick " if quick else ""}--resume "{run_dir}"'
                ),
            },
        )
        print(f"[{position}/{len(selected)}] {category}: quality gate evaluated", flush=True)

    aggregate_rows = _aggregate(prediction_rows)
    macro = _macro_summary(aggregate_rows)
    _atomic_csv(run_dir / "quality_metrics.csv", aggregate_rows)
    summary: dict[str, object] = {
        "mode": mode,
        "seed": settings.seed,
        "categories": len(selected),
        "image_size": settings.image_size,
        "quality_alpha": settings.alpha,
        "fit_source": "training-normal only",
        "calibration_source": "training-normal-derived validation only",
        "official_test_preserved": True,
        "policy": "quality p <= alpha forces recapture before anomaly inference",
        "metrics": macro,
        "source_m5_run": str(m5_run),
        "limitations": [
            "the gate detects capture-profile departures, not every possible defect",
            "a passed capture is not a product-quality guarantee",
            "category selection must be correct for category-conditional validity",
            "rejected captures require recapture or human review and are never auto-classified",
        ],
    }
    _atomic_json(run_dir / "summary.json", summary)
    _atomic_json(
        run_dir / "progress.json",
        {
            "stage": "complete",
            "completed_categories": selected,
            "completed_count": len(selected),
            "total_categories": len(selected),
        },
    )
    metadata["status"] = "complete"
    metadata["completed_at_utc"] = datetime.now(UTC).isoformat()
    for filename in (
        "quality_model.json",
        "quality_predictions.csv",
        "quality_metrics.csv",
        "summary.json",
        "progress.json",
    ):
        path = run_dir / filename
        metadata[f"{path.stem}_sha256"] = sha256_file(path)
    _atomic_json(run_dir / "metadata.json", metadata)
    return M8RunResult(run_dir, summary)

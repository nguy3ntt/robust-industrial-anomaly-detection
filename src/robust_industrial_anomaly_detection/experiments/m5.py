"""M5 normal-only calibration and selective human-review experiment."""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from urllib.parse import urlparse

import numpy as np
import yaml
from dotenv import load_dotenv

from robust_industrial_anomaly_detection.baselines.multiscale import (
    FrozenPyramidResNet18,
    ProjectedPatchMemory,
    compose_feature_map,
    load_projected_index,
)
from robust_industrial_anomaly_detection.data.visa import load_trusted_manifest, sha256_file
from robust_industrial_anomaly_detection.evaluation.calibration import (
    binary_decisions,
    conformal_p_values,
    empirical_quantile_threshold,
    selective_decisions,
    selective_metrics,
)
from robust_industrial_anomaly_detection.evaluation.robustness import bootstrap_mean_interval
from robust_industrial_anomaly_detection.experiments.m2 import (
    Sample,
    _atomic_csv,
    _atomic_json,
    _environment,
    _git_metadata,
    _read_csv,
    _samples,
)


class M5Error(RuntimeError):
    """Raised when M5 cannot satisfy its frozen experiment contract."""


@dataclass(frozen=True)
class M5Settings:
    name: str
    seed: int
    bootstrap_samples: int
    confidence: float
    source_experiment: str
    reference_model: str
    candidate_model: str
    primary_model: str
    image_size: int
    batch_size: int
    image_score_fraction: float
    quantile: float
    anomaly_alpha: float
    review_alpha: float
    scopes: tuple[str, ...]
    review_alphas: tuple[float, ...]
    minimum_useful_coverage: float
    quick_validation: int
    quick_test_normal: int
    quick_test_anomaly: int
    quick_bootstrap_samples: int

    @property
    def models(self) -> tuple[str, str, str]:
        return self.reference_model, self.candidate_model, self.primary_model


@dataclass(frozen=True)
class M5RunResult:
    run_dir: Path
    summary: dict[str, object]


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise M5Error(f"{name} must be a mapping")
    return cast(Mapping[str, object], value)


def _string(section: Mapping[str, object], name: str) -> str:
    value = section.get(name)
    if not isinstance(value, str) or not value:
        raise M5Error(f"{name} must be a non-empty string")
    return value


def _integer(section: Mapping[str, object], name: str) -> int:
    value = section.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise M5Error(f"{name} must be a positive integer")
    return value


def _number(section: Mapping[str, object], name: str) -> float:
    value = section.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise M5Error(f"{name} must be numeric")
    return float(value)


def _numbers(section: Mapping[str, object], name: str) -> tuple[float, ...]:
    value = section.get(name)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise M5Error(f"{name} must be a sequence")
    result = tuple(float(item) for item in value if isinstance(item, (int, float)))
    if len(result) != len(value):
        raise M5Error(f"{name} must contain only numbers")
    return result


def load_m5_settings(path: Path) -> M5Settings:
    """Load and strictly validate the frozen M5 protocol."""

    try:
        root = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "configuration")
    except (OSError, yaml.YAMLError) as exc:
        raise M5Error(f"cannot load M5 configuration {path}: {exc}") from exc
    experiment = _mapping(root.get("experiment"), "experiment")
    source = _mapping(root.get("source_m4"), "source_m4")
    calibration = _mapping(root.get("calibration"), "calibration")
    coverage = _mapping(root.get("coverage_risk"), "coverage_risk")
    quick = _mapping(root.get("quick"), "quick")
    scopes_raw = calibration.get("scopes")
    if not isinstance(scopes_raw, Sequence) or isinstance(scopes_raw, (str, bytes)):
        raise M5Error("calibration.scopes must be a sequence")
    scopes = tuple(item for item in scopes_raw if isinstance(item, str))
    settings = M5Settings(
        name=_string(experiment, "name"),
        seed=_integer(experiment, "seed"),
        bootstrap_samples=_integer(experiment, "bootstrap_samples"),
        confidence=_number(experiment, "confidence"),
        source_experiment=_string(source, "experiment_name"),
        reference_model=_string(source, "reference_model"),
        candidate_model=_string(source, "candidate_model"),
        primary_model=_string(source, "primary_model"),
        image_size=_integer(source, "image_size"),
        batch_size=_integer(source, "batch_size"),
        image_score_fraction=_number(source, "image_score_fraction"),
        quantile=_number(calibration, "quantile"),
        anomaly_alpha=_number(calibration, "anomaly_alpha"),
        review_alpha=_number(calibration, "review_alpha"),
        scopes=scopes,
        review_alphas=_numbers(coverage, "review_alphas"),
        minimum_useful_coverage=_number(coverage, "minimum_useful_coverage"),
        quick_validation=_integer(quick, "validation_per_category"),
        quick_test_normal=_integer(quick, "test_normal_per_category"),
        quick_test_anomaly=_integer(quick, "test_anomaly_per_category"),
        quick_bootstrap_samples=_integer(quick, "bootstrap_samples"),
    )
    probabilities = (
        settings.confidence,
        settings.image_score_fraction,
        settings.quantile,
        settings.anomaly_alpha,
        settings.review_alpha,
        settings.minimum_useful_coverage,
        *settings.review_alphas,
    )
    if any(not 0.0 < value < 1.0 for value in probabilities):
        raise M5Error("all probability settings must be strictly between zero and one")
    if settings.scopes != ("category", "pooled"):
        raise M5Error("calibration.scopes must be exactly [category, pooled]")
    if not settings.anomaly_alpha <= settings.review_alpha:
        raise M5Error("anomaly_alpha cannot exceed review_alpha")
    if sorted(settings.review_alphas) != list(settings.review_alphas):
        raise M5Error("coverage-risk review levels must be sorted")
    if settings.anomaly_alpha not in settings.review_alphas:
        raise M5Error("coverage-risk levels must include the no-review anomaly alpha")
    if len(set(settings.models)) != 3:
        raise M5Error("source model identifiers must be distinct")
    return settings


def _stable_seed(seed: int, key: str) -> int:
    return seed + int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 1_000_000


def _find_source_run(artifact_root: Path, experiment: str, explicit: Path | None) -> Path:
    runs_root = (artifact_root / "runs").resolve(strict=True)
    if explicit is not None:
        candidates = [explicit.resolve(strict=True)]
        try:
            candidates[0].relative_to(runs_root)
        except ValueError as exc:
            raise M5Error("source M4 run must be below the configured run root") from exc
    else:
        candidates = sorted(runs_root.glob(f"{experiment}-full-*"), reverse=True)
    for candidate in candidates:
        try:
            metadata = json.loads((candidate / "metadata.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if metadata.get("status") == "complete" and metadata.get("mode") == "full":
            return candidate
    raise M5Error("no complete full M4 source run is available")


def _verify_entry(root: Path, relative: str, manifest: Mapping[str, object]) -> None:
    raw = manifest.get(relative)
    if not isinstance(raw, Mapping):
        raise M5Error(f"source index manifest has no entry for {relative}")
    path = root / relative
    if (
        not path.is_file()
        or path.stat().st_size != raw.get("bytes")
        or sha256_file(path) != raw.get("sha256")
    ):
        raise M5Error(f"source index failed integrity verification: {relative}")


def _verify_sources(
    source_m4: Path,
    *,
    manifest_hash: str,
    audit_hash: str,
    categories: Sequence[str],
    settings: M5Settings,
) -> tuple[Path, list[dict[str, object]], dict[str, object]]:
    try:
        metadata = cast(
            dict[str, object], json.loads((source_m4 / "metadata.json").read_text("utf-8"))
        )
        summary = cast(
            dict[str, object], json.loads((source_m4 / "summary.json").read_text("utf-8"))
        )
        index_manifest = cast(
            dict[str, object],
            json.loads((source_m4 / "index_manifest.json").read_text("utf-8")),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise M5Error(f"cannot read M4 source artifacts: {exc}") from exc
    if metadata.get("status") != "complete" or metadata.get("mode") != "full":
        raise M5Error("M4 source must be a complete full run")
    if (
        metadata.get("manifest_sha256") != manifest_hash
        or metadata.get("audit_sha256") != audit_hash
    ):
        raise M5Error("M4 source used different trusted data inputs")
    predictions_path = source_m4 / "predictions.csv"
    if sha256_file(predictions_path) != metadata.get("predictions_sha256"):
        raise M5Error("M4 predictions failed integrity verification")
    if sha256_file(source_m4 / "index_manifest.json") != metadata.get("index_manifest_sha256"):
        raise M5Error("M4 index manifest failed integrity verification")
    for category in categories:
        _verify_entry(source_m4, f"indexes/intervention/{category}.npz", index_manifest)
    source_m3_raw = summary.get("source_m3_run")
    if not isinstance(source_m3_raw, str):
        raise M5Error("M4 summary does not identify its M3 source")
    source_m3 = Path(source_m3_raw).resolve(strict=True)
    try:
        m3_metadata = json.loads((source_m3 / "metadata.json").read_text("utf-8"))
        m3_manifest = json.loads((source_m3 / "index_manifest.json").read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise M5Error(f"cannot read M3 source artifacts: {exc}") from exc
    if m3_metadata.get("status") != "complete" or m3_metadata.get("mode") != "full":
        raise M5Error("M3 source must be a complete full run")
    if (
        m3_metadata.get("manifest_sha256") != manifest_hash
        or m3_metadata.get("audit_sha256") != audit_hash
    ):
        raise M5Error("M3 source used different trusted data inputs")
    if sha256_file(source_m3 / "metadata.json") != metadata.get(
        "source_m3_metadata_sha256"
    ) or sha256_file(source_m3 / "index_manifest.json") != metadata.get(
        "source_m3_index_manifest_sha256"
    ):
        raise M5Error("M3 source no longer matches the provenance frozen by M4")
    for model in (settings.reference_model, settings.candidate_model):
        for category in categories:
            _verify_entry(
                source_m3,
                f"indexes/category_specific/{model}/{category}.npz",
                cast(Mapping[str, object], m3_manifest),
            )
    predictions = _read_csv(predictions_path)
    return source_m3, predictions, metadata


def _memory(path: Path, settings: M5Settings) -> ProjectedPatchMemory:
    bank, projection = load_projected_index(path)
    result = ProjectedPatchMemory(
        seed=settings.seed,
        bank_size=len(bank),
        projected_dimension=projection.shape[1],
        patches_per_image=1,
        candidate_limit=len(bank),
        image_score_fraction=settings.image_score_fraction,
        selection="kcenter",
    )
    result.fit_projected(bank, projection)
    return result


def _validation_and_test(
    samples: Sequence[Sample], settings: M5Settings, quick: bool
) -> tuple[list[Sample], list[Sample]]:
    validation = [sample for sample in samples if sample.split == "validation"]
    test = [sample for sample in samples if sample.split == "test"]
    if quick:
        validation = validation[: settings.quick_validation]
        test = [sample for sample in test if sample.label == "normal"][
            : settings.quick_test_normal
        ] + [sample for sample in test if sample.label == "anomaly"][: settings.quick_test_anomaly]
    if not validation or any(sample.label != "normal" for sample in validation):
        raise M5Error("calibration requires non-empty normal-only validation data")
    if {sample.label for sample in test} != {"normal", "anomaly"}:
        raise M5Error("evaluation requires both normal and anomalous official-test samples")
    return validation, test


def _calibration_for_category(
    *,
    category: str,
    validation: Sequence[Sample],
    source_m3: Path,
    source_m4: Path,
    extractor: FrozenPyramidResNet18,
    settings: M5Settings,
) -> list[dict[str, object]]:
    pyramids = extractor.extract([sample.image_path for sample in validation])
    mid = tuple(compose_feature_map(item, "mid") for item in pyramids)
    multiscale = tuple(compose_feature_map(item, "multiscale") for item in pyramids)
    rows: list[dict[str, object]] = []
    for model in settings.models:
        if model == settings.primary_model:
            path = source_m4 / "indexes" / "intervention" / f"{category}.npz"
        else:
            path = source_m3 / "indexes" / "category_specific" / model / f"{category}.npz"
        features = mid if model == settings.reference_model else multiscale
        scores, _ = _memory(path, settings).score(features)
        rows.extend(
            {
                "model": model,
                "category": category,
                "sample_id": sample.sample_id,
                "split": "validation",
                "label": "normal",
                "score": float(score),
            }
            for sample, score in zip(validation, scores, strict=True)
        )
    return rows


def _source_prediction_subset(
    rows: Sequence[Mapping[str, object]],
    *,
    categories: Sequence[str],
    models: Sequence[str],
    selected_test_ids: Mapping[str, set[str]],
) -> list[dict[str, object]]:
    category_set, model_set = set(categories), set(models)
    selected = [
        dict(row)
        for row in rows
        if row.get("category") in category_set
        and row.get("model") in model_set
        and row.get("sample_id") in selected_test_ids[cast(str, row.get("category"))]
    ]
    seen: set[tuple[object, object, object, object]] = set()
    for row in selected:
        key = (row.get("model"), row.get("category"), row.get("condition"), row.get("sample_id"))
        if key in seen:
            raise M5Error(f"duplicate M4 prediction key: {key}")
        seen.add(key)
    if not selected:
        raise M5Error("no M4 predictions match the selected evaluation samples")
    conditions = {cast(str, row["condition"]) for row in selected}
    if len(conditions) != 13 or "clean" not in conditions:
        raise M5Error("M4 predictions do not contain the expected clean plus 12 shifts")
    for model in models:
        for category in categories:
            for condition in conditions:
                actual = {
                    cast(str, row["sample_id"])
                    for row in selected
                    if row["model"] == model
                    and row["category"] == category
                    and row["condition"] == condition
                }
                if actual != selected_test_ids[category]:
                    raise M5Error(
                        f"M4 prediction coverage mismatch for {model}/{category}/{condition}"
                    )
    return selected


def _group_calibration(
    rows: Sequence[Mapping[str, object]], settings: M5Settings
) -> dict[tuple[str, str, str], np.ndarray]:
    result: dict[tuple[str, str, str], np.ndarray] = {}
    for model in settings.models:
        pooled = [float(cast(int | float, row["score"])) for row in rows if row["model"] == model]
        result[(model, "pooled", "__pooled__")] = np.asarray(pooled, dtype=np.float64)
        categories = sorted({cast(str, row["category"]) for row in rows if row["model"] == model})
        for category in categories:
            scores = [
                float(cast(int | float, row["score"]))
                for row in rows
                if row["model"] == model and row["category"] == category
            ]
            result[(model, "category", category)] = np.asarray(scores, dtype=np.float64)
    return result


def _add_clean_deltas(rows: list[dict[str, object]]) -> None:
    metrics = ("normal_fpr", "review_rate", "auto_coverage", "selective_risk", "anomaly_miss_rate")
    clean = {
        (row["model"], row["scope"], row["method"], row["category"]): row
        for row in rows
        if row["condition"] == "clean"
    }
    for row in rows:
        baseline = clean[(row["model"], row["scope"], row["method"], row["category"])]
        for metric in metrics:
            row[f"delta_{metric}"] = float(cast(int | float, row[metric])) - float(
                cast(int | float, baseline[metric])
            )


def _evaluate(
    source_predictions: Sequence[Mapping[str, object]],
    calibration_rows: Sequence[Mapping[str, object]],
    settings: M5Settings,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    calibration = _group_calibration(calibration_rows, settings)
    parameters: list[dict[str, object]] = []
    for (model, scope, category), scores in sorted(calibration.items()):
        parameters.append(
            {
                "model": model,
                "scope": scope,
                "category": category,
                "calibration_samples": len(scores),
                "quantile": settings.quantile,
                "quantile_threshold": empirical_quantile_threshold(scores, settings.quantile),
                "anomaly_alpha": settings.anomaly_alpha,
                "review_alpha": settings.review_alpha,
                "minimum_conformal_p": 1.0 / (len(scores) + 1.0),
                "exchangeability_scope": scope,
            }
        )
    parameter_index = {(row["model"], row["scope"], row["category"]): row for row in parameters}

    grouped: dict[tuple[str, str, str, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in source_predictions:
        grouped[
            (
                cast(str, row["model"]),
                cast(str, row["category"]),
                cast(str, row["condition"]),
                int(cast(int | float, row["severity"])),
            )
        ].append(row)
    operating: list[dict[str, object]] = []
    decisions_output: list[dict[str, object]] = []
    coverage_risk: list[dict[str, object]] = []
    for (model, category, condition, severity), group in sorted(grouped.items()):
        group = sorted(group, key=lambda row: cast(str, row["sample_id"]))
        scores = np.asarray([float(cast(int | float, row["score"])) for row in group])
        labels = np.asarray([int(cast(int | float, row["target"])) for row in group])
        family = cast(str, group[0]["family"])
        for scope in settings.scopes:
            calibration_category = category if scope == "category" else "__pooled__"
            calibration_scores = calibration[(model, scope, calibration_category)]
            parameter = parameter_index[(model, scope, calibration_category)]
            p_values = conformal_p_values(calibration_scores, scores)
            methods = {
                "empirical_quantile": (
                    binary_decisions(
                        scores,
                        float(cast(int | float, parameter["quantile_threshold"])),
                    ),
                    np.full(scores.shape, np.nan),
                ),
                "split_conformal": (
                    selective_decisions(
                        p_values,
                        anomaly_alpha=settings.anomaly_alpha,
                        review_alpha=settings.review_alpha,
                    ),
                    p_values,
                ),
            }
            for method, (actions, method_p_values) in methods.items():
                metrics = selective_metrics(labels, actions, confidence=settings.confidence)
                operating.append(
                    {
                        "model": model,
                        "scope": scope,
                        "method": method,
                        "category": category,
                        "condition": condition,
                        "family": family,
                        "severity": severity,
                        "target_false_alarm_rate": 1.0 - settings.quantile
                        if method == "empirical_quantile"
                        else settings.anomaly_alpha,
                        "anomaly_alpha": settings.anomaly_alpha,
                        "review_alpha": 0.0
                        if method == "empirical_quantile"
                        else settings.review_alpha,
                        **metrics,
                    }
                )
                decisions_output.extend(
                    {
                        "model": model,
                        "scope": scope,
                        "method": method,
                        "category": category,
                        "condition": condition,
                        "family": family,
                        "severity": severity,
                        "sample_id": source["sample_id"],
                        "label": source["label"],
                        "target": int(cast(int | float, source["target"])),
                        "score": float(score),
                        "conformal_p": float(p_value) if np.isfinite(p_value) else "",
                        "decision": str(action),
                    }
                    for source, score, p_value, action in zip(
                        group, scores, method_p_values, actions, strict=True
                    )
                )
            for review_alpha in settings.review_alphas:
                actions = selective_decisions(
                    p_values,
                    anomaly_alpha=settings.anomaly_alpha,
                    review_alpha=review_alpha,
                )
                coverage_risk.append(
                    {
                        "model": model,
                        "scope": scope,
                        "category": category,
                        "condition": condition,
                        "family": family,
                        "severity": severity,
                        "anomaly_alpha": settings.anomaly_alpha,
                        "review_alpha": review_alpha,
                        **selective_metrics(labels, actions, confidence=settings.confidence),
                    }
                )
    _add_clean_deltas(operating)
    return parameters, decisions_output, operating, coverage_risk


def _macro_rows(
    rows: Sequence[Mapping[str, object]], settings: M5Settings, bootstrap_samples: int
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str, str, int], list[Mapping[str, object]]] = defaultdict(
        list
    )
    for row in rows:
        grouped[
            (
                cast(str, row["model"]),
                cast(str, row["scope"]),
                cast(str, row["method"]),
                cast(str, row["condition"]),
                cast(str, row["family"]),
                int(cast(int | float, row["severity"])),
            )
        ].append(row)
    output: list[dict[str, object]] = []
    for key, group in sorted(grouped.items()):
        result: dict[str, object] = {
            "model": key[0],
            "scope": key[1],
            "method": key[2],
            "condition": key[3],
            "family": key[4],
            "severity": key[5],
            "categories": len(group),
        }
        for metric in (
            "normal_fpr",
            "review_rate",
            "auto_coverage",
            "selective_risk",
            "anomaly_confident_recall",
            "anomaly_miss_rate",
            "delta_normal_fpr",
            "delta_selective_risk",
        ):
            values = [
                value
                for row in group
                if np.isfinite(value := float(cast(int | float, row[metric])))
            ]
            result[f"{metric}_defined_categories"] = len(values)
            if not values:
                result[f"macro_{metric}"] = float("nan")
                result[f"macro_{metric}_ci_low"] = float("nan")
                result[f"macro_{metric}_ci_high"] = float("nan")
                continue
            low, high = bootstrap_mean_interval(
                values,
                seed=_stable_seed(settings.seed, f"macro:{key}:{metric}"),
                samples=bootstrap_samples,
                confidence=settings.confidence,
            )
            result[f"macro_{metric}"] = float(np.mean(values))
            result[f"macro_{metric}_ci_low"] = low
            result[f"macro_{metric}_ci_high"] = high
        output.append(result)
    return output


def _finite_mean(group: Sequence[Mapping[str, object]], metric: str) -> float:
    values = [value for row in group if np.isfinite(value := float(cast(int | float, row[metric])))]
    return float(np.mean(values)) if values else float("nan")


def _family_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                cast(str, row["model"]),
                cast(str, row["scope"]),
                cast(str, row["method"]),
                cast(str, row["family"]),
                int(cast(int | float, row["severity"])),
            )
        ].append(row)
    return [
        {
            "model": key[0],
            "scope": key[1],
            "method": key[2],
            "family": key[3],
            "severity": key[4],
            "category_conditions": len(group),
            **{
                f"macro_{metric}": _finite_mean(group, metric)
                for metric in (
                    "normal_fpr",
                    "review_rate",
                    "auto_coverage",
                    "selective_risk",
                    "anomaly_miss_rate",
                )
            },
        }
        for key, group in sorted(grouped.items())
    ]


def _hypotheses(
    operating: Sequence[Mapping[str, object]],
    coverage: Sequence[Mapping[str, object]],
    settings: M5Settings,
    bootstrap_samples: int,
) -> list[dict[str, object]]:
    clean = [
        row
        for row in operating
        if row["model"] == settings.primary_model
        and row["method"] == "split_conformal"
        and row["condition"] == "clean"
    ]
    by_category = {(cast(str, row["scope"]), cast(str, row["category"])): row for row in clean}
    h3_values = []
    for category in sorted({cast(str, row["category"]) for row in clean}):
        category_error = abs(
            float(cast(int | float, by_category[("category", category)]["normal_fpr"]))
            - settings.anomaly_alpha
        )
        pooled_error = abs(
            float(cast(int | float, by_category[("pooled", category)]["normal_fpr"]))
            - settings.anomaly_alpha
        )
        h3_values.append(pooled_error - category_error)
    h3_low, h3_high = bootstrap_mean_interval(
        h3_values,
        seed=_stable_seed(settings.seed, "H3"),
        samples=bootstrap_samples,
        confidence=settings.confidence,
    )

    selected_coverage = [
        row
        for row in coverage
        if row["model"] == settings.primary_model
        and row["scope"] == "category"
        and float(cast(int | float, row["review_alpha"]))
        in {settings.anomaly_alpha, settings.review_alpha}
    ]
    coverage_index = {
        (
            cast(str, row["category"]),
            cast(str, row["condition"]),
            float(cast(int | float, row["review_alpha"])),
        ): row
        for row in selected_coverage
    }
    unit_keys = sorted({(key[0], key[1]) for key in coverage_index})
    h4_values: list[float] = []
    for category, condition in unit_keys:
        no_review_risk = float(
            cast(
                int | float,
                coverage_index[(category, condition, settings.anomaly_alpha)]["selective_risk"],
            )
        )
        triage_risk = float(
            cast(
                int | float,
                coverage_index[(category, condition, settings.review_alpha)]["selective_risk"],
            )
        )
        if np.isfinite(no_review_risk) and np.isfinite(triage_risk):
            h4_values.append(no_review_risk - triage_risk)
    triage_coverage = [
        float(
            cast(
                int | float,
                coverage_index[(category, condition, settings.review_alpha)]["auto_coverage"],
            )
        )
        for category, condition in unit_keys
    ]
    h4_low, h4_high = (
        bootstrap_mean_interval(
            h4_values,
            seed=_stable_seed(settings.seed, "H4"),
            samples=bootstrap_samples,
            confidence=settings.confidence,
        )
        if h4_values
        else (float("nan"), float("nan"))
    )
    h4_coverage = float(np.mean(triage_coverage))
    return [
        {
            "test": "H3_category_conditional_false_alarm_alignment",
            "estimate": float(np.mean(h3_values)),
            "ci_low": h3_low,
            "ci_high": h3_high,
            "confidence": settings.confidence,
            "units": len(h3_values),
            "positive_direction": "category calibration is closer to the clean false-alarm target",
            "conclusion": "supported" if h3_low > 0.0 else "inconclusive",
        },
        {
            "test": "H4_selective_review_reduces_automatic_risk",
            "estimate": float(np.mean(h4_values)),
            "ci_low": h4_low,
            "ci_high": h4_high,
            "confidence": settings.confidence,
            "units": len(h4_values),
            "undefined_risk_units": len(unit_keys) - len(h4_values),
            "mean_auto_coverage": h4_coverage,
            "minimum_useful_coverage": settings.minimum_useful_coverage,
            "positive_direction": "review lowers risk among automatic decisions",
            "conclusion": "supported"
            if h4_values and h4_low > 0.0 and h4_coverage >= settings.minimum_useful_coverage
            else "inconclusive",
        },
    ]


def run_m5(
    *,
    project_root: Path,
    experiment_config: Path,
    data_config: Path,
    categories: Sequence[str] | None,
    device: str,
    quick: bool,
    source_m4_run: Path | None = None,
    resume_run: Path | None = None,
) -> M5RunResult:
    """Run the frozen M5 calibration and selective-review protocol."""

    load_dotenv(project_root / ".env")
    settings = load_m5_settings(experiment_config)
    data_root_raw = os.environ.get("ANOMALY_DATA_ROOT")
    artifact_root_raw = os.environ.get("ANOMALY_ARTIFACT_ROOT")
    if not data_root_raw or not artifact_root_raw:
        raise M5Error("ANOMALY_DATA_ROOT and ANOMALY_ARTIFACT_ROOT must be configured")
    data_root, artifact_root = Path(data_root_raw), Path(artifact_root_raw)
    manifest_dir = data_root / "manifests" / "visa-20220922"
    image_root = data_root / "sources" / "visa-20220922"
    manifest_path, audit_path = manifest_dir / "manifest.jsonl", manifest_dir / "audit.json"
    manifest_hash, audit_hash = sha256_file(manifest_path), sha256_file(audit_path)
    records = load_trusted_manifest(manifest_path, audit_path, data_config, image_root)
    all_samples = _samples(records, image_root)
    available_categories = sorted({sample.category for sample in all_samples})
    selected_categories = list(categories) if categories else available_categories
    if unknown := sorted(set(selected_categories) - set(available_categories)):
        raise M5Error(f"unknown categories: {unknown}")
    source_m4 = _find_source_run(artifact_root, settings.source_experiment, source_m4_run)
    source_m3, source_predictions, source_metadata = _verify_sources(
        source_m4,
        manifest_hash=manifest_hash,
        audit_hash=audit_hash,
        categories=selected_categories,
        settings=settings,
    )
    partitions = {
        category: _validation_and_test(
            [sample for sample in all_samples if sample.category == category], settings, quick
        )
        for category in selected_categories
    }
    selected_test_ids = {
        category: {sample.sample_id for sample in partitions[category][1]}
        for category in selected_categories
    }
    selected_predictions = _source_prediction_subset(
        source_predictions,
        categories=selected_categories,
        models=settings.models,
        selected_test_ids=selected_test_ids,
    )
    mode = "smoke" if quick else "full"
    bootstrap_samples = settings.quick_bootstrap_samples if quick else settings.bootstrap_samples
    resolved = {
        **asdict(settings),
        "scopes": list(settings.scopes),
        "review_alphas": list(settings.review_alphas),
        "mode": mode,
        "categories": selected_categories,
        "device": device,
        "source_m4_run": str(source_m4),
        "source_m3_run": str(source_m3),
    }
    hashes = {
        "manifest_sha256": manifest_hash,
        "audit_sha256": audit_hash,
        "experiment_config_sha256": sha256_file(experiment_config),
        "data_config_sha256": sha256_file(data_config),
        "source_m4_metadata_sha256": sha256_file(source_m4 / "metadata.json"),
        "source_m4_predictions_sha256": sha256_file(source_m4 / "predictions.csv"),
        "source_m4_index_manifest_sha256": sha256_file(source_m4 / "index_manifest.json"),
        "source_m3_index_manifest_sha256": sha256_file(source_m3 / "index_manifest.json"),
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
            "source_m4": {
                "run": str(source_m4),
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
        calibration_rows: list[dict[str, object]] = []
        _atomic_json(run_dir / "metadata.json", metadata)
        _atomic_json(run_dir / "environment.json", _environment(device))
    else:
        run_dir = resume_run.resolve(strict=True)
        try:
            run_dir.relative_to((artifact_root / "runs").resolve(strict=True))
        except ValueError as exc:
            raise M5Error("resume directory must be below the configured run root") from exc
        if yaml.safe_load((run_dir / "resolved_config.yaml").read_text("utf-8")) != resolved:
            raise M5Error("resume settings differ from the interrupted run")
        metadata = cast(
            dict[str, object], json.loads((run_dir / "metadata.json").read_text("utf-8"))
        )
        if metadata.get("status") != "running" or any(
            metadata.get(key) != value for key, value in hashes.items()
        ):
            raise M5Error("M5 resume inputs changed or run is not incomplete")
        calibration_rows = _read_csv(run_dir / "calibration_scores.csv")
        progress_path = run_dir / "progress.json"
        if progress_path.is_file():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            checkpoint_hash = progress.get("calibration_scores_sha256")
            if checkpoint_hash and checkpoint_hash != sha256_file(
                run_dir / "calibration_scores.csv"
            ):
                raise M5Error("calibration-score checkpoint failed integrity verification")
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
        raise M5Error(f"model checkpoint is unavailable: {checkpoint}")
    metadata["checkpoint_sha256"] = sha256_file(checkpoint)
    metadata["checkpoint_bytes"] = checkpoint.stat().st_size
    _atomic_json(run_dir / "metadata.json", metadata)

    completed = {
        category
        for category in selected_categories
        if all(
            sum(
                row.get("category") == category and row.get("model") == model
                for row in calibration_rows
            )
            == len(partitions[category][0])
            for model in settings.models
        )
    }
    for position, category in enumerate(selected_categories, start=1):
        if category in completed:
            continue
        calibration_rows = [row for row in calibration_rows if row.get("category") != category]
        calibration_rows.extend(
            _calibration_for_category(
                category=category,
                validation=partitions[category][0],
                source_m3=source_m3,
                source_m4=source_m4,
                extractor=extractor,
                settings=settings,
            )
        )
        _atomic_csv(run_dir / "calibration_scores.csv", calibration_rows)
        completed.add(category)
        calibration_checkpoint_hash = sha256_file(run_dir / "calibration_scores.csv")
        _atomic_json(
            run_dir / "progress.json",
            {
                "stage": "calibration_scores",
                "completed_categories": sorted(completed),
                "completed_count": len(completed),
                "total_categories": len(selected_categories),
                "calibration_scores_sha256": calibration_checkpoint_hash,
                "resume_command_hint": (
                    "python scripts/run_m5_calibration.py "
                    f'{"--quick " if quick else ""}--resume "{run_dir}"'
                ),
            },
        )
        print(
            f"[{position}/{len(selected_categories)}] {category}: calibration scores saved",
            flush=True,
        )

    parameters, decision_rows, operating_rows, coverage_rows = _evaluate(
        selected_predictions, calibration_rows, settings
    )
    summary_rows = _macro_rows(operating_rows, settings, bootstrap_samples)
    family_rows = _family_rows(operating_rows)
    category_rows = [row for row in operating_rows if row["condition"] == "clean"]
    hypothesis_rows = _hypotheses(operating_rows, coverage_rows, settings, bootstrap_samples)
    _atomic_csv(run_dir / "calibration_parameters.csv", parameters)
    _atomic_csv(run_dir / "decisions.csv", decision_rows)
    _atomic_csv(run_dir / "operating_metrics.csv", operating_rows)
    _atomic_csv(run_dir / "coverage_risk.csv", coverage_rows)
    _atomic_csv(run_dir / "calibration_summary.csv", summary_rows)
    _atomic_csv(run_dir / "family_summary.csv", family_rows)
    _atomic_csv(run_dir / "category_conditional.csv", category_rows)
    _atomic_csv(run_dir / "hypothesis_tests.csv", hypothesis_rows)
    summary: dict[str, object] = {
        "mode": mode,
        "seed": settings.seed,
        "categories": len(selected_categories),
        "models": list(settings.models),
        "primary_model": settings.primary_model,
        "calibration_source": "training-normal-derived validation only",
        "official_test_preserved": True,
        "policy": {
            "anomalous": f"conformal p <= {settings.anomaly_alpha}",
            "review": f"{settings.anomaly_alpha} < conformal p <= {settings.review_alpha}",
            "normal": f"conformal p > {settings.review_alpha}",
        },
        "hypotheses": hypothesis_rows,
        "source_m4_run": str(source_m4),
        "assumptions": [
            "clean false-alarm validity is marginal and requires exchangeability "
            "with calibration normals",
            "category-conditional validity additionally requires the category label "
            "to be known and stable",
            "shifted inputs are not guaranteed exchangeable; their false-positive "
            "rates are empirical stress tests",
            "review decisions defer uncertainty and do not certify product quality "
            "or diagnose root cause",
        ],
    }
    _atomic_json(run_dir / "summary.json", summary)
    _atomic_json(
        run_dir / "progress.json",
        {
            "stage": "complete",
            "completed_categories": selected_categories,
            "completed_count": len(selected_categories),
            "total_categories": len(selected_categories),
        },
    )
    metadata["status"] = "complete"
    metadata["completed_at_utc"] = datetime.now(UTC).isoformat()
    for filename in (
        "calibration_scores.csv",
        "calibration_parameters.csv",
        "decisions.csv",
        "operating_metrics.csv",
        "coverage_risk.csv",
        "calibration_summary.csv",
        "family_summary.csv",
        "category_conditional.csv",
        "hypothesis_tests.csv",
        "summary.json",
        "progress.json",
    ):
        path = run_dir / filename
        metadata[f"{path.stem}_sha256"] = sha256_file(path)
    _atomic_json(run_dir / "metadata.json", metadata)
    return M5RunResult(run_dir, summary)

"""Hash-gated reconstruction of the M7 publication tables and figures."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

PRIMARY_MODEL = "multiscale_augmented_coreset_512"
MANIFEST_SHA256 = "73283c2704ed18183e5394fff699b0f8d916113393e53ba53986adaf6f32cc6e"
BG = "#0d0b0e"
SURFACE = "#171318"
GRID = "#3a2d33"
TEXT = "#f4ecef"
MUTED = "#ab949d"
RED = "#e5484d"
GREEN = "#55bd8b"
AMBER = "#e7a253"
BLUE = "#68a8d8"


class M7Error(RuntimeError):
    """Raised when a publication input or output fails an integrity contract."""


@dataclass(frozen=True)
class SourceSpec:
    name: str
    run: str
    metadata_sha256: str
    required_files: dict[str, str]


@dataclass(frozen=True)
class M7Settings:
    seed: int
    manifest_sha256: str
    output_directory: Path
    sources: tuple[SourceSpec, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expect_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise M7Error(f"{label} must be a string-keyed mapping")
    return value


def load_m7_settings(path: Path) -> M7Settings:
    """Load the frozen source-run and output contract."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _expect_mapping(raw, "M7 configuration")
    if int(root.get("schema_version", 0)) != 1:
        raise M7Error("unsupported M7 configuration schema")
    sources_raw = _expect_mapping(root.get("sources"), "sources")
    sources: list[SourceSpec] = []
    for name, source_value in sources_raw.items():
        source = _expect_mapping(source_value, f"source {name}")
        files_raw = _expect_mapping(source.get("required_files"), f"source {name} files")
        files = {filename: str(metadata_key) for filename, metadata_key in files_raw.items()}
        sources.append(
            SourceSpec(
                name=name,
                run=str(source["run"]),
                metadata_sha256=str(source["metadata_sha256"]),
                required_files=files,
            )
        )
    required_sources = {"m2", "m3", "m4", "m5", "m6"}
    if {source.name for source in sources} != required_sources:
        raise M7Error("M7 sources must contain exactly m2 through m6")
    manifest_sha256 = str(root["manifest_sha256"])
    if manifest_sha256 != MANIFEST_SHA256:
        raise M7Error("M7 must use the audited canonical manifest")
    return M7Settings(
        seed=int(root["seed"]),
        manifest_sha256=manifest_sha256,
        output_directory=Path(str(root["output_directory"])),
        sources=tuple(sources),
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return _expect_mapping(value, str(path))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _number(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        raise M7Error(f"missing numeric field {key}")
    return float(value)


def _as_float(value: object) -> float:
    if isinstance(value, (int, float, str)):
        return float(value)
    raise M7Error(f"expected a numeric value, got {type(value).__name__}")


def _as_int(value: object) -> int:
    if isinstance(value, (int, float, str)):
        return int(value)
    raise M7Error(f"expected an integer value, got {type(value).__name__}")


def _verify_sources(
    artifact_root: Path, settings: M7Settings
) -> tuple[dict[str, Path], dict[str, Any]]:
    runs_root = artifact_root / "runs"
    paths: dict[str, Path] = {}
    record: dict[str, Any] = {}
    for source in settings.sources:
        run = runs_root / source.run
        metadata_path = run / "metadata.json"
        if not metadata_path.is_file():
            raise M7Error(f"missing source metadata: {metadata_path}")
        observed_metadata_hash = _sha256(metadata_path)
        if observed_metadata_hash != source.metadata_sha256:
            raise M7Error(f"source metadata hash mismatch for {source.name}")
        metadata = _read_json(metadata_path)
        if metadata.get("status") != "complete":
            raise M7Error(f"source run is not complete: {source.run}")
        if metadata.get("manifest_sha256") != settings.manifest_sha256:
            raise M7Error(f"source run manifest mismatch: {source.run}")
        file_records: dict[str, Any] = {}
        for filename, metadata_key in source.required_files.items():
            source_file = run / filename
            expected = metadata.get(metadata_key)
            if not source_file.is_file() or not isinstance(expected, str):
                raise M7Error(f"source contract missing {filename} / {metadata_key}")
            observed = _sha256(source_file)
            if observed != expected:
                raise M7Error(f"source file hash mismatch: {source.run}/{filename}")
            file_records[filename] = {"bytes": source_file.stat().st_size, "sha256": observed}
        paths[source.name] = run
        record[source.name] = {
            "run": source.run,
            "metadata_sha256": observed_metadata_hash,
            "files": file_records,
        }
    return paths, record


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _canvas(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (1600, 900), BG)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1600, 11), fill=RED)
    draw.text((88, 62), title, font=_font(46, bold=True), fill=TEXT)
    draw.text((90, 124), subtitle, font=_font(21), fill=MUTED)
    return image, draw


def _save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.png")
    image.save(temporary, format="PNG", optimize=True)
    os.replace(temporary, path)


def _clean_rows(paths: dict[str, Path]) -> list[dict[str, object]]:
    m2_summary = _read_json(paths["m2"] / "summary.json")
    m2_results = _expect_mapping(m2_summary["results"], "M2 results")
    labels = {
        "handcrafted": "Colour/texture NN",
        "global_nn": "ResNet-18 global NN",
        "patch_memory": "Mid-scale patch memory",
    }
    rows: list[dict[str, object]] = []
    for key in ("handcrafted", "global_nn", "patch_memory"):
        result = _expect_mapping(m2_results[key], key)
        rows.append(
            {
                "model": labels[key],
                "scope": "category",
                "image_auroc": result["macro_image_auroc"],
                "image_ap": result["macro_image_ap"],
                "normal_fpr": result["macro_test_normal_fpr"],
                "pixel_auroc": result.get("macro_pixel_auroc", ""),
                "pixel_ap": result.get("macro_pixel_ap", ""),
                "mean_iou": result.get("macro_pixel_iou_mean", ""),
                "aupro_30": result.get("macro_aupro_30", ""),
                "score_latency_ms": "",
                "index_bytes": "",
                "source": "M2 corrected full run",
            }
        )
    variants = {
        "compact_mid_random_512": "Mid-scale random 512",
        "multiscale_random_512": "Multi-scale random 512",
        "multiscale_bounded_random_512": "Multi-scale bounded random 512",
        "multiscale_coreset_512": "Multi-scale k-center 512",
        "multiscale_coreset_balanced": "Unified balanced k-center 1536",
    }
    for row in _read_csv(paths["m3"] / "ablation_summary.csv"):
        variant = row["variant"]
        if variant not in variants:
            continue
        rows.append(
            {
                "model": variants[variant],
                "scope": row["scope"],
                "image_auroc": row["macro_image_auroc"],
                "image_ap": row["macro_image_ap"],
                "normal_fpr": row["macro_test_normal_fpr"],
                "pixel_auroc": row["macro_pixel_auroc"],
                "pixel_ap": row["macro_pixel_ap"],
                "mean_iou": row["macro_pixel_iou_mean"],
                "aupro_30": row["macro_aupro_30"],
                "score_latency_ms": row["macro_latency_ms_per_image"],
                "index_bytes": float(row["macro_compressed_index_bytes"]) * 12,
                "source": "M3 corrected full run",
            }
        )
    clean_augmented = [
        row
        for row in _read_csv(paths["m4"] / "category_metrics.csv")
        if row["model"] == PRIMARY_MODEL and row["condition"] == "clean"
    ]
    if len(clean_augmented) != 12:
        raise M7Error("expected 12 clean augmented-category rows")

    def mean(key: str) -> float:
        return sum(_number(row, key) for row in clean_augmented) / len(clean_augmented)

    rows.append(
        {
            "model": "Augmented multi-scale k-center 512",
            "scope": "category_specific",
            "image_auroc": mean("image_auroc"),
            "image_ap": mean("image_ap"),
            "normal_fpr": mean("test_normal_fpr"),
            "pixel_auroc": mean("pixel_auroc"),
            "pixel_ap": mean("pixel_ap"),
            "mean_iou": mean("pixel_iou_mean"),
            "aupro_30": mean("aupro_30"),
            "score_latency_ms": "",
            "index_bytes": "",
            "source": "M4 full run",
        }
    )
    return rows


def _robustness_rows(paths: dict[str, Path]) -> list[dict[str, object]]:
    rows = [
        row
        for row in _read_csv(paths["m4"] / "family_summary.csv")
        if row["model"] == PRIMARY_MODEL
    ]
    order = {
        name: index
        for index, name in enumerate(
            ("jpeg", "resize", "blur", "noise", "photometric", "geometric")
        )
    }
    rows.sort(key=lambda row: order[row["family"]])
    return [
        {
            "family": row["family"],
            "delta_image_auroc": row["mean_delta_image_auroc"],
            "delta_image_auroc_ci_low": row["mean_delta_image_auroc_ci_low"],
            "delta_image_auroc_ci_high": row["mean_delta_image_auroc_ci_high"],
            "delta_normal_fpr": row["mean_delta_test_normal_fpr"],
            "delta_normal_fpr_ci_low": row["mean_delta_test_normal_fpr_ci_low"],
            "delta_normal_fpr_ci_high": row["mean_delta_test_normal_fpr_ci_high"],
            "delta_pixel_auroc": row["mean_delta_pixel_auroc"],
            "rank_spearman": row["mean_score_rank_spearman"],
            "heatmap_correlation": row["mean_heatmap_correlation"],
        }
        for row in rows
    ]


def _calibration_rows(paths: dict[str, Path]) -> list[dict[str, object]]:
    rows = [
        row
        for row in _read_csv(paths["m5"] / "calibration_summary.csv")
        if row["model"] == PRIMARY_MODEL
        and row["scope"] == "category"
        and row["method"] == "split_conformal"
    ]
    order = {
        name: index
        for index, name in enumerate(
            ("clean", "jpeg", "resize", "blur", "noise", "photometric", "geometric")
        )
    }
    rows.sort(key=lambda row: (order[row["family"]], int(row["severity"])))
    return [
        {
            "condition": row["condition"],
            "family": row["family"],
            "severity": row["severity"],
            "normal_fpr": row["macro_normal_fpr"],
            "normal_fpr_ci_low": row["macro_normal_fpr_ci_low"],
            "normal_fpr_ci_high": row["macro_normal_fpr_ci_high"],
            "review_rate": row["macro_review_rate"],
            "auto_coverage": row["macro_auto_coverage"],
            "selective_risk": row["macro_selective_risk"],
            "anomaly_miss_rate": row["macro_anomaly_miss_rate"],
        }
        for row in rows
    ]


def _coverage_rows(paths: dict[str, Path]) -> list[dict[str, object]]:
    grouped: dict[float, list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(paths["m5"] / "coverage_risk.csv"):
        if (
            row["model"] == PRIMARY_MODEL
            and row["scope"] == "category"
            and row["condition"] == "clean"
        ):
            grouped[float(row["review_alpha"])].append(row)
    output: list[dict[str, object]] = []
    for alpha, rows in sorted(grouped.items()):
        if len(rows) != 12:
            raise M7Error(f"expected 12 clean coverage rows at review alpha {alpha}")
        output.append(
            {
                "review_alpha": alpha,
                "auto_coverage": sum(_number(row, "auto_coverage") for row in rows) / 12,
                "selective_risk": sum(_number(row, "selective_risk") for row in rows) / 12,
                "review_rate": sum(_number(row, "review_rate") for row in rows) / 12,
            }
        )
    return output


def _hypothesis_rows(paths: dict[str, Path]) -> list[dict[str, object]]:
    friendly = {
        "H2_multiscale_image_robustness_advantage": "H2 Multi-scale robustness",
        "augmentation_intervention_image_robustness_advantage": "Intervention robustness",
        "H5_explanation_stability_failure_separation": "H5 Explanation stability",
        "H3_category_conditional_false_alarm_alignment": "H3 Category calibration",
        "H4_selective_review_reduces_automatic_risk": "H4 Selective review",
    }
    output: list[dict[str, object]] = []
    for stage in ("m4", "m5"):
        for row in _read_csv(paths[stage] / "hypothesis_tests.csv"):
            output.append(
                {
                    "test": row["test"],
                    "label": friendly[row["test"]],
                    "estimate": row["estimate"],
                    "ci_low": row["ci_low"],
                    "ci_high": row["ci_high"],
                    "units": row["units"],
                    "conclusion": row["conclusion"],
                    "stage": stage.upper(),
                }
            )
    return output


def _error_taxonomy(
    paths: dict[str, Path],
    robustness: list[dict[str, object]],
    calibration: list[dict[str, object]],
) -> list[dict[str, object]]:
    failures = [
        row
        for row in _read_csv(paths["m4"] / "failure_clusters.csv")
        if row["model"] == PRIMARY_MODEL and row["severity"] == "2"
    ]
    counts: dict[str, tuple[int, int]] = {}
    for family in {row["family"] for row in failures}:
        selected = [row for row in failures if row["family"] == family]
        counts[family] = (
            sum(int(row["new_false_positives"]) for row in selected),
            sum(int(row["new_false_negatives"]) for row in selected),
        )
    transfer = _read_csv(paths["m4"] / "category_transfer.csv")
    transfer_auroc = sum(_number(row, "image_auroc") for row in transfer) / len(transfer)
    transfer_gap = sum(
        _number(row, "delta_image_auroc_vs_category_specific") for row in transfer
    ) / len(transfer)
    robust_by_family = {str(row["family"]): row for row in robustness}
    severe_calibration = {
        str(row["family"]): row for row in calibration if str(row["severity"]) == "2"
    }
    blur_fp, _blur_fn = counts["blur"]
    noise_fp, _noise_fn = counts["noise"]
    jpeg_fp, _jpeg_fn = counts["jpeg"]
    photo_fp, photo_fn = counts["photometric"]
    severe_blur_fpr = _as_float(severe_calibration["blur"]["normal_fpr"])
    severe_noise_fpr = _as_float(severe_calibration["noise"]["normal_fpr"])
    severe_jpeg_fpr = _as_float(severe_calibration["jpeg"]["normal_fpr"])
    geometric_heatmap_correlation = _as_float(robust_by_family["geometric"]["heatmap_correlation"])
    return [
        {
            "code": "E1",
            "failure_mode": "Capture nuisance false positives",
            "evidence": (
                f"Severe blur/noise/JPEG add {blur_fp}/{noise_fp}/{jpeg_fp} false "
                f"positives; severe blur FPR {severe_blur_fpr:.3f}."
            ),
            "operational_effect": "Normal parts receive confident anomaly alerts.",
            "mitigation": (
                "Reject degraded capture, monitor score drift, and recalibrate after "
                "validated capture changes."
            ),
        },
        {
            "code": "E2",
            "failure_mode": "Photometric anomaly misses",
            "evidence": (
                f"Severe photometric shift creates {photo_fn} new false negatives "
                f"versus {photo_fp} false positives."
            ),
            "operational_effect": "Darkened defects can move toward the normal score region.",
            "mitigation": (
                "Control illumination and retain human review for low-quality acquisitions."
            ),
        },
        {
            "code": "E3",
            "failure_mode": "Spatial evidence instability",
            "evidence": (
                "Geometric-shift heatmap correlation averages "
                f"{geometric_heatmap_correlation:.3f} after inverse alignment."
            ),
            "operational_effect": "Highlighted support can move despite a similar image decision.",
            "mitigation": (
                "Treat heatmaps as supporting evidence and inspect alignment before acting."
            ),
        },
        {
            "code": "E4",
            "failure_mode": "Category transfer failure",
            "evidence": (
                f"Leave-one-category-out AUROC is {transfer_auroc:.3f}, a "
                f"{abs(transfer_gap):.3f} loss from category-specific memory."
            ),
            "operational_effect": "Unknown categories receive unreliable scores.",
            "mitigation": "Require a known category and category-specific normal memory.",
        },
        {
            "code": "E5",
            "failure_mode": "Calibration exchangeability failure",
            "evidence": (
                "Severe blur/noise/JPEG normal FPR reaches "
                f"{severe_blur_fpr:.3f}/{severe_noise_fpr:.3f}/{severe_jpeg_fpr:.3f}."
            ),
            "operational_effect": "Clean conformal false-alarm control no longer applies.",
            "mitigation": (
                "Detect shift, pause automatic use, then correct capture or recalibrate "
                "with approved normal data."
            ),
        },
        {
            "code": "E6",
            "failure_mode": "Residual thresholded localization trade-off",
            "evidence": (
                "The coreset improves ranking and AUPRO but mean IoU remains below "
                "the 2,048-entry reference."
            ),
            "operational_effect": (
                "Pixel support may be diffuse or incomplete at one frozen threshold."
            ),
            "mitigation": (
                "Review continuous heatmaps and report both threshold-free and "
                "thresholded localization metrics."
            ),
        },
    ]


def _figure_clean(rows: list[dict[str, object]], path: Path) -> None:
    wanted = [
        "Mid-scale patch memory",
        "Multi-scale k-center 512",
        "Augmented multi-scale k-center 512",
    ]
    selected = [next(row for row in rows if row["model"] == name) for name in wanted]
    image, draw = _canvas(
        "Clean detection and localization",
        "Macro means across the 12 VisA categories; higher is better",
    )
    metrics = (
        ("image_auroc", "Image AUROC", RED),
        ("pixel_auroc", "Pixel AUROC", BLUE),
        ("aupro_30", "AUPRO@30", GREEN),
    )
    left, top, width, height = 125, 230, 1360, 540
    for tick in range(6):
        value = 0.5 + tick * 0.1
        y = top + height - (value - 0.5) / 0.5 * height
        draw.line((left, y, left + width, y), fill=GRID, width=2)
        draw.text((55, y - 12), f"{value:.1f}", font=_font(18), fill=MUTED)
    group_width = width / len(selected)
    bar_width = 82
    for group_index, row in enumerate(selected):
        center = left + group_width * (group_index + 0.5)
        for metric_index, (key, _label, color) in enumerate(metrics):
            value = _as_float(row[key])
            x0 = center + (metric_index - 1) * (bar_width + 14) - bar_width / 2
            y0 = top + height - (value - 0.5) / 0.5 * height
            draw.rectangle((x0, y0, x0 + bar_width, top + height), fill=color)
            draw.text((x0 + 4, y0 - 29), f"{value:.3f}", font=_font(16, bold=True), fill=TEXT)
        label = str(row["model"]).replace(" 512", "\n512")
        draw.multiline_text(
            (center - 165, top + height + 22),
            label,
            font=_font(19, bold=True),
            fill=TEXT,
            anchor="ma",
            align="center",
            spacing=4,
        )
    legend_x = 940
    for index, (_key, label, color) in enumerate(metrics):
        x = legend_x + index * 185
        draw.rectangle((x, 160, x + 18, 178), fill=color)
        draw.text((x + 27, 157), label, font=_font(17), fill=MUTED)
    _save_image(image, path)


def _figure_robustness(rows: list[dict[str, object]], path: Path) -> None:
    image, draw = _canvas(
        "Robustness degradation by shift family",
        "Primary augmented coreset; means across two severities and 12 categories",
    )
    families = [str(row["family"]).title() for row in rows]
    values_left = [_as_float(row["delta_image_auroc"]) for row in rows]
    values_right = [_as_float(row["delta_normal_fpr"]) for row in rows]
    panels = (
        (100, "Δ image AUROC", values_left, -0.10, 0.01, RED),
        (850, "Δ normal FPR", values_right, -0.02, 0.52, AMBER),
    )
    for x, label, values, minimum, maximum, color in panels:
        draw.text((x, 190), label, font=_font(23, bold=True), fill=TEXT)
        axis_left, axis_top, axis_width = x + 175, 250, 500
        zero = axis_left + (0 - minimum) / (maximum - minimum) * axis_width
        draw.line((zero, axis_top - 12, zero, axis_top + 6 * 78), fill=MUTED, width=2)
        for index, (family, value) in enumerate(zip(families, values, strict=True)):
            y = axis_top + index * 78
            draw.text((x, y + 8), family, font=_font(18), fill=MUTED)
            endpoint = axis_left + (value - minimum) / (maximum - minimum) * axis_width
            draw.rectangle((min(zero, endpoint), y, max(zero, endpoint), y + 36), fill=color)
            anchor_x = endpoint + (8 if value >= 0 else -8)
            draw.text(
                (anchor_x, y + 18),
                f"{value:+.3f}",
                font=_font(16, bold=True),
                fill=TEXT,
                anchor="lm" if value >= 0 else "rm",
            )
    _save_image(image, path)


def _figure_calibration(rows: list[dict[str, object]], path: Path) -> None:
    image, draw = _canvas(
        "False positives under capture shift",
        "Category conformal policy; normal FPR by family and severity",
    )
    families = ("clean", "jpeg", "resize", "blur", "noise", "photometric", "geometric")
    lookup = {
        (str(row["family"]), _as_int(row["severity"])): _as_float(row["normal_fpr"]) for row in rows
    }
    left, top, width, height = 120, 235, 1370, 500
    for tick in range(6):
        value = tick * 0.2
        y = top + height - value * height
        draw.line((left, y, left + width, y), fill=GRID, width=2)
        draw.text((60, y - 11), f"{value:.1f}", font=_font(18), fill=MUTED)
    target_y = top + height - 0.05 * height
    draw.line((left, target_y, left + width, target_y), fill=GREEN, width=3)
    draw.text(
        (left + width - 3, target_y - 12),
        "clean target 0.05",
        font=_font(15, bold=True),
        fill=GREEN,
        anchor="rs",
    )
    group = width / len(families)
    for index, family in enumerate(families):
        center = left + group * (index + 0.5)
        severities = (0,) if family == "clean" else (1, 2)
        for offset, severity in enumerate(severities):
            value = lookup[(family, severity)]
            x0 = center - 46 + offset * 52
            y0 = top + height - value * height
            color = BLUE if severity in (0, 1) else RED
            draw.rectangle((x0, y0, x0 + 42, top + height), fill=color)
            draw.text(
                (x0 + 21, y0 - 24),
                f"{value:.2f}",
                font=_font(14, bold=True),
                fill=TEXT,
                anchor="ma",
            )
        draw.text(
            (center, top + height + 28),
            family.title(),
            font=_font(17, bold=True),
            fill=TEXT,
            anchor="ma",
        )
    draw.rectangle((1120, 164, 1138, 182), fill=BLUE)
    draw.text((1147, 162), "Moderate", font=_font(17), fill=MUTED)
    draw.rectangle((1270, 164, 1288, 182), fill=RED)
    draw.text((1297, 162), "Severe", font=_font(17), fill=MUTED)
    _save_image(image, path)


def _figure_coverage(rows: list[dict[str, object]], path: Path) -> None:
    image, draw = _canvas(
        "Selective review coverage and risk",
        "Clean category conformal policy; each point is a frozen review boundary",
    )
    left, top, width, height = 170, 215, 1220, 550
    for tick in range(6):
        coverage = 0.5 + tick * 0.1
        x = left + (coverage - 0.5) / 0.5 * width
        draw.line((x, top, x, top + height), fill=GRID, width=2)
        draw.text(
            (x, top + height + 24), f"{coverage:.1f}", font=_font(17), fill=MUTED, anchor="ma"
        )
    for tick in range(7):
        risk = 0.12 + tick * 0.03
        y = top + height - (risk - 0.12) / 0.18 * height
        draw.line((left, y, left + width, y), fill=GRID, width=2)
        draw.text((left - 25, y), f"{risk:.2f}", font=_font(17), fill=MUTED, anchor="rm")
    points: list[tuple[float, float, float]] = []
    for row in rows:
        x = left + (_as_float(row["auto_coverage"]) - 0.5) / 0.5 * width
        y = top + height - (_as_float(row["selective_risk"]) - 0.12) / 0.18 * height
        points.append((x, y, _as_float(row["review_alpha"])))
    draw.line([(x, y) for x, y, _alpha in points], fill=RED, width=5, joint="curve")
    for x, y, alpha in points:
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=RED, outline=TEXT, width=2)
        draw.text((x + 15, y - 16), f"review={alpha:.2f}", font=_font(15, bold=True), fill=TEXT)
    draw.text(
        (left + width / 2, 840),
        "Automatic coverage",
        font=_font(20, bold=True),
        fill=TEXT,
        anchor="ma",
    )
    draw.text(
        (left, top - 22),
        "Selective risk",
        font=_font(20, bold=True),
        fill=TEXT,
        anchor="ls",
    )
    _save_image(image, path)


def _figure_hypotheses(rows: list[dict[str, object]], path: Path) -> None:
    image, draw = _canvas(
        "Confirmatory hypothesis tests",
        "Predeclared 95% intervals; positive values support the stated direction",
    )
    left, top, width = 600, 220, 850
    minimum, maximum = -0.04, 0.24
    zero = left + (0 - minimum) / (maximum - minimum) * width
    draw.line((zero, top - 30, zero, top + len(rows) * 105), fill=MUTED, width=3)
    for tick in range(-4, 25, 4):
        value = tick / 100
        x = left + (value - minimum) / (maximum - minimum) * width
        draw.text((x, 790), f"{value:+.2f}", font=_font(15), fill=MUTED, anchor="ma")
    for index, row in enumerate(rows):
        y = top + index * 105
        estimate = _as_float(row["estimate"])
        low = _as_float(row["ci_low"])
        high = _as_float(row["ci_high"])
        x = left + (estimate - minimum) / (maximum - minimum) * width
        x_low = left + (low - minimum) / (maximum - minimum) * width
        x_high = left + (high - minimum) / (maximum - minimum) * width
        color = GREEN if row["conclusion"] == "supported" else AMBER
        draw.text((85, y), str(row["label"]), font=_font(20, bold=True), fill=TEXT)
        draw.text(
            (85, y + 33),
            f"{row['stage']} · n={row['units']} · {row['conclusion']}",
            font=_font(15),
            fill=MUTED,
        )
        draw.line((x_low, y + 18, x_high, y + 18), fill=color, width=6)
        draw.line((x_low, y + 7, x_low, y + 29), fill=color, width=3)
        draw.line((x_high, y + 7, x_high, y + 29), fill=color, width=3)
        draw.ellipse((x - 9, y + 9, x + 9, y + 27), fill=color, outline=TEXT, width=2)
        draw.text(
            (x_high + 12, y + 18),
            f"{estimate:+.4f}",
            font=_font(15, bold=True),
            fill=TEXT,
            anchor="lm",
        )
    _save_image(image, path)


def build_m7_package(project_root: Path, config_path: Path) -> dict[str, Any]:
    """Regenerate all public M7 data products from verified completed runs."""

    settings = load_m7_settings(config_path)
    load_dotenv(project_root / ".env")
    artifact_root_raw = os.environ.get("ANOMALY_ARTIFACT_ROOT")
    if not artifact_root_raw:
        raise M7Error("ANOMALY_ARTIFACT_ROOT must be configured")
    paths, source_records = _verify_sources(Path(artifact_root_raw), settings)
    output = project_root / settings.output_directory
    tables = output / "tables"
    figures = output / "figures"
    output.mkdir(parents=True, exist_ok=True)

    clean = _clean_rows(paths)
    robustness = _robustness_rows(paths)
    calibration = _calibration_rows(paths)
    coverage = _coverage_rows(paths)
    hypotheses = _hypothesis_rows(paths)
    errors = _error_taxonomy(paths, robustness, calibration)

    _write_csv(tables / "clean_model_comparison.csv", clean, list(clean[0]))
    _write_csv(tables / "robustness_by_family.csv", robustness, list(robustness[0]))
    _write_csv(tables / "calibration_by_condition.csv", calibration, list(calibration[0]))
    _write_csv(tables / "coverage_risk_clean.csv", coverage, list(coverage[0]))
    _write_csv(tables / "hypothesis_tests.csv", hypotheses, list(hypotheses[0]))
    _write_csv(tables / "error_taxonomy.csv", errors, list(errors[0]))

    _figure_clean(clean, figures / "clean_performance.png")
    _figure_robustness(robustness, figures / "robustness_degradation.png")
    _figure_calibration(calibration, figures / "shift_false_positive_control.png")
    _figure_coverage(coverage, figures / "coverage_risk.png")
    _figure_hypotheses(hypotheses, figures / "hypothesis_intervals.png")

    generated_files = sorted(
        [path for path in output.rglob("*") if path.is_file() and path.name != "manifest.json"]
    )
    output_records = {
        path.relative_to(output).as_posix(): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in generated_files
    }
    canonical = json.dumps(output_records, sort_keys=True, separators=(",", ":")).encode()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "seed": settings.seed,
        "manifest_sha256": settings.manifest_sha256,
        "output_directory": settings.output_directory.as_posix(),
        "source_runs": source_records,
        "outputs": output_records,
        "package_sha256": hashlib.sha256(canonical).hexdigest(),
        "privacy": {
            "dataset_images_included": False,
            "private_images_included": False,
            "model_weights_included": False,
        },
    }
    _atomic_text(output / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest

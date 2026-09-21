"""Privacy-preserving inspection report construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

REPORT_FIELDS = (
    "filename",
    "category",
    "model",
    "decision",
    "anomaly_score",
    "conformal_p",
    "anomaly_alpha",
    "review_alpha",
    "empirical_score_threshold",
    "pixel_threshold",
    "calibration_samples",
    "anomalous_area_fraction",
    "input_width",
    "input_height",
    "model_input_size",
    "warnings",
    "timing_ms",
)


class ReportError(ValueError):
    """Raised when report input does not satisfy the export contract."""


def build_inspection_report(
    results: Sequence[Mapping[str, object]],
    *,
    runtime_info: Mapping[str, object],
    max_results: int,
) -> dict[str, object]:
    """Build a metadata-only report; image fields are never copied."""

    if not results or len(results) > max_results:
        raise ReportError(f"report requires 1 to {max_results} inspection results")
    redacted: list[dict[str, object]] = []
    for result in results:
        if not all(field in result for field in ("filename", "category", "decision")):
            raise ReportError("inspection result is missing required fields")
        redacted.append({field: result[field] for field in REPORT_FIELDS if field in result})
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "local visual anomaly inspection decision support",
        "source_images_embedded": False,
        "result_count": len(redacted),
        "runtime": {
            key: runtime_info[key]
            for key in (
                "model",
                "device",
                "anomaly_alpha",
                "review_alpha",
                "source_m5_run",
            )
            if key in runtime_info
        },
        "limitations": runtime_info.get("limitations", []),
        "results": redacted,
    }

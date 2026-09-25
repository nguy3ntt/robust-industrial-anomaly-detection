from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from robust_industrial_anomaly_detection.reporting.m7 import (
    MANIFEST_SHA256,
    M7Error,
    M7Settings,
    SourceSpec,
    _figure_coverage,
    _verify_sources,
    load_m7_settings,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_checked_in_m7_contract_pins_all_completed_stages() -> None:
    settings = load_m7_settings(Path("configs/reporting/m7.yaml"))

    assert settings.manifest_sha256 == MANIFEST_SHA256
    assert {source.name for source in settings.sources} == {"m2", "m3", "m4", "m5", "m6"}
    assert all(source.run and "latest" not in source.run for source in settings.sources)
    assert all(len(source.metadata_sha256) == 64 for source in settings.sources)


def test_m7_contract_rejects_noncanonical_manifest(tmp_path: Path) -> None:
    source = yaml.safe_load(Path("configs/reporting/m7.yaml").read_text(encoding="utf-8"))
    source["manifest_sha256"] = "0" * 64
    config = tmp_path / "m7.yaml"
    config.write_text(yaml.safe_dump(source), encoding="utf-8")

    with pytest.raises(M7Error, match="canonical manifest"):
        load_m7_settings(config)


def test_source_verifier_refuses_changed_input(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "frozen-run"
    run.mkdir(parents=True)
    table = run / "table.csv"
    table.write_text("metric,value\nauroc,0.8\n", encoding="utf-8")
    metadata = {
        "status": "complete",
        "manifest_sha256": MANIFEST_SHA256,
        "table_sha256": _sha256(table),
    }
    metadata_path = run / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
    settings = M7Settings(
        seed=20250915,
        manifest_sha256=MANIFEST_SHA256,
        output_directory=Path("reports/m7"),
        sources=(
            SourceSpec(
                name="m2",
                run="frozen-run",
                metadata_sha256=_sha256(metadata_path),
                required_files={"table.csv": "table_sha256"},
            ),
        ),
    )

    paths, records = _verify_sources(tmp_path, settings)
    assert paths["m2"] == run
    assert records["m2"]["files"]["table.csv"]["sha256"] == _sha256(table)

    table.write_text("metric,value\nauroc,0.9\n", encoding="utf-8")
    with pytest.raises(M7Error, match="source file hash mismatch"):
        _verify_sources(tmp_path, settings)


def test_coverage_figure_has_publication_dimensions(tmp_path: Path) -> None:
    path = tmp_path / "coverage.png"
    rows: list[dict[str, object]] = [
        {"review_alpha": 0.5, "auto_coverage": 0.6, "selective_risk": 0.12},
        {"review_alpha": 0.2, "auto_coverage": 0.8, "selective_risk": 0.19},
        {"review_alpha": 0.05, "auto_coverage": 1.0, "selective_risk": 0.28},
    ]

    _figure_coverage(rows, path)

    with Image.open(path) as image:
        assert image.size == (1600, 900)
        assert image.mode == "RGB"

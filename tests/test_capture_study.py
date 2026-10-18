"""Generated fixtures test workflow safeguards, NOT real fresh-capture accuracy."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from robust_industrial_anomaly_detection.data import capture_study as study
from robust_industrial_anomaly_detection.prototype.engine import load_prototype_settings

ROOT = Path(__file__).parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def write_table(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def capture(root, index, *, session="dev-a", split="development", label="normal", object_id=None):
    path = root / split / f"photo-{index}.png"
    # Explicitly synthetic, deterministic and independent test arrays.
    rng = np.random.default_rng(index)
    Image.fromarray(rng.integers(40, 210, (48, 64, 3), dtype=np.uint8)).save(path)
    return {
        "sample_id": f"capture-{index}",
        "image_path": f"{split}/{path.name}",
        "object_id": object_id or f"object-{index}",
        "session_id": session,
        "label": label,
        "captured_at": datetime.now(UTC).isoformat(),
        "defect_note": "synthetic fixture, not research" if label == "anomaly" else "",
    }


@pytest.fixture
def development(tmp_path):
    root = tmp_path / "study"
    study.initialize(root, category="candle")
    protocol = json.loads((root / "protocol.json").read_text())
    protocol.update(
        product_description="Test fixture only",
        domain_match_reason="Not a real product study",
        rights_confirmed=True,
        fresh_photographs_confirmed=True,
    )
    write_json(root / "protocol.json", protocol)
    sessions = study._table(root / "sessions.csv", study.SESSION_FIELDS)
    for session in sessions:
        session["description"] = "Synthetic unit test metadata, not physical capture evidence"
    write_table(root / "sessions.csv", study.SESSION_FIELDS, sessions)
    rows = [capture(root, 1), capture(root, 2, label="anomaly")]
    write_table(root / "captures.csv", study.CAPTURE_FIELDS, rows)
    return root, rows


class FixtureEngine:
    """Never a real camera/model; exercises only state transitions and counting."""

    def __init__(self, root):
        self.m5_run = root / "fixture-m5"
        self.quality_run = root / "fixture-gate"
        self.localization_run = root / "fixture-localization"
        self.categories = ("candle",)
        self.settings = load_prototype_settings(ROOT / "configs/prototype/m10.yaml")
        self.reference = root / "reference.bin"
        self.reference.write_bytes(b"synthetic frozen normal reference")
        self.calls = 0
        self.fail_after = None

    def runtime_info(self):
        return {"categories": self.categories, "device": "cpu", "anomaly_alpha": 0.05}

    def provenance_files(self):
        return (self.reference,)

    def inspect_bytes(self, payload, *, filename, category):
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError("synthetic interruption")
        self.calls += 1
        return SimpleNamespace(
            decision="recapture",
            model_evaluated=False,
            anomaly_score=None,
            conformal_p=None,
            quality_p=0.01,
            timing_ms={"total": 1.0},
        )


def register_final(root, rows):
    for index in range(40):
        rows.append(
            capture(
                root,
                100 + index,
                session=("eval-a", "eval-b", "eval-c", "eval-d")[index // 10],
                split="evaluation",
                label="normal" if index % 10 < 5 else "anomaly",
            )
        )
    write_table(root / "captures.csv", study.CAPTURE_FIELDS, rows)


def test_initialization_is_empty_pending_and_never_overwrites(tmp_path):
    root = tmp_path / "pending"
    result = study.initialize(root, category="candle")
    assert result["status"] == "awaiting_real_development_photographs"
    assert study._table(root / "captures.csv", study.CAPTURE_FIELDS) == []
    assert not list((root / "evaluation").iterdir())
    assert not (root / "summary.json").exists() and not (root / "freeze.json").exists()
    with pytest.raises(study.CaptureStudyError, match="No photographs"):
        study.audit(root)
    with pytest.raises(FileExistsError):
        study.initialize(root, category="other")


def test_audit_requires_real_attestation_and_both_development_labels(development):
    root, rows = development
    protocol = json.loads((root / "protocol.json").read_text())
    protocol["fresh_photographs_confirmed"] = False
    write_json(root / "protocol.json", protocol)
    with pytest.raises(study.CaptureStudyError, match="genuine fresh"):
        study.audit(root)
    protocol["fresh_photographs_confirmed"] = True
    write_json(root / "protocol.json", protocol)
    rows[1]["label"] = "normal"
    write_table(root / "captures.csv", study.CAPTURE_FIELDS, rows)
    with pytest.raises(study.CaptureStudyError, match="both labels"):
        study.audit(root)


def test_inventory_and_paths_prevent_selective_or_external_files(development):
    root, rows = development
    extra = root / "development/extra.png"
    extra.write_bytes(b"unregistered")
    with pytest.raises(study.CaptureStudyError, match="Unregistered"):
        study.audit(root)
    extra.unlink()
    rows[0]["image_path"] = "development/../../outside.png"
    write_table(root / "captures.csv", study.CAPTURE_FIELDS, rows)
    with pytest.raises(study.CaptureStudyError, match="must stay inside"):
        study.audit(root)


def test_duplicate_decoded_content_is_rejected(development):
    root, rows = development
    with Image.open(root / rows[0]["image_path"]) as image:
        image.save(root / rows[1]["image_path"], compress_level=0)
    with pytest.raises(study.CaptureStudyError, match="decoded-content duplicate"):
        study.audit(root)


def test_holdout_cannot_be_audited_before_freeze_or_collected_before_commitment(development):
    root, _ = development
    engine = FixtureEngine(root)
    capture(root, 50, split="evaluation", session="eval-a")
    with pytest.raises(study.CaptureStudyError, match="uncollected"):
        study.freeze(root, ROOT, engine)
    with pytest.raises(study.CaptureStudyError, match=r"freeze\.json"):
        study.audit(root, evaluation=True)
    assert not engine.calls


def test_freeze_commits_model_code_plan_and_development_without_inference(development):
    root, _ = development
    engine = FixtureEngine(root)
    result = study.freeze(root, ROOT, engine)
    assert result["status"] == "frozen_collect_final_photographs_now"
    assert study.verify_freeze(root)["development_sha256"]
    assert engine.calls == 0
    with pytest.raises(study.CaptureStudyError, match="already frozen"):
        study.freeze(root, ROOT, engine)
    engine.reference.write_bytes(b"changed")
    with pytest.raises(study.CaptureStudyError, match="Frozen runtime changed"):
        study.verify_freeze(root)


def test_plan_and_original_development_changes_refuse_reuse(development):
    root, rows = development
    engine = FixtureEngine(root)
    study.freeze(root, ROOT, engine)
    register_final(root, rows)
    # Metadata changes count too, not only image bytes.
    rows[0]["defect_note"] = "changed after freeze"
    write_table(root / "captures.csv", study.CAPTURE_FIELDS, rows)
    with pytest.raises(study.CaptureStudyError, match="Development membership"):
        study.audit(root, evaluation=True)
    (root / "sessions.csv").write_text((root / "sessions.csv").read_text() + "\n")
    with pytest.raises(study.CaptureStudyError, match="session plan changed"):
        study.verify_freeze(root)


def test_object_leakage_and_perceptual_cross_partition_duplicates(development):
    root, rows = development
    engine = FixtureEngine(root)
    study.freeze(root, ROOT, engine)
    final = capture(root, 90, split="evaluation", session="eval-a", object_id=rows[0]["object_id"])
    write_table(root / "captures.csv", study.CAPTURE_FIELDS, [*rows, final])
    with pytest.raises(study.CaptureStudyError, match="physical object crosses"):
        study.audit(root, evaluation=True)
    final["object_id"] = "independent-object"
    with Image.open(root / rows[0]["image_path"]) as image:
        pixels = np.asarray(image).astype(np.uint8) + 1
    Image.fromarray(pixels).save(root / final["image_path"])
    write_table(root / "captures.csv", study.CAPTURE_FIELDS, [*rows, final])
    with pytest.raises(study.CaptureStudyError, match="perceptual duplicate"):
        study.audit(root, evaluation=True)


def test_evaluation_requires_minimum_coverage_without_scoring(development):
    root, rows = development
    engine = FixtureEngine(root)
    study.freeze(root, ROOT, engine)
    rows.append(capture(root, 80, split="evaluation", session="eval-a"))
    write_table(root / "captures.csv", study.CAPTURE_FIELDS, rows)
    with pytest.raises(study.CaptureStudyError, match="Insufficient pilot coverage"):
        study.evaluate(root, ROOT, engine)
    assert engine.calls == 0 and not (root / "holdout-opened.json").exists()


def test_real_state_machine_with_synthetic_fixtures_interrupts_resumes_and_counts_recapture(
    development,
):
    root, rows = development
    engine = FixtureEngine(root)
    study.freeze(root, ROOT, engine)
    register_final(root, rows)
    engine.fail_after = 2
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        study.evaluate(root, ROOT, engine)
    assert (root / "holdout-opened.json").is_file()
    assert len(list((root / "predictions").glob("*.json"))) == 2
    engine.fail_after = None
    result = study.evaluate(root, ROOT, engine)
    assert engine.calls == 40 and result["images"] == 40
    assert result["normal_recapture_rate"] == result["defect_recapture_rate"] == 1
    assert result["normal_false_alert_rate_all"] == result["confident_defect_recall_all"] == 0
    assert result["automatic_coverage"] == 0 and result["operator_action_rate"] == 1
    assert result["model_evaluated_images"] == 0
    assert result["normal_false_alert_rate_evaluated_only"] is None
    assert result["confident_defect_recall_evaluated_only"] is None
    assert not result["localization_accuracy_evaluated"]
    assert study.evaluate(root, ROOT, engine) == result and engine.calls == 40
    assert "data:image" not in (root / "summary.json").read_text()
    checkpoint = next((root / "predictions").glob("*.json"))
    value = json.loads(checkpoint.read_text())
    value["result"]["decision"] = "anomalous"
    write_json(checkpoint, value)
    with pytest.raises(study.CaptureStudyError, match="prediction failed resume"):
        study.evaluate(root, ROOT, engine)


def test_cameras_and_labels_cannot_be_artificially_confounded(development):
    root, rows = development
    engine = FixtureEngine(root)
    study.freeze(root, ROOT, engine)
    register_final(root, rows)
    for row in rows:
        if row["session_id"] == "eval-a":
            row["label"] = "normal"
        elif row["session_id"] == "eval-b":
            row["label"] = "anomaly"
            row["defect_note"] = "unit test only"
    write_table(root / "captures.csv", study.CAPTURE_FIELDS, rows)
    with pytest.raises(study.CaptureStudyError, match="avoid capture/label confounding"):
        study.audit(root, evaluation=True)


def test_environment_and_different_runtime_cannot_reuse_holdout(development, monkeypatch):
    root, rows = development
    engine = FixtureEngine(root)
    study.freeze(root, ROOT, engine)
    register_final(root, rows)
    original = study._environment()
    monkeypatch.setattr(study, "_environment", lambda: {"python": "different"})
    with pytest.raises(study.CaptureStudyError, match="environment changed"):
        study.verify_freeze(root)
    monkeypatch.setattr(study, "_environment", lambda: original)
    monkeypatch.setattr(engine, "runtime_info", lambda: {"device": "other"})
    with pytest.raises(study.CaptureStudyError, match="loaded runtime differs"):
        study.evaluate(root, ROOT, engine)
    assert engine.calls == 0


def test_case_collisions_and_windows_reserved_ids_are_refused(development):
    root, rows = development
    for value in ("CON", "nul", "COM1", "LPT9"):
        with pytest.raises(study.CaptureStudyError):
            study._identifier(value, "sample_id")
    rows[1]["sample_id"] = rows[0]["sample_id"].upper()
    write_table(root / "captures.csv", study.CAPTURE_FIELDS, rows)
    with pytest.raises(study.CaptureStudyError, match="Duplicate sample ID"):
        study.audit(root)


def test_research_gate_cannot_be_silently_frozen_as_the_deployed_policy(development):
    root, _ = development
    engine = FixtureEngine(root)
    engine.settings = load_prototype_settings(ROOT / "configs/prototype/m12.yaml")
    with pytest.raises(study.CaptureStudyError, match="unchanged quality-gated M10"):
        study.freeze(root, ROOT, engine)
    assert not (root / "freeze.json").exists()

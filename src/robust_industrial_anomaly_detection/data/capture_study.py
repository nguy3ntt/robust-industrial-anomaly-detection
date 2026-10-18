"""Preregistered, local-only fresh captures for the frozen deployed product model.

No supplied images are synthesized, no reference is adapted, and final images
cannot be scored until the runtime and development cohort have been locked.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import warnings
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from platform import python_version
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image, ImageOps

from robust_industrial_anomaly_detection.data.visa import _write_json, sha256_file

if TYPE_CHECKING:
    from robust_industrial_anomaly_detection.prototype.engine import InspectionEngine

CAPTURE_FIELDS = (
    "sample_id",
    "image_path",
    "object_id",
    "session_id",
    "label",
    "captured_at",
    "defect_note",
)
SESSION_FIELDS = ("session_id", "split", "camera_id", "lighting_id", "description")
ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}\Z")
MAX_CAPTURES = 500


class CaptureStudyError(ValueError):
    """The study cannot proceed without violating its registered boundary."""


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    _write_json(value, path)


def _json(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CaptureStudyError(f"Cannot read study file {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise CaptureStudyError(f"Expected a JSON object: {path}")
    return result


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _environment() -> dict[str, Any]:
    return {
        "python": python_version(),
        "packages": {
            name: version(name)
            for name in (
                "numpy",
                "pillow",
                "opencv-python-headless",
                "torch",
                "torchvision",
                "scipy",
                "scikit-learn",
                "pyyaml",
                "python-dotenv",
            )
        },
    }


def _new_json(path: Path, value: Any) -> None:
    # Exclusive creation is intentional: never replace a scientific commitment.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def initialize(root: Path, *, category: str) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=False)
    for name in ("development", "evaluation"):
        (root / name).mkdir()
    _new_json(
        root / "protocol.json",
        {
            "schema_version": 1,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "category": category,
            "product_description": "",
            "domain_match_reason": "",
            "rights_confirmed": False,
            "fresh_photographs_confirmed": False,
            "scope": "frozen_deployed_product_capture_transfer",
            "minimum_evaluation_normals": 20,
            "minimum_evaluation_anomalies": 20,
            "minimum_evaluation_sessions": 4,
            "minimum_evaluation_cameras": 2,
            "minimum_evaluation_lighting_conditions": 2,
            "notes": (
                "Pilot only; no guarantees of product quality or independent capture provenance."
            ),
        },
    )
    with (root / "captures.csv").open("x", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerow(CAPTURE_FIELDS)
    with (root / "sessions.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(SESSION_FIELDS)
        for name, split, camera, light in (
            ("dev-a", "development", "camera-a", "diffuse"),
            ("dev-b", "development", "camera-b", "side-light"),
            ("eval-a", "evaluation", "camera-a", "diffuse"),
            ("eval-b", "evaluation", "camera-a", "side-light"),
            ("eval-c", "evaluation", "camera-b", "diffuse"),
            ("eval-d", "evaluation", "camera-b", "side-light"),
        ):
            writer.writerow((name, split, camera, light, "Describe the real physical session"))
    return {"study": str(root.resolve()), "status": "awaiting_real_development_photographs"}


def _table(path: Path, fields: tuple[str, ...]) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(fields):
                raise CaptureStudyError(
                    f"Unexpected columns in {path.name}; see the collection guide."
                )
            rows = list(reader)
    except OSError as exc:
        raise CaptureStudyError(f"Cannot read {path}: {exc}") from exc
    if len(rows) > MAX_CAPTURES:
        raise CaptureStudyError(f"This pilot permits at most {MAX_CAPTURES} rows per table.")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise CaptureStudyError(f"Malformed CSV row in {path.name}.")
    return [{key: value.strip() for key, value in row.items()} for row in rows]


def _identifier(value: str, name: str) -> None:
    reserved = {"con", "prn", "aux", "nul"} | {
        prefix + str(number) for prefix in ("com", "lpt") for number in range(1, 10)
    }
    if not ID.fullmatch(value) or value.casefold() in reserved:
        raise CaptureStudyError(f"{name} must use 1-80 letters, digits, hyphens or underscores.")


def _image_path(root: Path, relative: str, split: str) -> Path:
    # Accept only portable relative paths. Symlinks/junction escapes are rejected.
    parts = relative.replace("\\", "/").split("/")
    if len(parts) < 2 or parts[0] != split or any(part in {"", ".", ".."} for part in parts):
        raise CaptureStudyError(f"Image path must stay inside {split}/: {relative}")
    candidate = root.joinpath(*parts)
    try:
        candidate.resolve(strict=True).relative_to(root.resolve(strict=True) / split)
    except (OSError, ValueError) as exc:
        raise CaptureStudyError(
            f"Missing image or path escapes its study partition: {relative}"
        ) from exc
    if any(parent.is_symlink() for parent in (candidate, *candidate.parents) if parent != root):
        raise CaptureStudyError("Linked source images are not allowed in a capture study.")
    return candidate


def _fingerprints(path: Path) -> tuple[str, list[tuple[int, np.ndarray]]]:
    if path.stat().st_size > 20_971_520:
        raise CaptureStudyError(f"Capture exceeds the 20 MiB upload boundary: {path.name}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as source:
                if source.width * source.height > 40_000_000 or getattr(
                    source, "is_animated", False
                ):
                    raise CaptureStudyError("Capture must be a static image at most 40 MP.")
                rgb = ImageOps.exif_transpose(source).convert("RGB")
                decoded = hashlib.sha256(f"{rgb.size}:".encode() + rgb.tobytes()).hexdigest()
                result = []
                for view in (rgb, ImageOps.fit(rgb, (224, 224), Image.Resampling.BILINEAR)):
                    gray = view.convert("L")
                    small = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS))
                    bits = (small[:, 1:] > small[:, :-1]).reshape(-1)
                    dhash = sum(int(bit) << index for index, bit in enumerate(bits))
                    thumb = np.asarray(
                        gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32
                    )
                    thumb = (thumb - thumb.mean()) / max(float(thumb.std()), 1.0)
                    result.append((dhash, thumb))
                return decoded, result
    except (
        OSError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise CaptureStudyError(f"Cannot safely decode {path.name}: {exc}") from exc


def _protocol(root: Path) -> dict[str, Any]:
    value = _json(root / "protocol.json")
    if value.get("schema_version") != 1 or value.get("scope") != (
        "frozen_deployed_product_capture_transfer"
    ):
        raise CaptureStudyError("Unsupported study protocol.")
    for name in ("category", "product_description", "domain_match_reason"):
        if not isinstance(value.get(name), str) or not value[name].strip():
            raise CaptureStudyError(f"Complete protocol.json field {name} before auditing.")
    if (
        value.get("rights_confirmed") is not True
        or value.get("fresh_photographs_confirmed") is not True
    ):
        raise CaptureStudyError(
            "Confirm image rights and genuine fresh photographs in protocol.json."
        )
    minimums = {
        "minimum_evaluation_normals": 20,
        "minimum_evaluation_anomalies": 20,
        "minimum_evaluation_sessions": 4,
        "minimum_evaluation_cameras": 2,
        "minimum_evaluation_lighting_conditions": 2,
    }
    for name, floor in minimums.items():
        count = value.get(name)
        if type(count) is not int or count < floor:
            raise CaptureStudyError(f"{name} must be an integer at least {floor}.")
    return value


def verify_freeze(root: Path) -> dict[str, Any]:
    lock = _json(root / "freeze.json")
    digest = lock.get("commitment_sha256")
    if digest != _digest({key: value for key, value in lock.items() if key != "commitment_sha256"}):
        raise CaptureStudyError("The freeze commitment has changed or is incomplete.")
    if lock["environment"] != _environment():
        raise CaptureStudyError("The installed runtime environment changed after freeze.")
    for path, expected in lock["runtime_files"].items():
        if not Path(path).is_file() or sha256_file(Path(path)) != expected:
            raise CaptureStudyError(
                f"Frozen runtime changed: {path}. Do not reuse this holdout for tuning."
            )
    if sha256_file(root / "protocol.json") != lock["protocol_sha256"] or (
        sha256_file(root / "sessions.csv") != lock["sessions_sha256"]
    ):
        raise CaptureStudyError("The registered protocol/session plan changed after freeze.")
    return lock


def audit(root: Path, *, evaluation: bool = False) -> dict[str, Any]:
    """Check provenance and leakage; never infer or render an evaluation image."""

    captures = _table(root / "captures.csv", CAPTURE_FIELDS)
    if not captures:
        raise CaptureStudyError(
            "No photographs registered yet. Add real development photos and CSV rows."
        )
    protocol = _protocol(root)
    sessions = _table(root / "sessions.csv", SESSION_FIELDS)
    session_map = {}
    for row in sessions:
        for field in ("session_id", "camera_id", "lighting_id"):
            _identifier(row[field], field)
        if row["session_id"] in session_map or row["split"] not in {"development", "evaluation"}:
            raise CaptureStudyError("Session IDs must be unique and have a valid partition.")
        if not row["description"] or row["description"].startswith("Describe "):
            raise CaptureStudyError("Describe each planned real session in sessions.csv.")
        session_map[row["session_id"]] = row
    lock = verify_freeze(root) if evaluation else None
    seen_ids: set[str] = set()
    paths: set[Path] = set()
    objects: dict[str, str] = {}
    inspected: list[dict[str, Any]] = []
    fingerprints: list[tuple[str, str, str, list[tuple[int, np.ndarray]]]] = []
    for row in sorted(captures, key=lambda item: item["sample_id"]):
        for key in ("sample_id", "object_id", "session_id"):
            _identifier(row[key], key)
        if row["sample_id"].casefold() in seen_ids or row["session_id"] not in session_map:
            raise CaptureStudyError("Duplicate sample ID or unknown session.")
        seen_ids.add(row["sample_id"].casefold())
        session = session_map[row["session_id"]]
        split = session["split"]
        if split == "evaluation" and not evaluation:
            raise CaptureStudyError(
                "Evaluation rows exist before holdout unlock. Use the frozen evaluation audit."
            )
        if row["label"] not in {"normal", "anomaly"} or (
            row["label"] == "anomaly" and not row["defect_note"]
        ):
            raise CaptureStudyError("Use normal/anomaly labels and describe every physical defect.")
        try:
            captured = datetime.fromisoformat(row["captured_at"])
            if captured.tzinfo is None:
                raise ValueError("timezone required")
            if captured > datetime.now(UTC):
                raise ValueError("future capture date")
            if (
                split == "evaluation"
                and lock is not None
                and captured <= datetime.fromisoformat(lock["frozen_at_utc"])
            ):
                raise ValueError("final captures must be collected after the freeze")
        except ValueError as exc:
            raise CaptureStudyError(f"Invalid captured_at for {row['sample_id']}: {exc}") from exc
        prior = objects.setdefault(row["object_id"].casefold(), split)
        if prior != split:
            raise CaptureStudyError(
                "The same physical object crosses development/evaluation partitions."
            )
        path = _image_path(root, row["image_path"], split)
        if path.resolve() in paths:
            raise CaptureStudyError("An image path is registered more than once.")
        paths.add(path.resolve())
        file_hash = sha256_file(path)
        decoded_hash, views = _fingerprints(path)
        for other_split, other_file, other_decoded, other_views in fingerprints:
            if file_hash == other_file or decoded_hash == other_decoded:
                raise CaptureStudyError(
                    "Exact or decoded-content duplicate photographs are not allowed."
                )
            if split != other_split and any(
                (left_hash ^ right_hash).bit_count() <= 2
                and float(np.mean(np.abs(left - right))) <= 0.1
                for (left_hash, left), (right_hash, right) in zip(views, other_views, strict=True)
            ):
                raise CaptureStudyError("Confirmed perceptual duplicate crosses study partitions.")
        fingerprints.append((split, file_hash, decoded_hash, views))
        inspected.append(
            {
                **row,
                **{
                    key: session[key]
                    for key in (
                        "split",
                        "camera_id",
                        "lighting_id",
                    )
                },
                "sha256": file_hash,
                "decoded_sha256": decoded_hash,
            }
        )
    inventory = {
        path.resolve()
        for split in ("development", "evaluation")
        for path in (root / split).rglob("*")
        if path.is_file()
    }
    if inventory != paths:
        raise CaptureStudyError(
            "Unregistered files exist in the capture folders; "
            "include every capture, not selected outcomes."
        )
    development = [row for row in inspected if row["split"] == "development"]
    if set(row["label"] for row in development) != {"normal", "anomaly"}:
        raise CaptureStudyError(
            "Register development examples of both labels to check the workflow first."
        )
    if lock is not None and _digest(development) != lock["development_sha256"]:
        raise CaptureStudyError("Development membership or photographs changed after freeze.")
    final = [row for row in inspected if row["split"] == "evaluation"]
    if evaluation:
        requirements = {
            "minimum_evaluation_normals": sum(row["label"] == "normal" for row in final),
            "minimum_evaluation_anomalies": sum(row["label"] == "anomaly" for row in final),
            "minimum_evaluation_sessions": len({row["session_id"] for row in final}),
            "minimum_evaluation_cameras": len({row["camera_id"] for row in final}),
            "minimum_evaluation_lighting_conditions": len({row["lighting_id"] for row in final}),
        }
        for name, observed in requirements.items():
            if observed < protocol[name]:
                raise CaptureStudyError(
                    f"Insufficient pilot coverage: {name} needs {protocol[name]}, has {observed}."
                )
        for key in ("session_id", "camera_id", "lighting_id"):
            for value in {row[key] for row in final}:
                if {row["label"] for row in final if row[key] == value} != {"normal", "anomaly"}:
                    raise CaptureStudyError(
                        f"Both labels are required in each {key}; avoid capture/label confounding."
                    )
        if len({row["object_id"] for row in final}) < 20:
            raise CaptureStudyError("The final pilot needs at least 20 distinct physical objects.")
    result = {
        "schema_version": 1,
        "status": "audited_not_evaluated",
        "evaluation": evaluation,
        "category": protocol["category"],
        "captures": inspected,
        "development_sha256": _digest(development),
        "capture_inventory_sha256": _digest(inspected),
        "counts": dict(Counter(row["split"] + ":" + row["label"] for row in inspected)),
        "freshness_is_owner_attested_not_independently_verified": True,
    }
    _atomic_json(
        root / ("audit-evaluation.json" if evaluation else "audit-development.json"), result
    )
    return result


def runtime_files(project: Path, engine: InspectionEngine) -> dict[str, str]:
    paths = [
        *engine.provenance_files(),
        *sorted((project / "src").rglob("*.py")),
        project / "configs/prototype/m10.yaml",
        project / "scripts/capture_study.py",
        project / "scripts/run_inspector.py",
        project / "uv.lock",
        project / "pyproject.toml",
    ]
    return {str(path.resolve(strict=True)): sha256_file(path) for path in sorted(set(paths))}


def freeze(root: Path, project: Path, engine: InspectionEngine) -> dict[str, Any]:
    from robust_industrial_anomaly_detection.prototype.engine import load_prototype_settings

    if (root / "freeze.json").exists():
        raise CaptureStudyError("This study is already frozen; its commitment cannot be replaced.")
    if any((root / "evaluation").rglob("*")):
        raise CaptureStudyError("Keep final photographs uncollected until the runtime freeze.")
    inspected = audit(root)
    if inspected["category"] not in engine.categories:
        raise CaptureStudyError(
            "This product has no deployed normal reference; a separate adaptation study is needed."
        )
    if engine.settings != load_prototype_settings(project / "configs/prototype/m10.yaml"):
        raise CaptureStudyError("This protocol requires the unchanged quality-gated M10 runtime.")
    lock = {
        "schema_version": 1,
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "environment": _environment(),
        "protocol_sha256": sha256_file(root / "protocol.json"),
        "sessions_sha256": sha256_file(root / "sessions.csv"),
        "development_sha256": inspected["development_sha256"],
        "runtime": engine.runtime_info(),
        "runtime_files": runtime_files(project, engine),
        "m5_run": str(engine.m5_run.resolve()),
        "quality_run": str(engine.quality_run.resolve()) if engine.quality_run else None,
        "localization_run": str(engine.localization_run.resolve())
        if engine.localization_run
        else None,
        "holdout_opened": False,
    }
    lock["commitment_sha256"] = _digest(lock)
    _new_json(root / "freeze.json", lock)
    return {
        "status": "frozen_collect_final_photographs_now",
        "commitment_sha256": lock["commitment_sha256"],
    }


def _rates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    normals = [row for row in rows if row["label"] == "normal"]
    defects = [row for row in rows if row["label"] == "anomaly"]
    evaluated_normals = [row for row in normals if row["model_evaluated"]]
    evaluated_defects = [row for row in defects if row["model_evaluated"]]

    def fraction(group: list[dict[str, Any]], decisions: set[str]) -> float | None:
        return sum(row["decision"] in decisions for row in group) / len(group) if group else None

    return {
        "images": len(rows),
        "normal_images": len(normals),
        "anomaly_images": len(defects),
        "model_evaluated_images": len(evaluated_normals) + len(evaluated_defects),
        "physical_objects": len({row["object_id"] for row in rows}),
        "capture_sessions": len({row["session_id"] for row in rows}),
        "normal_false_alert_rate_all": fraction(normals, {"anomalous"}),
        "normal_false_alert_rate_evaluated_only": fraction(evaluated_normals, {"anomalous"}),
        "normal_recapture_rate": fraction(normals, {"recapture"}),
        "normal_review_rate": fraction(normals, {"review"}),
        "confident_defect_recall_all": fraction(defects, {"anomalous"}),
        "confident_defect_recall_evaluated_only": fraction(evaluated_defects, {"anomalous"}),
        "defect_recapture_rate": fraction(defects, {"recapture"}),
        "defect_false_accept_rate": fraction(defects, {"normal"}),
        "operator_action_rate": fraction(rows, {"review", "recapture", "anomalous"}),
        "automatic_coverage": fraction(rows, {"normal", "anomalous"}),
        "median_inspection_ms": float(np.median([row["inspection_ms"] for row in rows])),
    }


def evaluate(root: Path, project: Path, engine: InspectionEngine) -> dict[str, Any]:
    """Open once, then resume only the identical locked cohort; metadata stays local."""

    from robust_industrial_anomaly_detection.experiments.m2 import _atomic_csv

    lock = verify_freeze(root)
    if runtime_files(project, engine) != lock["runtime_files"] or (
        _digest(engine.runtime_info()) != _digest(lock["runtime"])
    ):
        raise CaptureStudyError("The loaded runtime differs from the frozen model/policy/device.")
    inspected = audit(root, evaluation=True)
    identity = {
        "freeze_sha256": lock["commitment_sha256"],
        "capture_inventory_sha256": inspected["capture_inventory_sha256"],
    }
    opened = root / "holdout-opened.json"
    if opened.exists():
        if _json(opened)["identity"] != identity:
            raise CaptureStudyError(
                "This holdout was already opened with different inputs. Do not retune it."
            )
    else:
        _new_json(opened, {"identity": identity, "opened_at_utc": datetime.now(UTC).isoformat()})
    directory = root / "predictions"
    directory.mkdir(exist_ok=True)
    predictions = []
    for row in inspected["captures"]:
        if row["split"] != "evaluation":
            continue
        target = directory / f"{row['sample_id']}.json"
        if target.exists():
            checkpoint = _json(target)
            if (
                checkpoint["identity"] != identity
                or checkpoint["source"] != row
                or (checkpoint.get("result_sha256") != _digest(checkpoint["result"]))
            ):
                raise CaptureStudyError("A saved prediction failed resume verification.")
            prediction = checkpoint["result"]
        else:
            path = _image_path(root, row["image_path"], "evaluation")
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != row["sha256"]:
                raise CaptureStudyError("A capture changed between audit and inference.")
            result = engine.inspect_bytes(
                payload, filename=row["sample_id"], category=inspected["category"]
            )
            prediction = {
                **{
                    key: row[key]
                    for key in (
                        "sample_id",
                        "label",
                        "object_id",
                        "session_id",
                        "camera_id",
                        "lighting_id",
                    )
                },
                "decision": result.decision,
                "model_evaluated": result.model_evaluated,
                "anomaly_score": result.anomaly_score,
                "conformal_p": result.conformal_p,
                "quality_p": result.quality_p,
                "inspection_ms": result.timing_ms["total"],
            }
            _atomic_json(
                target,
                {
                    "identity": identity,
                    "source": row,
                    "result": prediction,
                    "result_sha256": _digest(prediction),
                },
            )
        predictions.append(prediction)
    grouped = []
    for key in ("session_id", "camera_id", "lighting_id"):
        for value in sorted({row[key] for row in predictions}):
            grouped.append(
                {
                    "group": key,
                    "value": value,
                    **_rates([row for row in predictions if row[key] == value]),
                }
            )
    _atomic_csv(root / "group_metrics.csv", grouped)
    summary = {
        "status": "complete_exploratory_fresh_capture_pilot",
        **identity,
        **_rates(predictions),
        "source_images_published": False,
        "model_or_thresholds_adapted": False,
        "localization_accuracy_evaluated": False,
        "limitations": [
            "Owner-attested real captures, not independently certified acquisition.",
            "Known product references only; not new-category normal adaptation.",
            "Recapture/review are unresolved; all captures remain in primary denominators.",
            "Repeated objects/sessions are correlated; no image-independent confidence claim.",
            "No mask-based localization or untouched-factory generalization claim.",
        ],
        "output_hashes": {
            str(path.relative_to(root)): sha256_file(path)
            for path in [
                root / "audit-evaluation.json",
                root / "group_metrics.csv",
                *sorted(directory.glob("*.json")),
            ]
        },
    }
    _atomic_json(root / "summary.json", summary)
    return summary

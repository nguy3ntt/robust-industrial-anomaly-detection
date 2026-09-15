"""Leakage-safe FaceForensics++ video-manifest construction.

The official filenames encode source lineage. Original sequences are named with
one numeric sequence identifier. Manipulated videos use
``<target_sequence>_<source_sequence>``. Splits are therefore assigned to the
entire lineage, never independently to files or frames.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, cast

import yaml

Split = Literal["train", "val", "test", "unassigned"]
Severity = Literal["error", "warning"]
ProbeMode = Literal["none", "optional", "required"]
ChecksumMode = Literal["none", "sha256"]

SCHEMA_VERSION = "1.0"
DATASET_NAME = "faceforensicspp"
VALID_SPLITS = ("train", "val", "test")
_ENVIRONMENT_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_ORIGINAL_STEM = re.compile(r"^(\d{3})$")
_MANIPULATED_STEM = re.compile(r"^(\d{3})_(\d{3})$")
_SAFE_PATH_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_KNOWN_COMPRESSIONS = {"raw", "c23", "c40"}
_KNOWN_MANIPULATION_METHODS = {
    "Deepfakes",
    "Face2Face",
    "FaceSwap",
    "NeuralTextures",
    "FaceShifter",
}


class ManifestConfigurationError(ValueError):
    """Raised when a manifest configuration is unsafe or incomplete."""


class ManifestVerificationError(ValueError):
    """Raised when a published manifest and audit do not form a trusted pair."""


@dataclass(frozen=True)
class FaceForensicsConfig:
    """Validated configuration for one FaceForensics++ manifest build."""

    dataset_root: Path
    dataset_release: str
    access_date: date
    output_dir: Path
    split_dir: Path
    compressions: tuple[str, ...]
    manipulation_methods: tuple[str, ...]
    expected_counts: Mapping[str, int]
    identity_map_path: Path | None = None
    checksum_mode: ChecksumMode = "sha256"
    probe_mode: ProbeMode = "required"
    workers: int = 4
    require_complete_official_sources: bool = True
    manifest_filename: str = "faceforensicspp.jsonl"
    audit_filename: str = "faceforensicspp.audit.json"


@dataclass(frozen=True)
class ProbeMetadata:
    """Selected video-stream metadata returned by ffprobe."""

    codec: str | None = None
    width: int | None = None
    height: int | None = None
    average_fps: float | None = None
    frame_count: int | None = None
    duration_seconds: float | None = None


@dataclass(frozen=True)
class VideoRecord:
    """One deterministic, root-relative FaceForensics++ video record."""

    schema_version: str
    dataset: str
    dataset_release: str
    access_date: str
    video_id: str
    relative_path: str
    split: Split
    label: int
    class_name: Literal["real", "manipulated"]
    manipulation_method: str
    compression: str
    target_sequence_id: str
    donor_sequence_id: str | None
    lineage_sequence_ids: tuple[str, ...]
    identity_ids: tuple[str, ...]
    size_bytes: int
    sha256: str | None
    probe: ProbeMetadata
    probe_error: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["lineage_sequence_ids"] = list(self.lineage_sequence_ids)
        payload["identity_ids"] = list(self.identity_ids)
        return cast(dict[str, object], payload)


@dataclass(frozen=True)
class AuditIssue:
    """A machine-readable manifest audit finding."""

    severity: Severity
    code: str
    detail: str
    video_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["video_ids"] = list(self.video_ids)
        return cast(dict[str, object], payload)


@dataclass(frozen=True)
class AuditReport:
    """Audit result and aggregate evidence for a candidate manifest."""

    passed: bool
    generated_at_utc: str
    manifest_sha256: str
    summary: Mapping[str, object]
    official_split_sha256: Mapping[str, str]
    issues: tuple[AuditIssue, ...]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "generated_at_utc": self.generated_at_utc,
            "manifest_sha256": self.manifest_sha256,
            "summary": dict(self.summary),
            "official_split_sha256": dict(self.official_split_sha256),
            "issues": [issue.to_dict() for issue in self.issues],
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class BuildResult:
    """Candidate records plus their mandatory audit."""

    records: tuple[VideoRecord, ...]
    audit: AuditReport


def _require_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ManifestConfigurationError(f"{name} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestConfigurationError(f"{name} must be a non-empty string")
    return value.strip()


def _require_string_list(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ManifestConfigurationError(f"{name} must be a non-empty list")
    strings = tuple(_require_string(item, name) for item in value)
    if len(strings) != len(set(strings)):
        raise ManifestConfigurationError(f"{name} contains duplicates")
    return strings


def _expand_path(value: object, name: str, base_dir: Path) -> Path:
    raw_path = _require_string(value, name)
    missing = sorted(
        variable
        for variable in set(_ENVIRONMENT_VARIABLE.findall(raw_path))
        if not os.environ.get(variable)
    )
    if missing:
        raise ManifestConfigurationError(
            f"{name} references unset environment variable(s): {', '.join(missing)}"
        )
    expanded = _ENVIRONMENT_VARIABLE.sub(lambda match: os.environ[match.group(1)], raw_path)
    path = Path(os.path.expanduser(expanded))
    return (path if path.is_absolute() else base_dir / path).resolve()


def load_config(path: Path) -> FaceForensicsConfig:
    """Load and validate a YAML manifest configuration."""

    config_path = path.resolve()
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = _require_mapping(loaded, "configuration")
    if root.get("schema_version") != 1:
        raise ManifestConfigurationError("schema_version must be 1")
    dataset = _require_mapping(root.get("dataset"), "dataset")
    selection = _require_mapping(root.get("selection"), "selection")
    split = _require_mapping(root.get("split"), "split")
    integrity = _require_mapping(root.get("integrity"), "integrity")
    output = _require_mapping(root.get("output"), "output")
    base_dir = config_path.parent

    access_date_value = dataset.get("access_date")
    if isinstance(access_date_value, date):
        parsed_access_date = access_date_value
    else:
        access_date_raw = _require_string(access_date_value, "dataset.access_date")
        try:
            parsed_access_date = date.fromisoformat(access_date_raw)
        except ValueError as exc:
            raise ManifestConfigurationError("dataset.access_date must use YYYY-MM-DD") from exc
    if parsed_access_date > date.today():
        raise ManifestConfigurationError("dataset.access_date cannot be in the future")

    compressions = _require_string_list(selection.get("compressions"), "selection.compressions")
    unknown_compressions = sorted(set(compressions) - _KNOWN_COMPRESSIONS)
    if unknown_compressions:
        raise ManifestConfigurationError(f"unsupported compression values: {unknown_compressions}")
    methods = _require_string_list(
        selection.get("manipulation_methods"), "selection.manipulation_methods"
    )
    unknown_methods = sorted(set(methods) - _KNOWN_MANIPULATION_METHODS)
    if unknown_methods:
        raise ManifestConfigurationError(f"unsupported manipulation methods: {unknown_methods}")
    unsafe_components = [
        value for value in (*compressions, *methods) if not _SAFE_PATH_COMPONENT.fullmatch(value)
    ]
    if unsafe_components:
        raise ManifestConfigurationError(f"unsafe path components: {unsafe_components}")

    expected_raw = _require_mapping(selection.get("expected_counts"), "selection.expected_counts")
    expected_counts: dict[str, int] = {}
    for method, count in expected_raw.items():
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ManifestConfigurationError(
                f"selection.expected_counts.{method} must be a non-negative integer"
            )
        expected_counts[method] = count
    selected_categories = {"original", *methods}
    if set(expected_counts) != selected_categories:
        raise ManifestConfigurationError(
            "selection.expected_counts keys must exactly match original and the selected methods"
        )

    checksum_mode = _require_string(integrity.get("checksum"), "integrity.checksum")
    if checksum_mode not in ("none", "sha256"):
        raise ManifestConfigurationError("integrity.checksum must be none or sha256")
    probe_mode = _require_string(integrity.get("ffprobe"), "integrity.ffprobe")
    if probe_mode not in ("none", "optional", "required"):
        raise ManifestConfigurationError("integrity.ffprobe must be none, optional, or required")
    workers = integrity.get("workers", 4)
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1 or workers > 32:
        raise ManifestConfigurationError("integrity.workers must be an integer from 1 to 32")

    identity_map_value = split.get("identity_map")
    identity_map_path = (
        None
        if identity_map_value is None
        else _expand_path(identity_map_value, "split.identity_map", base_dir)
    )
    require_complete = split.get("require_complete_official_sources", True)
    if not isinstance(require_complete, bool):
        raise ManifestConfigurationError(
            "split.require_complete_official_sources must be true or false"
        )

    manifest_filename = _require_string(output.get("manifest"), "output.manifest")
    audit_filename = _require_string(output.get("audit"), "output.audit")
    for value, name, suffix in (
        (manifest_filename, "output.manifest", ".jsonl"),
        (audit_filename, "output.audit", ".json"),
    ):
        if Path(value).name != value or not value.endswith(suffix):
            raise ManifestConfigurationError(f"{name} must be a plain {suffix} filename")
    if manifest_filename == audit_filename:
        raise ManifestConfigurationError("manifest and audit filenames must differ")

    return FaceForensicsConfig(
        dataset_root=_expand_path(dataset.get("root"), "dataset.root", base_dir),
        dataset_release=_require_string(dataset.get("release"), "dataset.release"),
        access_date=parsed_access_date,
        output_dir=_expand_path(output.get("directory"), "output.directory", base_dir),
        split_dir=_expand_path(
            split.get("official_directory"), "split.official_directory", base_dir
        ),
        compressions=compressions,
        manipulation_methods=methods,
        expected_counts=expected_counts,
        identity_map_path=identity_map_path,
        checksum_mode=cast(ChecksumMode, checksum_mode),
        probe_mode=cast(ProbeMode, probe_mode),
        workers=workers,
        require_complete_official_sources=require_complete,
        manifest_filename=manifest_filename,
        audit_filename=audit_filename,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_official_splits(split_dir: Path) -> tuple[dict[str, Split], dict[str, str]]:
    """Load the three official split files and reject overlapping source IDs."""

    assignments: dict[str, Split] = {}
    hashes: dict[str, str] = {}
    for split_name in VALID_SPLITS:
        path = split_dir / f"{split_name}.json"
        if not path.is_file():
            raise ManifestConfigurationError(f"official split file is missing: {path}")
        hashes[split_name] = _sha256_file(path)
        payload: object = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ManifestConfigurationError(f"official split file must contain a list: {path}")
        for pair_index, pair in enumerate(payload):
            if not isinstance(pair, list) or len(pair) != 2:
                raise ManifestConfigurationError(
                    f"{path} pair {pair_index} must contain exactly two sequence IDs"
                )
            for raw_sequence_id in pair:
                sequence_id = _require_string(raw_sequence_id, f"{path} sequence ID")
                if not _ORIGINAL_STEM.fullmatch(sequence_id):
                    raise ManifestConfigurationError(
                        f"invalid official sequence ID {sequence_id!r} in {path}"
                    )
                previous = assignments.get(sequence_id)
                if previous is not None:
                    raise ManifestConfigurationError(
                        f"sequence {sequence_id} appears in both {previous} and {split_name}"
                    )
                assignments[sequence_id] = cast(Split, split_name)
    return assignments, hashes


def load_identity_map(path: Path | None) -> dict[str, str]:
    """Load an optional, user-supplied sequence-to-identity map."""

    if path is None:
        return {}
    if not path.is_file():
        raise ManifestConfigurationError(f"identity map is missing: {path}")
    identities: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"sequence_id", "identity_id"}.issubset(
            reader.fieldnames
        ):
            raise ManifestConfigurationError(
                "identity map must contain sequence_id and identity_id columns"
            )
        for row_number, row in enumerate(reader, start=2):
            sequence_id = (row.get("sequence_id") or "").strip()
            identity_id = (row.get("identity_id") or "").strip()
            if not _ORIGINAL_STEM.fullmatch(sequence_id) or not identity_id:
                raise ManifestConfigurationError(f"invalid identity map row {row_number}")
            previous = identities.get(sequence_id)
            if previous is not None and previous != identity_id:
                raise ManifestConfigurationError(
                    f"sequence {sequence_id} maps to multiple identities"
                )
            identities[sequence_id] = identity_id
    return identities


def _parse_rate(raw_rate: object) -> float | None:
    if not isinstance(raw_rate, str) or raw_rate in {"", "0/0", "N/A"}:
        return None
    try:
        return float(Fraction(raw_rate))
    except (ValueError, ZeroDivisionError):
        return None


def _optional_int(value: object) -> int | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return int(cast(Any, value))
    except (TypeError, ValueError):
        return None


def _optional_float(value: object) -> float | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError):
        return None


def probe_video(path: Path, executable: str = "ffprobe") -> ProbeMetadata:
    """Read the first video stream without decoding the complete file."""

    result = subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,nb_frames,duration:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        check=False,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(detail or f"ffprobe exited with code {result.returncode}")
    payload: object = json.loads(result.stdout)
    root = _require_mapping(payload, "ffprobe output")
    streams = root.get("streams")
    if not isinstance(streams, list) or not streams:
        raise RuntimeError("ffprobe found no video stream")
    stream = _require_mapping(streams[0], "ffprobe video stream")
    format_data = root.get("format")
    container = _require_mapping(format_data, "ffprobe format") if format_data else {}
    duration = _optional_float(stream.get("duration")) or _optional_float(container.get("duration"))
    return ProbeMetadata(
        codec=str(stream["codec_name"]) if stream.get("codec_name") else None,
        width=_optional_int(stream.get("width")),
        height=_optional_int(stream.get("height")),
        average_fps=_parse_rate(stream.get("avg_frame_rate")),
        frame_count=_optional_int(stream.get("nb_frames")),
        duration_seconds=duration,
    )


def _stable_video_id(dataset_release: str, relative_path: str) -> str:
    value = f"{DATASET_NAME}\0{dataset_release}\0{relative_path}".encode()
    return f"ffpp_{hashlib.sha256(value).hexdigest()[:20]}"


def _split_for_lineage(
    lineage: tuple[str, ...], assignments: Mapping[str, Split]
) -> tuple[Split, AuditIssue | None]:
    present = {assignments[item] for item in lineage if item in assignments}
    if not present:
        return "unassigned", None
    if len(present) > 1:
        return (
            "unassigned",
            AuditIssue(
                "error",
                "cross_split_lineage",
                f"lineage {lineage} maps to multiple official splits: {sorted(present)}",
            ),
        )
    if len(present) == 1 and not all(item in assignments for item in lineage):
        missing = sorted(item for item in lineage if item not in assignments)
        return (
            "unassigned",
            AuditIssue(
                "error",
                "partially_assigned_lineage",
                f"lineage {lineage} has unassigned sequence IDs: {missing}",
            ),
        )
    return next(iter(present)), None


def _record_from_path(
    path: Path,
    config: FaceForensicsConfig,
    assignments: Mapping[str, Split],
    identities: Mapping[str, str],
    ffprobe_executable: str | None,
) -> tuple[VideoRecord, tuple[AuditIssue, ...]]:
    initial_stat = path.stat()
    relative_path = path.relative_to(config.dataset_root).as_posix()
    parts = Path(relative_path).parts
    issues: list[AuditIssue] = []
    lineage: tuple[str, ...]
    if parts[:2] == ("original_sequences", "youtube") and len(parts) >= 5:
        compression = parts[2]
        method = "original"
        match = _ORIGINAL_STEM.fullmatch(path.stem)
        if match is None:
            raise ManifestConfigurationError(f"invalid original filename: {relative_path}")
        target_id = match.group(1)
        donor_id = None
        lineage = (target_id,)
        label = 0
        class_name: Literal["real", "manipulated"] = "real"
    elif parts[0] == "manipulated_sequences" and len(parts) >= 5:
        method = parts[1]
        compression = parts[2]
        match = _MANIPULATED_STEM.fullmatch(path.stem)
        if match is None:
            raise ManifestConfigurationError(f"invalid manipulated filename: {relative_path}")
        target_id, donor_id = match.groups()
        lineage = tuple(sorted({target_id, donor_id}))
        label = 1
        class_name = "manipulated"
        if target_id == donor_id:
            issues.append(
                AuditIssue(
                    "error",
                    "self_manipulation_lineage",
                    f"target and donor sequence are both {target_id}",
                )
            )
    else:
        raise ManifestConfigurationError(f"unsupported FaceForensics++ path: {relative_path}")

    split, split_issue = _split_for_lineage(lineage, assignments)
    if split_issue is not None:
        issues.append(split_issue)
    identity_ids = tuple(sorted({identities[item] for item in lineage if item in identities}))
    if identities:
        missing_identities = sorted(item for item in lineage if item not in identities)
        if missing_identities:
            issues.append(
                AuditIssue(
                    "error",
                    "missing_identity_mapping",
                    f"sequence IDs missing from identity map: {missing_identities}",
                )
            )

    sha256 = _sha256_file(path) if config.checksum_mode == "sha256" else None
    probe = ProbeMetadata()
    probe_error = None
    if ffprobe_executable is not None:
        try:
            probe = probe_video(path, ffprobe_executable)
            incomplete = [
                field
                for field, valid in (
                    ("codec", bool(probe.codec)),
                    ("width", probe.width is not None and probe.width > 0),
                    ("height", probe.height is not None and probe.height > 0),
                    ("average_fps", probe.average_fps is not None and probe.average_fps > 0),
                    (
                        "duration_seconds",
                        probe.duration_seconds is not None and probe.duration_seconds > 0,
                    ),
                )
                if not valid
            ]
            if incomplete:
                probe_severity: Severity = "error" if config.probe_mode == "required" else "warning"
                issues.append(
                    AuditIssue(
                        probe_severity,
                        "incomplete_video_probe",
                        f"ffprobe metadata is missing/invalid: {incomplete}",
                    )
                )
        except (
            OSError,
            RuntimeError,
            subprocess.SubprocessError,
            json.JSONDecodeError,
            ManifestConfigurationError,
        ) as exc:
            probe_error = str(exc)
            probe_failure_severity: Severity = (
                "error" if config.probe_mode == "required" else "warning"
            )
            issues.append(AuditIssue(probe_failure_severity, "video_probe_failed", probe_error))

    final_stat = path.stat()
    size_bytes = final_stat.st_size
    if (initial_stat.st_size, initial_stat.st_mtime_ns) != (
        final_stat.st_size,
        final_stat.st_mtime_ns,
    ):
        issues.append(
            AuditIssue("error", "file_changed_during_audit", "file size or timestamp changed")
        )
    if size_bytes == 0:
        issues.append(AuditIssue("error", "empty_video", "video file contains zero bytes"))
    record = VideoRecord(
        schema_version=SCHEMA_VERSION,
        dataset=DATASET_NAME,
        dataset_release=config.dataset_release,
        access_date=config.access_date.isoformat(),
        video_id=_stable_video_id(config.dataset_release, relative_path),
        relative_path=relative_path,
        split=split,
        label=label,
        class_name=class_name,
        manipulation_method=method,
        compression=compression,
        target_sequence_id=target_id,
        donor_sequence_id=donor_id,
        lineage_sequence_ids=lineage,
        identity_ids=identity_ids,
        size_bytes=size_bytes,
        sha256=sha256,
        probe=probe,
        probe_error=probe_error,
    )
    return record, tuple(issues)


def _inventory_paths(config: FaceForensicsConfig) -> tuple[list[Path], list[AuditIssue]]:
    paths: list[Path] = []
    issues: list[AuditIssue] = []
    expected_directories: list[Path] = []
    for compression in config.compressions:
        expected_directories.append(
            config.dataset_root / "original_sequences" / "youtube" / compression / "videos"
        )
        expected_directories.extend(
            config.dataset_root / "manipulated_sequences" / method / compression / "videos"
            for method in config.manipulation_methods
        )
    for directory in expected_directories:
        if not directory.is_dir():
            issues.append(
                AuditIssue(
                    "error",
                    "missing_expected_directory",
                    f"expected dataset directory is missing: {directory}",
                )
            )
            continue
        for path in directory.rglob("*.mp4"):
            if path.is_symlink():
                issues.append(
                    AuditIssue(
                        "error",
                        "symlink_video",
                        f"video symlinks are not accepted in an audited dataset: {path}",
                    )
                )
            elif path.is_file():
                paths.append(path)
    return sorted(set(paths), key=lambda item: item.as_posix().casefold()), issues


def _canonical_manifest(records: Sequence[VideoRecord]) -> bytes:
    lines = [
        json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")) for record in records
    ]
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def _issue_with_video(issue: AuditIssue, video_id: str) -> AuditIssue:
    return replace(issue, video_ids=(video_id,))


def audit_records(
    records: Sequence[VideoRecord],
    *,
    expected_counts: Mapping[str, int],
    official_assignments: Mapping[str, Split],
    official_split_hashes: Mapping[str, str],
    require_complete_official_sources: bool,
    identity_map_supplied: bool,
    initial_issues: Sequence[AuditIssue] = (),
) -> AuditReport:
    """Audit duplicates and every available source/identity leakage boundary."""

    issues = list(initial_issues)
    sorted_records = tuple(sorted(records, key=lambda record: record.relative_path.casefold()))

    by_path: defaultdict[str, list[VideoRecord]] = defaultdict(list)
    by_video_id: defaultdict[str, list[VideoRecord]] = defaultdict(list)
    by_checksum: defaultdict[str, list[VideoRecord]] = defaultdict(list)
    source_splits: defaultdict[str, set[Split]] = defaultdict(set)
    identity_splits: defaultdict[str, set[Split]] = defaultdict(set)
    for record in sorted_records:
        by_path[record.relative_path.casefold()].append(record)
        by_video_id[record.video_id].append(record)
        if record.sha256:
            by_checksum[record.sha256].append(record)
        if record.split == "unassigned":
            issues.append(
                AuditIssue(
                    "error",
                    "unassigned_video",
                    f"video has no valid split: {record.relative_path}",
                    (record.video_id,),
                )
            )
        for sequence_id in record.lineage_sequence_ids:
            source_splits[sequence_id].add(record.split)
        for identity_id in record.identity_ids:
            identity_splits[identity_id].add(record.split)

    for key, duplicates in by_path.items():
        if len(duplicates) > 1:
            issues.append(
                AuditIssue(
                    "error",
                    "duplicate_relative_path",
                    f"relative path occurs {len(duplicates)} times: {key}",
                    tuple(record.video_id for record in duplicates),
                )
            )
    for video_id, duplicates in by_video_id.items():
        if len(duplicates) > 1:
            issues.append(
                AuditIssue(
                    "error",
                    "duplicate_video_id",
                    f"video ID occurs {len(duplicates)} times: {video_id}",
                    tuple(record.video_id for record in duplicates),
                )
            )
    for digest, duplicates in by_checksum.items():
        unique_paths = {record.relative_path.casefold() for record in duplicates}
        if len(unique_paths) <= 1:
            continue
        splits = {record.split for record in duplicates}
        severity: Severity = "error" if len(splits) > 1 else "warning"
        issues.append(
            AuditIssue(
                severity,
                "cross_split_content_duplicate" if severity == "error" else "content_duplicate",
                f"SHA-256 {digest} occurs at {len(unique_paths)} paths in splits {sorted(splits)}",
                tuple(record.video_id for record in duplicates),
            )
        )
    for sequence_id, splits in source_splits.items():
        assigned = {split for split in splits if split != "unassigned"}
        if len(assigned) > 1:
            issues.append(
                AuditIssue(
                    "error",
                    "source_sequence_leakage",
                    f"source sequence {sequence_id} occurs in splits {sorted(assigned)}",
                )
            )
    for identity_id, splits in identity_splits.items():
        assigned = {split for split in splits if split != "unassigned"}
        if len(assigned) > 1:
            issues.append(
                AuditIssue(
                    "error",
                    "identity_leakage",
                    f"identity {identity_id} occurs in splits {sorted(assigned)}",
                )
            )

    method_counts = Counter(record.manipulation_method for record in sorted_records)
    for method, expected in expected_counts.items():
        actual = method_counts.get(method, 0)
        if actual != expected:
            issues.append(
                AuditIssue(
                    "error",
                    "unexpected_method_count",
                    f"{method}: expected {expected}, found {actual}",
                )
            )

    observed_sources = set(source_splits)
    official_sources = set(official_assignments)
    if require_complete_official_sources and observed_sources != official_sources:
        missing = sorted(official_sources - observed_sources)
        unexpected = sorted(observed_sources - official_sources)
        detail = (
            f"official source coverage mismatch: {len(missing)} missing, "
            f"{len(unexpected)} unexpected"
        )
        if missing:
            detail += f"; first missing={missing[:10]}"
        if unexpected:
            detail += f"; first unexpected={unexpected[:10]}"
        issues.append(AuditIssue("error", "official_source_coverage", detail))

    if sorted_records and all(record.sha256 is None for record in sorted_records):
        issues.append(
            AuditIssue(
                "warning",
                "checksums_disabled",
                "content-duplicate detection was not performed",
            )
        )

    limitations: list[str] = []
    if not identity_map_supplied:
        limitations.append(
            "FaceForensics++ does not provide a verified identity map in the selected public "
            "metadata. Source-sequence disjointness is verified; identity disjointness is not "
            "claimed until an approved identity map is supplied."
        )
    limitations.append(
        "All extracted frames and clips must inherit their parent video_id and split; this "
        "pipeline intentionally does not create independent frame-level splits."
    )

    summary: dict[str, object] = {
        "records": len(sorted_records),
        "by_split": dict(sorted(Counter(record.split for record in sorted_records).items())),
        "by_label": dict(sorted(Counter(record.class_name for record in sorted_records).items())),
        "by_method": dict(sorted(method_counts.items())),
        "by_compression": dict(
            sorted(Counter(record.compression for record in sorted_records).items())
        ),
        "unique_source_sequences": len(observed_sources),
        "unique_identities": len(identity_splits),
        "checksummed_records": sum(record.sha256 is not None for record in sorted_records),
        "successfully_probed_records": sum(
            record.probe_error is None and record.probe.codec is not None
            for record in sorted_records
        ),
        "errors": sum(issue.severity == "error" for issue in issues),
        "warnings": sum(issue.severity == "warning" for issue in issues),
    }
    manifest_bytes = _canonical_manifest(sorted_records)
    return AuditReport(
        passed=not any(issue.severity == "error" for issue in issues),
        generated_at_utc=datetime.now(UTC).isoformat(),
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        summary=summary,
        official_split_sha256=dict(sorted(official_split_hashes.items())),
        issues=tuple(issues),
        limitations=tuple(limitations),
    )


def build_manifest(config: FaceForensicsConfig) -> BuildResult:
    """Inventory selected videos, assign lineage-safe splits, and audit the result."""

    if not config.dataset_root.is_dir():
        raise ManifestConfigurationError(f"dataset root is missing: {config.dataset_root}")
    if (config.dataset_root / ".git").exists():
        raise ManifestConfigurationError("dataset root must not be a Git repository")
    assignments, split_hashes = load_official_splits(config.split_dir)
    identities = load_identity_map(config.identity_map_path)
    paths, inventory_issues = _inventory_paths(config)
    if config.identity_map_path is not None:
        missing_identity_rows = sorted(set(assignments) - set(identities))
        unexpected_identity_rows = sorted(set(identities) - set(assignments))
        if missing_identity_rows or unexpected_identity_rows:
            inventory_issues.append(
                AuditIssue(
                    "error",
                    "identity_map_coverage",
                    (
                        f"identity map coverage mismatch: {len(missing_identity_rows)} missing, "
                        f"{len(unexpected_identity_rows)} unexpected"
                    ),
                )
            )

    ffprobe_executable = None
    if config.probe_mode != "none":
        ffprobe_executable = shutil.which("ffprobe")
        if ffprobe_executable is None:
            inventory_issues.append(
                AuditIssue(
                    "error" if config.probe_mode == "required" else "warning",
                    "ffprobe_missing",
                    f"ffprobe is {config.probe_mode} but is not on PATH",
                )
            )

    records: list[VideoRecord] = []
    record_issues: list[AuditIssue] = []

    def process(path: Path) -> tuple[VideoRecord, tuple[AuditIssue, ...]]:
        return _record_from_path(path, config, assignments, identities, ffprobe_executable)

    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        for record, issues in executor.map(process, paths):
            records.append(record)
            record_issues.extend(_issue_with_video(issue, record.video_id) for issue in issues)

    sorted_records = tuple(sorted(records, key=lambda record: record.relative_path.casefold()))
    report = audit_records(
        sorted_records,
        expected_counts=config.expected_counts,
        official_assignments=assignments,
        official_split_hashes=split_hashes,
        require_complete_official_sources=config.require_complete_official_sources,
        identity_map_supplied=config.identity_map_path is not None,
        initial_issues=(*inventory_issues, *record_issues),
    )
    return BuildResult(sorted_records, report)


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def _verification_mapping(value: object, name: str) -> dict[str, object]:
    try:
        return _require_mapping(value, name)
    except ManifestConfigurationError as exc:
        raise ManifestVerificationError(str(exc)) from exc


def load_verified_manifest(manifest_path: Path, audit_path: Path) -> tuple[dict[str, object], ...]:
    """Load only a passing, hash-matched manifest/audit pair for downstream use."""

    try:
        audit_payload: object = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestVerificationError(f"cannot read audit: {exc}") from exc
    audit = _verification_mapping(audit_payload, "audit")
    if audit.get("passed") is not True:
        raise ManifestVerificationError("audit did not pass")
    expected_hash = audit.get("manifest_sha256")
    if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ManifestVerificationError("audit manifest_sha256 is invalid")
    try:
        manifest_content = manifest_path.read_bytes()
    except OSError as exc:
        raise ManifestVerificationError(f"cannot read manifest: {exc}") from exc
    actual_hash = hashlib.sha256(manifest_content).hexdigest()
    if actual_hash != expected_hash:
        raise ManifestVerificationError(
            f"manifest hash mismatch: expected {expected_hash}, found {actual_hash}"
        )

    records: list[dict[str, object]] = []
    required_fields = {
        "schema_version",
        "dataset",
        "video_id",
        "relative_path",
        "split",
        "lineage_sequence_ids",
        "sha256",
        "probe",
    }
    for line_number, line in enumerate(manifest_content.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload: object = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ManifestVerificationError(f"invalid JSON on manifest line {line_number}") from exc
        record = _verification_mapping(payload, f"manifest line {line_number}")
        missing = sorted(required_fields - set(record))
        if missing:
            raise ManifestVerificationError(
                f"manifest line {line_number} is missing fields: {missing}"
            )
        if record.get("schema_version") != SCHEMA_VERSION or record.get("dataset") != DATASET_NAME:
            raise ManifestVerificationError(
                f"manifest line {line_number} has an unsupported dataset/schema"
            )
        if record.get("split") not in VALID_SPLITS:
            raise ManifestVerificationError(f"manifest line {line_number} has an invalid split")
        relative_path = record.get("relative_path")
        relative_parts = Path(relative_path).parts if isinstance(relative_path, str) else ()
        if (
            not isinstance(relative_path, str)
            or Path(relative_path).is_absolute()
            or "\\" in relative_path
            or not relative_parts
            or any(part in {".", ".."} for part in relative_parts)
        ):
            raise ManifestVerificationError(
                f"manifest line {line_number} does not contain a safe relative path"
            )
        video_id = record.get("video_id")
        if not isinstance(video_id, str) or not re.fullmatch(r"ffpp_[0-9a-f]{20}", video_id):
            raise ManifestVerificationError(f"manifest line {line_number} has an invalid video ID")
        digest = record.get("sha256")
        if digest is not None and (
            not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise ManifestVerificationError(f"manifest line {line_number} has an invalid SHA-256")
        records.append(record)

    summary = audit.get("summary")
    summary_mapping = _verification_mapping(summary, "audit summary")
    if summary_mapping.get("records") != len(records):
        raise ManifestVerificationError("manifest record count does not match its audit")
    return tuple(records)


def _quarantine_manifest(path: Path) -> Path:
    digest = _sha256_file(path)[:12]
    candidate = path.with_name(f"{path.name}.stale-{digest}")
    suffix = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.stale-{digest}-{suffix}")
        suffix += 1
    path.replace(candidate)
    return candidate


def write_build_outputs(
    result: BuildResult, config: FaceForensicsConfig
) -> tuple[Path | None, Path, Path | None]:
    """Write the audit always, and publish a manifest only after a passing audit."""

    audit_path = config.output_dir / config.audit_filename
    audit_content = (json.dumps(result.audit.to_dict(), indent=2, sort_keys=True) + "\n").encode()
    _atomic_write(audit_path, audit_content)

    expected_manifest_path = config.output_dir / config.manifest_filename
    manifest_path: Path | None = None
    quarantined_path: Path | None = None
    if result.audit.passed:
        manifest_path = expected_manifest_path
        content = _canonical_manifest(result.records)
        if hashlib.sha256(content).hexdigest() != result.audit.manifest_sha256:
            raise RuntimeError("manifest changed after audit")
        _atomic_write(manifest_path, content)
    elif expected_manifest_path.exists():
        quarantined_path = _quarantine_manifest(expected_manifest_path)
    return manifest_path, audit_path, quarantined_path

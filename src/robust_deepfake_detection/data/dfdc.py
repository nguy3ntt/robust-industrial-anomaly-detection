"""Leakage-safe manifest construction for the DFDC Preview dataset.

The provider metadata identifies the target and swapped actors for every clip.
Those identities, rather than filenames or frames, are the unit of splitting.
The provider test split is immutable; validation is derived deterministically
from connected identity groups in the provider training split.
"""

from __future__ import annotations

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
from pathlib import Path, PurePosixPath
from typing import Literal, cast

import yaml

from robust_deepfake_detection.data.faceforensics import (
    AuditIssue,
    ChecksumMode,
    ManifestVerificationError,
    ProbeMetadata,
    ProbeMode,
    Severity,
    probe_video,
)

Split = Literal["train", "val", "test", "unassigned"]
ProviderSplit = Literal["train", "test"]

SCHEMA_VERSION = "1.0"
DATASET_NAME = "dfdc_preview"
VALID_SPLITS: tuple[Split, ...] = ("train", "val", "test")
_ENVIRONMENT_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_VIDEO_ID = re.compile(r"dfdcp_[0-9a-f]{20}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_METHOD_BY_DIRECTORY = {
    "original_videos": "original",
    "method_A": "method_A",
    "method_B": "method_B",
}
_TARGET_KEYS = ("target", "target_id", "target_identity", "target_identity_id")
_SWAPPED_KEYS = (
    "swapped",
    "swapped_id",
    "swapped_identity",
    "swapped_identity_id",
    "source",
    "source_id",
    "source_identity",
)


class DfdcConfigurationError(ValueError):
    """Raised when DFDC configuration or provider metadata is unsafe."""


@dataclass(frozen=True)
class DfdcConfig:
    """Validated settings for one DFDC Preview manifest build."""

    dataset_root: Path
    metadata_path: Path
    dataset_release: str
    access_date: date
    output_dir: Path
    validation_fraction: float = 0.15
    split_seed: int = 20260914
    minimum_records: int = 5000
    expected_identity_count: int | None = 66
    checksum_mode: ChecksumMode = "sha256"
    probe_mode: ProbeMode = "required"
    workers: int = 4
    manifest_filename: str = "dfdc-preview.jsonl"
    audit_filename: str = "dfdc-preview.audit.json"


@dataclass(frozen=True)
class ProviderEntry:
    """Normalized information from one provider dataset.json entry."""

    relative_path: str
    provider_split: ProviderSplit
    label: int
    class_name: Literal["real", "manipulated"]
    manipulation_method: str
    target_identity_id: str
    swapped_identity_id: str | None
    lineage_identity_ids: tuple[str, ...]
    augmentations: tuple[str, ...]


@dataclass(frozen=True)
class DfdcVideoRecord:
    """One deterministic, root-relative DFDC Preview video record."""

    schema_version: str
    dataset: str
    dataset_release: str
    access_date: str
    video_id: str
    relative_path: str
    split: Split
    provider_split: ProviderSplit
    label: int
    class_name: Literal["real", "manipulated"]
    manipulation_method: str
    target_identity_id: str
    swapped_identity_id: str | None
    lineage_identity_ids: tuple[str, ...]
    augmentations: tuple[str, ...]
    size_bytes: int
    sha256: str | None
    probe: ProbeMetadata
    probe_error: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["lineage_identity_ids"] = list(self.lineage_identity_ids)
        payload["augmentations"] = list(self.augmentations)
        return cast(dict[str, object], payload)


@dataclass(frozen=True)
class DfdcAuditReport:
    """Audit evidence for a candidate DFDC Preview manifest."""

    passed: bool
    generated_at_utc: str
    manifest_sha256: str
    provider_metadata_sha256: str
    split_policy: Mapping[str, object]
    summary: Mapping[str, object]
    issues: tuple[AuditIssue, ...]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "generated_at_utc": self.generated_at_utc,
            "manifest_sha256": self.manifest_sha256,
            "provider_metadata_sha256": self.provider_metadata_sha256,
            "split_policy": dict(self.split_policy),
            "summary": dict(self.summary),
            "issues": [issue.to_dict() for issue in self.issues],
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class DfdcBuildResult:
    """Candidate DFDC records and their mandatory audit."""

    records: tuple[DfdcVideoRecord, ...]
    audit: DfdcAuditReport


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise DfdcConfigurationError(f"{name} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DfdcConfigurationError(f"{name} must be a non-empty string")
    return value.strip()


def _expand_path(value: object, name: str, base_dir: Path) -> Path:
    raw_path = _string(value, name)
    missing = sorted(
        variable
        for variable in set(_ENVIRONMENT_VARIABLE.findall(raw_path))
        if not os.environ.get(variable)
    )
    if missing:
        raise DfdcConfigurationError(
            f"{name} references unset environment variable(s): {', '.join(missing)}"
        )
    expanded = _ENVIRONMENT_VARIABLE.sub(lambda match: os.environ[match.group(1)], raw_path)
    path = Path(os.path.expanduser(expanded))
    return (path if path.is_absolute() else base_dir / path).resolve()


def load_config(path: Path) -> DfdcConfig:
    """Load and validate a DFDC Preview YAML configuration."""

    config_path = path.resolve()
    root = _mapping(yaml.safe_load(config_path.read_text(encoding="utf-8")), "configuration")
    if root.get("schema_version") != 1:
        raise DfdcConfigurationError("schema_version must be 1")
    dataset = _mapping(root.get("dataset"), "dataset")
    split = _mapping(root.get("split"), "split")
    integrity = _mapping(root.get("integrity"), "integrity")
    output = _mapping(root.get("output"), "output")
    base_dir = config_path.parent

    access_date_value = dataset.get("access_date")
    if isinstance(access_date_value, date):
        access_date = access_date_value
    else:
        try:
            access_date = date.fromisoformat(_string(access_date_value, "dataset.access_date"))
        except ValueError as exc:
            raise DfdcConfigurationError("dataset.access_date must use YYYY-MM-DD") from exc
    if access_date > date.today():
        raise DfdcConfigurationError("dataset.access_date cannot be in the future")

    validation_fraction = split.get("validation_fraction", 0.15)
    if (
        not isinstance(validation_fraction, (int, float))
        or isinstance(validation_fraction, bool)
        or not 0.0 < float(validation_fraction) < 0.5
    ):
        raise DfdcConfigurationError("split.validation_fraction must be between 0 and 0.5")
    split_seed = split.get("seed", 20260914)
    if not isinstance(split_seed, int) or isinstance(split_seed, bool) or split_seed < 0:
        raise DfdcConfigurationError("split.seed must be a non-negative integer")

    minimum_records = dataset.get("minimum_records", 5000)
    if not isinstance(minimum_records, int) or isinstance(minimum_records, bool):
        raise DfdcConfigurationError("dataset.minimum_records must be an integer")
    if minimum_records < 1:
        raise DfdcConfigurationError("dataset.minimum_records must be positive")
    expected_identity_count = dataset.get("expected_identity_count", 66)
    if expected_identity_count is not None and (
        not isinstance(expected_identity_count, int)
        or isinstance(expected_identity_count, bool)
        or expected_identity_count < 1
    ):
        raise DfdcConfigurationError(
            "dataset.expected_identity_count must be null or a positive integer"
        )

    checksum_mode = _string(integrity.get("checksum"), "integrity.checksum")
    if checksum_mode not in ("none", "sha256"):
        raise DfdcConfigurationError("integrity.checksum must be none or sha256")
    probe_mode = _string(integrity.get("ffprobe"), "integrity.ffprobe")
    if probe_mode not in ("none", "optional", "required"):
        raise DfdcConfigurationError("integrity.ffprobe must be none, optional, or required")
    workers = integrity.get("workers", 4)
    if not isinstance(workers, int) or isinstance(workers, bool) or not 1 <= workers <= 32:
        raise DfdcConfigurationError("integrity.workers must be an integer from 1 to 32")

    manifest_filename = _string(output.get("manifest"), "output.manifest")
    audit_filename = _string(output.get("audit"), "output.audit")
    for value, name, suffix in (
        (manifest_filename, "output.manifest", ".jsonl"),
        (audit_filename, "output.audit", ".json"),
    ):
        if Path(value).name != value or not value.endswith(suffix):
            raise DfdcConfigurationError(f"{name} must be a plain {suffix} filename")

    return DfdcConfig(
        dataset_root=_expand_path(dataset.get("root"), "dataset.root", base_dir),
        metadata_path=_expand_path(dataset.get("metadata"), "dataset.metadata", base_dir),
        dataset_release=_string(dataset.get("release"), "dataset.release"),
        access_date=access_date,
        output_dir=_expand_path(output.get("directory"), "output.directory", base_dir),
        validation_fraction=float(validation_fraction),
        split_seed=split_seed,
        minimum_records=minimum_records,
        expected_identity_count=expected_identity_count,
        checksum_mode=cast(ChecksumMode, checksum_mode),
        probe_mode=cast(ProbeMode, probe_mode),
        workers=workers,
        manifest_filename=manifest_filename,
        audit_filename=audit_filename,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative_path(raw_path: object) -> str:
    value = _string(raw_path, "provider metadata path").replace("\\", "/")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise DfdcConfigurationError(f"unsafe provider metadata path: {value!r}")
    if path.suffix.casefold() != ".mp4":
        raise DfdcConfigurationError(f"provider metadata path is not an MP4: {value!r}")
    return path.as_posix()


def _identity(entry: Mapping[str, object], keys: Sequence[str], name: str) -> str | None:
    present = [(key, entry[key]) for key in keys if key in entry]
    if not present:
        return None
    normalized: set[str] = set()
    for _, value in present:
        if value is None or value == "":
            continue
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise DfdcConfigurationError(f"{name} must be a string or integer identity")
        normalized.add(str(value).strip())
    if len(normalized) > 1:
        raise DfdcConfigurationError(f"conflicting aliases for {name}")
    return next(iter(normalized), None)


def _augmentations(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, list):
        return tuple(sorted({_string(item, "augmentation") for item in value}))
    if isinstance(value, Mapping):
        return tuple(sorted(str(key) for key, enabled in value.items() if enabled))
    raise DfdcConfigurationError("augmentations must be null, a string, list, or mapping")


def load_provider_metadata(path: Path) -> tuple[dict[str, ProviderEntry], str]:
    """Normalize provider dataset.json and fingerprint the original bytes."""

    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise DfdcConfigurationError(f"cannot read provider metadata: {exc}") from exc
    try:
        root = _mapping(json.loads(raw), "provider metadata")
    except json.JSONDecodeError as exc:
        raise DfdcConfigurationError(f"provider metadata is invalid JSON: {exc}") from exc

    entries: dict[str, ProviderEntry] = {}
    seen_paths: set[str] = set()
    for raw_path, raw_entry in root.items():
        relative_path = _safe_relative_path(raw_path)
        folded = relative_path.casefold()
        if folded in seen_paths:
            raise DfdcConfigurationError(
                f"duplicate case-insensitive metadata path: {relative_path}"
            )
        seen_paths.add(folded)
        entry = _mapping(raw_entry, f"provider entry {relative_path}")
        label_raw = _string(entry.get("label"), f"{relative_path}.label").casefold()
        if label_raw not in {"real", "fake"}:
            raise DfdcConfigurationError(f"{relative_path}.label must be real or fake")
        split_raw = _string(entry.get("split"), f"{relative_path}.split").casefold()
        if split_raw not in {"train", "test"}:
            raise DfdcConfigurationError(f"{relative_path}.split must be train or test")
        directory = PurePosixPath(relative_path).parts[0]
        method = _METHOD_BY_DIRECTORY.get(directory)
        if method is None:
            raise DfdcConfigurationError(f"unsupported DFDC Preview directory: {directory}")
        if (label_raw == "real") != (method == "original"):
            raise DfdcConfigurationError(
                f"label/directory disagreement for {relative_path}: {label_raw} in {directory}"
            )

        target = _identity(entry, _TARGET_KEYS, f"{relative_path}.target identity")
        swapped = _identity(entry, _SWAPPED_KEYS, f"{relative_path}.swapped identity")
        if target is None:
            raise DfdcConfigurationError(f"{relative_path} has no target identity")
        if label_raw == "fake" and swapped is None:
            raise DfdcConfigurationError(f"{relative_path} has no swapped identity")
        if label_raw == "real" and swapped is not None:
            raise DfdcConfigurationError(f"real video {relative_path} has a swapped identity")
        lineage = tuple(sorted({target, *(() if swapped is None else (swapped,))}))
        entries[relative_path] = ProviderEntry(
            relative_path=relative_path,
            provider_split=cast(ProviderSplit, split_raw),
            label=0 if label_raw == "real" else 1,
            class_name="real" if label_raw == "real" else "manipulated",
            manipulation_method=method,
            target_identity_id=target,
            swapped_identity_id=swapped,
            lineage_identity_ids=lineage,
            augmentations=_augmentations(entry.get("augmentations")),
        )
    return entries, hashlib.sha256(raw).hexdigest()


class _DisjointSets:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, first: str, second: str) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self.parent[max(first_root, second_root)] = min(first_root, second_root)


def derive_splits(
    entries: Sequence[ProviderEntry], validation_fraction: float, seed: int
) -> tuple[dict[str, Split], tuple[AuditIssue, ...], Mapping[str, object]]:
    """Preserve provider test and derive validation from train identity components."""

    issues: list[AuditIssue] = []
    provider_splits: defaultdict[str, set[ProviderSplit]] = defaultdict(set)
    train_sets = _DisjointSets()
    for entry in entries:
        for identity in entry.lineage_identity_ids:
            provider_splits[identity].add(entry.provider_split)
            if entry.provider_split == "train":
                train_sets.add(identity)
        if entry.provider_split == "train":
            for identity in entry.lineage_identity_ids[1:]:
                train_sets.union(entry.lineage_identity_ids[0], identity)

    for identity, splits in sorted(provider_splits.items()):
        if len(splits) > 1:
            issues.append(
                AuditIssue(
                    "error",
                    "provider_identity_leakage",
                    f"identity {identity} occurs in provider splits {sorted(splits)}",
                )
            )

    components: defaultdict[str, set[str]] = defaultdict(set)
    for identity in train_sets.parent:
        components[train_sets.find(identity)].add(identity)
    ordered_components = sorted(
        (tuple(sorted(component)) for component in components.values()),
        key=lambda component: hashlib.sha256(f"{seed}\0{'|'.join(component)}".encode()).hexdigest(),
    )
    desired_val_identities = max(1, round(len(train_sets.parent) * validation_fraction))
    validation_identities: set[str] = set()
    for index, component in enumerate(ordered_components):
        if len(validation_identities) >= desired_val_identities:
            break
        if index == len(ordered_components) - 1:
            break
        validation_identities.update(component)
    if len(ordered_components) < 2:
        issues.append(
            AuditIssue(
                "error",
                "validation_split_impossible",
                "provider training identities form fewer than two disconnected groups",
            )
        )

    assignments: dict[str, Split] = {}
    for identity, provider_split_set in provider_splits.items():
        if provider_split_set == {"test"}:
            assignments[identity] = "test"
        elif provider_split_set == {"train"}:
            assignments[identity] = "val" if identity in validation_identities else "train"
        else:
            assignments[identity] = "unassigned"
    policy: Mapping[str, object] = {
        "provider_test_preserved": True,
        "validation_source": "provider_train_connected_identity_groups",
        "validation_fraction_requested": validation_fraction,
        "seed": seed,
        "training_identity_components": len(ordered_components),
        "validation_identities": len(validation_identities),
    }
    return assignments, tuple(issues), policy


def _record_split(entry: ProviderEntry, assignments: Mapping[str, Split]) -> Split:
    splits = {assignments.get(identity, "unassigned") for identity in entry.lineage_identity_ids}
    return next(iter(splits)) if len(splits) == 1 else "unassigned"


def _stable_video_id(dataset_release: str, relative_path: str) -> str:
    raw = f"{DATASET_NAME}\0{dataset_release}\0{relative_path}".encode()
    return f"dfdcp_{hashlib.sha256(raw).hexdigest()[:20]}"


def _record_from_entry(
    entry: ProviderEntry,
    config: DfdcConfig,
    assignments: Mapping[str, Split],
    ffprobe_executable: str | None,
) -> tuple[DfdcVideoRecord, tuple[AuditIssue, ...]]:
    path = config.dataset_root.joinpath(*PurePosixPath(entry.relative_path).parts)
    issues: list[AuditIssue] = []
    initial_stat = path.stat()
    split = _record_split(entry, assignments)
    if split == "unassigned":
        issues.append(
            AuditIssue(
                "error",
                "cross_split_lineage",
                f"identity lineage does not resolve to one split: {entry.lineage_identity_ids}",
            )
        )
    if entry.swapped_identity_id == entry.target_identity_id:
        issues.append(
            AuditIssue("error", "self_swap", "target and swapped identities are identical")
        )

    sha256 = _sha256_file(path) if config.checksum_mode == "sha256" else None
    probe = ProbeMetadata()
    probe_error = None
    if ffprobe_executable is not None:
        try:
            probe = probe_video(path, ffprobe_executable)
            incomplete = [
                name
                for name, valid in (
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
                severity: Severity = "error" if config.probe_mode == "required" else "warning"
                issues.append(
                    AuditIssue(
                        severity,
                        "incomplete_video_probe",
                        f"ffprobe metadata is missing/invalid: {incomplete}",
                    )
                )
        except (OSError, RuntimeError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            probe_error = str(exc)
            severity = "error" if config.probe_mode == "required" else "warning"
            issues.append(AuditIssue(severity, "video_probe_failed", probe_error))

    final_stat = path.stat()
    if (initial_stat.st_size, initial_stat.st_mtime_ns) != (
        final_stat.st_size,
        final_stat.st_mtime_ns,
    ):
        issues.append(AuditIssue("error", "file_changed_during_audit", "file changed"))
    if final_stat.st_size == 0:
        issues.append(AuditIssue("error", "empty_video", "video file contains zero bytes"))
    return (
        DfdcVideoRecord(
            schema_version=SCHEMA_VERSION,
            dataset=DATASET_NAME,
            dataset_release=config.dataset_release,
            access_date=config.access_date.isoformat(),
            video_id=_stable_video_id(config.dataset_release, entry.relative_path),
            relative_path=entry.relative_path,
            split=split,
            provider_split=entry.provider_split,
            label=entry.label,
            class_name=entry.class_name,
            manipulation_method=entry.manipulation_method,
            target_identity_id=entry.target_identity_id,
            swapped_identity_id=entry.swapped_identity_id,
            lineage_identity_ids=entry.lineage_identity_ids,
            augmentations=entry.augmentations,
            size_bytes=final_stat.st_size,
            sha256=sha256,
            probe=probe,
            probe_error=probe_error,
        ),
        tuple(issues),
    )


def _inventory(
    config: DfdcConfig, entries: Mapping[str, ProviderEntry]
) -> tuple[list[ProviderEntry], list[AuditIssue]]:
    issues: list[AuditIssue] = []
    found: dict[str, Path] = {}
    for directory in _METHOD_BY_DIRECTORY:
        root = config.dataset_root / directory
        if not root.is_dir():
            issues.append(
                AuditIssue("error", "missing_expected_directory", f"missing directory: {root}")
            )
            continue
        for path in root.rglob("*.mp4"):
            relative_path = path.relative_to(config.dataset_root).as_posix()
            if path.is_symlink():
                issues.append(
                    AuditIssue("error", "symlink_video", f"video symlink is not accepted: {path}")
                )
            elif path.is_file():
                found[relative_path] = path

    metadata_paths = set(entries)
    media_paths = set(found)
    missing_media = sorted(metadata_paths - media_paths)
    unlisted_media = sorted(media_paths - metadata_paths)
    if missing_media:
        issues.append(
            AuditIssue(
                "error",
                "metadata_media_missing",
                f"{len(missing_media)} metadata entries lack media; first={missing_media[:10]}",
            )
        )
    if unlisted_media:
        issues.append(
            AuditIssue(
                "error",
                "unlisted_media",
                f"{len(unlisted_media)} media files lack metadata; first={unlisted_media[:10]}",
            )
        )
    available = [entry for relative_path, entry in entries.items() if relative_path in found]
    return sorted(available, key=lambda item: item.relative_path.casefold()), issues


def _canonical_manifest(records: Sequence[DfdcVideoRecord]) -> bytes:
    lines = [
        json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")) for record in records
    ]
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def _with_video(issue: AuditIssue, video_id: str) -> AuditIssue:
    return replace(issue, video_ids=(video_id,))


def audit_records(
    records: Sequence[DfdcVideoRecord],
    *,
    config: DfdcConfig,
    metadata_sha256: str,
    split_policy: Mapping[str, object],
    initial_issues: Sequence[AuditIssue] = (),
) -> DfdcAuditReport:
    """Audit metadata coverage, duplicates, labels, and identity boundaries."""

    issues = list(initial_issues)
    sorted_records = tuple(sorted(records, key=lambda item: item.relative_path.casefold()))
    by_path: defaultdict[str, list[DfdcVideoRecord]] = defaultdict(list)
    by_id: defaultdict[str, list[DfdcVideoRecord]] = defaultdict(list)
    by_checksum: defaultdict[str, list[DfdcVideoRecord]] = defaultdict(list)
    identity_splits: defaultdict[str, set[Split]] = defaultdict(set)
    for record in sorted_records:
        by_path[record.relative_path.casefold()].append(record)
        by_id[record.video_id].append(record)
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
        for identity in record.lineage_identity_ids:
            identity_splits[identity].add(record.split)

    for path, duplicates in by_path.items():
        if len(duplicates) > 1:
            issues.append(
                AuditIssue(
                    "error",
                    "duplicate_relative_path",
                    f"relative path occurs {len(duplicates)} times: {path}",
                    tuple(item.video_id for item in duplicates),
                )
            )
    for video_id, duplicates in by_id.items():
        if len(duplicates) > 1:
            issues.append(
                AuditIssue(
                    "error",
                    "duplicate_video_id",
                    f"video ID occurs {len(duplicates)} times: {video_id}",
                    tuple(item.video_id for item in duplicates),
                )
            )
    for digest, duplicates in by_checksum.items():
        paths = {item.relative_path.casefold() for item in duplicates}
        if len(paths) <= 1:
            continue
        duplicate_splits = {item.split for item in duplicates}
        severity: Severity = "error" if len(duplicate_splits) > 1 else "warning"
        issues.append(
            AuditIssue(
                severity,
                "cross_split_content_duplicate" if severity == "error" else "content_duplicate",
                f"SHA-256 {digest} occurs at {len(paths)} paths in splits "
                f"{sorted(duplicate_splits)}",
                tuple(item.video_id for item in duplicates),
            )
        )
    for identity, derived_splits in sorted(identity_splits.items()):
        assigned = {split for split in derived_splits if split != "unassigned"}
        if len(assigned) > 1:
            issues.append(
                AuditIssue(
                    "error",
                    "identity_leakage",
                    f"identity {identity} occurs in derived splits {sorted(assigned)}",
                )
            )
    if len(sorted_records) < config.minimum_records:
        issues.append(
            AuditIssue(
                "error",
                "too_few_records",
                f"expected at least {config.minimum_records} records, found {len(sorted_records)}",
            )
        )
    identity_count = len(identity_splits)
    if (
        config.expected_identity_count is not None
        and identity_count != config.expected_identity_count
    ):
        issues.append(
            AuditIssue(
                "error",
                "unexpected_identity_count",
                f"expected {config.expected_identity_count} identities, found {identity_count}",
            )
        )
    split_counts = Counter(record.split for record in sorted_records)
    for split in VALID_SPLITS:
        if split_counts[split] == 0:
            issues.append(AuditIssue("error", "empty_split", f"derived {split} split is empty"))
            continue
        classes = {record.class_name for record in sorted_records if record.split == split}
        missing_classes = {"real", "manipulated"} - classes
        if missing_classes:
            issues.append(
                AuditIssue(
                    "error",
                    "empty_split_class",
                    f"derived {split} split lacks classes {sorted(missing_classes)}",
                )
            )
    method_counts = Counter(record.manipulation_method for record in sorted_records)
    for method in _METHOD_BY_DIRECTORY.values():
        if method_counts[method] == 0:
            issues.append(AuditIssue("error", "empty_method", f"method {method} is empty"))
    if sorted_records and all(record.sha256 is None for record in sorted_records):
        issues.append(
            AuditIssue("warning", "checksums_disabled", "duplicate-content checks were skipped")
        )

    limitations = (
        "The preview contains only two undisclosed face-swap methods; method labels do not "
        "represent modern manipulation diversity.",
        "Provider demographic figures are aggregate and self-contained subgroup annotations "
        "are not assumed; appearance-based demographic inference is prohibited.",
        "All extracted clips and frames must inherit their parent video_id and split.",
    )
    summary: dict[str, object] = {
        "records": len(sorted_records),
        "by_split": dict(sorted(split_counts.items())),
        "by_provider_split": dict(
            sorted(Counter(record.provider_split for record in sorted_records).items())
        ),
        "by_label": dict(sorted(Counter(record.class_name for record in sorted_records).items())),
        "by_method": dict(sorted(method_counts.items())),
        "by_augmentation": dict(
            sorted(
                Counter(
                    augmentation
                    for record in sorted_records
                    for augmentation in record.augmentations
                ).items()
            )
        ),
        "unique_identities": identity_count,
        "checksummed_records": sum(record.sha256 is not None for record in sorted_records),
        "successfully_probed_records": sum(
            record.probe_error is None and record.probe.codec is not None
            for record in sorted_records
        ),
        "errors": sum(issue.severity == "error" for issue in issues),
        "warnings": sum(issue.severity == "warning" for issue in issues),
    }
    return DfdcAuditReport(
        passed=not any(issue.severity == "error" for issue in issues),
        generated_at_utc=datetime.now(UTC).isoformat(),
        manifest_sha256=hashlib.sha256(_canonical_manifest(sorted_records)).hexdigest(),
        provider_metadata_sha256=metadata_sha256,
        split_policy=split_policy,
        summary=summary,
        issues=tuple(issues),
        limitations=limitations,
    )


def build_manifest(config: DfdcConfig) -> DfdcBuildResult:
    """Inventory DFDC Preview, derive identity-safe splits, and audit it."""

    if not config.dataset_root.is_dir():
        raise DfdcConfigurationError(f"dataset root is missing: {config.dataset_root}")
    if (config.dataset_root / ".git").exists():
        raise DfdcConfigurationError("dataset root must not be a Git repository")
    try:
        config.metadata_path.relative_to(config.dataset_root)
    except ValueError as exc:
        raise DfdcConfigurationError("provider metadata must be inside the dataset root") from exc
    entries, metadata_sha256 = load_provider_metadata(config.metadata_path)
    available, inventory_issues = _inventory(config, entries)
    assignments, split_issues, split_policy = derive_splits(
        tuple(entries.values()), config.validation_fraction, config.split_seed
    )
    split_policy = {**split_policy, "provider_metadata_entries": len(entries)}
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

    records: list[DfdcVideoRecord] = []
    record_issues: list[AuditIssue] = []

    def process(entry: ProviderEntry) -> tuple[DfdcVideoRecord, tuple[AuditIssue, ...]]:
        return _record_from_entry(entry, config, assignments, ffprobe_executable)

    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        for record, issues in executor.map(process, available):
            records.append(record)
            record_issues.extend(_with_video(issue, record.video_id) for issue in issues)
    sorted_records = tuple(sorted(records, key=lambda item: item.relative_path.casefold()))
    audit = audit_records(
        sorted_records,
        config=config,
        metadata_sha256=metadata_sha256,
        split_policy=split_policy,
        initial_issues=(*inventory_issues, *split_issues, *record_issues),
    )
    return DfdcBuildResult(sorted_records, audit)


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def _quarantine(path: Path) -> Path:
    digest = _sha256_file(path)[:12]
    candidate = path.with_name(f"{path.name}.stale-{digest}")
    index = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.stale-{digest}-{index}")
        index += 1
    path.replace(candidate)
    return candidate


def write_build_outputs(
    result: DfdcBuildResult, config: DfdcConfig
) -> tuple[Path | None, Path, Path | None]:
    """Always write the audit; publish the manifest only after it passes."""

    audit_path = config.output_dir / config.audit_filename
    audit_content = (json.dumps(result.audit.to_dict(), indent=2, sort_keys=True) + "\n").encode()
    _atomic_write(audit_path, audit_content)
    expected_manifest = config.output_dir / config.manifest_filename
    if not result.audit.passed:
        return (
            None,
            audit_path,
            _quarantine(expected_manifest) if expected_manifest.exists() else None,
        )
    content = _canonical_manifest(result.records)
    if hashlib.sha256(content).hexdigest() != result.audit.manifest_sha256:
        raise RuntimeError("manifest changed after audit")
    _atomic_write(expected_manifest, content)
    return expected_manifest, audit_path, None


def load_verified_manifest(manifest_path: Path, audit_path: Path) -> tuple[dict[str, object], ...]:
    """Load only a passing, hash-matched DFDC manifest/audit pair."""

    try:
        audit = _mapping(json.loads(audit_path.read_text(encoding="utf-8")), "audit")
    except (OSError, json.JSONDecodeError, DfdcConfigurationError) as exc:
        raise ManifestVerificationError(f"cannot read audit: {exc}") from exc
    if audit.get("passed") is not True:
        raise ManifestVerificationError("audit did not pass")
    expected_hash = audit.get("manifest_sha256")
    if not isinstance(expected_hash, str) or not _SHA256.fullmatch(expected_hash):
        raise ManifestVerificationError("audit manifest_sha256 is invalid")
    try:
        content = manifest_path.read_bytes()
    except OSError as exc:
        raise ManifestVerificationError(f"cannot read manifest: {exc}") from exc
    if hashlib.sha256(content).hexdigest() != expected_hash:
        raise ManifestVerificationError("manifest hash mismatch")

    required = {
        "schema_version",
        "dataset",
        "video_id",
        "relative_path",
        "split",
        "provider_split",
        "lineage_identity_ids",
        "sha256",
        "probe",
    }
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(content.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = _mapping(json.loads(line), f"manifest line {line_number}")
        except (json.JSONDecodeError, DfdcConfigurationError) as exc:
            raise ManifestVerificationError(f"invalid manifest line {line_number}: {exc}") from exc
        missing = sorted(required - set(record))
        if missing:
            raise ManifestVerificationError(
                f"manifest line {line_number} is missing fields: {missing}"
            )
        if record.get("schema_version") != SCHEMA_VERSION or record.get("dataset") != DATASET_NAME:
            raise ManifestVerificationError(
                f"manifest line {line_number} has an unsupported dataset/schema"
            )
        if record.get("split") not in VALID_SPLITS or record.get("provider_split") not in {
            "train",
            "test",
        }:
            raise ManifestVerificationError(f"manifest line {line_number} has an invalid split")
        if record.get("provider_split") == "test" and record.get("split") != "test":
            raise ManifestVerificationError(
                f"manifest line {line_number} changes the provider test boundary"
            )
        if record.get("provider_split") == "train" and record.get("split") == "test":
            raise ManifestVerificationError(
                f"manifest line {line_number} moves provider training data into test"
            )
        relative_path = record.get("relative_path")
        try:
            _safe_relative_path(relative_path)
        except DfdcConfigurationError as exc:
            raise ManifestVerificationError(str(exc)) from exc
        video_id = record.get("video_id")
        if not isinstance(video_id, str) or not _VIDEO_ID.fullmatch(video_id):
            raise ManifestVerificationError(f"manifest line {line_number} has an invalid video ID")
        digest = record.get("sha256")
        if digest is not None and (not isinstance(digest, str) or not _SHA256.fullmatch(digest)):
            raise ManifestVerificationError(f"manifest line {line_number} has an invalid SHA-256")
        lineage = record.get("lineage_identity_ids")
        if (
            not isinstance(lineage, list)
            or not lineage
            or any(not isinstance(identity, str) or not identity for identity in lineage)
            or len(lineage) != len(set(lineage))
        ):
            raise ManifestVerificationError(
                f"manifest line {line_number} has invalid identity lineage"
            )
        records.append(record)
    try:
        summary = _mapping(audit.get("summary"), "audit summary")
    except DfdcConfigurationError as exc:
        raise ManifestVerificationError(str(exc)) from exc
    if summary.get("records") != len(records):
        raise ManifestVerificationError("manifest record count does not match its audit")
    return tuple(records)

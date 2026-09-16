"""Secure acquisition and audited manifest construction for VisA."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast
from urllib.parse import urlparse

import numpy as np
import yaml
from PIL import Image

Split = Literal["train", "validation", "test"]
Label = Literal["normal", "anomaly"]
EXPECTED_HEADER = ("object", "split", "label", "image", "mask")
SCHEMA_VERSION = 1
BUFFER_SIZE = 1024 * 1024
MAX_ARCHIVE_MEMBERS = 100_000
MAX_EXTRACTED_BYTES = 20 * 1024**3


class VisaDataError(RuntimeError):
    """Raised when acquisition or audit cannot establish dataset integrity."""


@dataclass(frozen=True)
class VisaConfig:
    dataset_id: str
    version: str
    archive_url: str
    archive_sha256: str
    accessed_on: str
    expected_images: int
    expected_normal: int
    expected_anomalous: int
    expected_categories: int
    perceptual_max_dhash_distance: int
    perceptual_max_normalized_mae: float
    official_split_csv: str
    validation_fraction: float
    seed: int
    categories: tuple[str, ...]


@dataclass(frozen=True)
class DatasetLayout:
    source_root: Path
    image_root: Path
    split_csv: Path


@dataclass(frozen=True)
class SourceRow:
    category: str
    source_split: Literal["train", "test"]
    label: Label
    image_path: str
    mask_path: str | None


@dataclass(frozen=True)
class ManifestResult:
    manifest_path: Path
    audit_path: Path
    record_count: int
    manifest_sha256: str


@dataclass(frozen=True)
class _Inspected:
    record: dict[str, Any]
    dhash: int
    thumbnail: bytes


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise VisaDataError(f"{name} must be a mapping")
    return cast(Mapping[str, object], value)


def _string(mapping: Mapping[str, object], key: str, section: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise VisaDataError(f"{section}.{key} must be a non-empty string")
    return value.strip()


def _integer(mapping: Mapping[str, object], key: str, section: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise VisaDataError(f"{section}.{key} must be a positive integer")
    return value


def load_config(path: Path) -> VisaConfig:
    """Load and validate the research-relevant VisA configuration."""

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise VisaDataError(f"cannot read configuration {path}: {exc}") from exc
    root = _mapping(payload, "configuration")
    dataset = _mapping(root.get("dataset"), "dataset")
    integrity = _mapping(root.get("integrity"), "integrity")
    layout = _mapping(root.get("layout"), "layout")
    protocol = _mapping(root.get("protocol"), "protocol")
    categories_raw = root.get("categories")
    if not isinstance(categories_raw, list) or not categories_raw:
        raise VisaDataError("categories must be a non-empty list")
    if not all(isinstance(value, str) and value for value in categories_raw):
        raise VisaDataError("every category must be a non-empty string")
    categories = tuple(cast(list[str], categories_raw))
    if len(categories) != len(set(categories)):
        raise VisaDataError("categories must be unique")

    archive_sha256 = _string(integrity, "archive_sha256", "integrity").lower()
    if len(archive_sha256) != 64 or any(char not in "0123456789abcdef" for char in archive_sha256):
        raise VisaDataError("integrity.archive_sha256 must be a lowercase SHA-256 digest")
    fraction = protocol.get("validation_fraction")
    if not isinstance(fraction, float) or not 0.0 < fraction < 0.5:
        raise VisaDataError("protocol.validation_fraction must be a float between 0 and 0.5")
    if protocol.get("preserve_official_test") is not True:
        raise VisaDataError("protocol.preserve_official_test must remain true")
    dhash_distance = integrity.get("perceptual_max_dhash_distance")
    if (
        not isinstance(dhash_distance, int)
        or isinstance(dhash_distance, bool)
        or not 0 <= dhash_distance <= 8
    ):
        raise VisaDataError(
            "integrity.perceptual_max_dhash_distance must be an integer from 0 to 8"
        )
    normalized_mae = integrity.get("perceptual_max_normalized_mae")
    if not isinstance(normalized_mae, float) or not 0.0 <= normalized_mae <= 2.0:
        raise VisaDataError("integrity.perceptual_max_normalized_mae must be a float from 0 to 2")

    config = VisaConfig(
        dataset_id=_string(dataset, "id", "dataset"),
        version=_string(dataset, "version", "dataset"),
        archive_url=_string(dataset, "archive_url", "dataset"),
        archive_sha256=archive_sha256,
        accessed_on=_string(dataset, "accessed_on", "dataset"),
        expected_images=_integer(integrity, "expected_images", "integrity"),
        expected_normal=_integer(integrity, "expected_normal", "integrity"),
        expected_anomalous=_integer(integrity, "expected_anomalous", "integrity"),
        expected_categories=_integer(integrity, "expected_categories", "integrity"),
        perceptual_max_dhash_distance=dhash_distance,
        perceptual_max_normalized_mae=normalized_mae,
        official_split_csv=_string(layout, "official_split_csv", "layout"),
        validation_fraction=fraction,
        seed=_integer(protocol, "seed", "protocol"),
        categories=categories,
    )
    if len(categories) != config.expected_categories:
        raise VisaDataError("configured category count does not match expected_categories")
    if config.expected_normal + config.expected_anomalous != config.expected_images:
        raise VisaDataError("expected normal and anomalous counts do not sum to expected_images")
    return config


def sha256_file(path: Path) -> str:
    """Return the full-file SHA-256 digest without loading the file into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(BUFFER_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path: Path, expected_sha256: str) -> str:
    """Verify an archive against its recorded digest."""

    if not path.is_file():
        raise VisaDataError(f"archive does not exist: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise VisaDataError(
            f"archive SHA-256 mismatch: expected {expected_sha256}, observed {actual}"
        )
    return actual


def _download_name(url: str) -> str:
    name = Path(urlparse(url).path).name
    if not name or name in {".", ".."}:
        raise VisaDataError(f"archive URL has no safe filename: {url}")
    return name


def download_archive(config: VisaConfig, data_root: Path) -> Path:
    """Download the official archive with resume support and mandatory hashing."""

    download_root = data_root / "downloads"
    download_root.mkdir(parents=True, exist_ok=True)
    destination = download_root / _download_name(config.archive_url)
    partial = destination.with_name(f"{destination.name}.part")
    if destination.exists():
        verify_archive(destination, config.archive_sha256)
        return destination

    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "robust-industrial-anomaly-detection/0.1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(config.archive_url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except OSError as exc:
        raise VisaDataError(f"download request failed: {exc}") from exc

    with response:
        status = getattr(response, "status", None)
        if offset and status != 206:
            offset = 0
        mode = "ab" if offset else "wb"
        if status == 206:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {offset}-"):
                raise VisaDataError(f"server returned an invalid resume range: {content_range}")
        with partial.open(mode) as stream:
            while chunk := response.read(BUFFER_SIZE):
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())

    verify_archive(partial, config.archive_sha256)
    os.replace(partial, destination)
    return destination


def _safe_member_parts(name: str) -> tuple[str, ...]:
    if "\\" in name:
        raise VisaDataError(f"archive member uses a backslash path: {name}")
    path = PurePosixPath(name)
    parts = tuple(part for part in path.parts if part not in {"", "."})
    if path.is_absolute() or not parts or ".." in parts or ":" in parts[0]:
        raise VisaDataError(f"unsafe archive member path: {name}")
    return parts


def extract_archive(archive_path: Path, data_root: Path, dataset_id: str) -> Path:
    """Extract only regular files/directories into an atomic versioned source root."""

    source_parent = data_root / "sources"
    destination = source_parent / dataset_id
    if destination.is_dir():
        return destination
    source_parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{dataset_id}-extract-", dir=source_parent))
    total_size = 0
    count = 0
    try:
        with tarfile.open(archive_path, mode="r:*") as archive:
            for member in archive:
                count += 1
                total_size += max(member.size, 0)
                if count > MAX_ARCHIVE_MEMBERS or total_size > MAX_EXTRACTED_BYTES:
                    raise VisaDataError("archive exceeds the configured extraction safety limits")
                parts = _safe_member_parts(member.name)
                target = stage.joinpath(*parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise VisaDataError(f"unsupported archive member type: {member.name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise VisaDataError(f"cannot read archive member: {member.name}")
                with source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=BUFFER_SIZE)
        os.replace(stage, destination)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return destination


def locate_layout(source_root: Path, config: VisaConfig) -> DatasetLayout:
    """Locate the provider split table and image root across known archive wrappers."""

    relative_split = PurePosixPath(config.official_split_csv)
    split_matches = [
        path
        for path in source_root.rglob(relative_split.name)
        if path.as_posix().endswith(relative_split.as_posix())
    ]
    if len(split_matches) != 1:
        raise VisaDataError(
            f"expected exactly one {config.official_split_csv}, found {len(split_matches)}"
        )
    split_csv = split_matches[0]
    base = split_csv.parent.parent
    candidates = (base / "VisA", base)
    valid = [
        root for root in candidates if all((root / name).is_dir() for name in config.categories)
    ]
    if len(valid) != 1:
        raise VisaDataError(f"could not identify one VisA image root below {source_root}")
    return DatasetLayout(source_root=source_root, image_root=valid[0], split_csv=split_csv)


def _safe_relative_path(value: str, field: str) -> str:
    if not value or "\\" in value:
        raise VisaDataError(f"{field} must be a non-empty POSIX relative path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or ":" in path.parts[0]:
        raise VisaDataError(f"unsafe {field}: {value!r}")
    return path.as_posix()


def parse_split_table(path: Path, config: VisaConfig) -> tuple[SourceRow, ...]:
    """Parse the official one-class table and enforce its semantic contract."""

    rows: list[SourceRow] = []
    seen_images: set[str] = set()
    seen_masks: set[str] = set()
    try:
        stream = path.open(newline="", encoding="utf-8-sig")
    except OSError as exc:
        raise VisaDataError(f"cannot open split table {path}: {exc}") from exc
    with stream:
        reader = csv.reader(stream)
        try:
            header = tuple(next(reader))
        except StopIteration as exc:
            raise VisaDataError("split table is empty") from exc
        if header != EXPECTED_HEADER:
            raise VisaDataError(f"unexpected split-table header: {header}")
        for line_number, values in enumerate(reader, start=2):
            if len(values) != len(EXPECTED_HEADER):
                raise VisaDataError(f"split row {line_number} has {len(values)} fields, expected 5")
            category, source_split_raw, label_raw, image_raw, mask_raw = values
            if category not in config.categories:
                raise VisaDataError(f"split row {line_number} has unknown category {category!r}")
            if source_split_raw not in {"train", "test"}:
                raise VisaDataError(
                    f"split row {line_number} has invalid split {source_split_raw!r}"
                )
            if label_raw not in {"normal", "anomaly"}:
                raise VisaDataError(f"split row {line_number} has invalid label {label_raw!r}")
            source_split = cast(Literal["train", "test"], source_split_raw)
            label = cast(Label, label_raw)
            image_path = _safe_relative_path(image_raw, "image path")
            mask_path = _safe_relative_path(mask_raw, "mask path") if mask_raw else None
            if PurePosixPath(image_path).parts[0] != category:
                raise VisaDataError(f"split row {line_number} image/category mismatch")
            if source_split == "train" and (label != "normal" or mask_path is not None):
                raise VisaDataError(f"split row {line_number} violates normal-only training")
            if label == "normal" and mask_path is not None:
                raise VisaDataError(f"split row {line_number} gives a normal sample a mask")
            if label == "anomaly" and (source_split != "test" or mask_path is None):
                raise VisaDataError(f"split row {line_number} anomaly must be masked test data")
            if mask_path and PurePosixPath(mask_path).parts[0] != category:
                raise VisaDataError(f"split row {line_number} mask/category mismatch")
            if image_path in seen_images:
                raise VisaDataError(f"duplicate image path in split table: {image_path}")
            if mask_path and mask_path in seen_masks:
                raise VisaDataError(f"duplicate mask path in split table: {mask_path}")
            seen_images.add(image_path)
            if mask_path:
                seen_masks.add(mask_path)
            rows.append(SourceRow(category, source_split, label, image_path, mask_path))

    labels = Counter(row.label for row in rows)
    categories = {row.category for row in rows}
    if len(rows) != config.expected_images:
        raise VisaDataError(f"expected {config.expected_images} split rows, found {len(rows)}")
    if labels != Counter(normal=config.expected_normal, anomaly=config.expected_anomalous):
        raise VisaDataError(f"unexpected label counts: {dict(labels)}")
    if categories != set(config.categories):
        raise VisaDataError("split table does not cover the configured categories exactly")
    return tuple(rows)


def derive_splits(rows: Sequence[SourceRow], config: VisaConfig) -> dict[str, Split]:
    """Select calibration rows by stable per-category hashing, independent of row order."""

    assignments: dict[str, Split] = {
        row.image_path: "test" for row in rows if row.source_split == "test"
    }
    by_category: dict[str, list[SourceRow]] = defaultdict(list)
    for row in rows:
        if row.source_split == "train":
            by_category[row.category].append(row)
    for category in config.categories:
        candidates = by_category[category]
        if len(candidates) < 2:
            raise VisaDataError(f"category {category} has too few training rows for calibration")
        count = max(
            1,
            min(
                len(candidates) - 1, math.floor(len(candidates) * config.validation_fraction + 0.5)
            ),
        )
        ranked = sorted(
            candidates,
            key=lambda row: hashlib.sha256(
                f"{config.seed}\0{row.category}\0{row.image_path}".encode()
            ).digest(),
        )
        validation = {row.image_path for row in ranked[:count]}
        for row in candidates:
            assignments[row.image_path] = "validation" if row.image_path in validation else "train"
    if len(assignments) != len(rows):
        raise VisaDataError("split derivation did not assign every source row")
    return assignments


def _resolve_relative(root: Path, relative: str) -> Path:
    path = root.joinpath(*PurePosixPath(relative).parts)
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise VisaDataError(f"path escapes the dataset root: {relative}") from exc
    return path


def _dhash(image: Image.Image) -> int:
    pixels = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR))
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.flat:
        value = (value << 1) | int(bit)
    return value


def _inspect_row(
    row: SourceRow,
    split: Split,
    image_root: Path,
    dataset_id: str,
) -> _Inspected:
    image_path = _resolve_relative(image_root, row.image_path)
    if not image_path.is_file():
        raise VisaDataError(f"missing image: {row.image_path}")
    try:
        with Image.open(image_path) as image:
            image.load()
            width, height = image.size
            channels = len(image.getbands())
            if width <= 0 or height <= 0 or channels != 3:
                raise VisaDataError(
                    "invalid RGB image properties for "
                    f"{row.image_path}: {width}x{height}x{channels}"
                )
            dhash = _dhash(image)
            thumbnail = image.convert("L").resize((32, 32), Image.Resampling.BILINEAR).tobytes()
    except VisaDataError:
        raise
    except (OSError, ValueError) as exc:
        raise VisaDataError(f"cannot decode image {row.image_path}: {exc}") from exc

    mask_sha256: str | None = None
    mask_positive_pixels: int | None = None
    if row.mask_path is not None:
        mask_path = _resolve_relative(image_root, row.mask_path)
        if not mask_path.is_file():
            raise VisaDataError(f"missing mask: {row.mask_path}")
        try:
            with Image.open(mask_path) as mask:
                mask.load()
                if mask.size != (width, height):
                    raise VisaDataError(
                        "mask/image size mismatch for "
                        f"{row.image_path}: {mask.size} vs {(width, height)}"
                    )
                mask_positive_pixels = int(np.count_nonzero(np.asarray(mask)))
                if mask_positive_pixels == 0:
                    raise VisaDataError(f"anomaly mask is empty: {row.mask_path}")
        except VisaDataError:
            raise
        except (OSError, ValueError) as exc:
            raise VisaDataError(f"cannot decode mask {row.mask_path}: {exc}") from exc
        mask_sha256 = sha256_file(mask_path)

    sample_id = hashlib.sha256(f"{dataset_id}\0{row.image_path}".encode()).hexdigest()[:24]
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "dataset_version": dataset_id,
        "category": row.category,
        "split": split,
        "source_split": row.source_split,
        "label": row.label,
        "image_path": row.image_path,
        "mask_path": row.mask_path,
        "width": width,
        "height": height,
        "channels": channels,
        "image_sha256": sha256_file(image_path),
        "mask_sha256": mask_sha256,
        "mask_positive_pixels": mask_positive_pixels,
        "perceptual_hash": f"{dhash:016x}",
    }
    return _Inspected(record, dhash, thumbnail)


def _collect_inventory(root: Path, categories: Iterable[str], kind: str) -> set[str]:
    relative_paths: set[str] = set()
    extensions = {".jpg", ".jpeg", ".png"} if kind == "Images" else {".png"}
    for category in categories:
        folder = root / category / "Data" / kind
        if not folder.is_dir():
            raise VisaDataError(f"missing provider directory: {folder}")
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in extensions:
                relative_paths.add(path.relative_to(root).as_posix())
    return relative_paths


def _validate_inventory(
    rows: Sequence[SourceRow], layout: DatasetLayout, config: VisaConfig
) -> dict[str, str]:
    expected_images = {row.image_path for row in rows}
    actual_images = _collect_inventory(layout.image_root, config.categories, "Images")
    if expected_images != actual_images:
        missing = sorted(expected_images - actual_images)[:5]
        unexpected = sorted(actual_images - expected_images)[:5]
        raise VisaDataError(f"image inventory mismatch; missing={missing}, unexpected={unexpected}")
    expected_masks = {row.mask_path for row in rows if row.mask_path is not None}
    actual_masks = _collect_inventory(layout.image_root, config.categories, "Masks")
    if expected_masks != actual_masks:
        missing = sorted(expected_masks - actual_masks)[:5]
        unexpected = sorted(actual_masks - expected_masks)[:5]
        raise VisaDataError(f"mask inventory mismatch; missing={missing}, unexpected={unexpected}")
    annotation_hashes: dict[str, str] = {}
    for category in config.categories:
        annotation = layout.image_root / category / "image_anno.csv"
        if not annotation.is_file():
            raise VisaDataError(f"missing provider annotation file: {annotation}")
        annotation_hashes[category] = sha256_file(annotation)
    return annotation_hashes


def _check_exact_duplicates(items: Sequence[_Inspected]) -> None:
    exact: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        exact[cast(str, item.record["image_sha256"])].append(item.record)
    exact_groups = [records for records in exact.values() if len(records) > 1]
    if exact_groups:
        paths = [
            [cast(str, record["image_path"]) for record in group] for group in exact_groups[:5]
        ]
        raise VisaDataError(f"exact duplicate image content detected: {paths}")


def _near_duplicate_pairs(
    items: Sequence[_Inspected], config: VisaConfig
) -> Iterable[tuple[int, int]]:
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, item in enumerate(items):
        candidate_indexes: set[int] = set()
        for chunk in range(4):
            key = (chunk, (item.dhash >> (chunk * 16)) & 0xFFFF)
            candidate_indexes.update(buckets[key])
        for candidate_index in candidate_indexes:
            candidate = items[candidate_index]
            if item.record["category"] != candidate.record["category"]:
                continue
            distance = (item.dhash ^ candidate.dhash).bit_count()
            if distance > config.perceptual_max_dhash_distance:
                continue
            left = np.frombuffer(item.thumbnail, dtype=np.uint8).astype(np.int16)
            right = np.frombuffer(candidate.thumbnail, dtype=np.uint8).astype(np.int16)
            left -= int(np.mean(left))
            right -= int(np.mean(right))
            normalized_mae = float(np.mean(np.abs(left - right)))
            if normalized_mae <= config.perceptual_max_normalized_mae:
                yield candidate_index, index
        for chunk in range(4):
            buckets[(chunk, (item.dhash >> (chunk * 16)) & 0xFFFF)].append(index)


def _derive_grouped_splits(
    rows: Sequence[SourceRow], items: Sequence[_Inspected], config: VisaConfig
) -> dict[str, Split]:
    parent = list(range(len(items)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right in _near_duplicate_pairs(items, config):
        left_source = items[left].record["source_split"]
        right_source = items[right].record["source_split"]
        if left_source != right_source:
            raise VisaDataError(
                "perceptual duplicate crosses the official train/test boundary: "
                f"{items[left].record['image_path']} and {items[right].record['image_path']}"
            )
        if left_source == "train":
            union(left, right)

    assignments: dict[str, Split] = {
        row.image_path: "test" for row in rows if row.source_split == "test"
    }
    indexes_by_category: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row.source_split == "train":
            indexes_by_category[row.category].append(index)

    for category in config.categories:
        indexes = indexes_by_category[category]
        components: dict[int, list[int]] = defaultdict(list)
        for index in indexes:
            components[find(index)].append(index)
        groups = list(components.values())
        if len(groups) < 2:
            raise VisaDataError(
                f"category {category} has no leakage-safe train/validation separation"
            )
        target = max(1, math.floor(len(indexes) * config.validation_fraction + 0.5))
        ranked = sorted(
            groups,
            key=lambda group: hashlib.sha256(
                (
                    f"{config.seed}\0{category}\0"
                    + "\0".join(
                        sorted(cast(str, items[index].record["image_path"]) for index in group)
                    )
                ).encode()
            ).digest(),
        )
        selected: list[int] = []
        selected_count = 0
        for group_index, group in enumerate(ranked):
            if group_index == len(ranked) - 1:
                break
            new_count = selected_count + len(group)
            if not selected or abs(target - new_count) <= abs(target - selected_count):
                selected.extend(group)
                selected_count = new_count
        selected_set = set(selected)
        if not selected_set or len(selected_set) == len(indexes):
            raise VisaDataError(f"category {category} produced an invalid calibration partition")
        for index in indexes:
            image_path = cast(str, items[index].record["image_path"])
            assignments[image_path] = "validation" if index in selected_set else "train"
    return assignments


def _check_duplicates(items: Sequence[_Inspected], config: VisaConfig) -> None:
    _check_exact_duplicates(items)
    for left, right in _near_duplicate_pairs(items, config):
        if items[left].record["split"] != items[right].record["split"]:
            raise VisaDataError(
                "perceptual cross-split duplicate detected: "
                f"{items[left].record['image_path']} ({items[left].record['split']}) and "
                f"{items[right].record['image_path']} ({items[right].record['split']})"
            )


def _quarantine_canonical(output_dir: Path) -> None:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    for name in ("manifest.jsonl", "audit.json"):
        path = output_dir / name
        if path.exists():
            path.replace(output_dir / f"{path.stem}.stale-{timestamp}{path.suffix}")


def _write_manifest(records: Sequence[dict[str, Any]], destination: Path) -> str:
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        digest = sha256_file(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def _write_json(payload: Mapping[str, object], destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _nested_counts(records: Sequence[dict[str, Any]]) -> dict[str, object]:
    splits = Counter(cast(str, record["split"]) for record in records)
    labels = Counter(cast(str, record["label"]) for record in records)
    categories: dict[str, dict[str, int]] = {}
    for category in sorted({cast(str, record["category"]) for record in records}):
        selected = [record for record in records if record["category"] == category]
        categories[category] = {
            f"{split}:{label}": sum(
                record["split"] == split and record["label"] == label for record in selected
            )
            for split in ("train", "validation", "test")
            for label in ("normal", "anomaly")
        }
    return {
        "splits": dict(sorted(splits.items())),
        "labels": dict(sorted(labels.items())),
        "categories": categories,
    }


def build_manifest(
    layout: DatasetLayout,
    config: VisaConfig,
    config_path: Path,
    output_dir: Path,
    *,
    archive_path: Path | None = None,
    workers: int = 4,
) -> ManifestResult:
    """Audit VisA and atomically publish a canonical JSONL manifest/audit pair."""

    if config.accessed_on == "PENDING":
        raise VisaDataError(
            "dataset.accessed_on is PENDING; record the real acquisition date first"
        )
    if workers < 1:
        raise VisaDataError("workers must be at least one")
    output_dir.mkdir(parents=True, exist_ok=True)
    _quarantine_canonical(output_dir)

    rows = parse_split_table(layout.split_csv, config)
    annotation_hashes = _validate_inventory(rows, layout, config)
    archive_sha256: str | None = None
    if archive_path is not None:
        archive_sha256 = verify_archive(archive_path, config.archive_sha256)

    def inspect(row: SourceRow) -> _Inspected:
        provisional: Split = "test" if row.source_split == "test" else "train"
        return _inspect_row(row, provisional, layout.image_root, config.dataset_id)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        inspected = tuple(pool.map(inspect, rows))
    _check_exact_duplicates(inspected)
    assignments = _derive_grouped_splits(rows, inspected, config)
    for item in inspected:
        item.record["split"] = assignments[cast(str, item.record["image_path"])]
    _check_duplicates(inspected, config)

    records = sorted(
        (item.record for item in inspected),
        key=lambda record: (
            cast(str, record["category"]),
            {"train": 0, "validation": 1, "test": 2}[cast(str, record["split"])],
            cast(str, record["image_path"]),
        ),
    )
    source_test = {row.image_path for row in rows if row.source_split == "test"}
    derived_test = {
        cast(str, record["image_path"]) for record in records if record["split"] == "test"
    }
    if source_test != derived_test:
        raise VisaDataError("derived test membership differs from the official split")

    manifest_path = output_dir / "manifest.jsonl"
    audit_path = output_dir / "audit.json"
    manifest_sha256 = _write_manifest(records, manifest_path)
    audit: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "passed": True,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_id": config.dataset_id,
        "dataset_version": config.version,
        "accessed_on": config.accessed_on,
        "record_count": len(records),
        "manifest_sha256": manifest_sha256,
        "config_sha256": sha256_file(config_path),
        "source_split_sha256": sha256_file(layout.split_csv),
        "archive_sha256": archive_sha256,
        "validation_policy": {
            "source": "official training-normal rows only",
            "fraction_per_category": config.validation_fraction,
            "seed": config.seed,
            "selection": "SHA-256-ranked perceptual components per category",
        },
        "perceptual_duplicate_policy": {
            "hash": "64-bit difference hash on 9x8 grayscale",
            "confirmation_thumbnail": "32x32 grayscale with mean normalization",
            "max_dhash_distance": config.perceptual_max_dhash_distance,
            "max_normalized_mae": config.perceptual_max_normalized_mae,
        },
        "counts": _nested_counts(records),
        "checks": {
            "official_test_preserved": True,
            "exact_duplicates": 0,
            "perceptual_cross_split_duplicates": 0,
            "missing_or_unexpected_images": 0,
            "missing_or_unexpected_masks": 0,
            "invalid_images_or_masks": 0,
        },
        "annotation_sha256_by_category": annotation_hashes,
        "tools": {
            "numpy": importlib.metadata.version("numpy"),
            "pillow": importlib.metadata.version("pillow"),
            "python": sys.version.split()[0],
        },
    }
    _write_json(audit, audit_path)
    return ManifestResult(manifest_path, audit_path, len(records), manifest_sha256)


_MANIFEST_KEYS = {
    "schema_version",
    "sample_id",
    "dataset_version",
    "category",
    "split",
    "source_split",
    "label",
    "image_path",
    "mask_path",
    "width",
    "height",
    "channels",
    "image_sha256",
    "mask_sha256",
    "mask_positive_pixels",
    "perceptual_hash",
}


def load_trusted_manifest(
    manifest_path: Path,
    audit_path: Path,
    config_path: Path,
    image_root: Path | None = None,
) -> tuple[dict[str, Any], ...]:
    """Load a manifest only after validating its audit, hash, schema, and safe paths."""

    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VisaDataError(f"cannot read audit {audit_path}: {exc}") from exc
    if not isinstance(audit, dict) or audit.get("passed") is not True:
        raise VisaDataError("audit is missing or did not pass")
    if audit.get("config_sha256") != sha256_file(config_path):
        raise VisaDataError("configuration hash does not match the trusted audit")
    observed_manifest_hash = sha256_file(manifest_path)
    if audit.get("manifest_sha256") != observed_manifest_hash:
        raise VisaDataError("manifest hash does not match the trusted audit")

    records: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    with manifest_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise VisaDataError(f"invalid manifest JSON on line {line_number}") from exc
            if not isinstance(record, dict) or set(record) != _MANIFEST_KEYS:
                raise VisaDataError(f"manifest line {line_number} has an invalid schema")
            sample_id = record.get("sample_id")
            if not isinstance(sample_id, str) or sample_id in sample_ids:
                raise VisaDataError(
                    f"manifest line {line_number} has a duplicate/invalid sample ID"
                )
            sample_ids.add(sample_id)
            image_path = record.get("image_path")
            if not isinstance(image_path, str):
                raise VisaDataError(f"manifest line {line_number} has an invalid image path")
            _safe_relative_path(image_path, "manifest image path")
            mask_path = record.get("mask_path")
            if mask_path is not None:
                if not isinstance(mask_path, str):
                    raise VisaDataError(f"manifest line {line_number} has an invalid mask path")
                _safe_relative_path(mask_path, "manifest mask path")
            if image_root is not None and not _resolve_relative(image_root, image_path).is_file():
                raise VisaDataError(f"manifest image is unavailable: {image_path}")
            records.append(cast(dict[str, Any], record))
    if audit.get("record_count") != len(records):
        raise VisaDataError("manifest record count does not match the trusted audit")
    return tuple(records)

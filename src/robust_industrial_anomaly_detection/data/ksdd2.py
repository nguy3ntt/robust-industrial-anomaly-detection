"""Verified acquisition and normal-only audit of the official KolektorSDD2 split."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import yaml
from PIL import Image

from robust_industrial_anomaly_detection.data.visa import _dhash, sha256_file
from robust_industrial_anomaly_detection.experiments.m2 import _atomic_json


class KSDD2Error(ValueError):
    """The external dataset violates an acquisition, split, or integrity rule."""


def read_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise KSDD2Error("dataset configuration must be a mapping")
    return payload


def download_ksdd2(config_path: Path, data_root: Path) -> Path:
    """Download resumably from the official HTTPS source; print the observed hash."""

    config = read_config(config_path)
    archive = data_root / "downloads" / "KolektorSDD2.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = int(config["integrity"]["archive_bytes"])
    if not archive.is_file():
        partial = archive.with_suffix(".zip.part")
        offset = partial.stat().st_size if partial.is_file() else 0
        request = urllib.request.Request(
            config["dataset"]["archive_url"],
            headers={"Range": f"bytes={offset}-"} if offset else {},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            if offset and (
                response.status != 206
                or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")
            ):
                raise KSDD2Error("server did not honor the exact resume offset")
            with partial.open("ab" if offset else "wb") as stream:
                copied = offset
                last_reported = offset
                while chunk := response.read(1024 * 1024):
                    copied += len(chunk)
                    if copied > expected_bytes:
                        raise KSDD2Error("download exceeds the configured archive size")
                    stream.write(chunk)
                    if copied - last_reported >= 64 * 1024 * 1024:
                        print(f"Downloaded {copied:,}/{expected_bytes:,} bytes", flush=True)
                        last_reported = copied
                stream.flush()
                os.fsync(stream.fileno())
        if partial.stat().st_size != expected_bytes:
            raise KSDD2Error("incomplete dataset archive; rerun to resume")
        os.replace(partial, archive)
    if archive.stat().st_size != expected_bytes:
        raise KSDD2Error("archive byte count does not match the configured provider revision")
    observed = sha256_file(archive)
    print(f"Archive SHA-256: {observed}", flush=True)
    expected_hash = config["integrity"]["archive_sha256"]
    if expected_hash != "pending-acquisition" and observed != expected_hash:
        raise KSDD2Error("archive SHA-256 does not match the pinned acquisition")
    return archive


def safe_extract_zip(archive: Path, destination: Path) -> None:
    """Preflight every ZIP entry before publishing a bounded, link-free tree."""

    if destination.exists():
        raise KSDD2Error("extraction destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if len(members) > 15000 or sum(item.file_size for item in members) > 4_000_000_000:
            raise KSDD2Error("ZIP inventory exceeds bounded extraction limits")
        paths: list[tuple[zipfile.ZipInfo, tuple[str, ...]]] = []
        seen: set[str] = set()
        for item in members:
            path = PurePosixPath(item.filename)
            mode = item.external_attr >> 16
            if (
                path.is_absolute()
                or any(part in {"..", "."} or ":" in part for part in path.parts)
                or "\\" in item.filename
                or not path.parts
                or stat.S_ISLNK(mode)
                or item.flag_bits & 1
                or item.file_size > 100_000_000
            ):
                raise KSDD2Error(f"unsafe ZIP member: {item.filename}")
            normalized = path.as_posix().casefold()
            if normalized in seen:
                raise KSDD2Error(f"duplicate ZIP member: {item.filename}")
            seen.add(normalized)
            paths.append((item, path.parts))
        staging = Path(tempfile.mkdtemp(prefix=".ksdd2-extract-", dir=destination.parent))
        try:
            for item, parts in paths:
                target = staging.joinpath(*parts)
                if item.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(item) as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output)
            os.replace(staging, destination)
        except Exception:
            # Retain partial extraction for inspection; never publish it as trusted input.
            raise


def assign_normal_splits(
    records: list[dict[str, Any]],
    thumbnails: list[np.ndarray],
    *,
    seed: int,
    validation_fraction: float,
    dhash_distance: int,
    mae_limit: float,
) -> dict[str, object]:
    """Group duplicates before deterministic calibration splitting; preserve provider test."""

    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    exact: dict[str, int] = {}
    confirmed_pairs = 0
    for index, record in enumerate(records):
        candidates: set[int] = set()
        image_hash = record["image_sha256"]
        if image_hash in exact:
            candidates.add(exact[image_hash])
        exact[image_hash] = index
        dhash = int(record["perceptual_hash"], 16)
        for chunk in range(4):
            candidates.update(buckets[(chunk, (dhash >> (chunk * 16)) & 65535)])
        for previous in candidates:
            other = records[previous]
            same = image_hash == other["image_sha256"]
            if (
                not same
                and (dhash ^ int(other["perceptual_hash"], 16)).bit_count() <= dhash_distance
            ):
                left = thumbnails[index].astype(np.float32)
                right = thumbnails[previous].astype(np.float32)
                same = (
                    float(np.mean(np.abs((left - left.mean()) - (right - right.mean()))))
                    <= mae_limit
                )
            if same:
                confirmed_pairs += 1
                parent[find(index)] = find(previous)
        for chunk in range(4):
            buckets[(chunk, (dhash >> (chunk * 16)) & 65535)].append(index)

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        groups[find(index)].append(index)
    eligible: list[list[int]] = []
    exclusions: list[dict[str, str]] = []
    for group in groups.values():
        has_test = any(records[index]["source_split"] == "test" for index in group)
        train_normals: list[int] = []
        for index in group:
            record = records[index]
            if record["source_split"] == "test":
                record["split"] = "test"
            elif record["label"] == "anomaly" or has_test:
                record["split"] = "excluded"
                reason = (
                    "provider_training_anomaly"
                    if record["label"] == "anomaly"
                    else "test_duplicate"
                )
                exclusions.append({"image_path": record["image_path"], "reason": reason})
            else:
                train_normals.append(index)
        if train_normals:
            eligible.append(train_normals)
    eligible.sort(
        key=lambda group: hashlib.sha256(
            (str(seed) + ":" + "|".join(sorted(records[i]["image_path"] for i in group))).encode()
        ).digest()
    )
    total = sum(len(group) for group in eligible)
    target = round(total * validation_fraction)
    allocated = 0
    for position, group in enumerate(eligible):
        validation = allocated < target and position < len(eligible) - 1
        for index in group:
            records[index]["split"] = "validation" if validation else "train"
        if validation:
            allocated += len(group)
    if allocated < 49 or total - allocated < 16:
        raise KSDD2Error("not enough leakage-safe normal fitting/calibration images")
    return {
        "confirmed_duplicate_pairs": confirmed_pairs,
        "exclusions": exclusions,
        "normal_fit_count": total - allocated,
        "normal_calibration_count": allocated,
    }


def prepare_ksdd2(config_path: Path, data_root: Path) -> Path:
    """Verify archive, decode all image/mask pairs, and publish a hashed audit/manifest."""

    config = read_config(config_path)
    if len(config["integrity"]["archive_sha256"]) != 64:
        raise KSDD2Error("pin the observed official archive SHA-256 before extraction")
    archive = download_ksdd2(config_path, data_root)
    dataset_id = str(config["dataset"]["id"])
    root = data_root / "sources" / dataset_id
    if not root.exists():
        safe_extract_zip(archive, root)
    candidates = list(root.rglob("train"))
    candidates = [path for path in candidates if path.is_dir() and (path.parent / "test").is_dir()]
    if len(candidates) != 1:
        raise KSDD2Error("archive must contain exactly one provider train/test layout")
    provider_root = candidates[0].parent
    copy_exclusions = config["integrity"].get("verified_copy_exclusions", {})
    copy_audit: list[dict[str, str]] = []
    for copy_relative, original_relative in copy_exclusions.items():
        copy_path = provider_root / copy_relative
        original_path = provider_root / original_relative
        if not copy_path.is_file() or not original_path.is_file():
            raise KSDD2Error("configured provider copy exclusion is missing")
        copy_hash = sha256_file(copy_path)
        if copy_hash != sha256_file(original_path):
            raise KSDD2Error("provider copy exclusion is not an exact copy")
        copy_audit.append(
            {
                "image_path": copy_relative,
                "original_path": original_relative,
                "sha256": copy_hash,
                "reason": "verified_provider_archive_copy",
            }
        )
    records: list[dict[str, Any]] = []
    thumbnails: list[np.ndarray] = []
    counts: Counter[str] = Counter()
    for split in ("train", "test"):
        images = sorted(
            path
            for path in (provider_root / split).glob("*.png")
            if not path.stem.endswith("_GT")
            and path.relative_to(provider_root).as_posix() not in copy_exclusions
        )
        masks = set((provider_root / split).glob("*_GT.png"))
        expected_masks = {path.with_name(path.stem + "_GT.png") for path in images}
        if not images or masks != expected_masks:
            raise KSDD2Error(f"unmatched image/mask inventory in {split}")
        for path in images:
            mask_path = path.with_name(path.stem + "_GT.png")
            with Image.open(path) as image, Image.open(mask_path) as mask:
                image.load()
                mask.load()
                if image.mode != "RGB" or mask.size != image.size:
                    raise KSDD2Error(f"invalid image/mask properties: {path.name}")
                pixels = np.asarray(mask.convert("L"))
                if not set(np.unique(pixels)).issubset({0, 255}):
                    raise KSDD2Error(f"mask is not binary: {mask_path.name}")
                positive = int(np.count_nonzero(pixels))
                label = "anomaly" if positive else "normal"
                relative = path.relative_to(root).as_posix()
                records.append(
                    {
                        "schema_version": 1,
                        "dataset_version": dataset_id,
                        "sample_id": hashlib.sha256(
                            f"{dataset_id}:{relative}".encode()
                        ).hexdigest()[:24],
                        "category": "surface",
                        "source_split": split,
                        "label": label,
                        "image_path": relative,
                        "mask_path": mask_path.relative_to(root).as_posix(),
                        "width": image.width,
                        "height": image.height,
                        "image_sha256": sha256_file(path),
                        "mask_sha256": sha256_file(mask_path),
                        "mask_positive_pixels": positive,
                        "perceptual_hash": f"{_dhash(image):016x}",
                    }
                )
                thumbnails.append(
                    np.asarray(image.convert("L").resize((32, 32), Image.Resampling.BILINEAR))
                )
                counts[f"{split}_{label}"] += 1
    for key, count in counts.items():
        if count != config["integrity"][f"expected_{key}"]:
            raise KSDD2Error(f"provider count mismatch for {key}: {count}")
    split_audit = assign_normal_splits(
        records,
        thumbnails,
        seed=config["protocol"]["seed"],
        validation_fraction=config["protocol"]["validation_fraction"],
        dhash_distance=config["integrity"]["perceptual_max_dhash_distance"],
        mae_limit=config["integrity"]["perceptual_max_mean_centered_mae_uint8"],
    )
    output = data_root / "manifests" / dataset_id
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.jsonl"
    temporary = manifest.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8"
    )
    os.replace(temporary, manifest)
    _atomic_json(
        output / "audit.json",
        {
            "schema_version": 1,
            "status": "pass",
            "dataset": config["dataset"],
            "archive_sha256": sha256_file(archive),
            "config_sha256": sha256_file(config_path),
            "manifest_sha256": sha256_file(manifest),
            "provider_counts": dict(counts),
            "split_counts": dict(Counter(row["split"] for row in records)),
            **split_audit,
            "test_partition_preserved": True,
            "provider_copy_exclusions": copy_audit,
        },
    )
    print(
        json.dumps(
            {key: value for key, value in split_audit.items() if key != "exclusions"}, indent=2
        ),
        flush=True,
    )
    return manifest


def load_ksdd2_manifest(config_path: Path, data_root: Path) -> tuple[list[dict[str, Any]], Path]:
    """Require audited manifest/config identity and recheck every consumed image and mask."""

    config = read_config(config_path)
    dataset_id = str(config["dataset"]["id"])
    output = data_root / "manifests" / dataset_id
    manifest = output / "manifest.jsonl"
    audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    if (
        audit.get("status") != "pass"
        or audit.get("config_sha256") != sha256_file(config_path)
        or audit.get("manifest_sha256") != sha256_file(manifest)
    ):
        raise KSDD2Error("dataset audit/config/manifest identity no longer matches")
    root = (data_root / "sources" / dataset_id).resolve(strict=True)
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    for record in records:
        if record["split"] == "excluded":
            continue
        for field in ("image", "mask"):
            relative = PurePosixPath(record[f"{field}_path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise KSDD2Error("unsafe manifest path")
            path = root.joinpath(*relative.parts).resolve(strict=True)
            path.relative_to(root)
            if sha256_file(path) != record[f"{field}_sha256"]:
                raise KSDD2Error(f"modified dataset file: {relative}")
    return records, root

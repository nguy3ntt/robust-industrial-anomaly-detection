"""Dependency-free verification and explicit checkpoint acquisition for releases."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.request import urlopen

CHECKPOINT_NAME = "resnet18-f37072fd.pth"
CHECKPOINT_URL = f"https://download.pytorch.org/models/{CHECKPOINT_NAME}"
CHECKPOINT_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
FORMAT = "m16-portable-m10-v1"


class BundleError(ValueError):
    """A release fails integrity, provenance, or local safety checks."""


def digest_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BundleError("Duplicate JSON keys are not permitted.")
        result[key] = value
    return result


def read_json(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 2_000_000:
        raise BundleError(f"Oversized release JSON: {path.name}")

    def reject_constant(value: str) -> None:
        raise BundleError(f"Non-finite JSON constant is not permitted: {value}")

    result = json.loads(
        path.read_text("utf-8"), object_pairs_hook=_pairs, parse_constant=reject_constant
    )
    if not isinstance(result, dict):
        raise BundleError("Release JSON must be an object.")
    return result


def safe_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not re.fullmatch(r"[A-Za-z0-9_./-]+", relative):
        raise BundleError("Non-portable path in release manifest.")
    parts = relative.split("/")
    if any(part in {"", ".", ".."} for part in parts) or PurePosixPath(relative).is_absolute():
        raise BundleError("Release path escapes its root.")
    candidate = root.joinpath(*parts)
    try:
        candidate.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (ValueError, OSError) as exc:
        raise BundleError(f"Missing or escaping release file: {relative}") from exc
    if any(path.is_symlink() for path in (candidate, *candidate.parents) if path != root):
        raise BundleError("Linked files are not permitted in a release.")
    return candidate


def verify_bundle(root: Path) -> dict[str, Any]:
    manifest = read_json(root / "manifest.json")
    if manifest.get("format") != FORMAT or manifest.get("schema_version") != 1:
        raise BundleError("Unsupported demo bundle format.")
    files = manifest.get("files")
    if not isinstance(files, dict) or not 10 <= len(files) <= 500:
        raise BundleError("Invalid release file inventory.")
    folded: set[str] = set()
    total = 0
    for relative, expected in files.items():
        if relative.casefold() in folded:
            raise BundleError("Case-colliding release paths.")
        folded.add(relative.casefold())
        path = safe_path(root, relative)
        if not isinstance(expected, dict) or type(expected.get("bytes")) is not int:
            raise BundleError("Invalid release size entry.")
        if not 0 <= expected["bytes"] <= 32_000_000 or not re.fullmatch(
            r"[0-9a-f]{64}", str(expected.get("sha256", ""))
        ):
            raise BundleError("Invalid release hash or size boundary.")
        total += expected["bytes"]
        if total > 64_000_000:
            raise BundleError("Release exceeds the bounded runtime package size.")
        if (
            not path.is_file()
            or path.stat().st_size != expected["bytes"]
            or (digest_file(path) != expected["sha256"])
        ):
            raise BundleError(f"Release file failed SHA-256/size verification: {relative}")
    required = {
        "demo.py",
        "pyproject.toml",
        "uv.lock",
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "configs/prototype/m10.yaml",
        "runtime/runtime.json",
        "runtime/calibration.json",
        "runtime/quality.json",
        "README.md",
    }
    if not required <= set(files):
        raise BundleError("Release is missing mandatory runtime or attribution files.")
    for directory in ("runtime", "src"):
        actual = {
            path.relative_to(root).as_posix()
            for path in (root / directory).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        }
        declared = {name for name in files if name.startswith(directory + "/")}
        if actual != declared:
            raise BundleError(f"Unlisted or missing files in {directory}/.")
    forbidden = {".pth", ".pt", ".ckpt", ".safetensors", ".jpg", ".jpeg", ".png", ".webp"}
    if any(Path(name).suffix.lower() in forbidden or ".env" in Path(name).parts for name in files):
        raise BundleError("Weights, source images and private environment files cannot ship.")
    if (
        manifest.get("weights_included") is not False
        or manifest.get("source_images_included") is not False
    ):
        raise BundleError("Release privacy flags are invalid.")
    return manifest


def verify_checkpoint(path: Path) -> None:
    if (
        not path.is_file()
        or path.stat().st_size > 64_000_000
        or (digest_file(path) != CHECKPOINT_SHA256)
    ):
        raise BundleError(
            f"Missing or modified checkpoint: {path}. Supply --checkpoint <verified-file>, "
            "or explicitly use --prepare to fetch the official weights."
        )


def prepare_checkpoint(path: Path) -> None:
    """Explicit opt-in only. Never overwrite an existing corrupt or unrelated file."""

    if path.exists():
        verify_checkpoint(path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix="checkpoint-", suffix=".part", delete=False
        ) as output:
            temporary = Path(output.name)
            with urlopen(CHECKPOINT_URL, timeout=30) as response:  # fixed official HTTPS URL
                if response.geturl() != CHECKPOINT_URL:
                    raise BundleError("Unexpected checkpoint download redirect.")
                size = 0
                while block := response.read(1_048_576):
                    size += len(block)
                    if size > 64_000_000:
                        raise BundleError("Checkpoint download exceeds the size limit.")
                    output.write(block)
            output.flush()
            os.fsync(output.fileno())
        verify_checkpoint(temporary)
        # Same-volume hard-link publication is atomic and refuses an existing destination.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

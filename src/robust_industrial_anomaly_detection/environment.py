"""Reproducible runtime-environment inspection.

The checker intentionally depends only on the Python standard library so it can
explain an incomplete environment instead of failing during import.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

Status = Literal["pass", "warning", "fail"]


@dataclass(frozen=True)
class Check:
    """One environment assertion and its human-readable evidence."""

    name: str
    status: Status
    detail: str


@dataclass(frozen=True)
class EnvironmentReport:
    """Serializable collection of environment checks."""

    checks: tuple[Check, ...]

    @property
    def ok(self) -> bool:
        return all(check.status != "fail" for check in self.checks)

    def to_json(self) -> str:
        payload = {"ok": self.ok, "checks": [asdict(check) for check in self.checks]}
        return json.dumps(payload, indent=2, sort_keys=True)


REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "opencv-python-headless": "cv2",
    "pandas": "pandas",
    "pillow": "PIL",
    "python-dotenv": "dotenv",
    "pyyaml": "yaml",
    "scikit-learn": "sklearn",
    "scipy": "scipy",
    "tqdm": "tqdm",
}


def _package_check(distribution: str, module: str) -> Check:
    try:
        importlib.import_module(module)
        version = importlib.metadata.version(distribution)
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        return Check(distribution, "fail", f"not importable: {exc}")
    return Check(distribution, "pass", version)


def _path_check(variable: str, *, minimum_free_gib: float) -> Check:
    raw_path = os.environ.get(variable)
    if not raw_path:
        return Check(variable, "warning", "not configured; copy .env.example to .env")
    path = Path(raw_path).expanduser()
    if not path.exists():
        return Check(variable, "warning", f"configured path does not exist: {path}")
    free_gib = shutil.disk_usage(path).free / (1024**3)
    status: Status = "pass" if free_gib >= minimum_free_gib else "warning"
    return Check(variable, status, f"{path} ({free_gib:.1f} GiB free)")


def _torch_checks(accelerator: Literal["any", "cpu", "cuda"]) -> list[Check]:
    try:
        torch = importlib.import_module("torch")
        torchvision = importlib.import_module("torchvision")
    except ImportError as exc:
        return [Check("PyTorch", "fail", f"not importable: {exc}")]

    checks = [
        Check(
            "torch", "pass" if torch.__version__.startswith("2.13.0") else "fail", torch.__version__
        ),
        Check(
            "torchvision",
            "pass" if torchvision.__version__.startswith("0.28.0") else "fail",
            torchvision.__version__,
        ),
    ]
    cuda_available = bool(torch.cuda.is_available())
    if accelerator == "cuda" and not cuda_available:
        checks.append(Check("CUDA", "fail", "requested but torch.cuda.is_available() is false"))
    elif cuda_available:
        device_index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device_index)
        memory_gib = properties.total_memory / (1024**3)
        try:
            probe = torch.ones((16, 16), device=device_index)
            probe_total = float((probe @ probe).sum().item())
            torch.cuda.synchronize(device_index)
        except Exception as exc:
            checks.append(Check("CUDA", "fail", f"device found but compute probe failed: {exc}"))
            return checks
        status: Status = "pass" if memory_gib >= 8 else "warning"
        checks.append(
            Check(
                "CUDA",
                status,
                (
                    f"{properties.name}; {memory_gib:.1f} GiB; torch CUDA "
                    f"{torch.version.cuda}; compute probe={probe_total:.0f}"
                ),
            )
        )
    elif accelerator == "cpu":
        checks.append(Check("CUDA", "pass", "CPU environment selected"))
    else:
        checks.append(Check("CUDA", "warning", "not available; CPU execution only"))
    return checks


def inspect_environment(
    accelerator: Literal["any", "cpu", "cuda"] = "any",
) -> EnvironmentReport:
    """Inspect the interpreter, dependencies, local paths, and accelerator."""

    checks: list[Check] = []
    version_ok = sys.version_info[:2] == (3, 12)
    checks.append(
        Check(
            "Python",
            "pass" if version_ok else "fail",
            f"{platform.python_version()} ({sys.executable}); project requires 3.12.x",
        )
    )
    checks.append(
        Check("Platform", "pass", f"{platform.system()} {platform.release()} {platform.machine()}")
    )
    checks.extend(
        _package_check(distribution, module) for distribution, module in REQUIRED_PACKAGES.items()
    )
    checks.extend(_torch_checks(accelerator))
    checks.append(_path_check("ANOMALY_DATA_ROOT", minimum_free_gib=25))
    checks.append(_path_check("ANOMALY_ARTIFACT_ROOT", minimum_free_gib=20))
    return EnvironmentReport(tuple(checks))


def format_report(report: EnvironmentReport) -> str:
    """Render a compact terminal report."""

    markers = {"pass": "PASS", "warning": "WARN", "fail": "FAIL"}
    lines = [f"[{markers[check.status]}] {check.name}: {check.detail}" for check in report.checks]
    lines.append(f"RESULT: {'READY' if report.ok else 'NOT READY'}")
    return "\n".join(lines)

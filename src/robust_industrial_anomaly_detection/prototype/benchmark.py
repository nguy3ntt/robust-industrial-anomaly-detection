"""Reproducible latency, memory, and artifact benchmark for M6."""

from __future__ import annotations

import ctypes
import os
import time
import tracemalloc
from collections.abc import Sequence
from ctypes import wintypes
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
from dotenv import load_dotenv

from robust_industrial_anomaly_detection.data.visa import load_trusted_manifest, sha256_file
from robust_industrial_anomaly_detection.experiments.m2 import (
    Sample,
    _atomic_csv,
    _atomic_json,
    _environment,
    _git_metadata,
    _samples,
)
from robust_industrial_anomaly_detection.prototype.engine import (
    InspectionEngine,
    PrototypeError,
    PrototypeSettings,
)


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _working_set_bytes() -> int | None:
    if os.name != "nt":
        return None
    api = cast(Any, ctypes).windll
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    current_process = api.kernel32.GetCurrentProcess
    current_process.restype = wintypes.HANDLE
    get_memory = api.psapi.GetProcessMemoryInfo
    get_memory.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    get_memory.restype = wintypes.BOOL
    handle = current_process()
    if not get_memory(handle, ctypes.byref(counters), counters.cb):
        return None
    return int(counters.WorkingSetSize)


def _percentile(values: Sequence[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def _benchmark_samples(
    all_samples: Sequence[Sample], categories: Sequence[str], per_category: int
) -> list[Sample]:
    selected: list[Sample] = []
    for category in categories:
        candidates = sorted(
            (
                sample
                for sample in all_samples
                if sample.category == category and sample.split == "test"
            ),
            key=lambda sample: sample.sample_id,
        )
        if len(candidates) < per_category:
            raise PrototypeError(f"not enough benchmark samples for {category}")
        selected.extend(candidates[:per_category])
    return selected


def run_benchmark(
    *,
    project_root: Path,
    settings: PrototypeSettings,
    data_config: Path,
    device: str,
    m5_run: Path | None = None,
) -> tuple[Path, dict[str, object]]:
    """Benchmark deterministic test inputs without storing their pixels."""

    load_dotenv(project_root / ".env")
    data_root_raw = os.environ.get("ANOMALY_DATA_ROOT")
    artifact_root_raw = os.environ.get("ANOMALY_ARTIFACT_ROOT")
    if not data_root_raw or not artifact_root_raw:
        raise PrototypeError("ANOMALY_DATA_ROOT and ANOMALY_ARTIFACT_ROOT must be configured")
    data_root, artifact_root = Path(data_root_raw), Path(artifact_root_raw)
    manifest_dir = data_root / "manifests" / "visa-20220922"
    image_root = data_root / "sources" / "visa-20220922"
    manifest_path = manifest_dir / "manifest.jsonl"
    audit_path = manifest_dir / "audit.json"
    records = load_trusted_manifest(manifest_path, audit_path, data_config, image_root)
    all_samples = _samples(records, image_root)

    tracemalloc.start()
    rss_before = _working_set_bytes()
    startup_started = time.perf_counter()
    engine = InspectionEngine(
        project_root=project_root,
        settings=settings,
        device=device,
        m5_run=m5_run,
    )
    startup_ms = (time.perf_counter() - startup_started) * 1000.0
    rss_after_startup = _working_set_bytes()
    samples = _benchmark_samples(
        all_samples, engine.categories, settings.benchmark_samples_per_category
    )
    warmup = samples[: settings.benchmark_warmup_samples]
    for sample in warmup:
        engine.inspect_bytes(
            sample.image_path.read_bytes(),
            filename=sample.image_path.name,
            category=sample.category,
        )

    rows: list[dict[str, object]] = []
    benchmark_started = time.perf_counter()
    peak_rss = _working_set_bytes() or 0
    for position, sample in enumerate(samples, start=1):
        result = engine.inspect_bytes(
            sample.image_path.read_bytes(),
            filename=sample.image_path.name,
            category=sample.category,
        )
        current_rss = _working_set_bytes()
        if current_rss is not None:
            peak_rss = max(peak_rss, current_rss)
        rows.append(
            {
                "position": position,
                "sample_id": sample.sample_id,
                "category": sample.category,
                "source_label": sample.label,
                "decision": result.decision,
                "anomaly_score": result.anomaly_score,
                "conformal_p": result.conformal_p,
                **{f"{name}_ms": value for name, value in result.timing_ms.items()},
            }
        )
        print(
            f"[{position}/{len(samples)}] {sample.category}/{sample.sample_id}: "
            f"{result.timing_ms['total']:.1f} ms",
            flush=True,
        )
    benchmark_seconds = time.perf_counter() - benchmark_started
    _, python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    total_times = [float(cast(int | float, row["total_ms"])) for row in rows]
    feature_times = [float(cast(int | float, row["feature_extraction_ms"])) for row in rows]
    score_times = [float(cast(int | float, row["memory_scoring_ms"])) for row in rows]
    footprint = engine.resource_footprint()
    static_root = Path(__file__).with_name("static")
    static_bytes = sum(path.stat().st_size for path in static_root.iterdir() if path.is_file())
    summary: dict[str, object] = {
        "mode": "full",
        "device": device,
        "samples": len(rows),
        "categories": len(engine.categories),
        "startup_ms": startup_ms,
        "mean_total_ms": float(np.mean(total_times)),
        "median_total_ms": _percentile(total_times, 0.5),
        "p95_total_ms": _percentile(total_times, 0.95),
        "mean_feature_extraction_ms": float(np.mean(feature_times)),
        "mean_memory_scoring_ms": float(np.mean(score_times)),
        "sequential_images_per_second": len(rows) / benchmark_seconds,
        "rss_before_bytes": rss_before,
        "rss_after_startup_bytes": rss_after_startup,
        "peak_working_set_bytes": peak_rss or None,
        "python_tracemalloc_peak_bytes": python_peak,
        "static_ui_bytes": static_bytes,
        "runtime_artifacts": footprint,
        "source_m5_run": engine.m5_run.name,
        "source_images_persisted": False,
    }
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = artifact_root / "runs" / f"m6-prototype-benchmark-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    _atomic_csv(run_dir / "sample_timings.csv", rows)
    _atomic_json(run_dir / "summary.json", summary)
    _atomic_json(run_dir / "environment.json", _environment(device))
    resolved = {
        **asdict(settings),
        "device": device,
        "data_config": str(data_config),
        "source_m5_run": str(engine.m5_run),
    }
    _atomic_json(run_dir / "resolved_config.json", resolved)
    metadata = {
        "schema_version": 1,
        "status": "complete",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "git": _git_metadata(project_root),
        "manifest_sha256": sha256_file(manifest_path),
        "audit_sha256": sha256_file(audit_path),
        "data_config_sha256": sha256_file(data_config),
        "source_m5_metadata_sha256": sha256_file(engine.m5_run / "metadata.json"),
        "sample_timings_sha256": sha256_file(run_dir / "sample_timings.csv"),
        "summary_sha256": sha256_file(run_dir / "summary.json"),
        "environment_sha256": sha256_file(run_dir / "environment.json"),
        "resolved_config_sha256": sha256_file(run_dir / "resolved_config.json"),
    }
    _atomic_json(run_dir / "metadata.json", metadata)
    return run_dir, summary

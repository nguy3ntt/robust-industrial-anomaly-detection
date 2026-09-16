# Development environment and resource plan

This is the reference environment for the initial dataset tooling, baselines,
and single-GPU experiments.

## Audited workstation

Inventory was captured on 2026-09-14 using read-only operating-system and NVIDIA
queries.

| Resource | Detected value | Project implication |
| --- | --- | --- |
| Operating system | 64-bit Windows NT 10.0.26200 | Windows is the initial host; reusable Python remains portable to Linux. |
| CPU | Intel Core i5-13400F, 16 logical processors | Suitable for image hashing, decoding, and moderate data-loader parallelism. |
| Memory | 31.82 GiB | Use streaming manifests and bounded feature indexes. |
| GPU | NVIDIA GeForce RTX 4070 SUPER, 12 GiB VRAM | Suitable for frozen encoders and modest fine-tuning with mixed precision. |
| NVIDIA driver | 610.74, CUDA capability reported as 13.3 | Compatible with the selected PyTorch CUDA 13.0 build. |
| C: drive | Critically constrained during audit | Do not place caches, datasets, or artifacts here. |
| D: drive | About 293.7 GiB free during audit | Preferred for VisA, model caches, feature banks, checkpoints, and runs. |
| E: drive | Repository drive with limited headroom | Keep source and the local environment here, not datasets or checkpoints. |

## Selected software

- CPython 3.12 pinned by `.python-version`.
- `uv` 0.12.13 with a committed lockfile.
- Exact PyTorch 2.13.0 and torchvision 0.28.0 builds.
- Mutually exclusive `cpu` and `cu130` dependency choices.
- Bounded runtime dependencies and strict Ruff, mypy, pytest, and coverage tools.

The project does not install into the system Python. A separate CUDA toolkit is
not required for normal PyTorch wheel use.

## Local storage layout

```text
D:/industrial-anomaly-data/       original archive, extracted VisA, manifests
D:/industrial-anomaly-artifacts/  feature banks, checkpoints, runs, local reports
D:/industrial-anomaly-cache/      uv, PyTorch, and model-download caches
E:/.../                            tracked source and public documentation only
```

Copy `.env.example` to `.env` and create the selected D: directories before M1.
All of these paths are ignored or outside the repository.

## Delivery horizon

The complete academic/portfolio package retains the working target of
**2027-03-31**. This is a planning assumption, not an external deadline.

| Gate | Planning target |
| --- | --- |
| M1 data acquisition, governance, and audit | 2026-10-04 |
| M2 baselines and evaluation harness | 2026-11-08 |
| M3 localization model and ablations | 2026-12-13 |
| M4 robustness and generalization | 2027-01-31 |
| M5 calibration and selective review | 2027-02-21 |
| M6 practical prototype | 2027-03-07 |
| M7 academic and portfolio package | 2027-03-31 |

## Verification policy

Before data preparation or training, run the environment checker with the
accelerator actually intended for the run. It performs package/version checks
and a small CUDA computation when CUDA is selected. Warnings describe
operational setup still needed; failures mean the selected runtime is invalid.

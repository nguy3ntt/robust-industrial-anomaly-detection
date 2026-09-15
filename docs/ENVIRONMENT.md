# Development environment and resource plan

This record closes the environment and resource-selection portion of M0. It is
the reference configuration for early dataset tooling and baselines, not a
claim that every later experiment will fit on one workstation.

## Audited workstation

Inventory captured on 2026-09-14 using read-only operating-system and NVIDIA
queries.

| Resource | Detected value | Project implication |
| --- | --- | --- |
| Operating system | Windows 11 from Python's platform query; the legacy product-name API reports Windows 10 Pro; 64-bit NT 10.0.26200 | Windows is the initial supported development host. Keep reusable Python code portable to Linux. |
| CPU | 13th Gen Intel Core i5-13400F; 16 logical processors | Suitable for manifest construction and moderate parallel decoding. Worker counts must remain configurable. |
| Memory | 31.82 GiB physical; 18.99 GiB available during audit | Avoid loading full manifests or decoded video collections into memory. Use streaming/batched processing. |
| GPU | NVIDIA GeForce RTX 4070 SUPER; 12,282 MiB VRAM; compute capability 8.9 | Use mixed precision, sampled clips, gradient accumulation, and modest spatial/temporal batch sizes. |
| NVIDIA driver | 610.74; reports CUDA driver capability 13.3 | Compatible with the selected PyTorch CUDA 13.0 binaries. A separate CUDA toolkit is not required for ordinary wheel-based use. |
| C: | 3.1 GiB free of about 204.5 GiB | Do not place package caches, datasets, checkpoints, or experiment outputs here. Free space is critically low. |
| D: | 293.7 GiB free of about 931.5 GiB | Preferred location for dataset subsets, model/download caches, and experiment artifacts. Check dataset size before every download. |
| E: | 8.4 GiB free of about 260.3 GiB | Keep the repository and source here. Do not store datasets or checkpoints beside the source tree. |

At audit time, Git 2.47.1 and Python 3.14.5 were available. `uv`, FFmpeg,
Conda, and a supported Python 3.12 interpreter were not on `PATH`.

## Selected software environment

- CPython 3.12, pinned by `.python-version`. Python 3.14 is deliberately not
  used for the initial research environment to retain broad binary-package and
  research-library compatibility.
- `uv` 0.12.13 manages Python, the virtual environment, dependency resolution,
  and the committed `uv.lock` file. The required tool version is enforced in
  `pyproject.toml`.
- PyTorch 2.13.0 and torchvision 0.28.0 are exact direct pins. Two mutually
  exclusive extras are defined: `cpu` for portable checks and `cu130` for this
  NVIDIA workstation.
- CUDA 13.0 PyTorch wheels are selected from PyTorch's explicit package index.
  Other dependencies continue to come from PyPI, reducing dependency-confusion
  risk.
- Runtime libraries use bounded compatibility ranges in `pyproject.toml`; the
  lockfile captures exact transitive versions. Dependency upgrades must update
  the lockfile and pass the smallest relevant test suite before experiments use
  them.
- FFmpeg is a required external tool from M1 onward. Its exact version will be
  captured in every dataset-build or experiment record. It is not silently
  downloaded by project code.

The project does not install packages into the system Python. CPU and CUDA
environments are selected explicitly so a CPU-only wheel cannot be mistaken for
a GPU-enabled research setup.

This selection follows PyTorch's published pairing of PyTorch 2.13.0 with
torchvision 0.28.0 and its CUDA 13.0 wheel index, together with uv's documented
optional-dependency pattern for mutually exclusive accelerator builds:

- <https://pytorch.org/get-started/previous-versions/>
- <https://docs.astral.sh/uv/guides/integration/pytorch/>

## Storage layout

The local `.env.example` proposes the following roots for this workstation:

```text
D:/deepfake-data/       restricted raw data and derived private media
D:/deepfake-artifacts/  checkpoints, run outputs, and local reports
D:/deepfake-cache/      uv, PyTorch, and model-download caches
E:/robust-spatiotemporal-deepfake-detection/  tracked source and documentation
```

Copy `.env.example` to `.env` after creating or choosing these locations.
`.env`, data, media, weights, and run artifacts are ignored by Git. Because the
current D: capacity cannot be assumed to hold every dataset variant, M1 begins
with the licence-approved, checksum-recorded DFDC Preview package and retains
only derived data needed by active experiments.

## Delivery window

The planning target for the complete M7 academic package is **2027-03-31**,
roughly 28 weeks from this audit. This is a working assumption until an external
submission or assessment deadline is supplied.

| Gate | Planning target |
| --- | --- |
| M1 dataset governance and audit | 2026-10-11 |
| M2 baselines and evaluation harness | 2026-11-15 |
| M3 spatiotemporal model | 2026-12-20 |
| M4 generalization research | 2027-01-31 |
| M5 trustworthiness and explanation | 2027-02-21 |
| M6 practical prototype | 2027-03-07 |
| M7 academic package and clean reproducibility audit | 2027-03-31 |

Dataset approval delays or insufficient local storage can move these dates.
The schedule prioritizes a valid leakage-safe baseline over premature model
complexity.

## Verification policy

Run `uv run --extra cu130 python scripts/check_environment.py --accelerator
cuda` before GPU data preparation or training. Use `--json` when the result
needs to be captured as machine-readable run metadata. Warnings identify
operational risks; failures identify conditions that make the selected
environment invalid. The check includes a small CUDA tensor operation rather
than relying only on device discovery.

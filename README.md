# Robust Spatiotemporal Deepfake Detection

Research and prototype development for deepfake video detection that remains reliable under unseen manipulation methods, video compression, resizing, re-encoding, and other distribution shifts.

## Project objective

The project will develop an explainable, uncertainty-aware detector that combines spatial, frequency-domain, and temporal evidence. Its main contribution is not simply higher in-dataset accuracy: it will test whether a model can generalize to manipulations and post-processing operations that were absent during training.

The intended output is a reproducible research pipeline and a practical media-forensics interface that reports:

- whether a video contains evidence of facial manipulation;
- calibrated confidence and an option to abstain when evidence is insufficient;
- spatial regions and temporal segments influencing the result;
- robustness results for compression, resizing, blur, noise, and re-encoding;
- a clear statement that the output is decision support, not proof of authenticity.

## Research questions

1. Does combining spatial, frequency, and temporal evidence improve cross-dataset and unseen-manipulation generalization?
2. Which signal families remain reliable after common social-media transformations?
3. Can calibration and selective prediction reduce confident errors on out-of-distribution videos?
4. Do localization explanations remain stable under compression and perturbation?
5. What accuracy, latency, and memory trade-offs are required for a practical forensic screening tool?

## Proposed method

The initial architecture has three branches:

1. **Spatial branch:** a pretrained image or video encoder operating on sampled face crops and contextual frames.
2. **Frequency branch:** learnable features derived from DCT or FFT representations to capture synthesis and blending artifacts.
3. **Temporal branch:** a lightweight temporal transformer or sequence model that detects inconsistent motion and frame-to-frame artifacts.

The fused representation will feed four heads:

- real/manipulated classification;
- manipulation-region localization;
- calibrated uncertainty estimation;
- optional manipulation-family classification for analysis.

The research will begin with simple, reproducible baselines before adding each branch. Every added component must be justified by an ablation study.

## Dataset plan

| Dataset | Primary use | Notes |
| --- | --- | --- |
| [DFDC Preview](https://ai.meta.com/datasets/dfdc/) | Primary baseline training and controlled evaluation | About 5,000 videos from 66 consenting actors, two face-swap methods, and provider identity/split metadata. Live access terms must be accepted. |
| [FaceForensics++](https://github.com/ondyari/FaceForensics) | Optional future method and localization benchmark | Not selected for the initial pipeline because its application requests academic affiliation and advisor details. |
| Another approved, method-diverse benchmark | Frozen cross-dataset evaluation | Add only after reviewing its licence, source-media provenance, and identity controls. |

Datasets will not be committed to Git. Each user must obtain data from the official source and accept the applicable terms.

## Evaluation protocol

The project will report more than ordinary accuracy:

- ROC-AUC and average precision;
- balanced accuracy, precision, recall, F1, and confusion matrices;
- expected calibration error, Brier score, and reliability diagrams;
- coverage versus risk for selective prediction;
- leave-one-manipulation-method-out performance;
- cross-dataset performance without test-set fine-tuning;
- degradation under compression, resizing, blur, noise, and frame-rate changes;
- localization IoU or pointing-game accuracy when masks are available;
- video-level latency, throughput, peak memory, and model size.

Splits must be identity-disjoint where the dataset permits it. Frames from one source video must never be distributed across training and test sets.

## Repository layout

```text
configs/       Experiment configurations
data/          Local datasets and generated manifests (ignored by Git)
notebooks/     Exploratory analysis only
reports/       Public figures and research outputs
scripts/       Dataset preparation and experiment entry points
src/           Reusable package code
tests/         Automated tests
```

Local project management and Codex instructions are intentionally excluded from version control.

## Development environment

The initial supported environment is CPython 3.12 managed by
[`uv`](https://docs.astral.sh/uv/). Dependency versions are resolved in the
committed lockfile, and CPU and NVIDIA CUDA 13.0 PyTorch builds are explicit,
mutually exclusive choices.

On the audited Windows workstation, keep datasets, caches, checkpoints, and run
artifacts on the roomier D: drive rather than the source or system drives. Copy
`.env.example` to `.env` and adjust the local paths before downloading data.
See [the environment and resource plan](docs/ENVIRONMENT.md) for the recorded
hardware, constraints, dependency policy, and planning timeline.

After installing `uv`, create the local GPU development environment:

```powershell
$env:UV_CACHE_DIR = "D:\deepfake-cache\uv"
uv sync --extra cu130 --group dev --locked
Copy-Item .env.example .env
uv run --extra cu130 python scripts/check_environment.py --accelerator cuda
uv run --extra cu130 pytest
```

For a CPU-only environment, replace every `cu130` with `cpu` and validate with
`--accelerator cpu`. Keep the extra on `uv run` commands so synchronization does
not remove the selected PyTorch build. FFmpeg is required before M1 processes
videos; its detected version will be recorded with dataset and experiment
metadata.

## Dataset governance and manifests

M1 provides a DFDC Preview manifest builder that preserves Meta's actor-disjoint
test set and derives validation only from disconnected identity groups within
the provider training set. Both identities in every swap stay together. The
builder reconciles every metadata entry with local media, performs full-file
SHA-256 hashing and ffprobe checks, and rejects duplicate or identity-leaking
data. A failed audit does not publish a canonical manifest.

Read the [dataset governance gate](docs/datasets/GOVERNANCE.md), the
[DFDC Preview dataset card](docs/datasets/DFDC.md), and the
[manifest contract](docs/MANIFESTS.md) before acquiring or processing data.
DFDC access requires the researcher to personally accept the current live
agreement. Until the official package is acquired and a real-data audit passes,
M1 remains active and model training is prohibited.

## Planned reproducibility standard

- Configuration-driven experiments.
- Fixed, recorded random seeds.
- Dataset checksums and versioned manifests.
- Environment and dependency lock file.
- Saved metrics and model-card metadata for every reported run.
- No manual test-set selection or tuning.
- Baselines reproduced before novel model development.

## Responsible-use boundaries

This system is intended for research and triage. A model score alone cannot establish that media is authentic or manipulated. Results may be affected by demographic imbalance, capture hardware, compression, editing software, and novel generation methods. The final interface must expose uncertainty and avoid presenting model output as conclusive forensic evidence.

## Status

M0 is complete. M1 governance, dataset cards, DFDC identity-safe splitting,
manifest construction, and synthetic leakage tests are implemented. DFDC
Preview access and the real-data audit remain the active external gate.

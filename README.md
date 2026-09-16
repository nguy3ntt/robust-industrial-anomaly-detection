# Robust Industrial Anomaly Detection

A reproducible research and portfolio project for detecting and localizing
visual manufacturing defects when only normal training images are available.
The central question is not merely whether a model works on a familiar test
set, but whether its scores and explanations remain useful under realistic
changes in lighting, scale, blur, compression, and object category.

## What the system will do

Given an inspection image, the completed prototype will provide:

- an image-level anomaly score;
- a pixel-level heatmap showing suspicious regions;
- a three-way recommendation: normal, anomalous, or send for human review;
- uncertainty and a stated operating threshold;
- robustness results for common image-quality and capture shifts;
- transparent limitations rather than a claim of guaranteed product quality.

This is an inspection decision-support system. It is not a certified quality
control device and it does not diagnose the physical root cause of a defect.

## Research questions

1. Which pretrained feature representations produce the strongest normal-only
   anomaly baseline on VisA?
2. How much do anomaly ranking, localization, and false-positive control degrade
   under realistic capture and post-processing shifts?
3. Can normal-only score calibration provide useful false-alarm guarantees and
   a safer human-review region without labelled validation defects?
4. Which explanations remain spatially stable under benign transformations?
5. How well do unified and category-specific models transfer to unseen object
   categories, and what accuracy/latency trade-off is practical on one GPU?

## Dataset

The primary source is Amazon's [Visual Anomaly (VisA) dataset](https://github.com/amazon-science/spot-diff):

- 10,821 high-resolution colour images;
- 9,621 normal and 1,200 anomalous samples;
- 12 object categories across three broad object types;
- image-level labels and pixel-level anomaly masks;
- a public direct download with no account, cloud setup, or academic adviser;
- released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The project preserves the official one-class test split. Validation is derived
only from official training-normal images, so test defects remain unavailable
for model selection. See the [dataset card](docs/datasets/VISA.md),
[governance gate](docs/datasets/GOVERNANCE.md), and
[manifest contract](docs/MANIFESTS.md).

## Planned method

Work proceeds from auditable baselines to measured additions:

1. colour/statistical and pretrained global-feature sanity baselines;
2. a patch-memory baseline in the PatchCore/nearest-neighbour family;
3. multi-scale spatial features for image scoring and defect localization;
4. robustness interventions selected by controlled ablation;
5. normal-only quantile or conformal calibration and selective review;
6. explanation-stability analysis and an efficient local demonstration.

No complex model is accepted without comparison to a simpler baseline.

## Evaluation

Primary reporting includes image AUROC, average precision, FPR at high recall,
pixel AUROC, pixel average precision, region-aware overlap, and per-category
confidence intervals. Operating-point metrics include normal false-positive
rate, anomaly recall, review coverage, and selective risk. Robustness tables
will isolate JPEG compression, resizing, blur, noise, brightness/contrast, and
small geometric changes. Latency, peak GPU memory, model size, and index size
are reported with quality metrics.

Pooled scores never replace per-category results. The untouched official test
set is evaluated only after the experiment configuration is frozen.

## Repository layout

```text
configs/       Dataset and experiment configurations
data/          Placeholders only; real data stays outside Git
docs/          Research specification, governance, and protocols
notebooks/     Exploration only, never the source of production logic
reports/       Public figures and final research outputs
scripts/       Thin command-line entry points
src/           Reusable typed Python package
tests/         Focused automated verification
```

## Environment setup

The supported environment is CPython 3.12 managed by
[`uv`](https://docs.astral.sh/uv/). CPU and CUDA 13.0 PyTorch builds are explicit
and mutually exclusive.

```powershell
$env:UV_CACHE_DIR = "D:\industrial-anomaly-cache\uv"
uv sync --extra cu130 --group dev --locked
Copy-Item .env.example .env
uv run --extra cu130 python scripts/check_environment.py --accelerator cuda
uv run --extra cu130 pytest
```

Use `cpu` in place of `cu130` for a CPU-only setup. The example local paths put
datasets, caches, and artifacts on the roomier D: drive; edit `.env` on another
machine. Full details are in [the environment plan](docs/ENVIRONMENT.md).

## Reproducibility rules

- Configuration-driven runs and recorded deterministic seeds.
- Versioned source URLs, licences, checksums, manifests, and data audits.
- No direct model access to an unaudited raw dataset tree.
- Frozen test protocol and no test-based tuning.
- Saved environment, code revision, configuration, thresholds, and metrics for
  every reportable run.
- Public code and small reports only; datasets, checkpoints, and local outputs
  remain outside Git.

See [the full project specification](docs/PROJECT_SPEC.md) and
[reproducibility standard](docs/REPRODUCIBILITY.md).

## Status

M0 and M1 are complete. The official VisA archive was downloaded directly,
verified against SHA-256, extracted through a traversal-safe atomic process,
and audited across all 10,821 samples. The canonical manifest preserves the
official test set and derives validation only from grouped training-normal
images. M2—reproducible statistical, global-feature, and patch-memory
baselines—is the active phase.

To reproduce M1 after configuring `.env`:

```powershell
uv run --extra cu130 python scripts/prepare_visa.py --workers 8
```

The command resumes interrupted downloads, verifies the archive before
extraction, and publishes a manifest only when every integrity gate passes.

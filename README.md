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

## Reproduce the M2 baselines

The M2 runner accepts only the trusted M1 manifest. Thresholds are frozen from
normal validation images (95th percentile for image scores and 99.5th
percentile for patch scores); test anomalies never select a model, threshold,
or hyperparameter.

```powershell
# Fast end-to-end verification on one category.
uv run --extra cpu python scripts/run_m2_baselines.py `
  --device cpu --quick --category candle

# Reportable run over all 12 categories and all three baselines.
uv run --extra cpu python scripts/run_m2_baselines.py --device cpu
```

Use `--extra cu130 --device cuda` after installing the CUDA profile. If a run
is interrupted, pass its directory with `--resume`; the runner refuses to
resume if the resolved settings, manifest, audit, or checked-in configurations
have changed.

Every run stores resolved configuration, seed, environment, Git state, exact
model-weight identity and hash, per-image predictions, per-category metrics,
bootstrap intervals, progress, summary, and output hashes. Patch-memory runs
also save two local qualitative panels per category (highest-scoring normal and
anomaly), with columns `input | anomaly overlay | ground-truth mask`.

The completed full VisA run produced these macro per-category means:

| Baseline | Image AUROC | Image AP | Normal FPR | Pixel AUROC | Pixel AP | AUPRO@30 |
|---|---:|---:|---:|---:|---:|---:|
| Colour/texture NN | 0.8145 | 0.8497 | 0.0629 | — | — | — |
| Frozen ResNet-18 global NN | 0.8227 | 0.8513 | 0.0657 | — | — | — |
| Frozen ResNet-18 patch memory | **0.8283** | **0.8527** | **0.0477** | 0.9488 | 0.3038 | 0.7156 |

These are baseline measurements, not tuned claims of industrial readiness.
Category-level variation is substantial and remains visible in the generated
`category_metrics.csv` rather than being hidden by the macro summary.

## Reproduce the M3 multi-scale ablation

M3 freezes a five-stage category-specific ablation: the M2 reference, reduced
memory capacity, multi-scale features, bounded candidates, and k-center coreset
selection. It also evaluates one category-balanced unified index. The image and
pixel thresholds remain validation-normal-only; test defects do not tune any
configuration or operating threshold.

```powershell
# Fast end-to-end verification on one category (six evaluations).
uv run --extra cpu python scripts/run_m3_multiscale.py `
  --device cpu --quick --category candle

# Reportable evaluation: 60 category-specific ablations plus 12 unified jobs.
uv run --extra cpu python scripts/run_m3_multiscale.py --device cpu
```

Interrupted runs can be continued with `--resume <run-directory>`. Each run saves
resolved settings, input/model hashes, per-image predictions, category metrics,
component deltas, resource measurements, reloadable indexes, index hashes, and
qualitative panels.

The frozen full VisA run produced these macro per-category means. Latency is CPU
nearest-neighbour scoring time per test image on the development workstation;
total index storage sums the 12 category indexes, except for the single unified
index.

| Model | Image AUROC | Normal FPR | Pixel AUROC | Pixel AP | Mean IoU | AUPRO@30 | Score latency | Index storage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Mid-scale random, 2,048/category (reference) | 0.8283 | 0.0477 | 0.9488 | 0.3038 | **0.2507** | 0.7156 | 2.82 ms | 6.63 MiB |
| Mid-scale random, 512/category | 0.7615 | 0.0433 | 0.9372 | 0.2562 | 0.1605 | 0.7005 | **0.87 ms** | 2.45 MiB |
| Multi-scale random, 512/category | 0.7465 | 0.0460 | 0.9427 | 0.2173 | 0.1081 | 0.7226 | 2.27 ms | 2.62 MiB |
| Multi-scale bounded random, 512/category | 0.7506 | **0.0411** | 0.9434 | 0.2363 | 0.0893 | 0.7156 | 2.23 ms | 2.62 MiB |
| Multi-scale k-center, 512/category | **0.8679** | 0.0471 | **0.9555** | **0.3210** | 0.1665 | **0.7692** | 2.25 ms | 2.62 MiB |
| Multi-scale balanced unified, 1,536 total | 0.7936 | 0.0566 | 0.9257 | 0.2227 | 0.1087 | 0.6942 | 4.64 ms | **0.45 MiB** |

The coreset leads image ranking, pixel ranking, pixel AP, and AUPRO, while the
reference retains the strongest thresholded mean IoU. M4 therefore carries both
frozen category-specific models forward rather than hiding the trade-off behind
a single aggregate rank. These localization values supersede the initial M2/M3
run: a geometry audit found that non-square images were center-cropped while
masks were stretched. The evaluator now applies the same centered crop to both;
image scores and thresholds were unaffected.

## Reproduce the M4 robustness study

M4 freezes two severities each for JPEG compression, downscale/upscale, Gaussian
blur, sensor-like noise, brightness/contrast, and rotation/translation. It
evaluates the two M3 frontier models plus an equal-capacity normal-only nuisance
augmentation intervention. The run also measures paired score-rank and heatmap
stability, leave-one-category-out transfer, failure clusters, and nuisance-score
correlations.

```powershell
# Full protocol on one small deterministic subset.
uv run --extra cpu python scripts/run_m4_robustness.py `
  --device cpu --quick --category candle

# Reportable 12-category study (468 model-condition evaluations).
uv run --extra cpu python scripts/run_m4_robustness.py --device cpu
```

The full study produced 84,318 predictions, 77,832 paired stability records, 12
zero-shot transfer evaluations, 12 hashed intervention indexes, and 320 focused
failure panels. Principal results are:

- multi-scale robustness advantage over the reference is inconclusive:
  −0.0072 image AUROC, 95% CI [−0.0235, +0.0083];
- equal-budget normal nuisance augmentation improves shifted image AUROC by
  0.0077, 95% CI [+0.0030, +0.0126], and reduces normal FPR by 0.0774;
- retained-correct predictions have heatmaps 0.2153 more correlated with clean
  explanations than shift failures, 95% CI [+0.2066, +0.2253];
- leave-one-category-out image AUROC is only 0.5783, a mean loss of 0.2896 from
  category-specific memory;
- blur is the largest false-alarm failure, increasing the plain coreset's normal
  FPR by 0.6426 on average across its two severities.

See the [M4 robustness report](reports/M4_ROBUSTNESS.md) for the frozen ladder,
family-level tables, failure clusters, transfer results, and limitations.

## Reproduce the M5 calibration study

M5 reuses the verified M4 test scores and computes only clean validation-normal
scores for empirical-quantile and split-conformal calibration. Category scores
are checkpointed atomically, so an interrupted run resumes without repeating
completed categories.

```powershell
# Fast 12-category end-to-end contract check.
uv run --extra cpu python scripts/run_m5_calibration.py --device cpu --quick

# Reportable 12-category calibration and selective-review study.
uv run --extra cpu python scripts/run_m5_calibration.py --device cpu

# Resume an interrupted run using the exact path shown in progress.json.
uv run --extra cpu python scripts/run_m5_calibration.py --device cpu `
  --resume D:\industrial-anomaly-artifacts\runs\m5-calibration-full-<timestamp>
```

The primary augmented-coreset policy uses category-conditional conformal
p-values: `p <= 0.05` is anomalous, `0.05 < p <= 0.20` is review, and
`p > 0.20` is normal. On clean test data it yields macro normal FPR 0.0430,
review rate 0.1958, automatic coverage 0.8042, and selective risk 0.1940.
Category calibration improves false-alarm alignment over pooled calibration by
0.0681 (95% CI [0.00495, 0.18088]). The review policy reduces automatic risk by
0.07125 (95% CI [0.06063, 0.08122]) while preserving 0.7790 mean coverage across
clean and shifted conditions.

These guarantees do not survive arbitrary distribution shift: severe blur,
noise, JPEG, and geometry produce normal FPR 0.8686, 0.6820, 0.4696, and 0.2179.
See the [M5 calibration report](reports/M5_CALIBRATION.md) for category behavior,
coverage-risk definitions, exchangeability assumptions, and failure modes.

## Run the M6 local inspector

The practical prototype serves a self-contained browser interface on loopback.
Images are decoded and inspected in memory; the application does not retain
uploads, and exported JSON reports exclude original and derived image pixels.

```powershell
# Start the frozen CPU prototype, then open http://127.0.0.1:8765.
uv run --extra cpu python scripts/run_m6_prototype.py --device cpu

# Optionally open the browser automatically.
uv run --extra cpu python scripts/run_m6_prototype.py --device cpu --open-browser

# Reproduce the 24-image latency, memory, throughput, and size benchmark.
uv run --extra cpu python scripts/benchmark_m6_prototype.py --device cpu
```

Select the known product category, choose up to 24 images, and run inspection.
The result shows the anomaly score, category conformal p-value, three-way
decision, score reference, thresholded heatmap support, per-stage timing, and
limitations. The server rejects non-loopback binding and verifies the full
M5→M4 model/calibration/index provenance before becoming ready.

The finalized CPU prototype reaches 77.3 ms median and 108.8 ms p95 end-to-end
latency across 24 images, or 12.55 sequential images/s. Peak process working set
is 548.7 MB and runtime model/calibration artifacts total 49.97 MB.

See the [M6 prototype report](reports/M6_PROTOTYPE.md),
[model card](docs/MODEL_CARD.md), and [system card](docs/SYSTEM_CARD.md) for the
runtime design, benchmark, privacy controls, intended use, and deployment limits.

## Status

M0 through M6 are complete. The official VisA archive was downloaded directly,
verified against SHA-256, extracted through a traversal-safe atomic process,
and audited across all 10,821 samples. The canonical manifest preserves the
official test set and derives validation only from grouped training-normal
images. Reproducible baselines, multi-scale/coreset ablations, category-specific
and unified comparisons, resource measurements, reloadable hashed indexes,
controlled robustness/transfer evaluation, and normal-only selective calibration
now feed a local, calibrated, privacy-preserving inspection prototype. M7 will
freeze the final academic and portfolio package.

To reproduce M1 after configuring `.env`:

```powershell
uv run --extra cu130 python scripts/prepare_visa.py --workers 8
```

The command resumes interrupted downloads, verifies the archive before
extraction, and publishes a manifest only when every integrity gate passes.

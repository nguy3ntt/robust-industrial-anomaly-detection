# Final evaluation protocol

Status: frozen for M7 on 2026-09-23. Changes to this document require a new
protocol version and cannot silently replace the reported results.

## Research boundary

The project studies normal-only visual anomaly detection and localization under
capture and post-processing shift. It is decision support for research and
portfolio demonstration, not a certified inspection system and not a physical
root-cause diagnostic tool.

## Data and split integrity

- Dataset: VisA, public 2022-09-22 release, CC BY 4.0.
- Canonical audited manifest SHA-256:
  `73283c2704ed18183e5394fff699b0f8d916113393e53ba53986adaf6f32cc6e`.
- Total: 10,821 images across 12 categories.
- Training memory: 7,358 official training-normal images.
- Calibration: 1,301 deterministic grouped samples derived only from official
  training-normal data.
- Final test: all 2,162 official one-class test images, including 1,200 anomaly
  and 962 normal images.
- Exact and confirmed perceptual duplicate groups may not cross splits. The
  official test partition is never reassigned.

## Frozen systems

1. Interpretable colour/texture nearest neighbour.
2. Frozen ImageNet ResNet-18 global nearest neighbour.
3. Mid-scale patch memory with 2,048 prototypes per category.
4. Category-specific multi-scale k-center memory with 512 prototypes.
5. Category-balanced unified multi-scale memory with 1,536 prototypes total.
6. Primary system: category-specific augmented multi-scale k-center memory with
   512 prototypes and category-conditional split-conformal calibration.

The primary policy declares `p <= 0.05` anomalous, `0.05 < p <= 0.20` review,
and `p > 0.20` normal. No threshold may be edited after observing final-test
labels.

## Outcomes

Detection reports macro per-category image AUROC, image average precision, and
normal false-positive rate at the frozen operating point. Localization reports
pixel AUROC, pixel average precision, mean IoU at a validation-normal-only pixel
threshold, and AUPRO integrated to 30% false-positive area. Operational reports
include review rate, automatic coverage, selective risk, latency, throughput,
working-set memory, and artifact size.

Each pooled summary retains its per-category source table. Threshold-free and
thresholded localization metrics are reported together because they answer
different questions.

## Shift protocol

The primary and reference systems are evaluated at two deterministic severities
for JPEG compression, downscale/upscale, blur, sensor-like noise, photometric
change, and rotation/translation. Image masks receive the same geometric
transform as images. The study measures clean-to-shift deltas, paired intervals,
normal FPR, score-rank correlation, and inverse-aligned heatmap stability.

Zero-shot category transfer excludes every target-category training prototype.
The robustness intervention retains the same prototype and patch budget as its
unaugmented comparator.

## Confirmatory claims

- H2: multi-scale features improve image robustness relative to the mid-scale
  reference.
- Intervention: normal-only nuisance augmentation improves shifted image AUROC.
- H5: retained-correct predictions have more stable heatmaps than shift failures.
- H3: category calibration improves category-level clean false-alarm alignment.
- H4: the review band reduces automatic-decision risk while preserving useful
  coverage.

All five use the predeclared 95% intervals produced in M4 or M5. An interval
crossing zero is inconclusive; it is not converted into a positive claim.

## Clean M7 reconstruction

M7 does not select models or recalculate neural features. It performs a fresh,
deterministic publication reconstruction from the completed reportable M2–M6
runs. Before reading a source, it verifies:

1. the exact pinned run directory;
2. the source `metadata.json` SHA-256;
3. `status == complete`;
4. the canonical dataset manifest SHA-256; and
5. the recorded hash of every required input table or summary.

It then regenerates every public M7 table and figure, records the hash and byte
size of every output, and emits `reports/m7/manifest.json`. Any source mutation,
implicit “latest run” lookup, manifest mismatch, or incomplete run aborts the
build. This reuses completed inference rather than spending hours repeating an
identical feature extraction; the integrity boundary makes that choice explicit
and independently auditable.

Run the reconstruction with:

```powershell
uv run --extra cpu python scripts/build_m7_package.py `
  --config configs/reporting/m7.yaml
```

## Reporting and privacy rules

- Do not redistribute VisA images, masks, private inspection images, weights,
  feature banks, or machine-specific run directories.
- Generated plots use aggregate measurements only.
- Separate clean guarantees from shifted observations.
- Present failure modes and negative results with the headline metrics.
- Do not describe heatmaps as causal explanations.
- Do not claim industrial readiness, guaranteed quality, or category transfer.


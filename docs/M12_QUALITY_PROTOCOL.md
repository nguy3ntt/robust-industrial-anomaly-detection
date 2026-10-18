# Capture-quality generalization protocol

## Frozen question and boundary

M12 is a development-driven follow-up to M9's known blur/noise gate failures,
not independent confirmation on a previously unseen dataset. Compare the
historical eight-feature gate (v1) with one shared ten-feature recipe (v2).
Do not search thresholds, categories, feature subsets, or test winners. Fit
references on original training normals only, calibrate once on original clean
validation normals, and preserve every official test image. Training defects
and transformed/test images never fit references or select thresholds.

Both gates use category/domain-specific median and max(MAD, IQR, 1e-5) scales,
maximum directional deviation, upper-tail plus-one tie-conservative conformal
p-values, and alpha 0.02. This is a marginal clean-normal calibration under
exchangeability, not a guarantee under capture shift or within every session.
The original eight measurements and v1 serialization remain unchanged.

## V2 measurements, fixed before the new test evaluation

Retain all eight v1 measurements. Add:

- `log_focus_ratio` (lower tail): log(mean squared 3x3 Laplacian /
  max(mean squared 3x3 Sobel magnitude, 1e-8) + 1e-8). Contrast normalization
  reduces dependence on object brightness/texture amplitude. Log scaling avoids
  the bounded lower-tail deviation of raw Laplacian variance.
- `log_flat_noise` (upper tail): estimate noise by the median absolute 2x2
  diagonal high-pass response / 0.67449 on the least structured 25% of pixels;
  structure is squared Sobel magnitude after sigma-2 smoothing. Discard a
  one-pixel border, divide by max(gray standard deviation, 1e-4), and take
  log(value + 1e-6). This reduces edge contamination, not all product texture.

All measurements use uint8 RGB on the exact 224x224 model input: VisA centered
square crop, KSDD2 full-frame bilinear resize. All twelve unchanged M4 shifts and
the original sample-key/seed are evaluated, with no feature extraction retraining.
Reuse exact hash-pinned M5/M9 decisions to isolate the gate intervention.
Evaluate the historical gates again and require matching saved dispositions.

## Outcomes and honest trade-offs

Report per-category, condition, label, and gate: recapture rate; clean-normal
rejection; anomalous-image rejection; false-alert interception (normal review
or anomalous calls intercepted / original normal alerts); remaining automatic
anomalous calls per all inputs; and total operator-action rate (recapture OR
review OR anomalous). Withheld defects are unresolved, never successful defect
detections. Reduction of guarded false positives is not proof of improved
detector ranking, reduced workload, or improved defect recall.

Per-condition rejection intervals are exact binomial 95% intervals under
independent-image assumptions. Repeated shifts share images; pooled shifted
summaries are descriptive, not independent trials or significance tests.
Session/item dependence is unknown. Publish all categories/conditions, not just
blur/noise improvements. Do not automatically replace the live M10 deployment.
Provide a separately configured v2 inspector for explicit local comparison.

## Reproducibility and privacy

Run `scripts/run_m12_quality_generalization.py --quick` before a full run.
Use `--stop-after-conditions 1` and then `--resume <printed-directory>` to
verify interruption without changing the frozen protocol. Fit models and each
category-condition prediction table are atomically checkpointed with hashes.
Resume validates code, config, source pins, trusted manifests, consumed image
hashes, calibration membership, and all committed checkpoints before reuse.
`--stop-after-conditions` is an execution limit, not a protocol change.

Data, per-image predictions, model references, and run artifacts stay outside
Git under the configured artifact root. Public reporting contains only aggregate
tables, interpretation, and exact source/output hash pins. No new data account,
network download, dependency, or dataset upload is needed.

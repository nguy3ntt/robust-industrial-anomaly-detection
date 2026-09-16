# Dataset governance gate

No data may enter an experiment until all applicable items below are complete.

## Source and licence

- Acquire data only from the official VisA project archive.
- Read and record the dataset licence, version, source URL, and access date.
- Keep the original archive unchanged until its checksum is recorded.
- Retain the required VisA attribution in reports and demonstrations.

## Storage and privacy

- Store the archive and extracted images under `ANOMALY_DATA_ROOT`, outside Git.
- Store checkpoints and experiment outputs under `ANOMALY_ARTIFACT_ROOT`.
- Do not upload dataset files or model weights without explicit authorization
  and a fresh licence review.
- Do not mix workplace or private production images into VisA experiments
  without a separate data-governance record.

## Split and audit

- Preserve the official one-class test set exactly.
- Derive validation only from official training-normal images using the recorded
  deterministic policy.
- Detect exact and perceptual near-duplicates before training.
- Validate all image/mask pairs and record exclusions; silent repair is forbidden.
- Require a passing audit and matching manifest hash in every experiment.

## Reporting

- Report per-category results and uncertainty, not only a pooled average.
- Separate validation-based choices from final test reporting.
- Record all robustness transformations and random seeds.
- Describe the system as inspection decision support, not a quality guarantee.

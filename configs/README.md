# Configuration

Tracked YAML files are the source of truth for dataset and experiment choices.
Commands may select a configuration but should not hide research-relevant
defaults in code. Every run stores its fully resolved configuration.

`data/visa.yaml` records the official source, licence, acquisition date,
integrity expectations, duplicate thresholds, categories, and frozen split
policy. Changes invalidate the trusted manifest until M1 is rerun.

`prototype/m6.yaml` freezes the loopback server safety limits, deployed M5
policy, batch bound, and benchmark sampling count for the local inspector.

`experiments/m8-quality-gate.yaml` freezes the category-normal capture features,
calibration level, M5 decision source, and reused M4 shift ladder.

`prototype/m8.yaml` enables the verified M8 gate ahead of the unchanged M6
anomaly model. The M6 file remains unchanged for historical reproduction.

`data/ksdd2.yaml` pins the official second dataset archive, separate noncommercial
licence, provider counts, verified copy exclusions, and normal-only split policy.
`experiments/m9-external-validation.yaml` freezes the full-frame geometry,
equal-budget plain/augmented memories, calibration policy, and reused M4/M8
protocols for independent second-dataset evaluation.

`reporting/m7.yaml` pins the completed M2–M6 source runs, metadata hashes,
required source-file hashes, final output directory, and publication style for
the deterministic academic/portfolio package reconstruction.

`experiments/m10-localization.yaml` fixes normal-reference/calibration settings,
component area, bounded growth, bootstrap reporting, source runs, and severe
conditions. `prototype/m10.yaml` enables the verified cleanup policy alongside
the unchanged M8 gate and M5 image decision. `reporting/m10.yaml` pins the exact
completed full comparison and metadata hash for aggregate reconstruction.

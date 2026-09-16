# Reproducibility standard

Every result intended for comparison or publication must be reconstructable
from immutable inputs and recorded metadata.

## Required run record

- UTC start time and generated run identifier;
- Git revision plus dirty-worktree status;
- resolved configuration and random seed;
- Python, package, operating-system, accelerator, and driver versions;
- dataset version, manifest hash, audit hash, and split-table hash;
- model/encoder identifier and weight checksum;
- preprocessing and robustness transformation parameters;
- calibration data boundary and frozen operating thresholds;
- raw per-sample predictions plus aggregate metrics;
- wall time, peak memory, model size, and feature-index size.

## Determinism

Use seeded Python, NumPy, and PyTorch generators. Record when an operation is
not deterministic and measure run-to-run variation for material results.
Dataset order may never define a split; split membership must be content-stable
under filesystem enumeration changes.

## Test-set discipline

The official VisA test set is not a development set. Do not select encoders,
layers, thresholds, augmentations, or perturbation severities from test results.
Final test evaluation begins only after the corresponding configuration is
frozen. Exploratory test runs, if unavoidable, must be labelled and cannot
support confirmatory claims.

## Publication boundary

Source, configuration, small manifests with relative paths, aggregate metrics,
and permitted report figures may be published. Raw VisA files, private images,
checkpoints, caches, and machine-specific paths stay out of Git unless a later
licence and privacy review explicitly changes that boundary.

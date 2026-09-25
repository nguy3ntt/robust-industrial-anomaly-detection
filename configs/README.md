# Configuration

Tracked YAML files are the source of truth for dataset and experiment choices.
Commands may select a configuration but should not hide research-relevant
defaults in code. Every run stores its fully resolved configuration.

`data/visa.yaml` records the official source, licence, acquisition date,
integrity expectations, duplicate thresholds, categories, and frozen split
policy. Changes invalidate the trusted manifest until M1 is rerun.

`prototype/m6.yaml` freezes the loopback server safety limits, deployed M5
policy, batch bound, and benchmark sampling count for the local inspector.

`reporting/m7.yaml` pins the completed M2–M6 source runs, metadata hashes,
required source-file hashes, final output directory, and publication style for
the deterministic academic/portfolio package reconstruction.

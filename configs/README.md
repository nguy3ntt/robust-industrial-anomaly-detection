# Configuration

Tracked YAML files are the source of truth for dataset and experiment choices.
Commands may select a configuration but should not hide research-relevant
defaults in code. Every run stores its fully resolved configuration.

`data/visa.yaml` records the official source, licence, acquisition date,
integrity expectations, duplicate thresholds, categories, and frozen split
policy. Changes invalidate the trusted manifest until M1 is rerun.

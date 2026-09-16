# Manifest and audit contract

This document defines the data boundary that M1 will implement. Models must not
walk the raw dataset tree directly.

## Canonical record

Each UTF-8 JSON Lines record will contain:

| Field | Meaning |
| --- | --- |
| `sample_id` | Deterministic opaque identifier derived from dataset version and relative path. |
| `dataset_version` | Immutable source version, initially `visa-20220922`. |
| `category` | One of the 12 configured VisA object categories. |
| `split` | `train`, `validation`, or `test`. |
| `label` | `normal` or `anomaly`. |
| `image_path` | Safe path relative to the configured dataset root. |
| `mask_path` | Safe relative mask path for anomalous test images, otherwise `null`. |
| `width`, `height`, `channels` | Decoded image properties. |
| `image_sha256` | Full-file content digest. |
| `mask_sha256` | Full-file mask digest when a mask exists. |
| `source_split` | Provider split from `split_csv/1cls.csv`. |

Unknown fields are rejected by the trusted loader. Absolute paths and path
traversal are invalid.

## Validation derivation

The official test partition is immutable. For each category, confirmed
perceptual-duplicate components are formed within official training-normal
data; validation then selects deterministic SHA-256-ranked components targeting
15% using seed `20260915`. A component can never cross train and validation,
and the selection cannot drift with filesystem enumeration order.

Validation normal data calibrates normal score distributions and operating
thresholds. It does not supply anomalous labels. Test anomalies remain unseen
until final evaluation; model selection must use normal-only objectives or a
separately documented synthetic validation protocol.

## Required audit

The audit will fail on:

- missing, unexpected, undecodable, or duplicate images;
- an official test row appearing in train or validation;
- the same content hash appearing across splits;
- cross-split perceptual near-duplicates above the frozen similarity threshold;
- an anomaly without a readable and spatially aligned non-empty mask;
- a normal sample with positive mask pixels;
- count, category, label, path, split, or schema disagreement;
- tampering with the manifest, audit, configuration, or source split table.

On success, the builder publishes a manifest plus an audit containing their
hashes, counts, configuration fingerprint, source checksums, and tool versions.
Downstream code must verify that pair before loading any sample.

The initial conservative perceptual gate requires a 64-bit dHash Hamming
distance no greater than 2 and mean-normalized 32×32 grayscale MAE no greater
than 0.1. These thresholds identify essentially identical decoded content while
avoiding false duplicate claims for VisA's deliberately aligned consecutive
product photographs. Thresholds are versioned in the dataset configuration and
recorded in every audit.

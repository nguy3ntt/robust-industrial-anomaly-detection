# Dataset card: FaceForensics++

## Status and provenance

| Field | Record |
| --- | --- |
| Official source | <https://github.com/ondyari/FaceForensics> |
| Terms | <https://kaldir.vc.in.tum.de/faceforensics_tos.pdf> |
| Access form | <https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform> |
| Paper | Rössler et al., *FaceForensics++: Learning to Detect Manipulated Facial Images*, ICCV 2019 |
| Review date | 2026-09-15 |
| Access status | Not requested; initial use deferred because the form requests academic affiliation and advisor/PI details |
| Project status | Optional future benchmark only if truthful access is approved and current terms permit use |
| Local storage | `${DEEPFAKE_DATA_ROOT}/faceforensicspp`; never committed |

This card summarizes the project team's understanding and is not legal advice.
The researcher must read the live terms before accepting because terms can
change after this review.

## Intended use in this project

FaceForensics++ is no longer the initial training source. DFDC Preview was
selected instead because the project owner cannot truthfully supply the
academic affiliation and advisor/PI details requested by the access form. This
dataset may be reconsidered only through truthful independent access or a
future change in provider requirements.

The model output remains screening and decision support. The dataset must not
be used to claim that a detector proves authenticity or manipulation.

## Content and structure

The official repository documents 1,000 original sequences derived from 977
YouTube videos, with four original manipulation methods and binary masks. It
documents original filenames as numeric sequence IDs and manipulated filenames
as `<target sequence>_<source sequence>`. The official split is 720 source
sequences for training, 140 for validation, and 140 for testing.

The authors estimate approximately 10 GB for all c23 FaceForensics++ H.264
videos, about 2 GB for c40, and about 500 GB for raw/c0. The c23 videos are
therefore the first download. Frames will be decoded or cached selectively;
the project will not download the roughly 2 TB extracted-PNG distribution.

## Terms and handling controls

The terms reviewed on 2026-09-14 state, in summary:

- use is limited to non-commercial research and educational purposes;
- the researcher accepts responsibility for use and receives no warranty;
- colleagues may receive access only after separately agreeing to the terms;
- access can be terminated;
- a researcher employed by a for-profit entity represents that they can bind
  that employer; and
- the requested dataset citation must be used.

Operational controls derived from those terms:

- every user obtains access independently from the official source;
- no videos, frames, face crops, masks, credentials, or download links enter
  Git, public artifacts, or unapproved services;
- local dataset directories remain access-controlled and are deleted if the
  provider requires it;
- only aggregate metrics and non-identifying, license-permitted figures may be
  considered for publication; and
- the access date, selected files, official download-script version/hash, and
  dataset checksums are recorded locally before training.

## Labels, lineage, and splitting

The manifest builder treats original numeric IDs as source-sequence IDs. For a
manipulated file, both the target and donor sequence IDs are lineage. Every
record receives a split only when all lineage IDs map to the same official
split. Extracted frames and clips inherit the parent video ID and split.

The official metadata does not provide a verified identity map suitable for
claiming person-disjoint evaluation. Consequently:

- source-sequence disjointness is mandatory and automatically verified;
- identity disjointness is reported as **not verified**, not silently assumed;
- an approved `sequence_id,identity_id` CSV can be supplied later, at which
  point identity leakage becomes a hard audit error; and
- results must describe the split as source-sequence-disjoint unless a verified
  identity audit has actually passed.

## Known limitations and bias risks

- Source videos were selected because they contain trackable, mostly frontal,
  unobstructed faces. This underrepresents difficult poses, occlusions, camera
  motion, and real-world capture failures.
- The source consists of YouTube material and does not provide project-usable,
  consent-verified demographic annotations. Demographic balance and subgroup
  performance cannot be established from filenames or appearance guesses.
- Manipulations represent a small, historically specific method set. Strong
  in-dataset performance does not imply generalization to current generators.
- Compression levels are controlled approximations and do not reproduce every
  social-media pipeline.
- Multiple manipulated videos share source lineage with originals. File-level
  or frame-level random splitting would produce severe leakage.
- Binary masks describe dataset-specific manipulated regions and may not equal
  perceptually meaningful or causally sufficient explanations.

## Required audit evidence

A training run may use this dataset only when its audit JSON reports `passed:
true`. The audit must include full SHA-256 coverage, successful media probes,
expected per-method counts, the official split hashes, 1,000 unique source
sequences, no source or identity crossing, no cross-split content duplicates,
and a manifest checksum.

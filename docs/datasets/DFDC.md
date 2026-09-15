# Dataset card: Deepfake Detection Challenge Preview

## Status and provenance

| Field | Record |
| --- | --- |
| Official source | <https://ai.meta.com/datasets/dfdc/> |
| Access portal | <https://dfdc.ai/> |
| Paper | Dolhansky et al., *The Deepfake Detection Challenge (DFDC) Preview Dataset* (2019) |
| Review date | 2026-09-15 |
| Access status | Selected; live agreement acceptance and download pending |
| Project role | Primary M1/M2 dataset for baseline training and controlled evaluation |
| Local storage | `${DEEPFAKE_DATA_ROOT}/dfdc-preview`; never committed |

This card records public official information and is not legal advice. The live
agreement displayed by the access portal is authoritative and must be reviewed
and accepted personally before downloading data.

## Why this dataset is the primary source

The preview dataset is proportionate to the available workstation while still
supporting video-level research. Meta documents about 5,000 videos made with two
facial-modification algorithms. The accompanying paper reports 66 paid actors,
all of whom agreed to the use and manipulation of their likenesses; it also
states that no public or social-media footage was used.

This gives the project a stronger consent and identity-governance foundation
than the original FaceForensics++ plan. The provider's `dataset.json` records
the target and swapped identities, provider train/test assignment, label, and
augmentations for each video, allowing leakage to be checked directly.

## Intended use

- Train reproducible frame and video baselines on the provider training data.
- Derive validation only from provider-training identities.
- Keep the provider test identities frozen for final in-dataset evaluation.
- Test robustness to the degradations identified in provider metadata.
- Use a separately approved dataset later for frozen cross-dataset evaluation.

The detector is a screening and decision-support prototype. Neither dataset
membership nor a model score establishes forensic certainty.

## Split and leakage policy

The provider paper states that actors were separated between its training and
test sets to avoid cross-set swaps. This project preserves that boundary.

For the three-way project split:

1. Every target and swapped identity appearing together is joined into one
   connected identity group.
2. Provider test identities remain `test` and are never considered for
   validation.
3. A deterministic, seed-recorded selection of disconnected provider-training
   groups becomes `val`; the remainder becomes `train`.
4. Every video, derived clip, and frame inherits that identity-level split.

The audit rejects provider identity crossings, derived identity crossings,
unassigned records, byte-identical files crossing splits, and any mismatch
between `dataset.json` and local MP4 files.

## Count policy

The paper describes the dataset as about 5,000 videos and reports 66 actors. It
also contains an internal count inconsistency: one table reports 5,214 videos,
while the described 4,464 training and 780 test clips sum to 5,244. The project
therefore does not silently choose one disputed total. Instead, the real audit
requires:

- at least 5,000 records;
- exactly 66 unique identities;
- an exact one-to-one match between every `dataset.json` entry and local MP4;
- non-empty original, method A, and method B groups; and
- non-empty derived train, validation, and test splits.

The observed exact counts and provider metadata SHA-256 are stored in the audit.

## Governance requirements before use

- The researcher personally reviews and accepts the live access agreement.
- AWS credentials, access URLs, and tokens remain outside the repository and logs.
- The access date, package/object names, byte counts, provider metadata hash,
  and terms record are stored privately in `.project/DATA_ACCESS.md`.
- Data is downloaded only from the official portal into the configured D: drive.
- No videos, frames, crops, identities, credentials, or access links are
  committed, redistributed, or sent to external services.
- FFmpeg/ffprobe is installed and its version recorded before the real audit.
- Training remains blocked until `dfdc-preview.audit.json` reports `passed: true`.

## Known limitations

- Only two undisclosed face-swap methods are represented. This is not enough to
  support an unseen-method generalization claim by itself.
- The class and dataset priors do not represent real-world deepfake prevalence.
- Aggregate demographic figures in the paper do not create reliable per-video
  subgroup labels. The project will not infer demographics from appearance.
- Controlled capture and 15-second clipping differ from uncontrolled media.
- Test-time degradations may be learnable shortcuts and must be analyzed.
- The dataset does not provide the localization masks that FaceForensics++ did;
  localization evaluation will require a separately approved source or a
  consented synthetic test set later.

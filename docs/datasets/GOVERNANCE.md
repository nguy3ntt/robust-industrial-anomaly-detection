# Dataset governance gate

No restricted dataset is eligible for training merely because files exist on
disk. The project owner must personally accept access terms, and each acquired
package must pass the reproducible audit before training.

## DFDC Preview access gate

1. Open the official page at <https://ai.meta.com/datasets/dfdc/> and follow its
   access link.
2. Read the live agreement in full and confirm it permits this non-commercial
   portfolio/academic project. Stop if it conflicts with the intended use.
3. Use only accurate personal and AWS account details. Do not share secret keys
   or access URLs with Codex, Git, logs, or documentation.
4. Record the acceptance and access dates privately in
   `.project/DATA_ACCESS.md`; retain a local copy or hash of the accepted terms
   where the portal permits it.
5. Review AWS transfer/storage costs, then download the official preview package
   to `D:\deepfake-data\dfdc-preview`.
6. Confirm the root contains `dataset.json`, `original_videos`, `method_A`, and
   `method_B`. Do not rearrange or rename provider files.
7. Install FFmpeg/ffprobe from a trusted distribution and record its version.
8. Set the real access date in `configs/data/dfdc-preview.yaml` and run the
   manifest builder.
9. Do not begin model development unless the audit reports `passed: true`.

## Required acquisition record

Record the source and access date, selected package/object names, total files
and bytes, provider metadata SHA-256, accepted-terms reference, download tool
version, retries, and ffprobe version. Never record passwords, AWS access keys,
session tokens, or private download URLs.

## Stop conditions

Stop processing and investigate if:

- the live terms conflict with the intended use or publication plan;
- access belongs to another person or organization;
- `dataset.json` is absent, unreadable, or structurally unexpected;
- the metadata identity count is not 66;
- media and metadata paths do not match exactly;
- an identity occurs in both provider train and test data;
- validation cannot be formed from disconnected provider-training identities;
- media probing or full-file hashing fails;
- an identity or byte-identical file crosses a derived split; or
- the audit is rejected or does not match the manifest SHA-256.

## Deferred datasets

FaceForensics++ remains documented but is not an active dependency. Never
invent affiliation or advisor details. DF40 is not a clean substitute because
much of it derives from FaceForensics++ and Celeb-DF. Any later dataset must
receive a separate licence, provenance, consent, and leakage review.

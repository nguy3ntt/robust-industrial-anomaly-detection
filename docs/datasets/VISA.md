# VisA dataset card

## Summary

The Visual Anomaly (VisA) dataset is the primary benchmark for this project. It
contains 10,821 high-resolution colour images from 12 industrial-object
categories: 9,621 normal images and 1,200 anomalous images. Anomalous samples
include surface defects such as scratches and cracks and structural defects
such as missing or misplaced parts. Image-level labels and pixel-level masks
are provided.

- Official project: <https://github.com/amazon-science/spot-diff>
- Public archive: <https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar>
- Versioned archive name: `VisA_20220922.tar`
- Licence: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)
- Citation: Zou et al., *SPot-the-Difference Self-Supervised Pre-training for
  Anomaly Detection and Segmentation*, ECCV 2022.

The archive is publicly downloadable over HTTPS. It does not require an AWS
account, access key, institutional affiliation, or adviser approval.

## Intended use

VisA supports normal-only training, image-level anomaly detection, and
pixel-level defect localization. This project uses the official one-class split
as the primary comparable protocol. The provider test partition is immutable.
A deterministic calibration subset will be taken only from official training
normal images.

The dataset is appropriate for a portfolio and academic research prototype. It
does not establish performance on a particular factory, camera, production
line, material, or defect prevalence. Deployment claims require representative
site-specific validation.

## Data and labels

The 12 categories cover printed circuit boards, multiple-object scenes, and
roughly aligned single objects. Each category provides normal and anomalous
images, per-image annotations, and masks for anomalous images. Normal images do
not have stored masks because their masks are implicitly empty.

The official repository distributes split tables for one-class and two-class
research settings. This project initially supports only `split_csv/1cls.csv`.
It will not infer labels from filenames when provider annotations are available.

## Integrity controls

Before a manifest can be trusted, M1 must verify:

1. the archive checksum before extraction;
2. the exact expected global and per-category counts;
3. every split row against a safe, relative, existing image path;
4. every anomalous test image against a readable, non-empty mask;
5. every normal sample against the absence of anomalous mask pixels;
6. full-file duplicates and perceptual near-duplicates across partitions;
7. image dimensions, channels, decode validity, and mask alignment;
8. that official test rows were neither moved nor used to derive validation.

VisA contains highly aligned and sometimes consecutive captures. Perceptual
duplicate checks therefore use a conservative, versioned two-stage threshold
to distinguish essentially identical decoded content from merely similar
products. Confirmed components in provider training remain in one derived
split; a confirmed component crossing provider train/test fails the audit.

Canonical manifests are published only after the audit passes. The access date,
archive checksum, split-table checksum, configuration hash, and tool versions
must be retained with the manifest.

## Known limitations

- The 12 categories do not represent all industrial materials or processes.
- Images are curated and may not reproduce the lighting, motion, contamination,
  or class prevalence of a live production line.
- Anomaly labels describe visible defects, not root cause or business severity.
- Pixel masks can contain annotation uncertainty near defect boundaries.
- Results can be dominated by category or capture shortcuts unless per-category
  and shift-specific metrics are reported.
- The benchmark does not justify autonomous rejection of manufactured items.

## Required attribution

Any published report, demonstration, or redistributed permitted derivative
must attribute the VisA authors and dataset under CC BY 4.0. Dataset files are
not committed to this repository; code licensing does not replace the dataset
licence.

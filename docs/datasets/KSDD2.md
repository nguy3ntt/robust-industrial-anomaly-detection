# KolektorSDD2 external validation dataset

Kolektor Surface-Defect Dataset 2 is a real industrial surface-defect benchmark
provided by ViCoS and Kolektor Group. Images contain scratches, spots, and other
surface imperfections, with binary pixel masks and an official train/test split.

- Official page: https://www.vicos.si/resources/kolektorsdd2/
- Official archive: https://data.vicos.si/datasets/KSDD/KolektorSDD2.zip
- Provider redirect: https://go.vicos.si/kolektorsdd2
- Accessed: 2026-10-13.
- Pinned revision: archive Last-Modified 2023-09-28; local ID `ksdd2-20230928`.
- Licence: CC BY-NC-SA 4.0. Academic and noncommercial portfolio use is permitted;
  commercial use requires contacting the provider. Dataset terms remain separate
  from this repository's MIT code licence.
- Archive bytes: 853,126,555.
- Archive SHA-256:
  `edcdb486809b24f1d17b785e30c52fafc5999554dd5fe18ddf77b61ceb6f36a8`.

The checksum is an observed pin from the official HTTPS download. The provider
page does not publish an independent SHA-256. Future runs verify this exact
acquisition, rather than claiming independent provider checksum authentication.

## Audited partition

| Provider partition | Normal | Defect | Total |
|---|---:|---:|---:|
| Train | 2,085 | 246 | 2,331 |
| Test | 894 | 110 | 1,004 |

The provider ZIP also contains `train/10301 (copy).png` and
`train/10301_GT (copy).png`. Their full hashes exactly match `10301.png` and
`10301_GT.png`. These two copy files are retained on disk but explicitly
excluded from the canonical inventory. The exclusions are configured and
recorded in the audit; unexpected or changed copies fail validation.

The 246 provider training defects are excluded from all fitting and calibration.
Training normals are grouped by exact file hashes and the same conservative
perceptual definition used in VisA: dHash distance at most 2 and mean-centered
32×32 grayscale MAE at most 0.1 on the uint8 intensity scale. A confirmed
train/test duplicate would exclude its training component while preserving the
official test membership. No confirmed pairs were found in this acquisition.

Whole normal components receive a deterministic seeded split, yielding 1,772
fitting normals and 313 calibration normals. Every image/mask pair is decoded,
checked for matching dimensions and binary masks, and hashed. Downstream use
verifies the audit/config/manifest identity and each consumed image and mask.

## Preprocessing and limits

Images are approximately 230×630 pixels. M9 preserves the full frame with a
bilinear resize to 224×224, and uses the corresponding nearest-neighbour mask
resize. This avoids dropping strip-end defects, but changes aspect ratio and
limits fine defect detail. Localization is reported on a declared 56×56 grid;
these metrics must not be presented as native-resolution measurements or
directly compared with VisA's 224×224 localization values.

This is one additional surface domain. It strengthens evidence that the method
can be adapted to another dataset using normals, but is not a camera deployment
study or zero-shot validation of VisA product memories. The provider does not
expose physical-item/session identifiers, so grouping checks confirmed image
duplicates rather than proving independent physical items.

## Attribution

Jakob Božič, Domen Tabernik, and Danijel Skočaj. *Mixed supervision for
surface-defect detection: from weakly to fully supervised learning*.
Computers in Industry, 2021. Cite this paper when using KolektorSDD2.

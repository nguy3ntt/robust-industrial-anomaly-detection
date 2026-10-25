# Portable demo — third-party attribution and distribution boundary

Reviewed against the sources below on **2026-10-25**. This records the package's
scope; it is not a warranty that every downstream use has all necessary rights.

## Project code

The project's own code is MIT-licensed; retain `LICENSE`. Third-party Python
packages are separately installed, not vendored or relicensed by this archive.
Their own notices and terms continue to apply.

## VisA-derived normal references

Visual Anomaly (VisA), released with *SPot-the-Difference Self-Supervised
Pre-training for Anomaly Detection and Segmentation* (ECCV 2022), by Yang Zou,
Jongheon Jeong, Latha Pemula, Dongqing Zhang and Onkar Dabeer.

Provider notice: Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

- [Official project and citation](https://github.com/amazon-science/spot-diff)
- [Provider dataset licence](https://github.com/amazon-science/spot-diff/blob/main/LICENSE-DATASET)
- [Provider notice](https://github.com/amazon-science/spot-diff/blob/main/NOTICE)
- [CC BY 4.0 licence](https://creativecommons.org/licenses/by/4.0/legalcode.en)

The provider identifies the dataset as CC BY 4.0. Preserve attribution, licence
links and the indication of changes when sharing permitted derivatives. In this
package the source photographs were transformed into frozen neural features,
projected normal memories, unlabelled calibration references and spatial/quality
statistics. Raw photographs and masks are not redistributed. The bundled
derivative references retain this CC BY 4.0 notice; the code's MIT licence does
not replace it. No endorsement is implied and no warranty is provided.

## ResNet/ImageNet checkpoint — not included

This demo uses torchvision's ResNet-18 ImageNet-1K V1 checkpoint. It is **not in
the ZIP**. Users may supply their verified existing file or explicitly request
the fixed official download with `--prepare`. The SHA-256 is pinned in code and
the runtime contract; startup does not silently fetch another model.

[Torchvision's model documentation](https://docs.pytorch.org/vision/stable/models.html)
warns that pretrained weights can have their own dataset-derived terms. Library
code licensing does not establish unrestricted rights to the checkpoint or its
outputs. Review the applicable terms for your use, especially before commercial
use or redistributing model-derived artifacts; no blanket rights clearance is
asserted here. The project's stated purpose remains academic/portfolio use.

## Explicit exclusions and publication review

No KolektorSDD2 images/references, private production/capture data, source images,
test predictions, sample identifiers, pretrained checkpoints or credentials are
eligible for this fixed exporter. Derived feature memories can retain source
information: absence of raw pixels is not proof of anonymity.

The exporter builds locally only. Before attaching a release candidate to a
public release, its owner must inspect the manifest/exclusions, retain these
notices and report checksums, and review intended-use rights. Nothing is uploaded
by the build command. Existing aggregate-only historical report packages remain
unchanged; this is a separate runtime distribution with an explicit inventory.

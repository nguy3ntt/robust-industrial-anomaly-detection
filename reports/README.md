# Reports

This directory contains small public figures, tables, and final written
outputs. Do not place raw dataset images, private inspection samples, model
weights, feature banks, or machine-specific run directories here.

Public milestone reports:

- [M6 practical prototype](M6_PROTOTYPE.md)
- [M7 final evaluation](M7_FINAL_EVALUATION.md)
- [Conference-style paper](CONFERENCE_PAPER.md)
- [M8 capture-quality gate](M8_CAPTURE_QUALITY_GATE.md)
- [M9 independent second-dataset validation](M9_EXTERNAL_VALIDATION.md)
- [M10 thresholded localization](M10_LOCALIZATION.md)
- [M11 modern encoder comparison](M11_ENCODER_COMPARISON.md)
- [M12 capture-quality generalization](M12_QUALITY_GENERALIZATION.md)
- [M13 small-defect resolution and tiling](M13_SMALL_DEFECT_LOCALIZATION.md)

M13 compares coarse/fine readout and overlapping detail tiles using common fine
ground truth and matched normal-memory budgets. Aggregate overlap, region-size,
false-positive, paired-difference and resource tables are under [`m13/`](m13/).
Rebuild with `scripts/build_m13_report.py`; licensed comparison images and private
per-image evidence remain in ignored local artifacts. Fine display resolution
is not itself additional encoder information or physical defect precision.

M12 aggregate gate trade-offs and exact source/output pins are under
[`m12/`](m12/). Its modest improvements do not resolve the cross-domain gate
weakness or automatically replace the historical inspector.

M11 aggregate category metrics, paired differences, macro intervals, resource
measurements, mask visibility, severe operating trade-offs, and their exact
source/output hashes are under [`m11/`](m11/). Rebuild with
`scripts/build_m11_report.py` using `configs/reporting/m11.yaml`. This comparison
does not replace the deployed ResNet inspector or rewrite historical M7 outputs.

M10 aggregate tables and source/output hashes are under [`m10/`](m10/).
Rebuild with `scripts/build_m10_report.py` using `configs/reporting/m10.yaml`.
Its five-policy comparison keeps visible-defect metrics separate from normal
empty-mask success and discloses unsuccessful spatial alternatives.

The M9 aggregate metrics and exact source/output hash manifest are under
[`m9/`](m9/). Rebuild them using `scripts/build_m9_report.py` and the exact run
pinned in `configs/reporting/m9.yaml`. Dataset images, per-image outcomes, and
feature banks remain outside this public directory.

The deterministic M7 tables, figures, and provenance manifest are under
[`m7/`](m7/). Presentation-ready artifacts are generated in `../output/`:

- [Conference paper PDF](../output/pdf/robust-industrial-anomaly-detection-paper.pdf)
- [Research poster PDF](../output/pdf/robust-industrial-anomaly-detection-poster.pdf)
- [Editable research deck](../output/presentations/robust-industrial-anomaly-detection-m7-final.pptx)
- [Caption-led demonstration video](../output/video/robust-industrial-anomaly-detection-demo.mp4)

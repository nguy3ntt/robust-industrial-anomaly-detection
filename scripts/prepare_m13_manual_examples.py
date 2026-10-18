"""Render local common-grid/tiling comparison panels without publishing dataset imagery."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from build_m13_report import verified_source
from PIL import Image, ImageDraw, ImageFont

from robust_industrial_anomaly_detection.baselines.tiling import evaluation_support
from robust_industrial_anomaly_detection.data.visa import sha256_file
from robust_industrial_anomaly_detection.evaluation.small_defects import (
    capture_measurements,
    region_measurements,
)
from robust_industrial_anomaly_detection.experiments.m2 import _atomic_json, _read_csv
from robust_industrial_anomaly_detection.experiments.m12 import verify_files


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("C:/Windows/Fonts/consola.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def _overlay(
    image: np.ndarray, values: np.ndarray, threshold: float, support: np.ndarray
) -> np.ndarray:
    relative = values / max(abs(threshold), 1e-8)
    heat = cv2.cvtColor(
        cv2.applyColorMap(
            np.asarray(np.clip(relative / 2, 0, 1) * 255, dtype=np.uint8), cv2.COLORMAP_INFERNO
        ),
        cv2.COLOR_BGR2RGB,
    )
    mixed = np.asarray(
        np.clip(image.astype(float) * 0.65 + heat.astype(float) * 0.35, 0, 255), dtype=np.uint8
    )
    contours, _ = cv2.findContours(
        support.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(mixed, contours, -1, (255, 65, 65), 1)
    return mixed


def make_panel(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    rows: list[dict[str, Any]],
    calibrations: dict[str, Any],
    destination: Path,
    title: str,
) -> dict[str, Any]:
    rgb, mask = arrays["rgb"], arrays["mask"].astype(bool)
    if rgb.shape != (336, 336, 3) or rgb.dtype != np.uint8 or mask.shape != (336, 336):
        raise ValueError("manual arrays have unexpected input geometry")
    maps = {
        "global56": arrays["global_map"],
        "global336": arrays["global_map"],
        "tiled336": arrays["tiled_map"],
    }
    if any(
        values.shape != (336, 336) or not np.all(np.isfinite(values)) for values in maps.values()
    ):
        raise ValueError("manual score maps are invalid")
    panels = [rgb.copy(), np.repeat((mask.astype(np.uint8) * 255)[:, :, None], 3, axis=2)]
    captions = ["Common 336 input", "Ground truth (positive coverage)"]
    metrics = []
    for name, values in maps.items():
        mode = "tiled" if name == "tiled336" else "global"
        grid = 56 if name == "global56" else 336
        threshold = calibrations[mode]["pixel_thresholds"][str(grid)]
        support = evaluation_support(values, threshold, grid=grid)
        measurements = capture_measurements(
            mask,
            support,
            metadata["label"],
            small_fraction=0.001,
            regions=region_measurements(mask, support),
        )
        prediction = next(
            row
            for row in rows
            if row["sample_id"] == metadata["sample_id"] and row["readout"] == name
        )
        if (
            measurements["mask_pixels"] != prediction["mask_pixels"]
            or measurements["flagged_pixels"] != prediction["flagged_pixels"]
        ):
            raise ValueError("manual support does not reproduce the evaluated mask/prediction")
        if measurements["foreground_iou"] is not None and not np.isclose(
            measurements["foreground_iou"], prediction["foreground_iou"]
        ):
            raise ValueError("manual foreground IoU differs from evaluated evidence")
        panels.append(_overlay(rgb, values, threshold, support))
        captions.append(name)
        metrics.append(
            {
                "readout": name,
                "decision": prediction["decision"],
                "conformal_p": prediction["conformal_p"],
                "pixel_threshold": threshold,
                **measurements,
            }
        )
    y, x = np.nonzero(mask)
    if len(x):
        center_x, center_y = int((x.min() + x.max()) / 2), int((y.min() + y.max()) / 2)
        half = max(24, int(max(x.max() - x.min() + 1, y.max() - y.min() + 1) / 2) + 8)
    else:
        center_x = center_y = 168
        half = 64
    left, top = max(0, center_x - half), max(0, center_y - half)
    right, bottom = min(336, center_x + half + 1), min(336, center_y + half + 1)
    canvas = Image.new("RGB", (1792, 920), "#15171b")
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 12), title, font=_font(24), fill="#f5e9e6")
    draw.text(
        (16, 46),
        f"Source label: {metadata['label']} | GT pixels: {int(mask.sum())} "
        "| fixed mask/crop convention",
        font=_font(17),
        fill="#c8bcb7",
    )
    for index, (pixels, caption) in enumerate(zip(panels, captions, strict=True)):
        offset = 16 + index * 352
        draw.text((offset, 80), caption, font=_font(16), fill="#ffffff")
        canvas.paste(Image.fromarray(pixels), (offset, 108))
        roi = pixels[top:bottom, left:right]
        zoom = Image.fromarray(roi).resize((336, 336), Image.Resampling.NEAREST)
        canvas.paste(zoom, (offset, 490))
        draw.text((offset, 459), "Same ROI, nearest-pixel zoom", font=_font(15), fill="#c8bcb7")
        if index >= 2:
            row = metrics[index - 2]
            iou = (
                "undefined (no positive GT)"
                if row["foreground_iou"] is None
                else f"IoU {row['foreground_iou']:.4f}"
            )
            draw.text((offset, 842), f"{iou}", font=_font(15), fill="#ffffff")
            draw.text(
                (offset, 865),
                f"{row['decision']} | p {row['conformal_p']:.4f}",
                font=_font(15),
                fill="#ffffff",
            )
    draw.text(
        (16, 896),
        "Heat: score / own frozen threshold (0-2). Red: exact support. "
        "Source label and model decision are NOT interchangeable.",
        font=_font(16),
        fill="#d8bab3",
    )
    canvas.save(destination)
    return {
        "panel": destination.name,
        **metadata,
        "roi": [left, top, right, bottom],
        "results": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("visa", "ksdd2"), default="visa")
    parser.add_argument("--category", default="candle")
    parser.add_argument(
        "--run", type=Path, help="Explicit completed run for local smoke/debug evidence only"
    )
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    if args.run is None:
        config = yaml.safe_load((project / "configs/reporting/m13.yaml").read_text("utf-8"))
        run = verified_source(config, project)
    else:
        run = args.run.resolve(strict=True)
        metadata = json.loads((run / "metadata.json").read_text("utf-8"))
        if metadata.get("status") != "complete":
            raise ValueError("local comparison requires a completed run")
        verify_files(run, metadata["output_hashes"])
    metadata = json.loads((run / "metadata.json").read_text("utf-8"))
    category = "surface" if args.dataset == "ksdd2" else args.category
    if args.dataset == "visa":
        visa = yaml.safe_load((project / "configs/data/visa.yaml").read_text("utf-8"))
        if category not in visa["categories"]:
            raise ValueError("choose a configured VisA category")
    directory = run / "units" / args.dataset / category
    directory.resolve(strict=True).relative_to(run)
    clean = directory / "conditions/clean"
    calibrations = {
        mode: json.loads((directory / mode / "calibration.json").read_text("utf-8"))
        for mode in ("global", "tiled")
    }
    rows: list[dict[str, Any]] = list(_read_csv(clean / "predictions.csv"))
    output = (
        project
        / "artifacts/manual"
        / f"m13-{args.dataset}-{category}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    )
    output.mkdir(parents=True, exist_ok=False)
    examples = []
    for name in ("normal", "smallest-anomaly"):
        path = clean / f"{name}.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
        sample = json.loads((clean / f"{name}.json").read_text("utf-8"))
        examples.append(
            make_panel(
                arrays,
                sample,
                rows,
                calibrations,
                output / f"{name}.png",
                f"{args.dataset}/{category} | {name} | {metadata['mode']} study",
            )
        )
    if not examples:
        raise ValueError("selected run/category has no saved comparison examples")
    _atomic_json(
        output / "expected.json",
        {
            "source_run": run.name,
            "source_metadata_sha256": sha256_file(run / "metadata.json"),
            "mode": metadata["mode"],
            "selection": (
                "first fixed test normal; smallest positive fine mask; no prediction-outcome search"
            ),
            "source_images_published": False,
            "examples": examples,
        },
    )
    print(
        json.dumps(
            {
                "folder": str(output),
                "panels": [row["panel"] for row in examples],
                "mode": metadata["mode"],
                "evaluated_support_reproduced": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

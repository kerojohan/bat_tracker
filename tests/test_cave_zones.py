from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from bat_tracker.cave_zones import annotation_to_mask, run_cave_zones


def test_annotation_to_mask_extracts_red_cave_overlay() -> None:
    annotation = np.zeros((80, 120, 3), dtype=np.uint8)
    cv2.ellipse(annotation, (35, 55), (24, 12), -20, 0, 360, (0, 0, 255), -1)

    mask = annotation_to_mask(annotation)

    assert mask.dtype == np.uint8
    assert int(np.count_nonzero(mask)) > 500
    ys, xs = np.nonzero(mask)
    assert xs.min() < 20
    assert xs.max() > 50
    assert ys.min() > 35


def test_run_cave_zones_prioritizes_annotation_and_limits_components(tmp_path: Path) -> None:
    background = np.full((100, 140), 90, dtype=np.uint8)
    annotation = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(annotation, (10, 60), (70, 92), (0, 0, 255), -1)
    cv2.rectangle(annotation, (105, 5), (132, 28), (0, 0, 255), -1)
    annotation_path = tmp_path / "background_cave.png"
    cv2.imwrite(str(annotation_path), annotation)

    result = run_cave_zones(
        background_gray=background,
        output_dir=tmp_path / "cave_zones",
        cfg={
            "enabled": True,
            "method": "hybrid",
            "input_annotation": str(annotation_path),
            "min_component_area_ratio": 0.001,
            "max_components": 1,
            "dilate_px": 0,
        },
    )

    assert result.mask is not None
    assert len(result.zones) == 1
    assert result.meta["source"] == "input_annotation"
    assert Path(result.outputs["cave_zones_mask_png"]).exists()
    assert Path(result.outputs["cave_zones_overlay_png"]).exists()
    zones_payload = json.loads(Path(result.outputs["cave_zones_zones_json"]).read_text(encoding="utf-8"))
    assert len(zones_payload["zones"]) == 1
    x1, y1, x2, y2 = zones_payload["zones"][0]["bbox"]
    assert x1 <= 12
    assert y1 >= 55
    assert x2 >= 65
    assert y2 >= 88


def test_run_cave_zones_hybrid_uses_motion_and_dark_region(tmp_path: Path) -> None:
    background = np.full((90, 120), 120, dtype=np.uint8)
    background[52:82, 12:58] = 18
    motion = np.zeros_like(background, dtype=np.float32)
    motion[56:78, 20:52] = 10.0

    result = run_cave_zones(
        background_gray=background,
        output_dir=tmp_path / "cave_zones",
        cfg={
            "enabled": True,
            "method": "hybrid",
            "input_mask": "",
            "input_annotation": "",
            "use_motion_heatmap": True,
            "use_dark_regions": True,
            "min_component_area_ratio": 0.001,
            "max_components": 1,
            "dilate_px": 2,
            "motion_percentile": 70.0,
            "dark_percentile": 20.0,
        },
        motion_heatmap=motion,
    )

    assert result.mask is not None
    ys, xs = np.nonzero(result.mask)
    assert xs.min() <= 20
    assert xs.max() >= 52
    assert ys.min() <= 56
    assert ys.max() >= 78

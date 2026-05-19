from __future__ import annotations

from bat_tracker.detection import Detection
from bat_tracker.detection_fusion import build_secondary_detection_config
from bat_tracker.detection_fusion import fuse_detections


def _det(x: float, y: float, size: int = 6) -> Detection:
    half = size // 2
    return Detection(
        x=x,
        y=y,
        bbox_x1=int(x) - half,
        bbox_y1=int(y) - half,
        bbox_x2=int(x) + half,
        bbox_y2=int(y) + half,
        area=float(size * size),
    )


def test_fuse_detections_keeps_primary_and_adds_secondary_missing() -> None:
    primary = [_det(20, 20), _det(60, 60)]
    secondary = [_det(22, 21), _det(95, 95)]

    fused, stats = fuse_detections(
        primary,
        secondary,
        dedupe_max_distance_px=5.0,
        dedupe_min_iou=0.10,
    )

    assert fused == [primary[0], primary[1], secondary[1]]
    assert stats.primary_count == 2
    assert stats.secondary_count == 2
    assert stats.secondary_added == 1
    assert stats.secondary_duplicates == 1


def test_secondary_detection_config_inherits_primary_and_applies_overrides() -> None:
    cfg = build_secondary_detection_config(
        {"threshold_mode": "otsu", "diff_threshold": 18, "min_area": 6},
        {
            "enabled": True,
            "inherit_primary": True,
            "dedupe_max_distance_px": 8.0,
            "threshold_mode": "fixed",
            "diff_threshold": 10,
        },
    )

    assert cfg == {"threshold_mode": "fixed", "diff_threshold": 10, "min_area": 6}

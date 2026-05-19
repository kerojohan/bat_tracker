from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .detection import Detection


@dataclass(frozen=True)
class DetectionFusionStats:
    primary_count: int
    secondary_count: int
    secondary_added: int
    secondary_duplicates: int


def build_secondary_detection_config(primary_cfg: dict, secondary_cfg: dict) -> dict:
    inherit_primary = bool(secondary_cfg.get("inherit_primary", True))
    if inherit_primary:
        cfg = dict(primary_cfg)
    else:
        cfg = {}

    control_keys = {
        "enabled",
        "inherit_primary",
        "dedupe_max_distance_px",
        "dedupe_min_iou",
    }
    for key, value in secondary_cfg.items():
        if key not in control_keys:
            cfg[key] = value
    return cfg


def fuse_detections(
    primary: Iterable[Detection],
    secondary: Iterable[Detection],
    *,
    dedupe_max_distance_px: float,
    dedupe_min_iou: float,
) -> tuple[List[Detection], DetectionFusionStats]:
    primary_list = list(primary)
    secondary_list = list(secondary)
    fused = list(primary_list)
    duplicate_count = 0

    for det in secondary_list:
        if _matches_any(det, fused, dedupe_max_distance_px, dedupe_min_iou):
            duplicate_count += 1
            continue
        fused.append(det)

    return fused, DetectionFusionStats(
        primary_count=len(primary_list),
        secondary_count=len(secondary_list),
        secondary_added=len(fused) - len(primary_list),
        secondary_duplicates=duplicate_count,
    )


def _matches_any(
    det: Detection,
    existing: list[Detection],
    max_distance_px: float,
    min_iou: float,
) -> bool:
    max_distance_sq = max(0.0, float(max_distance_px)) ** 2
    min_iou = max(0.0, float(min_iou))
    for other in existing:
        if max_distance_sq > 0.0:
            dx = det.x - other.x
            dy = det.y - other.y
            if dx * dx + dy * dy <= max_distance_sq:
                return True
        if min_iou > 0.0 and _bbox_iou(det, other) >= min_iou:
            return True
    return False


def _bbox_iou(a: Detection, b: Detection) -> float:
    x1 = max(a.bbox_x1, b.bbox_x1)
    y1 = max(a.bbox_y1, b.bbox_y1)
    x2 = min(a.bbox_x2, b.bbox_x2)
    y2 = min(a.bbox_y2, b.bbox_y2)
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = float(inter_w * inter_h)
    if inter_area <= 0.0:
        return 0.0

    area_a = float(max(0, a.bbox_x2 - a.bbox_x1) * max(0, a.bbox_y2 - a.bbox_y1))
    area_b = float(max(0, b.bbox_x2 - b.bbox_x1) * max(0, b.bbox_y2 - b.bbox_y1))
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union

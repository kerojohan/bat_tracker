from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

import cv2
import numpy as np


@dataclass
class CaveZonesResult:
    mask: np.ndarray | None
    zones: list[dict]
    meta: dict
    outputs: dict[str, str]


def _as_binary(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _load_binary_mask(path: str | Path, expected_shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Could not load cave zone mask: {path}")
    mask = _as_binary(mask)
    if mask.shape[:2] != expected_shape:
        raise ValueError(
            "cave_zones.input_mask shape does not match the processing frame size: "
            f"expected {expected_shape}, got {mask.shape[:2]}"
        )
    return mask


def annotation_to_mask(annotation_bgr: np.ndarray) -> np.ndarray:
    """Extract red cave-mouth annotations from an RGB/BGR visual overlay."""
    if annotation_bgr.ndim != 3 or annotation_bgr.shape[2] < 3:
        return _as_binary(annotation_bgr)

    hsv = cv2.cvtColor(annotation_bgr[:, :, :3], cv2.COLOR_BGR2HSV)
    red_low = cv2.inRange(hsv, np.array([0, 70, 70]), np.array([12, 255, 255]))
    red_high = cv2.inRange(hsv, np.array([168, 70, 70]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(red_low, red_high)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return _as_binary(mask)


def _resolve_path(path_value: str) -> Path | None:
    raw = str(path_value or "").strip()
    if raw:
        return Path(raw).expanduser()
    return None


def _component_rows(
    mask: np.ndarray,
    *,
    source: str,
    background_gray: np.ndarray,
    motion_heatmap: np.ndarray | None,
    cfg: Dict[str, Any],
) -> list[dict]:
    binary = (mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    h, w = mask.shape[:2]
    frame_area = max(1, h * w)
    min_area = max(1, int(round(float(cfg.get("min_component_area_ratio", 0.002)) * frame_area)))
    dark_threshold = float(np.percentile(background_gray, float(cfg.get("dark_percentile", 18.0))))
    motion_positive = motion_heatmap[motion_heatmap > 1e-6] if motion_heatmap is not None else np.array([])
    motion_scale = float(np.percentile(motion_positive, 95.0)) if motion_positive.size else 1.0
    motion_scale = max(motion_scale, 1e-6)

    rows: list[dict] = []
    for idx in range(1, num):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        cw = int(stats[idx, cv2.CC_STAT_WIDTH])
        ch = int(stats[idx, cv2.CC_STAT_HEIGHT])
        component = labels == idx
        dark_ratio = float(np.mean(background_gray[component] <= dark_threshold)) if area else 0.0
        motion_density = 0.0
        if motion_heatmap is not None:
            motion_density = float(np.mean(np.clip(motion_heatmap[component] / motion_scale, 0.0, 1.0)))
        area_ratio = area / float(frame_area)
        reasonable_size = 1.0 - min(1.0, abs(area_ratio - 0.04) / 0.12)
        score = 0.42 * motion_density + 0.34 * dark_ratio + 0.24 * reasonable_size
        if source in {"input_mask", "input_annotation"}:
            score = 1.0 + min(1.0, area_ratio * 10.0) + min(0.1, score)
        rows.append(
            {
                "source": source,
                "component_id": idx,
                "bbox": [x, y, x + cw - 1, y + ch - 1],
                "area": area,
                "area_ratio": round(area_ratio, 6),
                "score": round(float(score), 6),
                "motion_density": round(motion_density, 6),
                "dark_ratio": round(dark_ratio, 6),
            }
        )
    return rows


def _rows_to_mask(rows: Iterable[dict], candidate_masks: list[tuple[str, np.ndarray]], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=np.uint8)
    by_source = {source: mask for source, mask in candidate_masks}
    for row in rows:
        source = str(row.get("source", ""))
        source_mask = by_source.get(source)
        if source_mask is None:
            continue
        binary = (source_mask > 0).astype(np.uint8)
        num, labels, _, _ = cv2.connectedComponentsWithStats(binary, 8)
        component_id = int(row.get("component_id", 0))
        if component_id <= 0 or component_id >= num:
            continue
        out[labels == component_id] = 255
    return out


def _motion_candidate_mask(motion_heatmap: np.ndarray | None, cfg: Dict[str, Any]) -> np.ndarray | None:
    if motion_heatmap is None or not bool(cfg.get("use_motion_heatmap", True)):
        return None
    positive = motion_heatmap[motion_heatmap > 1e-6]
    if positive.size == 0:
        return np.zeros(motion_heatmap.shape[:2], dtype=np.uint8)
    percentile = float(np.clip(float(cfg.get("motion_percentile", 94.0)), 50.0, 99.5))
    threshold = float(np.percentile(positive, percentile))
    mask = np.where(motion_heatmap >= threshold, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def _dark_candidate_mask(background_gray: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray | None:
    if not bool(cfg.get("use_dark_regions", True)):
        return None
    percentile = float(np.clip(float(cfg.get("dark_percentile", 18.0)), 1.0, 60.0))
    threshold = float(np.percentile(background_gray, percentile))
    mask = np.where(background_gray <= threshold, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def _save_overlay(background_gray: np.ndarray, mask: np.ndarray, zones: list[dict], path: Path) -> None:
    overlay = cv2.cvtColor(background_gray, cv2.COLOR_GRAY2BGR)
    tint = np.zeros_like(overlay)
    tint[mask > 0] = (0, 210, 255)
    overlay = cv2.addWeighted(overlay, 0.78, tint, 0.38, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)
    for idx, zone in enumerate(zones, start=1):
        x1, y1, x2, y2 = zone["bbox"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 220, 0), 1, cv2.LINE_AA)
        cv2.putText(
            overlay,
            f"zone {idx} {zone['score']:.2f}",
            (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(overlay, "cave entry/exit zones", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.imwrite(str(path), overlay)


def run_cave_zones(
    *,
    background_gray: np.ndarray,
    output_dir: Path,
    cfg: Dict[str, Any],
    motion_heatmap: np.ndarray | None = None,
) -> CaveZonesResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "cave_zones_mask_png": str((output_dir / "mask.png").resolve()),
        "cave_zones_overlay_png": str((output_dir / "overlay.png").resolve()),
        "cave_zones_zones_json": str((output_dir / "zones.json").resolve()),
        "cave_zones_candidates_overlay_png": str((output_dir / "zone_candidates_overlay.png").resolve()),
        "cave_zones_diagnostics_json": str((output_dir / "zone_diagnostics.json").resolve()),
    }
    meta: dict[str, Any] = {
        "enabled": bool(cfg.get("enabled", False)),
        "method": str(cfg.get("method", "hybrid")),
        "source": "",
        "zones_total": 0,
        "mask_nonzero_px": 0,
        "outputs": outputs,
    }
    if not bool(cfg.get("enabled", False)):
        return CaveZonesResult(mask=None, zones=[], meta=meta, outputs=outputs)

    expected_shape = background_gray.shape[:2]
    source_mask: np.ndarray | None = None
    source = ""
    input_mask = str(cfg.get("input_mask", "")).strip()
    if input_mask:
        source_mask = _load_binary_mask(input_mask, expected_shape)
        source = "input_mask"
    else:
        annotation_path = _resolve_path(str(cfg.get("input_annotation", "")))
        if annotation_path is not None and annotation_path.exists():
            annotation = cv2.imread(str(annotation_path), cv2.IMREAD_COLOR)
            if annotation is None:
                raise RuntimeError(f"Could not load cave zone annotation: {annotation_path}")
            if annotation.shape[:2] != expected_shape:
                raise ValueError(
                    "cave_zones.input_annotation shape does not match the processing frame size: "
                    f"expected {expected_shape}, got {annotation.shape[:2]}"
                )
            source_mask = annotation_to_mask(annotation)
            source = "input_annotation"

    candidate_rows: list[dict] = []
    candidate_masks: list[tuple[str, np.ndarray]] = []
    method = str(cfg.get("method", "hybrid")).strip().lower()
    if source_mask is not None and np.any(source_mask):
        candidate_masks.append((source, source_mask))
    elif method != "annotation":
        motion_mask = _motion_candidate_mask(motion_heatmap, cfg)
        if method in {"hybrid", "motion"} and motion_mask is not None and np.any(motion_mask):
            candidate_masks.append(("motion_heatmap", motion_mask))
        dark_mask = _dark_candidate_mask(background_gray, cfg)
        if method in {"hybrid", "dark"} and dark_mask is not None and np.any(dark_mask):
            if motion_mask is not None and np.any(motion_mask):
                dilate_px = max(0, int(cfg.get("motion_dark_connect_dilate_px", 18)))
                gate = motion_mask
                if dilate_px > 0:
                    k = 2 * dilate_px + 1
                    gate = cv2.dilate(gate, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=1)
                dark_mask = np.where((dark_mask > 0) & (gate > 0), 255, 0).astype(np.uint8)
            if np.any(dark_mask):
                candidate_masks.append(("dark_regions", dark_mask))

    merged_candidate_mask = np.zeros(expected_shape, dtype=np.uint8)
    for source_name, mask in candidate_masks:
        rows = _component_rows(
            mask,
            source=source_name,
            background_gray=background_gray,
            motion_heatmap=motion_heatmap,
            cfg=cfg,
        )
        candidate_rows.extend(rows)
        merged_candidate_mask = cv2.bitwise_or(merged_candidate_mask, mask)

    max_components = max(1, int(cfg.get("max_components", 3)))
    selected_rows = sorted(candidate_rows, key=lambda row: float(row["score"]), reverse=True)[:max_components]
    final_mask = _rows_to_mask(selected_rows, candidate_masks, expected_shape) if selected_rows else np.zeros(expected_shape, dtype=np.uint8)
    dilate_px = max(0, int(cfg.get("dilate_px", 8)))
    if dilate_px > 0 and np.any(final_mask):
        k = 2 * dilate_px + 1
        final_mask = cv2.dilate(final_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=1)

    # Re-read components after dilation so the public contract matches the final mask.
    zones = _component_rows(
        final_mask,
        source="selected",
        background_gray=background_gray,
        motion_heatmap=motion_heatmap,
        cfg={**cfg, "min_component_area_ratio": 0.0},
    )
    zones = sorted(zones, key=lambda row: float(row["score"]), reverse=True)
    for idx, zone in enumerate(zones, start=1):
        zone["zone_id"] = idx
        if idx - 1 < len(selected_rows):
            zone["origin"] = selected_rows[idx - 1].get("source", "")
            zone["candidate_score"] = selected_rows[idx - 1].get("score", 0.0)

    cv2.imwrite(outputs["cave_zones_mask_png"], final_mask)
    _save_overlay(background_gray, final_mask, zones, Path(outputs["cave_zones_overlay_png"]))
    _save_overlay(background_gray, merged_candidate_mask, candidate_rows, Path(outputs["cave_zones_candidates_overlay_png"]))

    zones_payload = {"zones": zones}
    with Path(outputs["cave_zones_zones_json"]).open("w", encoding="utf-8") as handle:
        json.dump(zones_payload, handle, indent=2)

    meta.update(
        {
            "source": source or "hybrid",
            "zones_total": len(zones),
            "mask_nonzero_px": int(np.count_nonzero(final_mask)),
        }
    )
    diagnostics = {
        "meta": meta,
        "candidate_components": sorted(candidate_rows, key=lambda row: float(row["score"]), reverse=True),
        "selected_components": selected_rows,
    }
    with Path(outputs["cave_zones_diagnostics_json"]).open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)

    return CaveZonesResult(mask=final_mask if np.any(final_mask) else None, zones=zones, meta=meta, outputs=outputs)

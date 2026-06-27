from __future__ import annotations

import csv
import heapq
import json
import sys
from collections import Counter, defaultdict, deque
from dataclasses import asdict
from math import ceil
from math import hypot
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Dict, List

import cv2
import numpy as np

from .background import compute_background_median
from .cave_zones import annotation_to_mask, run_cave_zones
from .compute import build_execution_plan
from .config import load_config
from .detection import build_detection_context
from .detection import detect_foreground_blobs
from .detection_fusion import build_secondary_detection_config
from .detection_fusion import fuse_detections
from .fast_events import reconstruct_fast_events
from .fast_events import reconstruct_fast_events_from_candidates
from .fast_events import render_fast_events_overlay
from .fast_events import write_fast_events_csv
from .fast_events import write_fast_tracks_csv
from .heatmap_events import reconstruct_heatmap_events
from .heatmap_events import render_heatmap_events_overlay
from .heatmap_events import write_heatmap_events_csv
from .heatmap_events import write_heatmap_tracks_csv
from .kalman_tracker import KalmanTracker
from .kinetic_secondary import dedupe_secondary_track_points
from .kinetic_secondary import run_kinetic_secondary_tracks
from .kinetic_secondary import suppress_temporal_burst_track_points
from .perf import PerformanceCollector
from .render import export_tracks_render_json, export_tracks_svg, render_detections_overlay, render_tracks_overlay
from .track_quality import compute_track_quality
from .track_deduplication import deduplicate_track_points
from .track_deduplication import render_track_deduplication_overlay
from .track_deduplication import write_track_deduplication_csv
from .track_deduplication import write_track_deduplication_json
from .track_smoothing import smooth_track_points
from .tracker import GreedyTracker, TrackPoint
from .trails import export_realtime_trails_video
from .valid_region import load_image as load_valid_region_image
from .valid_region import load_mask as load_valid_region_mask
from .valid_region import run_valid_region
from .valid_region import save_precomputed_mask_outputs
from .video import iter_gray_frames, read_video_meta

# Two OpenCV worker threads gave the best real pipeline wall time on the
# validated CPU benchmark videos. Higher values improved some isolated kernels
# but regressed end-to-end throughput due to contention and oversubscription.
OPENCV_CPU_THREADS = 2
cv2.setNumThreads(OPENCV_CPU_THREADS)


def _background_context_bounds(meta, background_cfg: Dict) -> tuple[int, int | None]:
    start_sec = float(background_cfg.get("context_start_sec", 0.0))
    duration_sec = float(background_cfg.get("context_duration_sec", -1.0))
    fps = max(1e-6, float(meta.fps))

    start_frame = max(0, int(round(start_sec * fps)))
    if duration_sec < 0.0:
        return start_frame, None

    duration_frames = max(1, int(round(duration_sec * fps)))
    end_frame = start_frame + duration_frames - 1
    return start_frame, end_frame


def _valid_region_context_bounds(meta, valid_region_cfg: Dict, background_cfg: Dict) -> tuple[int, int | None]:
    start_sec = float(valid_region_cfg.get("context_start_sec", -1.0))
    duration_sec = float(valid_region_cfg.get("context_duration_sec", -1.0))
    if start_sec < 0.0 and duration_sec < 0.0:
        return _background_context_bounds(meta, background_cfg)

    if start_sec < 0.0:
        start_sec = float(background_cfg.get("context_start_sec", 0.0))
    if duration_sec < 0.0:
        duration_sec = float(background_cfg.get("context_duration_sec", -1.0))

    fps = max(1e-6, float(meta.fps))
    start_frame = max(0, int(round(start_sec * fps)))
    if duration_sec < 0.0:
        return start_frame, None

    duration_frames = max(1, int(round(duration_sec * fps)))
    end_frame = start_frame + duration_frames - 1
    return start_frame, end_frame


def _compute_auto_vegetation_mask(
    video_path: str,
    meta,
    *,
    sample_frames: int = 220,
    max_frame_for_sampling: int = 1250,
    percentile: float = 85.0,
    min_component_area: int = 24,
) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for vegetation mask estimation: {video_path}")
    try:
        n = max(1, int(meta.frame_count))
        stop = min(n - 1, max_frame_for_sampling)
        idxs = np.linspace(0, stop, max(20, sample_frames), dtype=int)
        frames: List[np.ndarray] = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if len(frames) < 20:
            return np.zeros((meta.height, meta.width), dtype=np.uint8)

        stack = np.stack(frames, axis=0).astype(np.float32)
        activity = np.mean(np.abs(np.diff(stack, axis=0)), axis=0)
        tstd = np.std(stack, axis=0)
        median = np.median(stack, axis=0).astype(np.uint8)
        gx = cv2.Sobel(median, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(median, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)

        an = (activity - activity.min()) / (float(np.ptp(activity)) + 1e-6)
        sn = (tstd - tstd.min()) / (float(np.ptp(tstd)) + 1e-6)
        gn = (grad - grad.min()) / (float(np.ptp(grad)) + 1e-6)
        score = 0.50 * an + 0.35 * sn + 0.15 * gn
        thr = float(np.percentile(score, float(np.clip(percentile, 50.0, 99.5))))
        mask = np.where(score >= thr, 255, 0).astype(np.uint8)

        k3 = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)
        mask = cv2.medianBlur(mask, 3)
        num, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
        clean = np.zeros_like(mask)
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area >= max(1, min_component_area):
                clean[labels == i] = 255
        mask = cv2.erode(clean, k3, iterations=1)
        return mask
    finally:
        cap.release()


def _save_vegetation_mask_overlay(
    background: np.ndarray,
    vegetation_mask: np.ndarray,
    output_path: Path,
) -> None:
    if background.ndim == 2:
        overlay = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    else:
        overlay = background.copy()

    tint = np.zeros_like(overlay)
    tint[vegetation_mask > 0] = (0, 255, 120)
    overlay = cv2.addWeighted(overlay, 0.82, tint, 0.42, 0)

    contours, _ = cv2.findContours(
        (vegetation_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(
        overlay,
        "vegetation mask (auto)",
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output_path), overlay)


def _exclude_mask_from_vegetation(
    background_gray: np.ndarray | None,
    vegetation_mask: np.ndarray | None,
    exclude_mask: np.ndarray | None,
    *,
    dilate_px: int = 0,
    mode: str = "weak_evidence",
    keep_texture_percentile: float = 88.0,
    keep_min_intensity_percentile: float = 35.0,
    keep_min_gradient: float = 4.0,
) -> tuple[np.ndarray | None, dict]:
    if vegetation_mask is None or exclude_mask is None:
        return vegetation_mask, {
            "vegetation_exclusion_enabled": False,
            "vegetation_exclusion_mode": mode,
            "vegetation_pixels_before_exclusion": int(np.count_nonzero(vegetation_mask)) if vegetation_mask is not None else 0,
            "vegetation_pixels_after_exclusion": int(np.count_nonzero(vegetation_mask)) if vegetation_mask is not None else 0,
            "vegetation_pixels_removed_by_exclusion": 0,
            "vegetation_pixels_kept_in_entry_exit_zone": 0,
        }
    if vegetation_mask.shape[:2] != exclude_mask.shape[:2]:
        raise ValueError(
            "vegetation exclusion mask shape does not match vegetation mask shape: "
            f"expected {vegetation_mask.shape[:2]}, got {exclude_mask.shape[:2]}"
        )

    exclusion = np.where(exclude_mask > 0, 255, 0).astype(np.uint8)
    if dilate_px > 0:
        k = 2 * int(dilate_px) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        exclusion = cv2.dilate(exclusion, kernel, iterations=1)

    before = int(np.count_nonzero(vegetation_mask))
    cleaned = vegetation_mask.copy()
    overlap = (exclusion > 0) & (vegetation_mask > 0)
    keep_inside = np.zeros_like(overlap, dtype=bool)
    mode = str(mode or "weak_evidence").strip().lower()
    if mode == "weak_evidence" and background_gray is not None:
        if background_gray.shape[:2] != vegetation_mask.shape[:2]:
            raise ValueError(
                "vegetation background shape does not match vegetation mask shape: "
                f"expected {vegetation_mask.shape[:2]}, got {background_gray.shape[:2]}"
            )
        gx = cv2.Sobel(background_gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(background_gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
        texture_threshold = max(
            float(keep_min_gradient),
            float(np.percentile(grad, float(np.clip(keep_texture_percentile, 50.0, 99.5)))),
        )
        intensity_threshold = float(
            np.percentile(background_gray, float(np.clip(keep_min_intensity_percentile, 1.0, 95.0)))
        )
        # Keep only structural, non-cave-shadow evidence inside the entrance zone.
        keep_inside = (grad >= texture_threshold) & (background_gray >= intensity_threshold)
    elif mode == "none":
        keep_inside = overlap

    removed = overlap & ~keep_inside
    cleaned[removed] = 0
    after = int(np.count_nonzero(cleaned))
    return cleaned, {
        "vegetation_exclusion_enabled": True,
        "vegetation_exclusion_mode": mode,
        "vegetation_exclusion_dilate_px": int(max(0, dilate_px)),
        "vegetation_pixels_before_exclusion": before,
        "vegetation_pixels_after_exclusion": after,
        "vegetation_pixels_removed_by_exclusion": before - after,
        "vegetation_pixels_kept_in_entry_exit_zone": int(np.count_nonzero(overlap & keep_inside)),
    }


def _save_binary_mask_overlay(
    background: np.ndarray,
    mask: np.ndarray,
    output_path: Path,
    *,
    title: str,
    color: tuple[int, int, int] = (0, 210, 255),
) -> None:
    if background.ndim == 2:
        overlay = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    else:
        overlay = background.copy()
    tint = np.zeros_like(overlay)
    tint[mask > 0] = color
    overlay = cv2.addWeighted(overlay, 0.80, tint, 0.38, 0)
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(output_path), overlay)


def _load_cavemark_mask(
    *,
    background: np.ndarray,
    cfg: Dict,
    output_dir: Path,
) -> tuple[np.ndarray | None, Dict, Dict[str, str]]:
    outputs = {
        "cavemark_mask_png": "",
        "cavemark_overlay_png": "",
    }
    meta: Dict = {
        "enabled": bool(cfg.get("enabled", False)),
        "source": "",
        "mask_nonzero_px": 0,
        "outputs": outputs,
    }
    if not bool(cfg.get("enabled", False)):
        return None, meta, outputs

    expected_shape = background.shape[:2]
    mask: np.ndarray | None = None
    input_mask = str(cfg.get("input_mask", "")).strip()
    input_annotation = str(cfg.get("input_annotation", "")).strip()
    if input_mask:
        mask = load_valid_region_mask(input_mask)
        meta["source"] = "input_mask"
        meta["input_mask"] = str(Path(input_mask).resolve())
    elif input_annotation:
        annotation = cv2.imread(str(Path(input_annotation).expanduser()), cv2.IMREAD_COLOR)
        if annotation is None:
            raise RuntimeError(f"Could not load cavemark annotation: {input_annotation}")
        mask = annotation_to_mask(annotation)
        meta["source"] = "input_annotation"
        meta["input_annotation"] = str(Path(input_annotation).resolve())

    if mask is None:
        return None, meta, outputs
    if mask.shape[:2] != expected_shape:
        raise ValueError(
            "cavemark mask shape does not match the processing frame size: "
            f"expected {expected_shape}, got {mask.shape[:2]}"
        )
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    dilate_px = max(0, int(cfg.get("dilate_px", 0)))
    if dilate_px > 0 and np.any(mask):
        k = 2 * dilate_px + 1
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = output_dir / "mask.png"
    overlay_path = output_dir / "overlay.png"
    cv2.imwrite(str(mask_path), mask)
    _save_binary_mask_overlay(background, mask, overlay_path, title="cavemark entry/exit")
    outputs.update(
        {
            "cavemark_mask_png": str(mask_path.resolve()),
            "cavemark_overlay_png": str(overlay_path.resolve()),
        }
    )
    meta.update({"mask_nonzero_px": int(np.count_nonzero(mask)), "outputs": outputs})
    return (mask if np.any(mask) else None), meta, outputs


def _points_by_track(points: List[TrackPoint]) -> Dict[int, List[TrackPoint]]:
    grouped: Dict[int, List[TrackPoint]] = defaultdict(list)
    for point in points:
        grouped[int(point.track_id)].append(point)
    for track_points in grouped.values():
        track_points.sort(key=lambda point: point.frame)
    return grouped


def _point_in_binary_mask(point: TrackPoint, mask: np.ndarray) -> bool:
    x = int(round(point.x))
    y = int(round(point.y))
    return 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] > 0


def _score_entry_exit_candidate(
    *,
    source: str,
    mask: np.ndarray,
    background: np.ndarray,
    motion_heatmap: np.ndarray | None,
    vegetation_mask: np.ndarray | None,
    raw_points: List[TrackPoint],
    cfg: Dict,
) -> Dict:
    binary = mask > 0
    area = int(np.count_nonzero(binary))
    frame_area = max(1, mask.shape[0] * mask.shape[1])
    area_ratio = area / float(frame_area)
    if area == 0:
        return {
            "source": source,
            "score": -999.0,
            "area_ratio": 0.0,
            "motion_support": 0.0,
            "dark_ratio": 0.0,
            "endpoint_support": 0.0,
            "vegetation_overlap_ratio": 0.0,
        }

    vegetation_overlap_ratio = 0.0
    if vegetation_mask is not None:
        vegetation_overlap_ratio = float(np.count_nonzero(binary & (vegetation_mask > 0))) / float(area)

    dark_threshold = float(np.percentile(background, float(np.clip(cfg.get("dark_percentile", 18.0), 1.0, 60.0))))
    dark_ratio = float(np.mean(background[binary] <= dark_threshold))

    motion_support = 0.0
    if motion_heatmap is not None:
        positive = motion_heatmap[motion_heatmap > 1e-6]
        motion_scale = float(np.percentile(positive, 95.0)) if positive.size else 1.0
        motion_support = float(np.mean(np.clip(motion_heatmap[binary] / max(motion_scale, 1e-6), 0.0, 1.0)))

    tracks = _points_by_track(raw_points)
    endpoint_hits = 0
    crossing_hits = 0
    for track_points in tracks.values():
        if not track_points:
            continue
        start_inside = _point_in_binary_mask(track_points[0], mask)
        end_inside = _point_in_binary_mask(track_points[-1], mask)
        if start_inside or end_inside:
            endpoint_hits += 1
        elif any(_point_in_binary_mask(point, mask) for point in track_points[1:-1]):
            crossing_hits += 1
    tracks_total = max(1, len(tracks))
    endpoint_support = min(1.0, (endpoint_hits + 0.5 * crossing_hits) / float(tracks_total))

    ideal_area_ratio = max(1e-6, float(cfg.get("ideal_area_ratio", 0.04)))
    max_reasonable_area_ratio = max(ideal_area_ratio, float(cfg.get("max_reasonable_area_ratio", 0.18)))
    area_score = 1.0 - min(1.0, abs(area_ratio - ideal_area_ratio) / max_reasonable_area_ratio)

    source_bias = float(cfg.get(f"{source}_bias", 0.0))
    score = (
        float(cfg.get("motion_weight", 0.25)) * motion_support
        + float(cfg.get("dark_weight", 0.25)) * dark_ratio
        + float(cfg.get("endpoint_weight", 0.30)) * endpoint_support
        + float(cfg.get("area_weight", 0.20)) * area_score
        + source_bias
        - float(cfg.get("vegetation_overlap_penalty", 0.45)) * vegetation_overlap_ratio
    )
    return {
        "source": source,
        "score": round(float(score), 6),
        "area_ratio": round(float(area_ratio), 6),
        "motion_support": round(float(motion_support), 6),
        "dark_ratio": round(float(dark_ratio), 6),
        "endpoint_support": round(float(endpoint_support), 6),
        "endpoint_tracks": int(endpoint_hits),
        "crossing_tracks": int(crossing_hits),
        "tracks_total": int(len(tracks)),
        "vegetation_overlap_ratio": round(float(vegetation_overlap_ratio), 6),
    }


def _select_entry_exit_mask(
    *,
    source_cfg: str,
    candidates: Dict[str, np.ndarray | None],
    background: np.ndarray,
    motion_heatmap: np.ndarray | None,
    vegetation_mask: np.ndarray | None,
    raw_points: List[TrackPoint],
    selection_cfg: Dict,
) -> tuple[np.ndarray | None, str, Dict]:
    source_cfg = str(source_cfg or "auto").strip().lower()
    if source_cfg != "auto":
        selected = candidates.get(source_cfg)
        if selected is None and source_cfg != "valid_region":
            selected = candidates.get("valid_region")
            selected_source = "valid_region" if selected is not None else "none"
        else:
            selected_source = source_cfg if selected is not None else "none"
        return selected, selected_source, {
            "mode": source_cfg,
            "selected_source": selected_source,
            "scores": [],
            "reason": "explicit_source",
        }

    scores = [
        _score_entry_exit_candidate(
            source=source,
            mask=mask,
            background=background,
            motion_heatmap=motion_heatmap,
            vegetation_mask=vegetation_mask,
            raw_points=raw_points,
            cfg=selection_cfg,
        )
        for source, mask in candidates.items()
        if mask is not None
    ]
    scores.sort(key=lambda row: float(row["score"]), reverse=True)
    if not scores:
        return None, "none", {"mode": "auto", "selected_source": "none", "scores": [], "reason": "no_candidates"}

    selected_source = str(scores[0]["source"])
    selected = candidates.get(selected_source)
    reason = "highest_score"
    if float(scores[0].get("vegetation_overlap_ratio", 0.0)) >= 0.5:
        reason = "highest_score_despite_high_vegetation_overlap"
    return selected, selected_source, {
        "mode": "auto",
        "selected_source": selected_source,
        "scores": scores,
        "reason": reason,
    }


def _accumulate_motion_heatmap(
    heatmap: np.ndarray,
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    *,
    blur_kernel: int,
    threshold: int,
) -> None:
    if blur_kernel < 1 or blur_kernel % 2 == 0:
        blur_kernel = 5
    previous_blur = cv2.GaussianBlur(previous_gray, (blur_kernel, blur_kernel), 0)
    current_blur = cv2.GaussianBlur(current_gray, (blur_kernel, blur_kernel), 0)
    diff = cv2.absdiff(current_blur, previous_blur)
    _, binary = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    heatmap += diff.astype(np.float32) * (binary.astype(np.float32) / 255.0)


def _save_motion_heatmap_overlay(
    background: np.ndarray,
    heatmap: np.ndarray,
    output_path: Path,
) -> None:
    if background.ndim == 2:
        base = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    else:
        base = background.copy()

    positive = heatmap[heatmap > 1e-6]
    if positive.size == 0:
        cv2.imwrite(str(output_path), base)
        return

    scale = float(np.percentile(positive, 97.5))
    scale = max(scale, 1e-6)
    normalized = np.clip(heatmap / scale, 0.0, 1.0)
    normalized = np.power(normalized, 0.75, dtype=np.float32)
    heatmap_u8 = np.uint8(np.round(normalized * 255.0))
    colored = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_TURBO)
    alpha = np.clip(normalized * 0.88, 0.0, 0.88).astype(np.float32)[..., None]
    out = base.astype(np.float32)
    out *= 1.0 - alpha
    out += colored.astype(np.float32) * alpha
    overlay = np.clip(out, 0.0, 255.0).astype(np.uint8)
    cv2.imwrite(str(output_path), overlay)


_DEBUG_OUTPUT_KEYS = {
    "primary_detections_overlay_png",
    "secondary_detections_overlay_png",
    "track_candidates_csv",
    "track_deduplication_csv",
    "track_deduplication_json",
    "track_deduplication_overlay_png",
    "secondary_kinetic_tracks_csv",
    "secondary_kinetic_tracks_overlay_png",
    "secondary_kinetic_added_tracks_csv",
    "secondary_kinetic_added_tracks_overlay_png",
    "tracks_overlay_raw_png",
    "tracks_overlay_smoothed_png",
    "vegetation_mask_png",
    "vegetation_mask_overlay_png",
    "valid_region_overlay_png",
    "valid_region_profile_png",
    "valid_region_gate_overlay_png",
}


def _cleanup_output_files(outputs: Dict[str, str], *, enabled: bool) -> list[str]:
    if not enabled:
        return []

    removed_keys: list[str] = []
    for key in _DEBUG_OUTPUT_KEYS:
        path_str = str(outputs.get(key, "")).strip()
        if not path_str:
            continue
        try:
            Path(path_str).unlink(missing_ok=True)
        except IsADirectoryError:
            continue
        removed_keys.append(key)
        outputs[key] = ""
    return sorted(removed_keys)


CSV_COLUMNS = [
    "video_id",
    "track_id",
    "frame",
    "time_sec",
    "x",
    "y",
    "vx",
    "vy",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "area",
]

EVENTS_CSV_COLUMNS = [
    "video_id",
    "track_id",
    "time_start_sec",
    "time_end_sec",
    "duration_sec",
    "frame_start",
    "frame_end",
    "num_detections",
    "x_start",
    "y_start",
    "x_end",
    "y_end",
    "displacement_px",
    "path_length_px",
    "straightness",
    "mean_speed_px_sec",
    "mean_area",
    "start_in_valid_region",
    "end_in_valid_region",
    "direction",
]

TRACK_CANDIDATES_CSV_COLUMNS = [
    "video_id",
    "track_id",
    "accepted",
    "reject_reasons",
    "score",
    "frame_start",
    "frame_end",
    "num_detections",
    "duration_sec",
    "x_start",
    "y_start",
    "x_end",
    "y_end",
    "displacement_px",
    "path_length_px",
    "straightness",
    "mean_speed_px_sec",
    "mean_area",
    "start_in_valid_region",
    "end_in_valid_region",
    "direction",
]


def _classify_direction(start_inside: bool, end_inside: bool) -> str:
    if start_inside and end_inside:
        return "inside"
    if start_inside and not end_inside:
        return "exit"
    if not start_inside and end_inside:
        return "entry"
    return "outside"


def _classify_direction_full(
    s_in: bool,
    e_in: bool,
    tps: List[TrackPoint],
    valid_mask: np.ndarray | None,
    frame_shape: tuple[int, int] | None = None,
) -> str:
    """Classify direction using the full trajectory.

    When start and end are both inside the valid region the initial
    classification is "inside", but we additionally check every
    intermediate point.  If any midpoint lies outside the valid region
    the track crossed the boundary during its flight and should be
    reclassified as "exit" rather than "inside".
    """
    direction = _classify_direction(s_in, e_in)
    if direction == "inside" and valid_mask is not None and len(tps) > 2:
        for tp in tps[1:-1]:
            if not _point_in_mask(tp, valid_mask):
                direction = "exit"
                break
    if direction == "outside" and valid_mask is not None and len(tps) > 2:
        inside_indices = [idx for idx, tp in enumerate(tps[1:-1], start=1) if _point_in_mask(tp, valid_mask)]
        if inside_indices:
            ys, xs = np.nonzero(valid_mask > 0)
            if xs.size > 0:
                cx = float(np.mean(xs))
                cy = float(np.mean(ys))
                start_vec = (tps[0].x - cx, tps[0].y - cy)
                end_vec = (tps[-1].x - cx, tps[-1].y - cy)
                if start_vec[0] * end_vec[0] + start_vec[1] * end_vec[1] > 0.0:
                    return "outside"
                start_dist = hypot(tps[0].x - cx, tps[0].y - cy)
                end_dist = hypot(tps[-1].x - cx, tps[-1].y - cy)
                if end_dist < start_dist * 0.92:
                    return "entry"
                if start_dist < end_dist * 0.92:
                    return "exit"
            return "inside"
    if direction == "outside" and frame_shape is not None:
        direction = _infer_outside_direction_from_motion(tps[0], tps[-1], frame_shape)
    return direction


def _infer_outside_direction_from_motion(
    start: TrackPoint,
    end: TrackPoint,
    frame_shape: tuple[int, int] | None,
) -> str:
    if frame_shape is None:
        return "outside"
    height, width = frame_shape
    dy = end.y - start.y
    min_vertical_move = max(40.0, 0.15 * float(height))
    top_band = 0.20 * float(height)
    if dy <= -min_vertical_move and end.y <= top_band:
        return "exit"
    if dy >= min_vertical_move and start.y <= top_band:
        return "entry"
    return "outside"


def _write_events_csv(
    path: Path,
    points: List[TrackPoint],
    valid_mask: np.ndarray | None,
) -> None:
    by_track: Dict[int, List[TrackPoint]] = defaultdict(list)
    for p in points:
        by_track[p.track_id].append(p)

    rows: list[dict] = []
    for track_id in sorted(by_track):
        tps = sorted(by_track[track_id], key=lambda p: p.frame)
        start = tps[0]
        end = tps[-1]

        displacement = hypot(end.x - start.x, end.y - start.y)
        pl = _path_length(tps)
        duration = end.time_sec - start.time_sec
        straightness = (displacement / pl) if pl > 0 else 0.0
        mean_speed = (pl / duration) if duration > 0 else 0.0
        avg_area = sum(p.area for p in tps) / len(tps)

        if valid_mask is not None:
            s_in = _point_in_mask(start, valid_mask)
            e_in = _point_in_mask(end, valid_mask)
            direction = _classify_direction_full(s_in, e_in, tps, valid_mask, valid_mask.shape[:2])
        else:
            s_in = None
            e_in = None
            direction = "unknown"

        rows.append({
            "video_id": start.video_id,
            "track_id": track_id,
            "time_start_sec": round(start.time_sec, 4),
            "time_end_sec": round(end.time_sec, 4),
            "duration_sec": round(duration, 4),
            "frame_start": start.frame,
            "frame_end": end.frame,
            "num_detections": len(tps),
            "x_start": round(start.x, 2),
            "y_start": round(start.y, 2),
            "x_end": round(end.x, 2),
            "y_end": round(end.y, 2),
            "displacement_px": round(displacement, 2),
            "path_length_px": round(pl, 2),
            "straightness": round(straightness, 4),
            "mean_speed_px_sec": round(mean_speed, 2),
            "mean_area": round(avg_area, 2),
            "start_in_valid_region": s_in if s_in is not None else "",
            "end_in_valid_region": e_in if e_in is not None else "",
            "direction": direction,
        })

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENTS_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_tracks_csv(path: Path, points: List[TrackPoint]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for point in sorted(points, key=lambda p: (p.track_id, p.frame)):
            row = asdict(point)
            writer.writerow(row)


def _write_track_candidates_csv(path: Path, rows: List[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACK_CANDIDATES_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_metrics(points: List[TrackPoint], frame_count: int) -> Dict:
    tracks_counter = Counter(p.track_id for p in points)
    tracks_lengths = list(tracks_counter.values())
    total_tracks = len(tracks_counter)

    return {
        "frames_processed": frame_count,
        "detections_kept": len(points),
        "tracks_total": total_tracks,
        "track_length_min": min(tracks_lengths) if tracks_lengths else 0,
        "track_length_max": max(tracks_lengths) if tracks_lengths else 0,
        "track_length_mean": mean(tracks_lengths) if tracks_lengths else 0.0,
    }


class TemporalBurstGate:
    def __init__(
        self,
        min_detections: int,
        window_frames: int,
        trigger_frames: int,
        cooldown_frames: int,
    ):
        self.min_detections = int(min_detections)
        self.window_frames = int(window_frames)
        self.trigger_frames = int(trigger_frames)
        self.cooldown_frames = int(cooldown_frames)
        self._recent_hits = deque()
        self._cooldown_until = -1

    @classmethod
    def from_detection_cfg(cls, detection_cfg: Dict) -> "TemporalBurstGate | None":
        min_detections = int(detection_cfg.get("temporal_burst_min_detections", 0))
        window_frames = int(detection_cfg.get("temporal_burst_window_frames", 0))
        trigger_frames = int(detection_cfg.get("temporal_burst_trigger_frames", 0))
        cooldown_frames = int(detection_cfg.get("temporal_burst_cooldown_frames", 0))

        if min_detections <= 0 or window_frames <= 0 or trigger_frames <= 0 or cooldown_frames <= 0:
            return None
        return cls(min_detections, window_frames, trigger_frames, cooldown_frames)

    def should_keep(self, frame_idx: int, det_count: int) -> bool:
        hit = det_count >= self.min_detections
        self._recent_hits.append((frame_idx, hit))

        oldest_valid = frame_idx - self.window_frames + 1
        while self._recent_hits and self._recent_hits[0][0] < oldest_valid:
            self._recent_hits.popleft()

        if frame_idx <= self._cooldown_until:
            return False

        hits_in_window = sum(1 for _, is_hit in self._recent_hits if is_hit)
        if hits_in_window >= self.trigger_frames:
            self._cooldown_until = frame_idx + self.cooldown_frames - 1
            return False

        return True


class ProgressReporter:
    def __init__(self, enabled: bool, step_percent: int, stages: list[tuple[str, float]]):
        self.enabled = bool(enabled) and bool(stages)
        self.step_percent = max(1, min(100, int(step_percent)))
        self._next_threshold = self.step_percent
        self._last_reported = 0
        self._current_stage = ""

        total_weight = sum(max(0.0, float(weight)) for _, weight in stages)
        if total_weight <= 0:
            self.enabled = False
            self._stage_base = {}
            self._stage_weight = {}
            return

        self._stage_base: Dict[str, float] = {}
        self._stage_weight: Dict[str, float] = {}
        base = 0.0
        for stage_name, weight in stages:
            w = max(0.0, float(weight)) / total_weight
            self._stage_base[stage_name] = base
            self._stage_weight[stage_name] = w
            base += w

    def start_stage(self, stage_name: str) -> None:
        if not self.enabled:
            return
        if stage_name not in self._stage_base:
            return
        self._current_stage = stage_name
        self.update_stage_fraction(stage_name, 0.0)

    def complete_stage(self, stage_name: str, detail: str | None = None) -> None:
        self.update_stage_fraction(stage_name, 1.0, detail=detail)

    def update_stage_fraction(self, stage_name: str, fraction: float, detail: str | None = None) -> None:
        if not self.enabled:
            return
        if stage_name not in self._stage_base:
            return
        frac = max(0.0, min(1.0, float(fraction)))
        overall = self._stage_base[stage_name] + self._stage_weight[stage_name] * frac
        pct = min(100, int(round(overall * 100.0)))
        self._emit(pct, detail=detail)

    def _emit(self, pct: int, detail: str | None = None) -> None:
        if pct < self._next_threshold:
            return
        self._last_reported = pct
        self._next_threshold = ((pct // self.step_percent) + 1) * self.step_percent
        suffix = f" - {detail}" if detail else (f" - {self._current_stage}" if self._current_stage else "")
        print(f"[progress] {pct}%{suffix}", file=sys.stderr, flush=True)

    def finish(self) -> None:
        if not self.enabled:
            return
        if self._last_reported >= 100:
            return
        print("[progress] 100% - done", file=sys.stderr, flush=True)


def _path_length(track_points: List[TrackPoint]) -> float:
    if len(track_points) < 2:
        return 0.0
    return sum(hypot(p1.x - p0.x, p1.y - p0.y) for p0, p1 in zip(track_points[:-1], track_points[1:]))


def _point_in_mask(point: TrackPoint, mask: np.ndarray) -> bool:
    xi = int(round(point.x))
    yi = int(round(point.y))
    if yi < 0 or yi >= mask.shape[0] or xi < 0 or xi >= mask.shape[1]:
        return False
    return bool(mask[yi, xi] > 0)


def _build_valid_region_gate_mask(
    valid_mask: np.ndarray | None,
    tracking_cfg: Dict,
    frame_size: tuple[int, int] | None = None,
) -> np.ndarray | None:
    gate_mask = valid_mask
    if frame_size is None and gate_mask is not None:
        frame_size = (int(gate_mask.shape[1]), int(gate_mask.shape[0]))
    valid_region_gate_dilate_px = max(
        0,
        int(
            round(
                _scale_linear_px_for_resolution(
                    float(tracking_cfg.get("valid_region_gate_dilate_px", 0)),
                    tracking_cfg,
                    frame_size,
                    reference_width_key="reference_width",
                    reference_height_key="reference_height",
                )
            )
        ),
    )
    if gate_mask is not None and valid_region_gate_dilate_px > 0:
        k = 2 * valid_region_gate_dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        gate_mask = cv2.dilate(gate_mask, kernel, iterations=1)
    return gate_mask


def _scale_vegetation_params_for_resolution(vegetation_cfg: Dict, width: int, height: int) -> tuple[int, int]:
    """Scale px-based vegetation parameters to preserve behavior across resolutions."""
    dilate_px = max(0, int(vegetation_cfg.get("mask_dilate_px", 0)))
    min_component_area = max(1, int(vegetation_cfg.get("auto_min_component_area", 24)))

    if not bool(vegetation_cfg.get("auto_scale_with_resolution", True)):
        return dilate_px, min_component_area

    ref_w = max(1, int(vegetation_cfg.get("reference_width", 1024)))
    ref_h = max(1, int(vegetation_cfg.get("reference_height", 576)))
    target_w = max(1, int(width))
    target_h = max(1, int(height))

    # Linear scale for radius-like params; area scale for connected-component area.
    ref_diag = max(1.0, hypot(float(ref_w), float(ref_h)))
    target_diag = max(1.0, hypot(float(target_w), float(target_h)))
    linear_scale = target_diag / ref_diag
    area_scale = (float(target_w) * float(target_h)) / (float(ref_w) * float(ref_h))

    scaled_dilate = int(round(float(dilate_px) * linear_scale))
    if dilate_px > 0 and scaled_dilate < 1:
        scaled_dilate = 1

    scaled_min_area = int(round(float(min_component_area) * area_scale))
    if min_component_area > 0 and scaled_min_area < 1:
        scaled_min_area = 1

    return max(0, scaled_dilate), max(1, scaled_min_area)


def _filter_points_start_or_end_in_mask(
    points: List[TrackPoint],
    mask: np.ndarray | None,
) -> tuple[List[TrackPoint], dict]:
    if mask is None:
        return points, {
            "mask_filter_enabled": False,
            "tracks_before_mask_filter": len({point.track_id for point in points}),
            "tracks_after_mask_filter": len({point.track_id for point in points}),
            "tracks_rejected_by_mask_filter": 0,
        }

    by_track: Dict[int, List[TrackPoint]] = defaultdict(list)
    for point in points:
        by_track[point.track_id].append(point)

    kept: List[TrackPoint] = []
    rejected = 0
    for track_points in by_track.values():
        track_points = sorted(track_points, key=lambda point: point.frame)
        if not track_points:
            continue
        if _point_in_mask(track_points[0], mask) or _point_in_mask(track_points[-1], mask):
            kept.extend(track_points)
        else:
            rejected += 1

    return kept, {
        "mask_filter_enabled": True,
        "tracks_before_mask_filter": len(by_track),
        "tracks_after_mask_filter": len({point.track_id for point in kept}),
        "tracks_rejected_by_mask_filter": rejected,
    }


def _filter_points_excluding_directions(
    points: List[TrackPoint],
    mask: np.ndarray | None,
    excluded_directions: set[str] | None = None,
) -> tuple[List[TrackPoint], dict]:
    """Keep final exported tracks aligned with valid-region events."""
    excluded = excluded_directions or {"outside"}
    if mask is None or not excluded:
        track_count = len({point.track_id for point in points})
        return points, {
            "direction_filter_enabled": False,
            "tracks_before_direction_filter": track_count,
            "tracks_after_direction_filter": track_count,
            "tracks_rejected_by_direction_filter": 0,
        }

    by_track: Dict[int, List[TrackPoint]] = defaultdict(list)
    for point in points:
        by_track[int(point.track_id)].append(point)

    kept: List[TrackPoint] = []
    rejected = 0
    rejected_by_direction: Counter = Counter()
    for track_points in by_track.values():
        track_points = sorted(track_points, key=lambda point: point.frame)
        if not track_points:
            continue
        direction = _classify_direction_full(
            _point_in_mask(track_points[0], mask),
            _point_in_mask(track_points[-1], mask),
            track_points,
            mask,
            mask.shape[:2],
        )
        if direction in excluded:
            rejected += 1
            rejected_by_direction[direction] += 1
            continue
        kept.extend(track_points)

    return sorted(kept, key=lambda point: (point.track_id, point.frame)), {
        "direction_filter_enabled": True,
        "tracks_before_direction_filter": len(by_track),
        "tracks_after_direction_filter": len({point.track_id for point in kept}),
        "tracks_rejected_by_direction_filter": rejected,
        "tracks_rejected_by_direction": dict(rejected_by_direction),
    }


def _save_valid_region_gate_overlay(
    background_gray: np.ndarray,
    gate_mask: np.ndarray,
    output_path: Path,
) -> None:
    overlay = cv2.cvtColor(background_gray, cv2.COLOR_GRAY2BGR)
    tint = np.zeros_like(overlay)
    tint[gate_mask > 0] = (255, 140, 0)
    overlay = cv2.addWeighted(overlay, 0.82, tint, 0.30, 0)
    contours, _ = cv2.findContours(gate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 165, 255), 2, lineType=cv2.LINE_AA)
    cv2.imwrite(str(output_path), overlay)


def _filter_track_points(
    points: List[TrackPoint],
    tracking_cfg: Dict,
    fps: float,
    valid_mask: np.ndarray | None = None,
    entry_exit_mask: np.ndarray | None = None,
    vegetation_mask: np.ndarray | None = None,
    vegetation_cfg: Dict | None = None,
) -> tuple[List[TrackPoint], List[dict]]:
    def _ratio(value: float, threshold: float) -> float:
        if threshold <= 0.0:
            return 1.0
        return min(1.0, max(0.0, value / threshold))

    min_track_length_cfg = int(tracking_cfg.get("min_track_length", 1))
    min_track_duration_sec = float(tracking_cfg.get("min_track_duration_sec", 0.0))
    min_track_length_from_sec = int(ceil(max(0.0, min_track_duration_sec) * max(1e-6, fps)))
    min_track_length = max(min_track_length_cfg, min_track_length_from_sec)
    min_track_displacement_cfg = float(tracking_cfg.get("min_track_displacement", 0.0))
    min_track_path_length_cfg = float(tracking_cfg.get("min_track_path_length", 0.0))
    min_track_straightness = float(tracking_cfg.get("min_track_straightness", 0.0))
    static_noise_filter_enabled = bool(tracking_cfg.get("static_noise_filter_enabled", False))
    static_noise_min_duration_sec = max(0.0, float(tracking_cfg.get("static_noise_min_duration_sec", 3.0)))
    static_noise_max_mean_speed_ratio = max(0.0, float(tracking_cfg.get("static_noise_max_mean_speed_ratio_per_sec", 0.0)))
    static_noise_max_displacement_ratio = max(0.0, float(tracking_cfg.get("static_noise_max_displacement_ratio_per_sec", 0.0)))
    max_track_internal_gap_frames = max(0, int(tracking_cfg.get("max_track_internal_gap_frames", 0)))
    loiter_filter_enabled = bool(tracking_cfg.get("loiter_filter_enabled", False))
    loiter_min_duration_sec = max(0.0, float(tracking_cfg.get("loiter_min_duration_sec", 10.0)))
    loiter_min_displacement_ratio = max(0.0, float(tracking_cfg.get("loiter_min_displacement_ratio", 0.0)))
    require_start_or_end_in_valid_region = bool(tracking_cfg.get("require_start_or_end_in_valid_region", False))
    # Compat v1.1.11: si require_start_or_end_in_valid_region=true, el track
    # debe tocar la máscara (inicio o fin) independientemente de valid_region_mode.
    # valid_region_mode="gate" mantiene además el comportamiento de descarte
    # explícito por gate cuando se evalúa región válida.
    valid_region_mode = str(tracking_cfg.get("valid_region_mode", "annotate")).strip().lower()
    gate_deletes = valid_region_mode == "gate"
    gate_mask = entry_exit_mask if entry_exit_mask is not None else _build_valid_region_gate_mask(valid_mask, tracking_cfg)
    direction_mask = gate_mask if gate_mask is not None else valid_mask
    strong_short_score_min = 0.9
    vegetation_cfg = vegetation_cfg or {}

    frame_source_mask = direction_mask if direction_mask is not None else valid_mask
    frame_h, frame_w = (frame_source_mask.shape[:2] if frame_source_mask is not None else (0, 0))
    if frame_h <= 0 or frame_w <= 0:
        max_x = max((point.x for point in points), default=1.0)
        max_y = max((point.y for point in points), default=1.0)
        frame_w = int(max(2.0, max_x + 1.0))
        frame_h = int(max(2.0, max_y + 1.0))
    frame_diag = max(1.0, hypot(float(frame_w), float(frame_h)))
    frame_size = (int(frame_w), int(frame_h))
    min_track_displacement = _scale_linear_px_for_resolution(
        min_track_displacement_cfg,
        tracking_cfg,
        frame_size,
        reference_width_key="reference_width",
        reference_height_key="reference_height",
    )
    min_track_path_length = _scale_linear_px_for_resolution(
        min_track_path_length_cfg,
        tracking_cfg,
        frame_size,
        reference_width_key="reference_width",
        reference_height_key="reference_height",
    )
    # Detector de "blob estàtic": descarta soroll fix de llarga durada (reflexos, punts
    # calents, vegetació quasi immòbil) que un tracker manté viu i que acumula
    # desplaçament per salts esporàdics. Tots els llindars escalen amb la diagonal del
    # frame (fracció per segon), de manera que s'adapten a la resolució del vídeo.
    # Només dispara quan coincideixen tres evidències independents (velocitat de
    # trajectòria baixa + avanç net baix + durada sostinguda), de manera que un
    # ratpenat real (ràpid, o que avança de debò, o de durada curta) no hi cau.
    static_noise_max_mean_speed = static_noise_max_mean_speed_ratio * frame_diag
    static_noise_max_displacement_rate = static_noise_max_displacement_ratio * frame_diag

    def _strip_vegetation_jitter(track_points: List[TrackPoint]) -> List[TrackPoint]:
        if vegetation_mask is None or not bool(vegetation_cfg.get("enabled", False)):
            return track_points
        if len(track_points) < 3:
            return track_points
        drop_all_points_in_mask = bool(vegetation_cfg.get("drop_all_points_in_mask", False))

        if drop_all_points_in_mask:
            kept = []
            for point in track_points:
                xi, yi = int(round(point.x)), int(round(point.y))
                inside = (
                    0 <= yi < vegetation_mask.shape[0]
                    and 0 <= xi < vegetation_mask.shape[1]
                    and vegetation_mask[yi, xi] > 0
                )
                if not inside:
                    kept.append(point)
            # Strict mode: never keep points inside vegetation mask.
            if len(kept) < 2:
                return kept
            return kept

        min_motion_ratio_per_sec = max(0.0, float(vegetation_cfg.get("min_motion_ratio_per_sec", 0.25)))
        min_consecutive_points = max(2, int(vegetation_cfg.get("min_consecutive_points", 3)))
        min_motion_px_per_sec = min_motion_ratio_per_sec * frame_diag

        noisy = [False] * len(track_points)
        for i, point in enumerate(track_points):
            xi, yi = int(round(point.x)), int(round(point.y))
            if yi < 0 or yi >= vegetation_mask.shape[0] or xi < 0 or xi >= vegetation_mask.shape[1]:
                continue
            if vegetation_mask[yi, xi] <= 0:
                continue
            p0 = track_points[max(0, i - 1)]
            p1 = track_points[min(len(track_points) - 1, i + 1)]
            dt = max(1e-6, p1.time_sec - p0.time_sec)
            speed = hypot(p1.x - p0.x, p1.y - p0.y) / dt
            if speed < min_motion_px_per_sec:
                noisy[i] = True

        kept = list(track_points)
        run_start = None
        for i, flag in enumerate(noisy + [False]):
            if flag and run_start is None:
                run_start = i
            elif not flag and run_start is not None:
                run_len = i - run_start
                if run_len >= min_consecutive_points:
                    for j in range(run_start, i):
                        kept[j] = None  # type: ignore[index]
                run_start = None

        kept = [p for p in kept if p is not None]
        if len(kept) < 2:
            return track_points
        return kept

    by_track: Dict[int, List[TrackPoint]] = defaultdict(list)
    for point in points:
        by_track[point.track_id].append(point)

    filtered: List[TrackPoint] = []
    assessments: List[dict] = []
    for track_points in by_track.values():
        track_points = sorted(track_points, key=lambda p: p.frame)
        original_points = track_points
        track_points = _strip_vegetation_jitter(track_points)
        if len(track_points) < 2:
            start = original_points[0]
            end = original_points[-1]
            duration = end.time_sec - start.time_sec
            displacement = hypot(end.x - start.x, end.y - start.y)
            path_length = _path_length(original_points)
            straightness = (displacement / path_length) if path_length > 0 else 0.0
            mean_speed = (path_length / duration) if duration > 0 else 0.0
            avg_area = sum(p.area for p in original_points) / len(original_points)
            assessments.append({
                "video_id": start.video_id,
                "track_id": start.track_id,
                "accepted": False,
                "reject_reasons": "vegetation_mask",
                "score": 0.0,
                "frame_start": start.frame,
                "frame_end": end.frame,
                "num_detections": len(original_points),
                "duration_sec": round(duration, 4),
                "x_start": round(start.x, 2),
                "y_start": round(start.y, 2),
                "x_end": round(end.x, 2),
                "y_end": round(end.y, 2),
                "displacement_px": round(displacement, 2),
                "path_length_px": round(path_length, 2),
                "straightness": round(straightness, 4),
                "mean_speed_px_sec": round(mean_speed, 2),
                "mean_area": round(avg_area, 2),
                "start_in_valid_region": "",
                "end_in_valid_region": "",
                "direction": "unknown",
            })
            continue
        start = track_points[0]
        end = track_points[-1]
        duration = end.time_sec - start.time_sec
        displacement = hypot(end.x - start.x, end.y - start.y)
        path_length = _path_length(track_points)
        straightness = (displacement / path_length) if path_length > 0 else 0.0
        mean_speed = (path_length / duration) if duration > 0 else 0.0
        avg_area = sum(p.area for p in track_points) / len(track_points)
        score = mean(
            [
                _ratio(float(len(track_points)), float(min_track_length)),
                _ratio(displacement, min_track_displacement),
                _ratio(path_length, min_track_path_length),
                _ratio(straightness, min_track_straightness) if min_track_straightness > 0.0 else 1.0,
            ]
        )
        reject_reasons: list[str] = []
        if len(track_points) < min_track_length:
            reject_reasons.append("min_track_length")
        if displacement < min_track_displacement:
            reject_reasons.append("min_track_displacement")
        if path_length < min_track_path_length:
            reject_reasons.append("min_track_path_length")
        if min_track_straightness > 0.0 and path_length > 0.0:
            if straightness < min_track_straightness:
                reject_reasons.append("min_track_straightness")
        if (
            static_noise_filter_enabled
            and duration >= static_noise_min_duration_sec
            and static_noise_max_mean_speed > 0.0
            and static_noise_max_displacement_rate > 0.0
        ):
            displacement_rate = displacement / duration if duration > 0.0 else 0.0
            if mean_speed < static_noise_max_mean_speed and displacement_rate < static_noise_max_displacement_rate:
                reject_reasons.append("static_noise")
        # Discontinuïtat temporal: un track real és temporalment dens (els forats
        # entre deteccions consecutives no superen max_missed ni la tolerància de
        # merge). Un forat intern molt gran delata que s'han cosit fragments no
        # relacionats (vol curt + punt fix d'una cantonada + ràfega de soroll) sota
        # un mateix id, cosa que infla path_length i burla els altres filtres.
        if max_track_internal_gap_frames > 0 and len(track_points) >= 2:
            max_internal_gap = max(
                track_points[i].frame - track_points[i - 1].frame
                for i in range(1, len(track_points))
            )
            if max_internal_gap > max_track_internal_gap_frames:
                reject_reasons.append("temporal_gap")
        # Merodeo: el vol d'un ratpenat sortint de la cova és un trànsit ràpid que
        # creua l'escena; no s'hi està molts segons. Un track de llarga durada que
        # NO transita (avanç net petit respecte a la mida del frame, encara que
        # acumuli molt recorregut donant voltes) no correspon a aquest comportament
        # i sol ser soroll persistent (insecte prop de l'òptica, reflex mòbil, etc.).
        # El llindar de desplaçament és una fracció de la diagonal, així que escala
        # amb la resolució del vídeo.
        if (
            loiter_filter_enabled
            and loiter_min_duration_sec > 0.0
            and loiter_min_displacement_ratio > 0.0
            and duration >= loiter_min_duration_sec
        ):
            displacement_ratio = displacement / frame_diag if frame_diag > 0.0 else 0.0
            if displacement_ratio < loiter_min_displacement_ratio:
                reject_reasons.append("loiter")

        s_in = None
        e_in = None
        direction = "unknown"
        if gate_mask is not None:
            s_in = _point_in_mask(start, gate_mask)
            e_in = _point_in_mask(end, gate_mask)
            direction = _classify_direction_full(s_in, e_in, track_points, gate_mask, gate_mask.shape[:2])
        elif direction_mask is not None:
            s_in = _point_in_mask(start, direction_mask)
            e_in = _point_in_mask(end, direction_mask)
            direction = _classify_direction_full(s_in, e_in, track_points, direction_mask, direction_mask.shape[:2])

        accepted = not reject_reasons
        if not accepted:
            reasons_set = set(reject_reasons)
            if (
                reasons_set.issubset({"min_track_length"})
                and len(track_points) >= min_track_length_from_sec
                and score >= strong_short_score_min
            ):
                accepted = True
                reject_reasons = []
        assessments.append({
            "video_id": start.video_id,
            "track_id": start.track_id,
            "accepted": accepted,
            "reject_reasons": ";".join(reject_reasons),
            "score": round(score, 4),
            "frame_start": start.frame,
            "frame_end": end.frame,
            "num_detections": len(track_points),
            "duration_sec": round(duration, 4),
            "x_start": round(start.x, 2),
            "y_start": round(start.y, 2),
            "x_end": round(end.x, 2),
            "y_end": round(end.y, 2),
            "displacement_px": round(displacement, 2),
            "path_length_px": round(path_length, 2),
            "straightness": round(straightness, 4),
            "mean_speed_px_sec": round(mean_speed, 2),
            "mean_area": round(avg_area, 2),
            "start_in_valid_region": s_in if s_in is not None else "",
            "end_in_valid_region": e_in if e_in is not None else "",
            "direction": direction,
        })

        if not accepted:
            continue

        filtered.extend(track_points)

    assessments.sort(key=lambda row: int(row["track_id"]))
    return filtered, assessments


def _vector_cosine(v0: tuple[float, float], v1: tuple[float, float]) -> float | None:
    n0 = hypot(v0[0], v0[1])
    n1 = hypot(v1[0], v1[1])
    if n0 <= 1e-6 or n1 <= 1e-6:
        return None
    return (v0[0] * v1[0] + v0[1] * v1[1]) / (n0 * n1)


def _track_edge_vectors(points: List[TrackPoint]) -> tuple[tuple[float, float], tuple[float, float]]:
    start_idx = min(2, len(points) - 1)
    end_idx = max(0, len(points) - 3)
    start_vec = (points[start_idx].x - points[0].x, points[start_idx].y - points[0].y)
    end_vec = (points[-1].x - points[end_idx].x, points[-1].y - points[end_idx].y)
    return start_vec, end_vec


def _auto_merge_track_points(
    points: List[TrackPoint],
    tracking_cfg: Dict,
    frame_size: tuple[int, int] | None = None,
) -> tuple[List[TrackPoint], List[Dict]]:
    if not bool(tracking_cfg.get("auto_merge_suggested", False)):
        return points, []

    by_track: Dict[int, List[TrackPoint]] = defaultdict(list)
    for point in points:
        by_track[point.track_id].append(point)
    for track_id in by_track:
        by_track[track_id] = sorted(by_track[track_id], key=lambda p: p.frame)

    if len(by_track) < 2:
        return points, []

    max_gap = int(tracking_cfg.get("merge_max_gap_frames", 8))
    max_endpoint_dist = _scale_linear_px_for_resolution(
        float(tracking_cfg.get("merge_max_endpoint_distance", 80.0)),
        tracking_cfg,
        frame_size,
        reference_width_key="reference_width",
        reference_height_key="reference_height",
    )
    min_overlap_common = int(tracking_cfg.get("merge_overlap_min_common_frames", 3))
    max_overlap_mean_dist = _scale_linear_px_for_resolution(
        float(tracking_cfg.get("merge_overlap_max_mean_distance", 60.0)),
        tracking_cfg,
        frame_size,
        reference_width_key="reference_width",
        reference_height_key="reference_height",
    )
    min_overlap_cos = float(tracking_cfg.get("merge_overlap_min_direction_cosine", 0.8))
    local_overlap_min_cos = max(0.65, min_overlap_cos - 0.15)
    proximity_override_dist = _scale_linear_px_for_resolution(
        float(tracking_cfg.get("merge_overlap_proximity_override_distance", 0.0)),
        tracking_cfg,
        frame_size,
        reference_width_key="reference_width",
        reference_height_key="reference_height",
    )
    # Guardas anti-transitividad / anti-coexistencia:
    # - max_group_overlap_frames: dos grupos no pueden fusionarse si comparten
    #   demasiados frames (murcielagos distintos que vuelan en paralelo por el
    #   mismo corredor), salvo que sean practicamente la misma deteccion.
    # - duplicate_max_distance: umbral de "misma deteccion" que permite fusionar
    #   pese al solape temporal (track duplicado real).
    # - max_group_size: tope de tracks distintos por grupo fusionado.
    max_group_overlap_frames = int(tracking_cfg.get("merge_max_group_overlap_frames", 6))
    duplicate_max_distance = _scale_linear_px_for_resolution(
        float(tracking_cfg.get("merge_duplicate_max_distance", 12.0)),
        tracking_cfg,
        frame_size,
        reference_width_key="reference_width",
        reference_height_key="reference_height",
    )
    max_group_size = int(tracking_cfg.get("merge_max_group_size", 0))

    parent: Dict[int, int] = {track_id: track_id for track_id in by_track}
    group_frames: Dict[int, set] = {
        track_id: {p.frame for p in pts} for track_id, pts in by_track.items()
    }
    group_members: Dict[int, set] = {track_id: {track_id} for track_id in by_track}

    def find(track_id: int) -> int:
        while parent[track_id] != track_id:
            parent[track_id] = parent[parent[track_id]]
            track_id = parent[track_id]
        return track_id

    def union(a: int, b: int) -> int:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return ra
        keep, drop = (ra, rb) if ra < rb else (rb, ra)
        parent[drop] = keep
        group_frames[keep] |= group_frames[drop]
        group_members[keep] |= group_members[drop]
        return keep

    merges_applied: List[Dict] = []
    track_ids = sorted(by_track.keys())

    # Precompute per-track data to avoid redundant work per pair.
    track_data: Dict[int, Dict] = {}
    for tid in track_ids:
        pts = by_track[tid]
        sv, ev = _track_edge_vectors(pts)
        track_data[tid] = {
            "start": pts[0],
            "end": pts[-1],
            "start_vec": sv,
            "end_vec": ev,
            "frames": {p.frame: p for p in pts},
        }

    # Generate candidate pairs via temporal sweep + handoff spatial pre-filter.
    # Temporal sweep: sort tracks by start_frame; maintain a min-heap of active
    # tracks (end_frame, index). Prune tracks that ended more than max_gap frames
    # before the current track starts. For handoff-type candidates (A ends before
    # B starts), additionally require endpoint proximity to avoid O(N²) behaviour.
    n = len(track_ids)
    sf_arr = [track_data[tid]["start"].frame for tid in track_ids]
    ef_arr = [track_data[tid]["end"].frame for tid in track_ids]
    sx_arr = [track_data[tid]["start"].x for tid in track_ids]
    sy_arr = [track_data[tid]["start"].y for tid in track_ids]
    ex_arr = [track_data[tid]["end"].x for tid in track_ids]
    ey_arr = [track_data[tid]["end"].y for tid in track_ids]
    max_ep_dist_sq = max_endpoint_dist * max_endpoint_dist

    sorted_pos = sorted(range(n), key=lambda i: sf_arr[i])
    heap: List[Tuple[int, int]] = []  # (end_frame, position)
    candidate_pairs: List[Tuple[int, int]] = []

    for b_pos in sorted_pos:
        b_sf = sf_arr[b_pos]
        b_id = track_ids[b_pos]

        while heap and heap[0][0] < b_sf - max_gap:
            heapq.heappop(heap)

        for a_ef, a_pos in heap:
            a_id = track_ids[a_pos]
            if a_ef < b_sf:
                # Handoff A→B: pre-filter by endpoint proximity.
                dx = ex_arr[a_pos] - sx_arr[b_pos]
                dy = ey_arr[a_pos] - sy_arr[b_pos]
                if dx * dx + dy * dy > max_ep_dist_sq:
                    continue
            candidate_pairs.append((min(a_id, b_id), max(a_id, b_id)))

        heapq.heappush(heap, (ef_arr[b_pos], b_pos))

    for track_a_id, track_b_id in candidate_pairs:
        td_a = track_data[track_a_id]
        td_b = track_data[track_b_id]
        a_start = td_a["start"]
        a_end = td_a["end"]
        a_start_vec = td_a["start_vec"]
        a_end_vec = td_a["end_vec"]
        a_frames = td_a["frames"]
        b_start = td_b["start"]
        b_end = td_b["end"]
        b_start_vec = td_b["start_vec"]
        b_end_vec = td_b["end_vec"]

        reason = None
        reason_data: Dict[str, float | int] = {}

        if a_end.frame < b_start.frame:
            gap = b_start.frame - a_end.frame
            dist = hypot(b_start.x - a_end.x, b_start.y - a_end.y)
            if gap <= max_gap and dist <= max_endpoint_dist:
                reason = "handoff"
                reason_data = {"gap_frames": gap, "endpoint_distance": dist}
        elif b_end.frame < a_start.frame:
            gap = a_start.frame - b_end.frame
            dist = hypot(a_start.x - b_end.x, a_start.y - b_end.y)
            if gap <= max_gap and dist <= max_endpoint_dist:
                reason = "handoff"
                reason_data = {"gap_frames": gap, "endpoint_distance": dist}
        else:
            b_frames = td_b["frames"]
            common_frames = sorted(set(a_frames.keys()).intersection(b_frames.keys()))
            if len(common_frames) >= 1:
                distances = []
                for frame in common_frames:
                    pa = a_frames[frame]
                    pb = b_frames[frame]
                    distances.append(hypot(pa.x - pb.x, pa.y - pb.y))

                mean_distance = sum(distances) / len(distances)
                start_cos = _vector_cosine(a_start_vec, b_start_vec)
                end_cos = _vector_cosine(a_end_vec, b_end_vec)
                connector_cos = _vector_cosine(a_end_vec, b_start_vec)
                global_cosines = [c for c in (start_cos, end_cos) if c is not None]
                mean_cos = (sum(global_cosines) / len(global_cosines)) if global_cosines else None

                overlap_reason = None
                # Proximity override: two detections within this distance for any shared frame
                # are almost certainly the same physical bat — skip direction checks.
                if (
                    proximity_override_dist > 0.0
                    and mean_distance <= proximity_override_dist
                    and len(common_frames) >= 1
                ):
                    overlap_reason = "overlap_proximity"
                elif len(common_frames) >= min_overlap_common:
                    if mean_distance <= max_overlap_mean_dist and (
                        mean_cos is None or mean_cos >= min_overlap_cos or connector_cos is not None and connector_cos >= min_overlap_cos
                    ):
                        overlap_reason = "overlap"
                    elif (
                        # Fragments of a curved trajectory: start vectors agree but end
                        # vectors diverge because they cover different phases of the arc.
                        mean_distance <= max_overlap_mean_dist
                        and start_cos is not None
                        and start_cos >= local_overlap_min_cos
                    ):
                        overlap_reason = "overlap_start_aligned"
                elif (
                    len(common_frames) >= 1
                    and mean_distance <= max_overlap_mean_dist
                    and connector_cos is not None
                    and connector_cos >= local_overlap_min_cos
                ):
                    overlap_reason = "overlap_local"

                if overlap_reason is not None:
                    direction_score = connector_cos
                    if direction_score is None:
                        direction_score = mean_cos if mean_cos is not None else 1.0
                    reason = overlap_reason
                    reason_data = {
                        "common_frames": len(common_frames),
                        "mean_distance": mean_distance,
                        "mean_direction_cosine": direction_score,
                    }

        if reason is None:
            continue

        ra = find(track_a_id)
        rb = find(track_b_id)
        if ra == rb:
            continue

        # Guarda anti-coexistencia: no fusionar grupos que ya comparten muchos
        # frames (vuelos paralelos distintos), salvo que sea practicamente la
        # misma deteccion (track duplicado real, distancia media minima).
        shared_frames = len(group_frames[ra] & group_frames[rb])
        mean_distance_val = reason_data.get("mean_distance")
        is_duplicate = (
            mean_distance_val is not None
            and float(mean_distance_val) <= duplicate_max_distance
        )
        if (
            max_group_overlap_frames >= 0
            and shared_frames > max_group_overlap_frames
            and not is_duplicate
        ):
            continue

        # Guarda anti-transitividad: tope de tracks distintos por grupo.
        if max_group_size > 0 and len(group_members[ra] | group_members[rb]) > max_group_size:
            continue

        union(track_a_id, track_b_id)
        merged_to = find(track_a_id)
        merges_applied.append(
            {
                "track_a": track_a_id,
                "track_b": track_b_id,
                "merged_to": merged_to,
                "reason": reason,
                **reason_data,
            }
        )

    remap: Dict[int, int] = {track_id: find(track_id) for track_id in track_ids}
    if all(src == dst for src, dst in remap.items()):
        return points, []

    merged_points: List[TrackPoint] = []
    for point in points:
        new_track_id = remap.get(point.track_id, point.track_id)
        if new_track_id == point.track_id:
            merged_points.append(point)
        else:
            merged_points.append(
                TrackPoint(
                    video_id=point.video_id,
                    track_id=new_track_id,
                    frame=point.frame,
                    time_sec=point.time_sec,
                    x=point.x,
                    y=point.y,
                    vx=point.vx,
                    vy=point.vy,
                    bbox_x1=point.bbox_x1,
                    bbox_y1=point.bbox_y1,
                    bbox_x2=point.bbox_x2,
                    bbox_y2=point.bbox_y2,
                    area=point.area,
                )
            )

    merged_points = sorted(merged_points, key=lambda p: (p.track_id, p.frame))
    by_track_frame: Dict[tuple[int, int], List[TrackPoint]] = defaultdict(list)
    for point in merged_points:
        by_track_frame[(point.track_id, point.frame)].append(point)

    consolidated: List[TrackPoint] = []
    for key in sorted(by_track_frame.keys()):
        candidates = by_track_frame[key]
        if len(candidates) == 1:
            consolidated.append(candidates[0])
            continue

        # Keep one point per merged track/frame. Prefer the strongest blob and
        # break ties deterministically to stabilize overlays and exports.
        best = max(
            candidates,
            key=lambda p: (
                p.area,
                -abs(p.vx) - abs(p.vy),
                -p.x,
                -p.y,
            ),
        )
        consolidated.append(best)

    return consolidated, merges_applied


def _scale_linear_px_for_resolution(
    value_px: float,
    cfg: Dict,
    frame_size: tuple[int, int] | None,
    *,
    reference_width_key: str,
    reference_height_key: str,
) -> float:
    if frame_size is None or not bool(cfg.get("auto_scale_with_resolution", True)):
        return float(value_px)

    width, height = frame_size
    ref_w = max(1, int(cfg.get(reference_width_key, 1024)))
    ref_h = max(1, int(cfg.get(reference_height_key, 576)))
    target_w = max(1, int(width))
    target_h = max(1, int(height))
    ref_diag = max(1.0, hypot(float(ref_w), float(ref_h)))
    target_diag = max(1.0, hypot(float(target_w), float(target_h)))
    return float(value_px) * target_diag / ref_diag


def _dedupe_coexisting_track_points(
    points: List[TrackPoint],
    tracking_cfg: Dict,
    frame_size: tuple[int, int] | None = None,
) -> tuple[List[TrackPoint], List[int]]:
    if not bool(tracking_cfg.get("dedupe_coexisting_tracks", False)):
        return points, []

    min_common_frames = int(tracking_cfg.get("dedupe_coexisting_min_common_frames", 3))
    max_mean_distance = _scale_linear_px_for_resolution(
        float(tracking_cfg.get("dedupe_coexisting_max_mean_distance", 45.0)),
        tracking_cfg,
        frame_size,
        reference_width_key="dedupe_coexisting_reference_width",
        reference_height_key="dedupe_coexisting_reference_height",
    )
    min_overlap_ratio = float(tracking_cfg.get("dedupe_coexisting_min_overlap_ratio", 0.8))
    max_short_track_length = int(tracking_cfg.get("dedupe_coexisting_max_short_track_length", 8))
    max_short_to_long_ratio = float(tracking_cfg.get("dedupe_coexisting_max_short_to_long_ratio", 0.7))
    min_direction_cosine = float(tracking_cfg.get("dedupe_coexisting_min_direction_cosine", 0.70))

    by_track: Dict[int, List[TrackPoint]] = defaultdict(list)
    for point in points:
        by_track[int(point.track_id)].append(point)
    for track_id in by_track:
        by_track[track_id].sort(key=lambda point: point.frame)

    duplicate_track_ids: set[int] = set()
    track_ids = sorted(by_track)
    frame_maps = {
        track_id: {point.frame: point for point in track_points}
        for track_id, track_points in by_track.items()
    }

    for i, track_a_id in enumerate(track_ids):
        if track_a_id in duplicate_track_ids:
            continue
        track_a = by_track[track_a_id]
        for track_b_id in track_ids[i + 1:]:
            if track_b_id in duplicate_track_ids:
                continue
            track_b = by_track[track_b_id]
            shorter_id, shorter = (
                (track_a_id, track_a)
                if len(track_a) <= len(track_b)
                else (track_b_id, track_b)
            )
            longer = track_b if shorter_id == track_a_id else track_a
            same_start = abs(track_a[0].frame - track_b[0].frame) <= 1
            same_end = abs(track_a[-1].frame - track_b[-1].frame) <= 1
            if len(shorter) > max_short_track_length:
                continue
            if shorter[0].frame < longer[0].frame or shorter[-1].frame > longer[-1].frame:
                continue
            # When two tracks start or end together, the old short-to-long
            # ratio rejected real duplicate splits like 4 vs 3 points. In that
            # case the spatial and directional evidence is already strong, so
            # the temporal containment check is enough and the ratio would be
            # too strict. Keep the ratio for the more ambiguous embedded
            # fragments that do not align on an edge.
            if not same_start and not same_end and len(shorter) / max(1, len(longer)) > max_short_to_long_ratio:
                continue

            common_frames = sorted(set(frame_maps[track_a_id]).intersection(frame_maps[track_b_id]))
            if len(common_frames) < min_common_frames:
                continue
            overlap_ratio = len(common_frames) / max(1, len(shorter))
            if overlap_ratio < min_overlap_ratio:
                continue

            distances = [
                hypot(
                    frame_maps[track_a_id][frame].x - frame_maps[track_b_id][frame].x,
                    frame_maps[track_a_id][frame].y - frame_maps[track_b_id][frame].y,
                )
                for frame in common_frames
            ]
            mean_distance = sum(distances) / len(distances)
            if mean_distance > max_mean_distance:
                continue

            vec_a = (track_a[-1].x - track_a[0].x, track_a[-1].y - track_a[0].y)
            vec_b = (track_b[-1].x - track_b[0].x, track_b[-1].y - track_b[0].y)
            direction_cosine = _vector_cosine(vec_a, vec_b)
            if direction_cosine is not None and direction_cosine < min_direction_cosine:
                continue

            duplicate_track_ids.add(shorter_id)

    if not duplicate_track_ids:
        return points, []

    deduped = [
        point
        for point in points
        if int(point.track_id) not in duplicate_track_ids
    ]
    return sorted(deduped, key=lambda point: (point.track_id, point.frame)), sorted(duplicate_track_ids)


def _rescue_crossing_continuation_points(
    source_points: List[TrackPoint],
    accepted_points: List[TrackPoint],
    assessments: List[dict],
    tracking_cfg: Dict,
    frame_size: tuple[int, int] | None = None,
) -> tuple[List[TrackPoint], List[Dict]]:
    if not bool(tracking_cfg.get("rescue_crossing_continuations", False)):
        return accepted_points, []

    accepted_track_ids = {int(point.track_id) for point in accepted_points}
    if not accepted_track_ids:
        return accepted_points, []

    rescue_reject_reasons = {
        reason.strip()
        for reason in str(tracking_cfg.get("rescue_crossing_reject_reasons", "valid_region_gate")).split(";")
        if reason.strip()
    }
    max_gap_frames = int(tracking_cfg.get("rescue_crossing_max_gap_frames", 2))
    max_start_distance = _scale_linear_px_for_resolution(
        float(tracking_cfg.get("rescue_crossing_max_start_distance", 70.0)),
        tracking_cfg,
        frame_size,
        reference_width_key="rescue_crossing_reference_width",
        reference_height_key="rescue_crossing_reference_height",
    )
    min_fragment_points = int(tracking_cfg.get("rescue_crossing_min_fragment_points", 4))
    min_new_points = int(tracking_cfg.get("rescue_crossing_min_new_points", 3))
    min_overlap_distance_from_other_tracks = _scale_linear_px_for_resolution(
        float(tracking_cfg.get("rescue_crossing_min_distance_from_other_tracks", 45.0)),
        tracking_cfg,
        frame_size,
        reference_width_key="rescue_crossing_reference_width",
        reference_height_key="rescue_crossing_reference_height",
    )

    source_by_track: Dict[int, List[TrackPoint]] = defaultdict(list)
    for point in source_points:
        source_by_track[int(point.track_id)].append(point)
    for track_id in source_by_track:
        source_by_track[track_id].sort(key=lambda point: point.frame)

    accepted_by_track: Dict[int, List[TrackPoint]] = defaultdict(list)
    for point in accepted_points:
        accepted_by_track[int(point.track_id)].append(point)
    for track_id in accepted_by_track:
        accepted_by_track[track_id].sort(key=lambda point: point.frame)

    rejected_track_ids: set[int] = set()
    for assessment in assessments:
        if bool(assessment.get("accepted")):
            continue
        track_id = int(assessment.get("track_id"))
        if track_id in accepted_track_ids:
            continue
        reasons = {
            reason.strip()
            for reason in str(assessment.get("reject_reasons", "")).split(";")
            if reason.strip()
        }
        if reasons and reasons.issubset(rescue_reject_reasons):
            rejected_track_ids.add(track_id)

    if not rejected_track_ids:
        return accepted_points, []

    accepted_frame_maps = {
        track_id: {point.frame: point for point in track_points}
        for track_id, track_points in accepted_by_track.items()
    }
    rescue_candidates: list[tuple[float, int, int, List[TrackPoint], float, int]] = []

    for accepted_track_id, accepted_track in accepted_by_track.items():
        accepted_end = accepted_track[-1]
        for rejected_track_id in sorted(rejected_track_ids):
            fragment = source_by_track.get(rejected_track_id, [])
            if len(fragment) < min_fragment_points:
                continue
            gap = fragment[0].frame - accepted_end.frame
            if gap < -1 or gap > max_gap_frames:
                continue
            start_distance = hypot(fragment[0].x - accepted_end.x, fragment[0].y - accepted_end.y)
            if start_distance > max_start_distance:
                continue

            new_points = [point for point in fragment if point.frame > accepted_end.frame]
            if len(new_points) < min_new_points:
                continue

            too_close_to_other_track = False
            for other_track_id, frame_map in accepted_frame_maps.items():
                if other_track_id == accepted_track_id:
                    continue
                distances = []
                for point in new_points:
                    other = frame_map.get(point.frame)
                    if other is not None:
                        distances.append(hypot(point.x - other.x, point.y - other.y))
                if distances and sum(distances) / len(distances) <= min_overlap_distance_from_other_tracks:
                    too_close_to_other_track = True
                    break
            if too_close_to_other_track:
                continue

            score = start_distance + max(0, gap) * max_start_distance - 2.0 * len(new_points)
            rescue_candidates.append(
                (score, accepted_track_id, rejected_track_id, new_points, start_distance, gap)
            )

    rescued_points = list(accepted_points)
    rescues: List[Dict] = []
    used_rejected_track_ids: set[int] = set()
    used_accepted_track_ids: set[int] = set()
    for score, accepted_track_id, rejected_track_id, new_points, start_distance, gap in sorted(rescue_candidates):
        if accepted_track_id in used_accepted_track_ids or rejected_track_id in used_rejected_track_ids:
            continue
        used_accepted_track_ids.add(accepted_track_id)
        used_rejected_track_ids.add(rejected_track_id)
        for point in new_points:
            rescued_points.append(
                TrackPoint(
                    video_id=point.video_id,
                    track_id=accepted_track_id,
                    frame=point.frame,
                    time_sec=point.time_sec,
                    x=point.x,
                    y=point.y,
                    vx=point.vx,
                    vy=point.vy,
                    bbox_x1=point.bbox_x1,
                    bbox_y1=point.bbox_y1,
                    bbox_x2=point.bbox_x2,
                    bbox_y2=point.bbox_y2,
                    area=point.area,
                )
            )
        rescues.append(
            {
                "track_id": accepted_track_id,
                "source_track_id": rejected_track_id,
                "points_added": len(new_points),
                "gap_frames": int(gap),
                "start_distance": round(float(start_distance), 3),
                "score": round(float(score), 3),
            }
        )

    if not rescues:
        return accepted_points, []
    return sorted(rescued_points, key=lambda point: (point.track_id, point.frame)), rescues


def _rescue_motion_candidate_points(
    source_points: List[TrackPoint],
    accepted_points: List[TrackPoint],
    assessments: List[dict],
    tracking_cfg: Dict,
    interaction_mask: np.ndarray | None = None,
    frame_size: tuple[int, int] | None = None,
) -> tuple[List[TrackPoint], List[Dict]]:
    """Recover short, high-motion candidates rejected by spatial masks.

    The primary failure mode in crowded bat emergence videos is not that the
    blob tracker never sees the animal; it is that post filters reject short
    fragments because the valid-region/vegetation masks are too conservative.
    This rescue keeps the noise filters for tiny/stationary blobs, but accepts
    candidates with enough independent motion evidence.
    """
    if not bool(tracking_cfg.get("rescue_motion_candidates", False)):
        return accepted_points, []
    if interaction_mask is None:
        return accepted_points, []

    rescue_reject_reasons = {
        reason.strip()
        for reason in str(
            tracking_cfg.get("rescue_motion_reject_reasons", "valid_region_gate;vegetation_mask")
        ).split(";")
        if reason.strip()
    }
    min_points = max(2, int(tracking_cfg.get("rescue_motion_min_points", 3)))
    min_displacement = _scale_linear_px_for_resolution(
        float(tracking_cfg.get("rescue_motion_min_displacement", 18.0)),
        tracking_cfg,
        frame_size,
        reference_width_key="reference_width",
        reference_height_key="reference_height",
    )
    min_path_length = _scale_linear_px_for_resolution(
        float(tracking_cfg.get("rescue_motion_min_path_length", 24.0)),
        tracking_cfg,
        frame_size,
        reference_width_key="reference_width",
        reference_height_key="reference_height",
    )
    min_mean_speed = _scale_linear_px_for_resolution(
        float(tracking_cfg.get("rescue_motion_min_mean_speed", 120.0)),
        tracking_cfg,
        frame_size,
        reference_width_key="reference_width",
        reference_height_key="reference_height",
    )
    min_straightness = float(tracking_cfg.get("rescue_motion_min_straightness", 0.0))
    interaction_dilate_px = max(
        0,
        int(
            round(
                _scale_linear_px_for_resolution(
                    float(tracking_cfg.get("rescue_motion_interaction_dilate_px", 0)),
                    tracking_cfg,
                    frame_size,
                    reference_width_key="reference_width",
                    reference_height_key="reference_height",
                )
            )
        ),
    )
    if interaction_dilate_px > 0:
        k = 2 * interaction_dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        interaction_mask = cv2.dilate(interaction_mask, kernel, iterations=1)

    accepted_track_ids = {int(point.track_id) for point in accepted_points}
    source_by_track: Dict[int, List[TrackPoint]] = defaultdict(list)
    for point in source_points:
        source_by_track[int(point.track_id)].append(point)
    for track_id in source_by_track:
        source_by_track[track_id].sort(key=lambda point: point.frame)

    rescued_points = list(accepted_points)
    rescues: List[Dict] = []
    for assessment in assessments:
        if bool(assessment.get("accepted")):
            continue
        track_id = int(assessment.get("track_id"))
        if track_id in accepted_track_ids:
            continue
        reasons = {
            reason.strip()
            for reason in str(assessment.get("reject_reasons", "")).split(";")
            if reason.strip()
        }
        if not reasons or not reasons.issubset(rescue_reject_reasons):
            continue
        num_points = int(assessment.get("num_detections", 0))
        displacement = float(assessment.get("displacement_px", 0.0))
        path_length = float(assessment.get("path_length_px", 0.0))
        mean_speed = float(assessment.get("mean_speed_px_sec", 0.0))
        straightness = float(assessment.get("straightness", 0.0))
        if num_points < min_points:
            continue
        if displacement < min_displacement or path_length < min_path_length:
            continue
        if mean_speed < min_mean_speed:
            continue
        if min_straightness > 0.0 and straightness < min_straightness:
            continue

        fragment = source_by_track.get(track_id, [])
        if not fragment:
            continue
        if not any(_point_in_mask(point, interaction_mask) for point in fragment):
            continue
        rescued_points.extend(fragment)
        accepted_track_ids.add(track_id)
        rescues.append(
            {
                "track_id": track_id,
                "reject_reasons": ";".join(sorted(reasons)),
                "points_added": len(fragment),
                "displacement_px": round(displacement, 3),
                "path_length_px": round(path_length, 3),
                "mean_speed_px_sec": round(mean_speed, 3),
                "straightness": round(straightness, 4),
            }
        )

    if not rescues:
        return accepted_points, []
    return sorted(rescued_points, key=lambda point: (point.track_id, point.frame)), rescues


def _candidate_recall_metrics(assessments: List[dict]) -> Dict:
    total_tracks = len(assessments)
    accepted_tracks = sum(1 for row in assessments if bool(row.get("accepted")))
    rejected_tracks = total_tracks - accepted_tracks
    accepted_points = 0
    rejected_points = 0
    rejection_reasons: Counter = Counter()
    for row in assessments:
        n = int(row.get("num_detections", 0))
        if bool(row.get("accepted")):
            accepted_points += n
        else:
            rejected_points += n
            for reason in str(row.get("reject_reasons", "")).split(";"):
                if reason:
                    rejection_reasons[reason] += 1
    total_points = accepted_points + rejected_points
    orphan_pct = (100.0 * rejected_points / total_points) if total_points else 0.0
    return {
        "track_candidates_total": total_tracks,
        "track_candidates_accepted": accepted_tracks,
        "track_candidates_rejected": rejected_tracks,
        "track_candidate_points_accepted": accepted_points,
        "track_candidate_points_rejected": rejected_points,
        "track_candidate_orphan_detection_pct": orphan_pct,
        "track_candidate_rejection_reasons": dict(rejection_reasons.most_common()),
    }


def _diagnose_entry_exit_zone(
    diagnostics_path: str,
    *,
    final_points: List[TrackPoint],
    candidate_points: List[TrackPoint],
    assessments: List[dict],
    entry_exit_mask: np.ndarray | None,
    valid_gate_mask: np.ndarray | None,
) -> Dict:
    if not diagnostics_path or entry_exit_mask is None:
        return {}

    def _endpoint_counts(points: List[TrackPoint], mask: np.ndarray | None) -> dict:
        by_track: Dict[int, List[TrackPoint]] = defaultdict(list)
        for point in points:
            by_track[int(point.track_id)].append(point)
        if mask is None:
            return {
                "tracks_total": len(by_track),
                "start_inside": 0,
                "end_inside": 0,
                "start_or_end_inside": 0,
                "pct_start_or_end_inside": 0.0,
            }
        start_inside = 0
        end_inside = 0
        any_inside = 0
        for track_points in by_track.values():
            track_points.sort(key=lambda point: point.frame)
            s_in = _point_in_mask(track_points[0], mask)
            e_in = _point_in_mask(track_points[-1], mask)
            start_inside += int(s_in)
            end_inside += int(e_in)
            any_inside += int(s_in or e_in)
        total = len(by_track)
        return {
            "tracks_total": total,
            "start_inside": start_inside,
            "end_inside": end_inside,
            "start_or_end_inside": any_inside,
            "pct_start_or_end_inside": round(100.0 * any_inside / total, 3) if total else 0.0,
        }

    candidate_by_track: Dict[int, List[TrackPoint]] = defaultdict(list)
    for point in candidate_points:
        candidate_by_track[int(point.track_id)].append(point)
    rejected_gate_track_ids = [
        int(row["track_id"])
        for row in assessments
        if not bool(row.get("accepted")) and "valid_region_gate" in str(row.get("reject_reasons", "")).split(";")
    ]
    rejected_touching_entry_exit = 0
    for track_id in rejected_gate_track_ids:
        track_points = candidate_by_track.get(track_id, [])
        if any(_point_in_mask(point, entry_exit_mask) for point in track_points):
            rejected_touching_entry_exit += 1

    diagnostics_summary = {
        "final_tracks_vs_entry_exit_zone": _endpoint_counts(final_points, entry_exit_mask),
        "final_tracks_vs_valid_region_gate": _endpoint_counts(final_points, valid_gate_mask),
        "rejected_by_gate_total": len(rejected_gate_track_ids),
        "rejected_by_gate_touching_entry_exit_zone": rejected_touching_entry_exit,
    }
    path = Path(diagnostics_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except json.JSONDecodeError:
        payload = {}
    payload["track_endpoint_diagnostics"] = diagnostics_summary
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return diagnostics_summary


def _export_track_clips(
    input_video: str,
    output_dir: Path,
    points: List[TrackPoint],
    fps: float,
    frame_size: tuple[int, int],
    clips_subdir: str,
    pad_frames: int,
) -> Dict[str, str]:
    by_track: Dict[int, List[TrackPoint]] = defaultdict(list)
    for point in points:
        by_track[point.track_id].append(point)

    if not by_track:
        return {}

    clips_dir = output_dir / clips_subdir
    clips_dir.mkdir(parents=True, exist_ok=True)

    intervals: Dict[int, tuple[int, int]] = {}
    for track_id, track_points in by_track.items():
        frames = sorted(p.frame for p in track_points)
        start = max(0, frames[0] - pad_frames)
        end = max(start, frames[-1] + pad_frames)
        intervals[track_id] = (start, end)

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for track clips export: {input_video}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    width, height = frame_size
    writers: Dict[int, cv2.VideoWriter] = {}
    clip_paths: Dict[str, str] = {}
    current_frame = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            for track_id, (start, end) in intervals.items():
                if current_frame < start or current_frame > end:
                    continue

                writer = writers.get(track_id)
                if writer is None:
                    clip_path = clips_dir / f"track_{track_id:04d}_{start:06d}-{end:06d}.mp4"
                    writer = cv2.VideoWriter(str(clip_path), fourcc, fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError(f"Cannot create clip writer for track {track_id}: {clip_path}")
                    writers[track_id] = writer
                    clip_paths[str(track_id)] = str(clip_path.resolve())

                if frame.ndim == 2:
                    out_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    out_frame = frame
                writer.write(out_frame)

            current_frame += 1
    finally:
        cap.release()
        for writer in writers.values():
            writer.release()

    return clip_paths


def _build_tracker(
    tracking_cfg: Dict,
    fps: float,
    video_id: str,
    frame_size: tuple[int, int] | None = None,
):
    """Construye el tracker según ``tracking.tracker`` (kalman|greedy)."""
    kind = str(tracking_cfg.get("tracker", "kalman")).strip().lower()
    max_distance = _scale_linear_px_for_resolution(
        float(tracking_cfg["max_distance"]),
        tracking_cfg,
        frame_size,
        reference_width_key="reference_width",
        reference_height_key="reference_height",
    )
    max_missed = int(tracking_cfg["max_missed"])
    if kind == "greedy":
        return GreedyTracker(
            max_distance=max_distance,
            max_missed=max_missed,
            fps=fps,
            video_id=video_id,
        )
    return KalmanTracker(
        max_distance=max_distance,
        max_missed=max_missed,
        fps=fps,
        video_id=video_id,
        sigma_acc=float(tracking_cfg.get("kalman_sigma_acc", 3.0)),
        measurement_std=float(tracking_cfg.get("kalman_measurement_std", 2.0)),
        high_area_threshold=float(tracking_cfg.get("kalman_high_area_threshold", 0.0)),
    )


def run_pipeline(input_video: str, output_dir: str, config_path: str | None = None) -> Dict:
    pipeline_started = perf_counter()
    cfg = load_config(config_path)
    execution_plan = build_execution_plan(cfg)
    execution_cfg = cfg.get("execution", {})
    strict_parity = bool(execution_cfg.get("strict_parity", True))
    background_runtime_stats: Dict[str, int] = {
        "background_gpu_used": 0,
        "background_gpu_unavailable": 0,
        "background_gpu_failures": 0,
        "background_gpu_parity_checked": 0,
        "background_gpu_parity_mismatch": 0,
    }
    detection_runtime_stats: Dict[str, int] = {
        "cuda_frames_used": 0,
        "cuda_runtime_failures": 0,
        "cuda_parity_checked_frames": 0,
        "cuda_parity_mismatch_frames": 0,
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = read_video_meta(input_video)
    perf = PerformanceCollector(meta.frame_count)
    valid_region_cfg = cfg.get("valid_region", {})
    valid_region_enabled = bool(valid_region_cfg.get("enabled", False))
    cave_zones_cfg = cfg.get("cave_zones", {})
    cave_zones_enabled = bool(cave_zones_cfg.get("enabled", False))
    secondary_detection_cfg = cfg.get("secondary_detection", {})
    secondary_detection_enabled = bool(secondary_detection_cfg.get("enabled", False))
    secondary_detection_algorithm = str(secondary_detection_cfg.get("algorithm", "foreground")).strip().lower()
    secondary_blob_detection_enabled = secondary_detection_enabled and secondary_detection_algorithm == "foreground"
    secondary_kinetic_enabled = secondary_detection_enabled and secondary_detection_algorithm == "kinetic"
    export_track_clips_enabled = bool(cfg["output"].get("export_track_clips", False))
    fast_events_enabled = bool(cfg.get("fast_events", {}).get("enabled", False))
    heatmap_events_enabled = bool(cfg.get("heatmap_events", {}).get("enabled", False))
    flight_trails_enabled = bool(cfg.get("flight_trails", {}).get("enabled", False))
    progress = ProgressReporter(
        enabled=bool(cfg["output"].get("progress_enabled", True)),
        step_percent=int(cfg["output"].get("progress_step_percent", 5)),
        stages=[
            ("background", 15.0),
            ("valid_region", 10.0 if valid_region_enabled else 0.0),
            ("frame_processing", 55.0),
            ("cave_zones", 5.0 if cave_zones_enabled else 0.0),
            ("postprocess", 8.0),
            ("fast_events", 6.0 if fast_events_enabled else 0.0),
            ("heatmap_events", 8.0 if heatmap_events_enabled else 0.0),
            ("exports_core", 12.0),
            ("flight_trails", 12.0 if flight_trails_enabled else 0.0),
            ("track_clips", 10.0 if export_track_clips_enabled else 0.0),
        ],
    )

    progress.start_stage("background")
    background_input = str(cfg["background"].get("input_image", "")).strip()
    background_source = "computed"
    background_context_start, background_context_end = _background_context_bounds(meta, cfg["background"])
    if background_input:
        background = load_valid_region_image(background_input)
        background_source = "input_image"
        if background.shape[:2] != (meta.height, meta.width):
            raise ValueError(
                "background.input_image shape does not match the input video: "
                f"expected {(meta.height, meta.width)}, got {background.shape[:2]}"
            )
    else:
        background = compute_background_median(
            video_path=input_video,
            meta=meta,
            sample_frames=int(cfg["background"]["sample_frames"]),
            uniform_sampling=bool(cfg["background"]["uniform_sampling"]),
            compute_device=execution_plan.selected_device,
            strict_parity=strict_parity,
            runtime_stats=background_runtime_stats,
            context_start_frame=background_context_start,
            context_end_frame=background_context_end,
        )
    background_path = out_dir / "background.png"
    cv2.imwrite(str(background_path), background)
    progress.complete_stage("background", detail="background ready")
    valid_mask: np.ndarray | None = None
    valid_mask_for_detection: np.ndarray | None = None
    valid_gate_mask: np.ndarray | None = None
    vegetation_mask: np.ndarray | None = None
    valid_region_meta: Dict = {"enabled": False}
    valid_region_outputs: Dict[str, str] = {}
    vegetation_outputs: Dict[str, str] = {
        "vegetation_mask_png": "",
        "vegetation_mask_overlay_png": "",
        "vegetation_mask_overlay_video": "",
    }
    cave_zones_meta: Dict = {"enabled": False}
    cave_zones_outputs: Dict[str, str] = {
        "cave_zones_mask_png": "",
        "cave_zones_overlay_png": "",
        "cave_zones_zones_json": "",
        "cave_zones_candidates_overlay_png": "",
        "cave_zones_diagnostics_json": "",
    }
    cavemark_meta: Dict = {"enabled": False}
    cavemark_outputs: Dict[str, str] = {
        "cavemark_mask_png": "",
        "cavemark_overlay_png": "",
    }
    entry_exit_zone_selection_meta: Dict = {
        "mode": str(cfg["tracking"].get("entry_exit_zone_source", "auto")),
        "selected_source": "none",
        "scores": [],
        "reason": "",
    }

    if valid_region_enabled:
        progress.start_stage("valid_region")
        valid_subdir = str(valid_region_cfg.get("output_subdir", "valid_region"))
        valid_output_dir = out_dir / valid_subdir
        mask_path = valid_output_dir / "mask.png"
        valid_mask_input = str(valid_region_cfg.get("input_mask", "")).strip()
        if valid_mask_input:
            valid_mask = load_valid_region_mask(valid_mask_input)
            if valid_mask.shape[:2] != background.shape[:2]:
                raise ValueError(
                    "valid_region.input_mask shape does not match the processing frame size: "
                    f"expected {background.shape[:2]}, got {valid_mask.shape[:2]}"
                )
            x_start, x_end = save_precomputed_mask_outputs(background, valid_mask, valid_output_dir)
            valid_region_meta = {
                "enabled": True,
                "x_start": int(x_start),
                "x_end": int(x_end),
                "width": int(max(1, x_end - x_start + 1)),
                "method": "input_mask",
                "input_mask": str(Path(valid_mask_input).resolve()),
                "output_dir": str(valid_output_dir.resolve()),
            }
        else:
            valid_input = str(valid_region_cfg.get("input_image", "")).strip()
            if valid_input:
                valid_image = load_valid_region_image(valid_input)
            else:
                vr_context_start, vr_context_end = _valid_region_context_bounds(meta, valid_region_cfg, cfg["background"])
                if vr_context_start == background_context_start and vr_context_end == background_context_end:
                    valid_image = background
                else:
                    valid_image = compute_background_median(
                        video_path=input_video,
                        meta=meta,
                        sample_frames=int(cfg["background"]["sample_frames"]),
                        uniform_sampling=bool(cfg["background"]["uniform_sampling"]),
                        compute_device=execution_plan.selected_device,
                        strict_parity=strict_parity,
                        context_start_frame=vr_context_start,
                        context_end_frame=vr_context_end,
                    )
            valid_region_meta = run_valid_region(
                image=valid_image,
                output_dir=valid_output_dir,
                config=valid_region_cfg,
            )
            valid_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if valid_mask is None:
                raise RuntimeError(f"Could not load valid-region mask from: {mask_path}")
        if bool(valid_region_cfg.get("apply_to_detection", True)):
            valid_mask_for_detection = valid_mask
        valid_gate_mask = _build_valid_region_gate_mask(valid_mask, cfg["tracking"])
        if valid_gate_mask is not None:
            gate_overlay_path = valid_output_dir / "gate_overlay.png"
            _save_valid_region_gate_overlay(background, valid_gate_mask, gate_overlay_path)
        valid_region_outputs = {
            "valid_region_mask_png": str(mask_path.resolve()),
            "valid_region_overlay_png": str((valid_output_dir / "overlay.png").resolve()),
            "valid_region_profile_png": str((valid_output_dir / "profile.png").resolve()),
            "valid_region_gate_overlay_png": str((valid_output_dir / "gate_overlay.png").resolve()),
        }
        progress.complete_stage("valid_region", detail="valid region ready")

    vegetation_cfg = cfg.get("vegetation_noise", {})
    if bool(vegetation_cfg.get("enabled", False)):
        scaled_dilate_px, scaled_min_component_area = _scale_vegetation_params_for_resolution(
            vegetation_cfg, meta.width, meta.height
        )
        vegetation_mask_input = str(vegetation_cfg.get("input_mask", "")).strip()
        if vegetation_mask_input:
            vegetation_mask = load_valid_region_mask(vegetation_mask_input)
            if vegetation_mask.shape[:2] != background.shape[:2]:
                raise ValueError(
                    "vegetation_noise.input_mask shape does not match the processing frame size: "
                    f"expected {background.shape[:2]}, got {vegetation_mask.shape[:2]}"
                )
            dilate_px = scaled_dilate_px
            if dilate_px > 0:
                k = 2 * dilate_px + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                vegetation_mask = cv2.dilate(vegetation_mask, kernel, iterations=1)
        else:
            vegetation_mask = _compute_auto_vegetation_mask(
                input_video,
                meta,
                sample_frames=int(vegetation_cfg.get("auto_sample_frames", 220)),
                max_frame_for_sampling=int(vegetation_cfg.get("auto_max_frame_for_sampling", 1250)),
                percentile=float(vegetation_cfg.get("auto_percentile", 85.0)),
                min_component_area=scaled_min_component_area,
            )
            dilate_px = scaled_dilate_px
            if dilate_px > 0:
                k = 2 * dilate_px + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                vegetation_mask = cv2.dilate(vegetation_mask, kernel, iterations=1)

    tracker = _build_tracker(
        cfg["tracking"],
        fps=meta.fps,
        video_id=meta.video_id,
        frame_size=(meta.width, meta.height),
    )
    burst_gate = TemporalBurstGate.from_detection_cfg(cfg["detection"])
    detection_context = build_detection_context(background, cfg["detection"])
    secondary_detection_runtime_stats: Dict[str, int] = {
        "secondary_cuda_frames_used": 0,
        "secondary_cuda_runtime_failures": 0,
        "secondary_cuda_parity_checked_frames": 0,
        "secondary_cuda_parity_mismatch_frames": 0,
    }
    secondary_detection_config = None
    secondary_detection_context = None
    if secondary_blob_detection_enabled:
        secondary_detection_config = build_secondary_detection_config(cfg["detection"], secondary_detection_cfg)
        secondary_detection_context = build_detection_context(background, secondary_detection_config)
    secondary_primary_total = 0
    secondary_raw_total = 0
    secondary_added_total = 0
    secondary_duplicate_total = 0
    detections_raw_total = 0
    detections_after_burst_total = 0
    frames_with_raw_detections = 0
    frames_with_detections_after_burst = 0
    primary_detection_debug = []
    secondary_detection_debug = []
    motion_heatmap = np.zeros((meta.height, meta.width), dtype=np.float32)
    motion_heatmap_previous_gray: np.ndarray | None = None
    motion_heatmap_cfg = cfg.get("heatmap_events", {})
    motion_heatmap_blur_kernel = int(motion_heatmap_cfg.get("blur_kernel", 5))
    motion_heatmap_threshold = int(motion_heatmap_cfg.get("threshold", 14))

    all_points: List[TrackPoint] = []
    frame_processed = 0
    suppressed_burst_frames = 0
    progress.start_stage("frame_processing")

    for frame_idx, gray in iter_gray_frames(input_video, perf=perf):
        frame_started = perf_counter()
        if motion_heatmap_previous_gray is not None:
            _accumulate_motion_heatmap(
                motion_heatmap,
                motion_heatmap_previous_gray,
                gray,
                blur_kernel=motion_heatmap_blur_kernel,
                threshold=motion_heatmap_threshold,
            )
        motion_heatmap_previous_gray = gray
        dets = detect_foreground_blobs(
            gray,
            background,
            cfg["detection"],
            valid_mask=valid_mask_for_detection,
            compute_device=execution_plan.selected_device,
            strict_parity=strict_parity,
            runtime_stats=detection_runtime_stats,
            context=detection_context,
            frame_idx=frame_idx,
            perf=perf,
        )
        if secondary_blob_detection_enabled:
            primary_detection_debug.extend(dets)
        if secondary_detection_config is not None and secondary_detection_context is not None:
            secondary_stats_raw: Dict[str, int] = {}
            secondary_dets = detect_foreground_blobs(
                gray,
                background,
                secondary_detection_config,
                valid_mask=valid_mask_for_detection,
                compute_device=execution_plan.selected_device,
                strict_parity=strict_parity,
                runtime_stats=secondary_stats_raw,
                context=secondary_detection_context,
                frame_idx=frame_idx,
                perf=perf,
            )
            secondary_detection_debug.extend(secondary_dets)
            fused_dets, fusion_stats = fuse_detections(
                dets,
                secondary_dets,
                dedupe_max_distance_px=float(secondary_detection_cfg.get("dedupe_max_distance_px", 8.0)),
                dedupe_min_iou=float(secondary_detection_cfg.get("dedupe_min_iou", 0.10)),
            )
            dets = fused_dets
            secondary_primary_total += fusion_stats.primary_count
            secondary_raw_total += fusion_stats.secondary_count
            secondary_added_total += fusion_stats.secondary_added
            secondary_duplicate_total += fusion_stats.secondary_duplicates
            for key, value in secondary_stats_raw.items():
                secondary_detection_runtime_stats[f"secondary_{key}"] = (
                    secondary_detection_runtime_stats.get(f"secondary_{key}", 0) + int(value)
                )
        raw_det_count = len(dets)
        detections_raw_total += raw_det_count
        if raw_det_count > 0:
            frames_with_raw_detections += 1
        if burst_gate is not None and not burst_gate.should_keep(frame_idx, len(dets)):
            dets = []
            suppressed_burst_frames += 1
        after_burst_count = len(dets)
        detections_after_burst_total += after_burst_count
        if after_burst_count > 0:
            frames_with_detections_after_burst += 1
        tracker_started = perf_counter()
        frame_points = tracker.step(frame_idx, dets)
        perf.record("tracker", perf_counter() - tracker_started, frame_idx=frame_idx)
        all_points.extend(frame_points)
        frame_processed += 1
        perf.mark_frame_processed(frame_idx)
        perf.record("total_frame", perf_counter() - frame_started, frame_idx=frame_idx)
        if meta.frame_count > 0:
            frame_fraction = frame_processed / float(meta.frame_count)
            progress.update_stage_fraction(
                "frame_processing",
                frame_fraction,
                detail=f"frames {frame_processed}/{meta.frame_count}",
            )

    progress.complete_stage("frame_processing", detail=f"frames {frame_processed}")

    cave_zone_mask: np.ndarray | None = None
    if cave_zones_enabled:
        progress.start_stage("cave_zones")
        cave_zones_subdir = str(cave_zones_cfg.get("output_subdir", "cave_zones"))
        cave_result = run_cave_zones(
            background_gray=background,
            output_dir=out_dir / cave_zones_subdir,
            cfg=cave_zones_cfg,
            motion_heatmap=motion_heatmap,
        )
        cave_zone_mask = cave_result.mask
        cave_zones_meta = cave_result.meta
        cave_zones_outputs = cave_result.outputs
        progress.complete_stage("cave_zones", detail=f"zones {len(cave_result.zones)}")

    cavemark_mask: np.ndarray | None = None
    cavemark_cfg = cfg.get("cavemark", {})
    if bool(cavemark_cfg.get("enabled", False)):
        cavemark_subdir = str(cavemark_cfg.get("output_subdir", "cavemark"))
        cavemark_mask, cavemark_meta, cavemark_outputs = _load_cavemark_mask(
            background=background,
            cfg=cavemark_cfg,
            output_dir=out_dir / cavemark_subdir,
        )

    entry_exit_zone_source_cfg = str(cfg["tracking"].get("entry_exit_zone_source", "auto")).strip().lower()
    entry_exit_mask, entry_exit_zone_source, entry_exit_zone_selection_meta = _select_entry_exit_mask(
        source_cfg=entry_exit_zone_source_cfg,
        candidates={
            "cavemark": cavemark_mask,
            "cave_zones": cave_zone_mask,
            "valid_region": valid_gate_mask,
        },
        background=background,
        motion_heatmap=motion_heatmap,
        vegetation_mask=vegetation_mask,
        raw_points=all_points,
        selection_cfg=cfg.get("entry_exit_zone_selection", {}),
    )

    vegetation_exclusion_meta: Dict = {
        "vegetation_exclusion_enabled": False,
        "vegetation_pixels_before_exclusion": int(np.count_nonzero(vegetation_mask)) if vegetation_mask is not None else 0,
        "vegetation_pixels_after_exclusion": int(np.count_nonzero(vegetation_mask)) if vegetation_mask is not None else 0,
        "vegetation_pixels_removed_by_exclusion": 0,
    }
    if vegetation_mask is not None and bool(vegetation_cfg.get("exclude_entry_exit_zones", True)):
        vegetation_mask, vegetation_exclusion_meta = _exclude_mask_from_vegetation(
            background,
            vegetation_mask,
            entry_exit_mask,
            dilate_px=max(0, int(vegetation_cfg.get("exclude_entry_exit_dilate_px", 12))),
            mode=str(vegetation_cfg.get("entry_exit_exclusion_mode", "weak_evidence")),
            keep_texture_percentile=float(vegetation_cfg.get("entry_exit_keep_texture_percentile", 88.0)),
            keep_min_intensity_percentile=float(vegetation_cfg.get("entry_exit_keep_min_intensity_percentile", 35.0)),
            keep_min_gradient=float(vegetation_cfg.get("entry_exit_keep_min_gradient", 4.0)),
        )

    if vegetation_mask is not None:
        mask_path = out_dir / "vegetation_mask.png"
        cv2.imwrite(str(mask_path), vegetation_mask)
        overlay_path = out_dir / "vegetation_mask_overlay.png"
        _save_vegetation_mask_overlay(background, vegetation_mask, overlay_path)
        vegetation_outputs = {
            "vegetation_mask_png": str(mask_path.resolve()),
            "vegetation_mask_overlay_png": str(overlay_path.resolve()),
            "vegetation_mask_overlay_video": "",
        }

    progress.start_stage("postprocess")
    postprocess_started = perf_counter()
    merged_points, merges_applied = _auto_merge_track_points(
        all_points,
        cfg["tracking"],
        frame_size=(meta.width, meta.height),
    )
    filtered_points, track_assessments = _filter_track_points(
        merged_points,
        cfg["tracking"],
        meta.fps,
        valid_mask=valid_mask,
        entry_exit_mask=entry_exit_mask,
        vegetation_mask=vegetation_mask,
        vegetation_cfg=vegetation_cfg,
    )
    filtered_points, crossing_continuation_rescues = _rescue_crossing_continuation_points(
        merged_points,
        filtered_points,
        track_assessments,
        cfg["tracking"],
        frame_size=(meta.width, meta.height),
    )
    filtered_points, motion_candidate_rescues = _rescue_motion_candidate_points(
        merged_points,
        filtered_points,
        track_assessments,
        cfg["tracking"],
        interaction_mask=entry_exit_mask,
        frame_size=(meta.width, meta.height),
    )
    filtered_points, coexisting_duplicate_track_ids = _dedupe_coexisting_track_points(
        filtered_points,
        cfg["tracking"],
        frame_size=(meta.width, meta.height),
    )
    track_dedup_result = deduplicate_track_points(
        filtered_points,
        cfg["tracking"],
        frame_size=(meta.width, meta.height),
    )
    filtered_points = track_dedup_result.points
    secondary_kinetic_points: List[TrackPoint] = []
    secondary_kinetic_added_points: List[TrackPoint] = []
    secondary_kinetic_meta: Dict = {"enabled": secondary_kinetic_enabled}
    secondary_kinetic_dedupe_meta: Dict = {}
    secondary_kinetic_mask_meta: Dict = {}
    secondary_kinetic_burst_meta: Dict = {}
    if secondary_kinetic_enabled:
        secondary_kinetic_points, secondary_kinetic_meta = run_kinetic_secondary_tracks(
            input_video,
            secondary_detection_cfg,
            video_id=meta.video_id,
            fps=meta.fps,
        )
        secondary_kinetic_points, secondary_kinetic_mask_meta = _filter_points_start_or_end_in_mask(
            secondary_kinetic_points,
            entry_exit_mask,
        )
        secondary_kinetic_points, secondary_kinetic_burst_meta = suppress_temporal_burst_track_points(
            secondary_kinetic_points,
            min_points_per_frame=int(
                secondary_detection_cfg.get("kinetic_temporal_burst_min_points_per_frame", 0)
            ),
            window_frames=int(secondary_detection_cfg.get("kinetic_temporal_burst_window_frames", 0)),
            trigger_frames=int(secondary_detection_cfg.get("kinetic_temporal_burst_trigger_frames", 0)),
            cooldown_frames=int(secondary_detection_cfg.get("kinetic_temporal_burst_cooldown_frames", 0)),
        )
        max_primary_track_id = max((point.track_id for point in filtered_points), default=0)
        secondary_kinetic_added_points, secondary_kinetic_dedupe_meta = dedupe_secondary_track_points(
            filtered_points,
            secondary_kinetic_points,
            max_overlap_distance_px=float(
                secondary_detection_cfg.get("kinetic_dedupe_max_overlap_distance_px", 45.0)
            ),
            min_overlap_frames=int(secondary_detection_cfg.get("kinetic_dedupe_min_overlap_frames", 4)),
            min_overlap_ratio=float(secondary_detection_cfg.get("kinetic_dedupe_min_overlap_ratio", 0.6)),
            secondary_track_id_offset=max_primary_track_id + 1,
        )
        filtered_points = sorted(
            [*filtered_points, *secondary_kinetic_added_points],
            key=lambda point: (point.track_id, point.frame),
        )
    filtered_points, final_direction_filter_meta = _filter_points_excluding_directions(
        filtered_points,
        entry_exit_mask,
        {"outside"},
    )
    perf.record("postprocess_stage", perf_counter() - postprocess_started, executions=1)
    progress.complete_stage("postprocess", detail="postprocess done")

    fast_events = []
    fast_event_outputs: Dict[str, str] = {
        "fast_events_csv": "",
        "fast_tracks_csv": "",
        "fast_events_overlay_png": "",
    }
    if fast_events_enabled:
        progress.start_stage("fast_events")
        fast_started = perf_counter()
        fast_events = reconstruct_fast_events(
            merged_points,
            track_assessments,
            cfg.get("fast_events", {}),
            frame_shape=(meta.height, meta.width),
        )
        used_fast_source_ids = {
            track_id
            for event in fast_events
            for track_id in event.source_track_ids
        }
        fast_events.extend(
            reconstruct_fast_events_from_candidates(
                track_assessments,
                cfg.get("fast_events", {}),
                fps=meta.fps,
                video_id=meta.video_id,
                frame_shape=(meta.height, meta.width),
                first_event_id=int(cfg["fast_events"].get("first_event_id", 10001)) + len(fast_events),
                exclude_source_track_ids=used_fast_source_ids,
            )
        )
        fast_events_csv_path = out_dir / "fast_events.csv"
        fast_tracks_csv_path = out_dir / "fast_tracks.csv"
        fast_overlay_path = out_dir / "fast_events_overlay.png"
        write_fast_events_csv(fast_events_csv_path, fast_events)
        write_fast_tracks_csv(fast_tracks_csv_path, fast_events, track_assessments)
        fast_overlay = render_fast_events_overlay(
            background,
            fast_events,
            line_thickness=int(cfg["fast_events"].get("overlay_line_thickness", 3)),
            start_radius=int(cfg["fast_events"].get("overlay_start_radius", 6)),
            alpha=float(cfg["fast_events"].get("overlay_alpha", 1.0)),
        )
        cv2.imwrite(str(fast_overlay_path), fast_overlay)
        fast_event_outputs = {
            "fast_events_csv": str(fast_events_csv_path.resolve()),
            "fast_tracks_csv": str(fast_tracks_csv_path.resolve()),
            "fast_events_overlay_png": str(fast_overlay_path.resolve()),
        }
        perf.record("fast_events_stage", perf_counter() - fast_started, executions=1)
        progress.complete_stage("fast_events", detail=f"fast events {len(fast_events)}")

    heatmap_events = []
    heatmap_event_outputs: Dict[str, str] = {
        "heatmap_events_csv": "",
        "heatmap_tracks_csv": "",
        "heatmap_events_overlay_png": "",
    }
    if heatmap_events_enabled:
        progress.start_stage("heatmap_events")
        heatmap_started = perf_counter()
        heatmap_events = reconstruct_heatmap_events(
            input_video,
            fast_events,
            cfg.get("heatmap_events", {}),
            fps=meta.fps,
            frame_count=meta.frame_count,
        )
        heatmap_events_csv_path = out_dir / "heatmap_events.csv"
        heatmap_tracks_csv_path = out_dir / "heatmap_tracks.csv"
        heatmap_overlay_path = out_dir / "heatmap_events_overlay.png"
        write_heatmap_events_csv(heatmap_events_csv_path, heatmap_events)
        write_heatmap_tracks_csv(heatmap_tracks_csv_path, heatmap_events)
        heatmap_overlay = render_heatmap_events_overlay(
            background,
            heatmap_events,
            line_thickness=int(cfg["heatmap_events"].get("overlay_line_thickness", 5)),
            alpha=float(cfg["heatmap_events"].get("overlay_alpha", 1.0)),
        )
        cv2.imwrite(str(heatmap_overlay_path), heatmap_overlay)
        heatmap_event_outputs = {
            "heatmap_events_csv": str(heatmap_events_csv_path.resolve()),
            "heatmap_tracks_csv": str(heatmap_tracks_csv_path.resolve()),
            "heatmap_events_overlay_png": str(heatmap_overlay_path.resolve()),
        }
        perf.record("heatmap_events_stage", perf_counter() - heatmap_started, executions=1)
        progress.complete_stage("heatmap_events", detail=f"heatmap events {len(heatmap_events)}")

    progress.start_stage("exports_core")
    tracks_csv_path = out_dir / "tracks.csv"
    _write_tracks_csv(tracks_csv_path, filtered_points)
    track_candidates_csv_path = out_dir / "track_candidates.csv"
    if bool(cfg["tracking"].get("export_track_candidates", False)):
        _write_track_candidates_csv(track_candidates_csv_path, track_assessments)

    track_dedup_outputs: Dict[str, str] = {
        "track_deduplication_csv": "",
        "track_deduplication_json": "",
        "track_deduplication_overlay_png": "",
    }
    if track_dedup_result.enabled:
        track_dedup_csv_path = out_dir / "track_deduplication.csv"
        track_dedup_json_path = out_dir / "track_deduplication.json"
        track_dedup_overlay_path = out_dir / "track_deduplication_overlay.png"
        write_track_deduplication_csv(track_dedup_csv_path, track_dedup_result.rows)
        write_track_deduplication_json(track_dedup_json_path, track_dedup_result)
        track_dedup_overlay = render_track_deduplication_overlay(
            background,
            merged_points,
            track_dedup_result.rows,
        )
        cv2.imwrite(str(track_dedup_overlay_path), track_dedup_overlay)
        track_dedup_outputs = {
            "track_deduplication_csv": str(track_dedup_csv_path.resolve()),
            "track_deduplication_json": str(track_dedup_json_path.resolve()),
            "track_deduplication_overlay_png": str(track_dedup_overlay_path.resolve()),
        }

    out_cfg_export = cfg.get("output", {})
    smoothing_on = bool(out_cfg_export.get("trajectory_smoothing_enabled", False))
    ts_window = int(out_cfg_export.get("trajectory_smoothing_window", 5))
    smoothed_points: List[TrackPoint] | None = None
    if smoothing_on:
        smoothed_points = smooth_track_points(filtered_points, ts_window)

    events_csv_path = out_dir / "events.csv"
    points_for_events = smoothed_points if smoothed_points is not None else filtered_points
    _write_events_csv(events_csv_path, points_for_events, entry_exit_mask)
    tracks_svg_path = out_dir / "tracks.svg"
    export_tracks_svg(
        tracks_svg_path,
        width=meta.width,
        height=meta.height,
        points=filtered_points,
        line_thickness=int(cfg["output"]["overlay_line_thickness"]),
        start_radius=int(cfg["output"]["overlay_start_radius"]),
        alpha=float(cfg["output"].get("overlay_alpha", 1.0)),
        draw_track_labels=bool(cfg["output"].get("overlay_draw_track_labels", False)),
        draw_track_labels_at_end=bool(cfg["output"].get("overlay_draw_track_labels_at_end", False)),
        label_font_scale=float(cfg["output"].get("overlay_label_font_scale", 0.5)),
        label_thickness=int(cfg["output"].get("overlay_label_thickness", 1)),
        valid_region_mask=valid_mask,
        direction_mask=entry_exit_mask,
    )
    tracks_render_json_path = out_dir / "tracks_render.json"
    export_tracks_render_json(
        tracks_render_json_path,
        width=meta.width,
        height=meta.height,
        points=filtered_points,
        valid_region_mask=valid_mask,
        direction_mask=entry_exit_mask,
    )

    overlay_line_t = int(cfg["output"]["overlay_line_thickness"])
    overlay_start_r = int(cfg["output"]["overlay_start_radius"])
    overlay_alpha_v = float(cfg["output"].get("overlay_alpha", 1.0))
    overlay_lbl = bool(cfg["output"].get("overlay_draw_track_labels", False))
    overlay_lbl_end = bool(cfg["output"].get("overlay_draw_track_labels_at_end", False))
    overlay_lbl_scale = float(cfg["output"].get("overlay_label_font_scale", 0.5))
    overlay_lbl_th = int(cfg["output"].get("overlay_label_thickness", 1))

    if smoothed_points is not None:
        overlay_raw = render_tracks_overlay(
            background_gray=background,
            points=filtered_points,
            line_thickness=overlay_line_t,
            start_radius=overlay_start_r,
            alpha=overlay_alpha_v,
            draw_track_labels=overlay_lbl,
            draw_track_labels_at_end=overlay_lbl_end,
            label_font_scale=overlay_lbl_scale,
            label_thickness=overlay_lbl_th,
        )
        cv2.imwrite(str(out_dir / "tracks_overlay_raw.png"), overlay_raw)
        overlay_smoothed = render_tracks_overlay(
            background_gray=background,
            points=smoothed_points,
            line_thickness=overlay_line_t,
            start_radius=overlay_start_r,
            alpha=overlay_alpha_v,
            draw_track_labels=overlay_lbl,
            draw_track_labels_at_end=overlay_lbl_end,
            label_font_scale=overlay_lbl_scale,
            label_thickness=overlay_lbl_th,
        )
        cv2.imwrite(str(out_dir / "tracks_overlay_smoothed.png"), overlay_smoothed)
        overlay = overlay_raw
    else:
        overlay = render_tracks_overlay(
            background_gray=background,
            points=filtered_points,
            line_thickness=overlay_line_t,
            start_radius=overlay_start_r,
            alpha=overlay_alpha_v,
            draw_track_labels=overlay_lbl,
            draw_track_labels_at_end=overlay_lbl_end,
            label_font_scale=overlay_lbl_scale,
            label_thickness=overlay_lbl_th,
        )
    overlay_path = out_dir / "tracks_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    detection_debug_outputs: Dict[str, str] = {
        "primary_detections_overlay_png": "",
        "secondary_detections_overlay_png": "",
    }
    if secondary_blob_detection_enabled:
        primary_detection_overlay_path = out_dir / "primary_detections_overlay.png"
        secondary_detection_overlay_path = out_dir / "secondary_detections_overlay.png"
        primary_detection_overlay = render_detections_overlay(
            background,
            primary_detection_debug,
            color=(0, 200, 255),
            alpha=0.85,
            line_thickness=1,
            point_radius=2,
        )
        secondary_detection_overlay = render_detections_overlay(
            background,
            secondary_detection_debug,
            color=(255, 80, 80),
            alpha=0.85,
            line_thickness=1,
            point_radius=2,
        )
        cv2.imwrite(str(primary_detection_overlay_path), primary_detection_overlay)
        cv2.imwrite(str(secondary_detection_overlay_path), secondary_detection_overlay)
        detection_debug_outputs = {
            "primary_detections_overlay_png": str(primary_detection_overlay_path.resolve()),
            "secondary_detections_overlay_png": str(secondary_detection_overlay_path.resolve()),
        }
    secondary_kinetic_outputs: Dict[str, str] = {
        "secondary_kinetic_tracks_csv": "",
        "secondary_kinetic_tracks_overlay_png": "",
        "secondary_kinetic_added_tracks_csv": "",
        "secondary_kinetic_added_tracks_overlay_png": "",
    }
    if secondary_kinetic_enabled:
        kinetic_tracks_csv_path = out_dir / "secondary_kinetic_tracks.csv"
        kinetic_overlay_path = out_dir / "secondary_kinetic_tracks_overlay.png"
        kinetic_added_csv_path = out_dir / "secondary_kinetic_added_tracks.csv"
        kinetic_added_overlay_path = out_dir / "secondary_kinetic_added_tracks_overlay.png"
        _write_tracks_csv(kinetic_tracks_csv_path, secondary_kinetic_points)
        _write_tracks_csv(kinetic_added_csv_path, secondary_kinetic_added_points)
        kinetic_overlay = render_tracks_overlay(
            background_gray=background,
            points=secondary_kinetic_points,
            line_thickness=overlay_line_t,
            start_radius=overlay_start_r,
            alpha=overlay_alpha_v,
            draw_track_labels=overlay_lbl,
            draw_track_labels_at_end=overlay_lbl_end,
            label_font_scale=overlay_lbl_scale,
            label_thickness=overlay_lbl_th,
        )
        kinetic_added_overlay = render_tracks_overlay(
            background_gray=background,
            points=secondary_kinetic_added_points,
            line_thickness=overlay_line_t,
            start_radius=overlay_start_r,
            alpha=overlay_alpha_v,
            draw_track_labels=overlay_lbl,
            draw_track_labels_at_end=overlay_lbl_end,
            label_font_scale=overlay_lbl_scale,
            label_thickness=overlay_lbl_th,
        )
        cv2.imwrite(str(kinetic_overlay_path), kinetic_overlay)
        cv2.imwrite(str(kinetic_added_overlay_path), kinetic_added_overlay)
        secondary_kinetic_outputs = {
            "secondary_kinetic_tracks_csv": str(kinetic_tracks_csv_path.resolve()),
            "secondary_kinetic_tracks_overlay_png": str(kinetic_overlay_path.resolve()),
            "secondary_kinetic_added_tracks_csv": str(kinetic_added_csv_path.resolve()),
            "secondary_kinetic_added_tracks_overlay_png": str(kinetic_added_overlay_path.resolve()),
        }
    progress.complete_stage("exports_core", detail="csv and overlay exported")

    flight_trails_output = ""
    if flight_trails_enabled:
        progress.start_stage("flight_trails")
        trail_points = smoothed_points if smoothed_points is not None else filtered_points
        trail_video_name = str(cfg.get("flight_trails", {}).get("video_filename", "flight_trails_overlay.mp4"))
        trail_video_path = out_dir / trail_video_name
        flight_trails_output = export_realtime_trails_video(
            input_video=input_video,
            output_path=trail_video_path,
            points=trail_points,
            frame_size=(meta.width, meta.height),
            fps=meta.fps,
            cfg=cfg.get("flight_trails", {}),
        )
        progress.complete_stage("flight_trails", detail="trail video exported")

    track_clip_outputs: Dict[str, str] = {}
    if export_track_clips_enabled:
        progress.start_stage("track_clips")
        track_clip_outputs = _export_track_clips(
            input_video=input_video,
            output_dir=out_dir,
            points=filtered_points,
            fps=meta.fps,
            frame_size=(meta.width, meta.height),
            clips_subdir=str(cfg["output"].get("track_clips_subdir", "track_clips")),
            pad_frames=max(0, int(cfg["output"].get("track_clips_padding_frames", 0))),
        )
        progress.complete_stage("track_clips", detail="track clips exported")

    perf.finish()
    perf_summary = perf.summary()
    perf_summary["pipeline_total_wall_sec"] = max(0.0, perf_counter() - pipeline_started)
    candidate_recall_metrics = _candidate_recall_metrics(track_assessments)
    motion_rescued_points = sum(int(row.get("points_added", 0)) for row in motion_candidate_rescues)
    candidate_total_points = (
        int(candidate_recall_metrics.get("track_candidate_points_accepted", 0))
        + int(candidate_recall_metrics.get("track_candidate_points_rejected", 0))
    )
    effective_orphan_points = max(
        0,
        int(candidate_recall_metrics.get("track_candidate_points_rejected", 0)) - motion_rescued_points,
    )
    candidate_recall_metrics["track_candidate_points_rescued_motion"] = motion_rescued_points
    candidate_recall_metrics["track_candidate_orphan_detection_pct_after_rescue"] = (
        100.0 * effective_orphan_points / candidate_total_points if candidate_total_points else 0.0
    )
    cave_zones_diagnostics_summary = _diagnose_entry_exit_zone(
        cave_zones_outputs.get("cave_zones_diagnostics_json", ""),
        final_points=filtered_points,
        candidate_points=merged_points,
        assessments=track_assessments,
        entry_exit_mask=entry_exit_mask,
        valid_gate_mask=valid_gate_mask,
    )
    if cave_zones_diagnostics_summary:
        cave_zones_meta["track_endpoint_diagnostics"] = cave_zones_diagnostics_summary

    cleanup_intermediate_outputs = bool(cfg["output"].get("cleanup_intermediate_outputs", True))
    trajectory_smoothing_meta = {
        "enabled": smoothing_on,
        "window": ts_window,
    }
    overlay_smoothing_paths: Dict[str, str] = {}
    if smoothed_points is not None:
        overlay_smoothing_paths = {
            "tracks_overlay_raw_png": str((out_dir / "tracks_overlay_raw.png").resolve()),
            "tracks_overlay_smoothed_png": str((out_dir / "tracks_overlay_smoothed.png").resolve()),
        }

    motion_heatmap_overlay_path = out_dir / "motion_heatmap_overlay.png"
    _save_motion_heatmap_overlay(background, motion_heatmap, motion_heatmap_overlay_path)

    outputs_payload = {
        "background_png": str(background_path.resolve()),
        "tracks_csv": str(tracks_csv_path.resolve()),
        "events_csv": str(events_csv_path.resolve()),
        "tracks_svg": str(tracks_svg_path.resolve()),
        "tracks_render_json": str(tracks_render_json_path.resolve()),
        "tracks_overlay_png": str(overlay_path.resolve()),
        "motion_heatmap_overlay_png": str(motion_heatmap_overlay_path.resolve()),
        **cave_zones_outputs,
        **cavemark_outputs,
        **detection_debug_outputs,
        **track_dedup_outputs,
        **secondary_kinetic_outputs,
        "flight_trails_overlay_video": flight_trails_output,
        "track_candidates_csv": (
            str(track_candidates_csv_path.resolve())
            if bool(cfg["tracking"].get("export_track_candidates", False))
            else ""
        ),
        **overlay_smoothing_paths,
        **fast_event_outputs,
        **heatmap_event_outputs,
        "track_clips": track_clip_outputs,
        **valid_region_outputs,
        **vegetation_outputs,
    }
    cleaned_output_keys = _cleanup_output_files(outputs_payload, enabled=cleanup_intermediate_outputs)

    meta_payload = {
        "trajectory_smoothing": trajectory_smoothing_meta,
        "video": {
            "input_path": str(Path(input_video).resolve()),
            "video_id": meta.video_id,
            "fps": meta.fps,
            "frame_count_reported": meta.frame_count,
            "width": meta.width,
            "height": meta.height,
        },
        "parameters": cfg,
        "background": {
            "source": background_source,
            "input_image": str(Path(background_input).resolve()) if background_input else "",
            "context_start_sec": float(cfg["background"].get("context_start_sec", 0.0)),
            "context_duration_sec": float(cfg["background"].get("context_duration_sec", -1.0)),
        },
        "valid_region": valid_region_meta,
        "cave_zones": cave_zones_meta,
        "cavemark": cavemark_meta,
        "entry_exit_zone_selection": entry_exit_zone_selection_meta,
        "metrics": {
            **_build_metrics(filtered_points, frame_processed),
            "detections_raw_total": detections_raw_total,
            "detections_after_burst_total": detections_after_burst_total,
            "detections_per_frame_raw": detections_raw_total / frame_processed if frame_processed else 0.0,
            "detections_per_frame_after_burst": detections_after_burst_total / frame_processed if frame_processed else 0.0,
            "frames_with_raw_detections": frames_with_raw_detections,
            "frames_with_detections_after_burst": frames_with_detections_after_burst,
            "frames_suppressed_temporal_burst": suppressed_burst_frames,
            "tracks_merged_auto": len(merges_applied),
            "tracks_rescued_crossing_continuations": len(crossing_continuation_rescues),
            "tracks_rescued_motion_candidates": len(motion_candidate_rescues),
            "tracks_deduped_coexisting": len(coexisting_duplicate_track_ids),
            "entry_exit_zone_source": entry_exit_zone_source,
            **vegetation_exclusion_meta,
            "cave_zones_rejected_by_gate_touching_entry_exit_zone": int(
                cave_zones_diagnostics_summary.get("rejected_by_gate_touching_entry_exit_zone", 0)
            ),
            **final_direction_filter_meta,
            "track_deduplication_groups": track_dedup_result.groups_total,
            "track_deduplication_pairs": track_dedup_result.pairs_total,
            "track_deduplication_tracks_discarded": track_dedup_result.tracks_discarded,
            "track_deduplication_tracks_merged": track_dedup_result.tracks_merged,
            **candidate_recall_metrics,
            "secondary_detection_primary_detections": secondary_primary_total,
            "secondary_detection_raw_detections": secondary_raw_total,
            "secondary_detection_added_detections": secondary_added_total,
            "secondary_detection_duplicate_detections": secondary_duplicate_total,
            "secondary_kinetic_tracks_raw": int(secondary_kinetic_dedupe_meta.get("secondary_tracks_raw", 0)),
            "secondary_kinetic_tracks_added": int(secondary_kinetic_dedupe_meta.get("secondary_tracks_added", 0)),
            "secondary_kinetic_tracks_duplicate": int(secondary_kinetic_dedupe_meta.get("secondary_tracks_duplicate", 0)),
            "secondary_kinetic_tracks_rejected_by_mask": int(
                secondary_kinetic_mask_meta.get("tracks_rejected_by_mask_filter", 0)
            ),
            "secondary_kinetic_temporal_burst_tracks_removed": int(
                secondary_kinetic_burst_meta.get("temporal_burst_tracks_removed", 0)
            ),
            **background_runtime_stats,
            **detection_runtime_stats,
            **secondary_detection_runtime_stats,
        },
        "execution": {
            "requested_device": execution_plan.requested_device,
            "selected_device": execution_plan.selected_device,
            "gpu_available": execution_plan.gpu_available,
            "strict_parity": strict_parity,
            "selection_reason": execution_plan.reason,
        },
        "performance": perf_summary,
        "outputs": outputs_payload,
        "cleanup": {
            "enabled": cleanup_intermediate_outputs,
            "removed_output_keys": cleaned_output_keys,
        },
        "postprocess": {
            "auto_merge_enabled": bool(cfg["tracking"].get("auto_merge_suggested", False)),
            "auto_merges_applied": merges_applied,
            "crossing_continuation_rescues": crossing_continuation_rescues,
            "motion_candidate_rescues": motion_candidate_rescues,
            "track_deduplication": {
                "enabled": track_dedup_result.enabled,
                "groups_total": track_dedup_result.groups_total,
                "pairs_total": track_dedup_result.pairs_total,
                "tracks_discarded": track_dedup_result.tracks_discarded,
                "tracks_merged": track_dedup_result.tracks_merged,
                "decisions": track_dedup_result.rows,
                "pairs": track_dedup_result.pairs,
            },
            "track_candidates_total": len(track_assessments),
            "track_candidates_kept": sum(1 for row in track_assessments if row["accepted"]),
            "track_candidates_rejected": sum(1 for row in track_assessments if not row["accepted"]),
            "track_candidates_top_rejections": dict(
                Counter(
                    reason
                    for row in track_assessments
                    for reason in str(row["reject_reasons"]).split(";")
                    if reason
                )
            ),
        },
        "secondary_kinetic": {
            **secondary_kinetic_meta,
            **secondary_kinetic_mask_meta,
            **secondary_kinetic_burst_meta,
            **secondary_kinetic_dedupe_meta,
        },
        "fast_events": {
            "enabled": fast_events_enabled,
            "events_total": len(fast_events),
            "source_tracks_total": sum(len(event.source_track_ids) for event in fast_events),
        },
        "heatmap_events": {
            "enabled": heatmap_events_enabled,
            "events_total": len(heatmap_events),
            "source_fast_events_total": len({event.source_fast_event_id for event in heatmap_events}),
        },
        "flight_trails": {
            "enabled": flight_trails_enabled,
            "source_track_points": len(smoothed_points) if smoothed_points is not None else len(filtered_points),
            "output_video": flight_trails_output,
        },
        "track_quality": compute_track_quality(
            filtered_points,
            fps=meta.fps,
            merges_applied=merges_applied,
        ),
    }

    meta_path = out_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta_payload, handle, indent=2)

    progress.finish()
    return meta_payload

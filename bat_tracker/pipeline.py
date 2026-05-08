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
from .compute import build_execution_plan
from .config import load_config
from .detection import build_detection_context
from .detection import detect_foreground_blobs
from .detection import detect_ffdiff_blobs
from .detection import TemporalAccumulator
from .fast_events import reconstruct_fast_events
from .fast_events import reconstruct_fast_events_from_candidates
from .fast_events import render_fast_events_overlay
from .fast_events import write_fast_events_csv
from .fast_events import write_fast_tracks_csv
from .heatmap_events import reconstruct_heatmap_events
from .heatmap_events import render_heatmap_events_overlay
from .heatmap_events import write_heatmap_events_csv
from .heatmap_events import write_heatmap_tracks_csv
from .perf import PerformanceCollector
from .render import export_tracks_render_json, export_tracks_svg, render_tracks_overlay
from .track_smoothing import smooth_track_points
from .tracker import GreedyTracker, TrackPoint, TrackingDebugRow
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
    "rescue_mode",
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

TRACKING_DEBUG_CSV_COLUMNS = [
    "frame",
    "track_id",
    "state",
    "pred_x",
    "pred_y",
    "gate_px",
    "matched",
    "det_x",
    "det_y",
    "match_dist",
    "det_score",
    "bbox_w",
    "bbox_h",
    "speed_px_sec",
    "missed_before",
    "missed_after",
    "birth_reason",
    "kill_reason",
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


def _deduplicate_detections(detections: list, proximity_px: float = 30.0) -> list:
    if len(detections) <= 1:
        return detections
    prox_sq = proximity_px * proximity_px
    dets = sorted(detections, key=lambda d: (-d.area, -d.score))
    kept: list = []
    for d in dets:
        if any((d.x - k.x) ** 2 + (d.y - k.y) ** 2 < prox_sq for k in kept):
            continue
        kept.append(d)
    return kept


def _point_in_mask(point: TrackPoint, mask: np.ndarray) -> bool:
    xi = int(round(point.x))
    yi = int(round(point.y))
    if yi < 0 or yi >= mask.shape[0] or xi < 0 or xi >= mask.shape[1]:
        return False
    return bool(mask[yi, xi] > 0)


def _build_valid_region_gate_mask(valid_mask: np.ndarray | None, tracking_cfg: Dict) -> np.ndarray | None:
    gate_mask = valid_mask
    valid_region_gate_dilate_px = max(0, int(tracking_cfg.get("valid_region_gate_dilate_px", 0)))
    if gate_mask is not None and valid_region_gate_dilate_px > 0:
        k = 2 * valid_region_gate_dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        gate_mask = cv2.dilate(gate_mask, kernel, iterations=1)
    return gate_mask


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
) -> tuple[List[TrackPoint], List[dict]]:
    def _ratio(value: float, threshold: float) -> float:
        if threshold <= 0.0:
            return 1.0
        return min(1.0, max(0.0, value / threshold))

    min_track_length_cfg = int(tracking_cfg.get("min_track_length", 1))
    min_track_duration_sec = float(tracking_cfg.get("min_track_duration_sec", 0.0))
    min_track_length_from_sec = int(ceil(max(0.0, min_track_duration_sec) * max(1e-6, fps)))
    min_track_length = max(min_track_length_cfg, min_track_length_from_sec)
    min_track_displacement = float(tracking_cfg.get("min_track_displacement", 0.0))
    min_track_path_length = float(tracking_cfg.get("min_track_path_length", 0.0))
    min_track_straightness = float(tracking_cfg.get("min_track_straightness", 0.0))
    require_start_or_end_in_valid_region = bool(tracking_cfg.get("require_start_or_end_in_valid_region", False))
    gate_mask = _build_valid_region_gate_mask(valid_mask, tracking_cfg)
    strong_short_score_min = 0.9

    fast_track_enabled = bool(tracking_cfg.get("fast_track_enabled", False))
    fast_track_speed_px_sec = float(tracking_cfg.get("fast_track_speed_px_sec", 500))
    fast_track_min_length = int(tracking_cfg.get("fast_track_min_length", 3))
    fast_track_min_duration_sec = float(tracking_cfg.get("fast_track_min_duration_sec", 0.08))
    fast_track_min_displacement = float(tracking_cfg.get("fast_track_min_displacement", 40))
    fast_track_min_path_length = float(tracking_cfg.get("fast_track_min_path_length", 60))
    fast_track_min_straightness = float(tracking_cfg.get("fast_track_min_straightness", 0.65))
    fast_track_require_gate_touch = bool(tracking_cfg.get("fast_track_require_gate_touch", True))
    fast_track_gate_margin = int(tracking_cfg.get("fast_track_gate_crossing_margin_px", 35))

    fast_gate_mask: np.ndarray | None = None
    if fast_track_enabled and gate_mask is not None and fast_track_gate_margin > 0:
        k = 2 * fast_track_gate_margin + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        fast_gate_mask = cv2.dilate(gate_mask, kernel, iterations=1)

    def _track_touches_mask(tps: List[TrackPoint], mask: np.ndarray) -> bool:
        return any(_point_in_mask(tp, mask) for tp in tps)

    by_track: Dict[int, List[TrackPoint]] = defaultdict(list)
    for point in points:
        by_track[point.track_id].append(point)

    filtered: List[TrackPoint] = []
    assessments: List[dict] = []
    for track_points in by_track.values():
        track_points = sorted(track_points, key=lambda p: p.frame)
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

        s_in = None
        e_in = None
        direction = "unknown"
        if require_start_or_end_in_valid_region and gate_mask is not None:
            s_in = _point_in_mask(start, gate_mask)
            e_in = _point_in_mask(end, gate_mask)
            direction = _classify_direction_full(s_in, e_in, track_points, valid_mask)
            if not (s_in or e_in):
                reject_reasons.append("valid_region_gate")
        elif valid_mask is not None:
            s_in = _point_in_mask(start, valid_mask)
            e_in = _point_in_mask(end, valid_mask)
            direction = _classify_direction_full(s_in, e_in, track_points, valid_mask, valid_mask.shape[:2])

        accepted = not reject_reasons
        rescue_mode = ""
        if not accepted:
            reasons_set = set(reject_reasons)
            if (
                reasons_set.issubset({"min_track_length"})
                and len(track_points) >= min_track_length_from_sec
                and score >= strong_short_score_min
            ):
                accepted = True
                reject_reasons = []
                rescue_mode = "strong_short"

        if not accepted and fast_track_enabled:
            is_fast = mean_speed >= fast_track_speed_px_sec
            if not is_fast:
                pass
            elif not fast_track_require_gate_touch:
                touches_gate = True
            elif fast_gate_mask is not None:
                touches_gate = _track_touches_mask(track_points, fast_gate_mask)
            else:
                touches_gate = False
            if is_fast and touches_gate:
                fast_ok = True
                fast_reasons: list[str] = []
                if len(track_points) < fast_track_min_length and duration < fast_track_min_duration_sec:
                    fast_ok = False
                    fast_reasons.append("fast_track_length")
                if displacement < fast_track_min_displacement:
                    fast_ok = False
                    fast_reasons.append("fast_track_displacement")
                if path_length < fast_track_min_path_length:
                    fast_ok = False
                    fast_reasons.append("fast_track_path_length")
                if straightness < fast_track_min_straightness:
                    fast_ok = False
                    fast_reasons.append("fast_track_straightness")
                if fast_ok:
                    accepted = True
                    reject_reasons = []
                    rescue_mode = "fast_track"

        assessments.append({
            "video_id": start.video_id,
            "track_id": start.track_id,
            "accepted": accepted,
            "reject_reasons": ";".join(reject_reasons),
            "score": round(score, 4),
            "rescue_mode": rescue_mode,
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


def _auto_merge_track_points(points: List[TrackPoint], tracking_cfg: Dict) -> tuple[List[TrackPoint], List[Dict]]:
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
    max_endpoint_dist = float(tracking_cfg.get("merge_max_endpoint_distance", 80.0))
    min_overlap_common = int(tracking_cfg.get("merge_overlap_min_common_frames", 3))
    max_overlap_mean_dist = float(tracking_cfg.get("merge_overlap_max_mean_distance", 60.0))
    min_overlap_cos = float(tracking_cfg.get("merge_overlap_min_direction_cosine", 0.8))
    local_overlap_min_cos = max(0.65, min_overlap_cos - 0.15)
    proximity_override_dist = float(tracking_cfg.get("merge_overlap_proximity_override_distance", 0.0))

    parent: Dict[int, int] = {track_id: track_id for track_id in by_track}

    def find(track_id: int) -> int:
        while parent[track_id] != track_id:
            parent[track_id] = parent[parent[track_id]]
            track_id = parent[track_id]
        return track_id

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

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
        union(track_a_id, track_b_id)
        merged_to = min(find(track_a_id), find(track_b_id))
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


def _merge_fast_overlap_track_points(
    points: List[TrackPoint],
    tracking_cfg: Dict,
) -> tuple[List[TrackPoint], List[Dict]]:
    if not bool(tracking_cfg.get("merge_fast_overlap_enabled", False)):
        return points, []

    max_mean_distance = float(tracking_cfg.get("merge_fast_overlap_max_mean_distance", 35))
    min_common_frames = int(tracking_cfg.get("merge_fast_overlap_min_common_frames", 2))
    max_gap_frames = int(tracking_cfg.get("merge_fast_handoff_max_gap_frames", 3))
    projection_tolerance = float(tracking_cfg.get("merge_fast_handoff_projection_tolerance_px", 45))

    by_track: Dict[int, List[TrackPoint]] = defaultdict(list)
    for point in points:
        by_track[point.track_id].append(point)
    for track_id in by_track:
        by_track[track_id] = sorted(by_track[track_id], key=lambda p: p.frame)

    if len(by_track) < 2:
        return points, []

    track_ids = sorted(by_track.keys())
    parent: Dict[int, int] = {tid: tid for tid in track_ids}

    def find(tid: int) -> int:
        while parent[tid] != tid:
            parent[tid] = parent[parent[tid]]
            tid = parent[tid]
        return tid

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    merges_applied: List[Dict] = []

    track_data: Dict[int, Dict] = {}
    for tid in track_ids:
        pts = by_track[tid]
        track_data[tid] = {
            "start": pts[0],
            "end": pts[-1],
            "frames": {p.frame: p for p in pts},
        }

    n = len(track_ids)
    sf_arr = [track_data[tid]["start"].frame for tid in track_ids]
    ef_arr = [track_data[tid]["end"].frame for tid in track_ids]
    ex_arr = [track_data[tid]["end"].x for tid in track_ids]
    ey_arr = [track_data[tid]["end"].y for tid in track_ids]
    sx_arr = [track_data[tid]["start"].x for tid in track_ids]
    sy_arr = [track_data[tid]["start"].y for tid in track_ids]
    max_gap_sq = (max_gap_frames + 1) * (max_gap_frames + 1)

    candidate_pairs: List[Tuple[int, int]] = []
    sorted_pos = sorted(range(n), key=lambda i: sf_arr[i])
    heap: List[Tuple[int, int]] = []
    for b_pos in sorted_pos:
        b_sf = sf_arr[b_pos]
        b_id = track_ids[b_pos]
        while heap and heap[0][0] < b_sf - max_gap_frames:
            heapq.heappop(heap)
        for a_ef, a_pos in heap:
            a_id = track_ids[a_pos]
            candidate_pairs.append((min(a_id, b_id), max(a_id, b_id)))
        heapq.heappush(heap, (ef_arr[b_pos], b_pos))

    for track_a_id, track_b_id in candidate_pairs:
        td_a = track_data[track_a_id]
        td_b = track_data[track_b_id]
        a_frames = td_a["frames"]
        a_end = td_a["end"]
        a_start = td_a["start"]
        b_start = td_b["start"]
        b_end = td_b["end"]

        reason = None
        reason_data: Dict[str, float | int] = {}

        if a_end.frame < b_start.frame:
            gap = b_start.frame - a_end.frame
            if gap <= max_gap_frames:
                proj_dist = hypot(
                    (a_end.x + (a_end.x - a_start.x)) - b_start.x,
                    (a_end.y + (a_end.y - a_start.y)) - b_start.y,
                )
                if proj_dist <= projection_tolerance:
                    reason = "handoff_fast"
                    reason_data = {"gap_frames": gap, "projection_distance_px": proj_dist}
        elif b_end.frame < a_start.frame:
            gap = a_start.frame - b_end.frame
            if gap <= max_gap_frames:
                proj_dist = hypot(
                    (b_end.x + (b_end.x - b_start.x)) - a_start.x,
                    (b_end.y + (b_end.y - b_start.y)) - a_start.y,
                )
                if proj_dist <= projection_tolerance:
                    reason = "handoff_fast"
                    reason_data = {"gap_frames": gap, "projection_distance_px": proj_dist}
        else:
            b_frames = td_b["frames"]
            common_frames = sorted(set(a_frames.keys()).intersection(b_frames.keys()))
            if len(common_frames) >= min_common_frames:
                distances = []
                for frame in common_frames:
                    pa = a_frames[frame]
                    pb = b_frames[frame]
                    distances.append(hypot(pa.x - pb.x, pa.y - pb.y))
                mean_distance = sum(distances) / len(distances)
                if mean_distance <= max_mean_distance:
                    reason = "overlap_fast"
                    reason_data = {
                        "common_frames": len(common_frames),
                        "mean_distance": mean_distance,
                    }
                elif reason is None:
                    a_vec = (a_end.x - a_start.x, a_end.y - a_start.y)
                    b_vec = (b_end.x - b_start.x, b_end.y - b_start.y)
                    dir_cos = _vector_cosine(a_vec, b_vec)
                    if dir_cos is not None:
                        if dir_cos >= 0.98:
                            mult = 4.0
                        elif dir_cos >= 0.95:
                            mult = 2.0
                        else:
                            mult = 0.0
                        if mult > 0.0 and mean_distance <= max_mean_distance * mult:
                            reason = "overlap_fast_aligned"
                            reason_data = {
                                "common_frames": len(common_frames),
                                "mean_distance": mean_distance,
                                "direction_cosine": dir_cos,
                            }

        if reason is None:
            continue

        ra = find(track_a_id)
        rb = find(track_b_id)
        if ra == rb:
            continue
        union(track_a_id, track_b_id)
        merged_to = min(find(track_a_id), find(track_b_id))
        merges_applied.append({
            "track_a": track_a_id,
            "track_b": track_b_id,
            "merged_to": merged_to,
            "reason": reason,
            **reason_data,
        })

    remap: Dict[int, int] = {tid: find(tid) for tid in track_ids}
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
        best = max(candidates, key=lambda p: (p.area, -abs(p.vx) - abs(p.vy), -p.x, -p.y))
        consolidated.append(best)

    return consolidated, merges_applied


def _write_tracking_debug_csv(path: Path, debug_rows: list) -> None:
    if not debug_rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACKING_DEBUG_CSV_COLUMNS)
        writer.writeheader()
        for row in debug_rows:
            writer.writerow(asdict(row))


def compute_noise_mask(
    video_path: str,
    background: np.ndarray,
    cfg: Dict,
    fps: float,
    total_frames: int,
) -> np.ndarray | None:
    if not bool(cfg.get("enabled", False)):
        return None

    sample_frames = int(cfg.get("sample_frames", 200))
    noise_std_threshold = float(cfg.get("noise_std_threshold", 8.0))
    dilate_kernel_size = int(cfg.get("dilate_kernel", 15))
    dilate_iterations = int(cfg.get("dilate_iterations", 2))
    blur_kernel_cfg = int(cfg.get("blur_kernel", 0))

    bg_gray = background
    if bg_gray.ndim == 3:
        bg_gray = cv2.cvtColor(bg_gray, cv2.COLOR_BGR2GRAY)

    if blur_kernel_cfg > 1 and blur_kernel_cfg % 2 == 1:
        bg_proc = cv2.GaussianBlur(bg_gray, (blur_kernel_cfg, blur_kernel_cfg), 0)
    else:
        bg_proc = bg_gray

    height, width = bg_proc.shape
    step = max(1, total_frames // sample_frames)
    frame_indices = list(range(0, total_frames, step))[:sample_frames]

    sum_arr = np.zeros((height, width), dtype=np.float64)
    sum_sq_arr = np.zeros((height, width), dtype=np.float64)
    n = 0

    cap = cv2.VideoCapture(str(video_path))
    try:
        frame_idx = 0
        sample_set = set(frame_indices)
        while cap.isOpened() and n < len(frame_indices):
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx not in sample_set:
                frame_idx += 1
                continue

            if frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            if blur_kernel_cfg > 1 and blur_kernel_cfg % 2 == 1:
                gray_proc = cv2.GaussianBlur(gray, (blur_kernel_cfg, blur_kernel_cfg), 0)
            else:
                gray_proc = gray

            diff = cv2.absdiff(gray_proc, bg_proc).astype(np.float64)
            sum_arr += diff
            sum_sq_arr += diff * diff
            n += 1
            frame_idx += 1
    finally:
        cap.release()

    if n == 0:
        return None

    mean_arr = sum_arr / float(n)
    temporal_std = np.sqrt(np.maximum(0.0, sum_sq_arr / float(n) - mean_arr * mean_arr))

    noise_mask = (temporal_std > noise_std_threshold).astype(np.uint8) * 255

    if dilate_kernel_size > 0 and dilate_iterations > 0:
        k = max(3, dilate_kernel_size)
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        noise_mask = cv2.dilate(noise_mask, kernel, iterations=dilate_iterations)

    return noise_mask


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


def run_pipeline(input_video: str, output_dir: str, config_path: str | None = None) -> Dict:
    pipeline_started = perf_counter()
    cfg = load_config(config_path)
    tracking_cfg = cfg["tracking"]
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
    valid_region_meta: Dict = {"enabled": False}
    valid_region_outputs: Dict[str, str] = {}

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
        valid_gate_mask = _build_valid_region_gate_mask(valid_mask, tracking_cfg)
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

    tracker = GreedyTracker(
        max_distance=float(tracking_cfg["max_distance"]),
        max_missed=int(tracking_cfg["max_missed"]),
        fps=meta.fps,
        video_id=meta.video_id,
        adaptive_max_distance_enabled=bool(tracking_cfg.get("adaptive_max_distance_enabled", False)),
        adaptive_max_distance_base=float(tracking_cfg.get("adaptive_max_distance_base", 120.0)),
        adaptive_max_distance_speed_gain=float(tracking_cfg.get("adaptive_max_distance_speed_gain", 1.25)),
        adaptive_max_distance_bbox_gain=float(tracking_cfg.get("adaptive_max_distance_bbox_gain", 0.35)),
        adaptive_max_distance_cap=float(tracking_cfg.get("adaptive_max_distance_cap", 220.0)),
        two_stage_association_enabled=bool(tracking_cfg.get("two_stage_association_enabled", False)),
        two_stage_association_score_threshold=float(tracking_cfg.get("two_stage_association_score_threshold", 0.4)),
        export_debug=bool(tracking_cfg.get("export_tracking_debug", False)),
    )
    burst_gate = TemporalBurstGate.from_detection_cfg(cfg["detection"])
    detection_context = build_detection_context(background, cfg["detection"])

    temporal_cfg = cfg.get("temporal_accumulation", {})
    temporal_acc = TemporalAccumulator(
        cfg=temporal_cfg,
        background=background,
        blur_kernel=detection_context.blur_kernel,
        process_roi=(detection_context.process_x0, detection_context.process_y0,
                      detection_context.process_x1, detection_context.process_y1),
    )

    noise_mask: np.ndarray | None = None
    noise_mask_cfg = cfg.get("noise_mask", {})
    noise_mask_path = str(cfg.get("detection", {}).get("noise_mask_path", "")).strip()
    if noise_mask_path:
        noise_mask = cv2.imread(noise_mask_path, cv2.IMREAD_GRAYSCALE)
        if noise_mask is None:
            raise RuntimeError(f"Cannot load noise mask from: {noise_mask_path}")
        if noise_mask.shape != background.shape:
            noise_mask = cv2.resize(noise_mask, (background.shape[1], background.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
        _, noise_mask = cv2.threshold(noise_mask, 127, 255, cv2.THRESH_BINARY)
    elif bool(noise_mask_cfg.get("enabled", False)):
        noise_mask = compute_noise_mask(
            video_path=input_video,
            background=background,
            cfg=noise_mask_cfg,
            fps=meta.fps,
            total_frames=meta.frame_count,
        )
        if noise_mask is not None and bool(noise_mask_cfg.get("save_mask_png", True)):
            noise_mask_path_out = out_dir / "noise_mask.png"
            cv2.imwrite(str(noise_mask_path_out), noise_mask)

    all_points: List[TrackPoint] = []
    frame_processed = 0
    suppressed_burst_frames = 0
    detection_cfg = cfg["detection"]
    clahe_enabled = bool(detection_cfg.get("clahe_enabled", False))
    ff_diff_enabled = bool(detection_cfg.get("ff_diff_enabled", False))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if clahe_enabled else None
    prev_gray: np.ndarray | None = None
    progress.start_stage("frame_processing")

    for frame_idx, gray in iter_gray_frames(input_video, perf=perf):
        frame_started = perf_counter()

        if clahe is not None:
            gray = clahe.apply(gray)

        dets = detect_foreground_blobs(
            gray,
            background,
            detection_cfg,
            valid_mask=valid_mask_for_detection,
            noise_mask=noise_mask,
            compute_device=execution_plan.selected_device,
            strict_parity=strict_parity,
            runtime_stats=detection_runtime_stats,
            context=detection_context,
            frame_idx=frame_idx,
            perf=perf,
        )

        if ff_diff_enabled and prev_gray is not None:
            ff_dets = detect_ffdiff_blobs(
                gray, prev_gray, detection_context,
                valid_mask=valid_mask_for_detection,
                noise_mask=noise_mask,
                perf=perf, frame_idx=frame_idx,
            )
            for fd in ff_dets:
                dets.append(fd)

        if ff_diff_enabled:
            prev_gray = gray.copy()
        if burst_gate is not None and not burst_gate.should_keep(frame_idx, len(dets)):
            dets = []
            suppressed_burst_frames += 1

        if temporal_acc.enabled:
            extra_dets = temporal_acc.feed(gray)
            if extra_dets:
                dets.extend(extra_dets)

        if len(dets) > 1:
            dedup_px = float(detection_cfg.get("dedup_proximity_px", 80.0))
            dets = _deduplicate_detections(dets, proximity_px=dedup_px)

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

    progress.start_stage("postprocess")
    postprocess_started = perf_counter()
    merged_points, merges_applied = _auto_merge_track_points(all_points, tracking_cfg)
    fast_merged_points, fast_merges_applied = _merge_fast_overlap_track_points(merged_points, tracking_cfg)
    filtered_points, track_assessments = _filter_track_points(
        fast_merged_points,
        tracking_cfg,
        meta.fps,
        valid_mask=valid_mask,
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
            fast_merged_points,
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
    if bool(tracking_cfg.get("export_track_candidates", False)):
        _write_track_candidates_csv(track_candidates_csv_path, track_assessments)

    tracking_debug_csv_path = out_dir / "tracking_debug.csv"
    if bool(tracking_cfg.get("export_tracking_debug", False)):
        _write_tracking_debug_csv(tracking_debug_csv_path, tracker.debug_rows)

    out_cfg_export = cfg.get("output", {})
    smoothing_on = bool(out_cfg_export.get("trajectory_smoothing_enabled", False))
    ts_window = int(out_cfg_export.get("trajectory_smoothing_window", 5))
    smoothed_points: List[TrackPoint] | None = None
    if smoothing_on:
        smoothed_points = smooth_track_points(filtered_points, ts_window)

    events_csv_path = out_dir / "events.csv"
    points_for_events = smoothed_points if smoothed_points is not None else filtered_points
    _write_events_csv(events_csv_path, points_for_events, valid_gate_mask)
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
        direction_mask=valid_gate_mask,
    )
    tracks_render_json_path = out_dir / "tracks_render.json"
    export_tracks_render_json(
        tracks_render_json_path,
        width=meta.width,
        height=meta.height,
        points=filtered_points,
        valid_region_mask=valid_mask,
        direction_mask=valid_gate_mask,
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

    all_tracks_overlay = render_tracks_overlay(
        background_gray=background,
        points=fast_merged_points,
        line_thickness=overlay_line_t,
        start_radius=overlay_start_r,
        alpha=overlay_alpha_v,
        draw_track_labels=overlay_lbl,
        draw_track_labels_at_end=overlay_lbl_end,
        label_font_scale=overlay_lbl_scale,
        label_thickness=overlay_lbl_th,
    )
    all_tracks_overlay_path = out_dir / "tracks_all_overlay.png"
    cv2.imwrite(str(all_tracks_overlay_path), all_tracks_overlay)
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
        "metrics": {
            **_build_metrics(filtered_points, frame_processed),
            "frames_suppressed_temporal_burst": suppressed_burst_frames,
            "tracks_merged_auto": len(merges_applied),
            "tracks_merged_fast_overlap": len(fast_merges_applied),
            **background_runtime_stats,
            **detection_runtime_stats,
        },
        "execution": {
            "requested_device": execution_plan.requested_device,
            "selected_device": execution_plan.selected_device,
            "gpu_available": execution_plan.gpu_available,
            "strict_parity": strict_parity,
            "selection_reason": execution_plan.reason,
        },
        "performance": perf_summary,
        "outputs": {
            "background_png": str(background_path.resolve()),
            "tracks_csv": str(tracks_csv_path.resolve()),
            "events_csv": str(events_csv_path.resolve()),
            "tracks_svg": str(tracks_svg_path.resolve()),
            "tracks_render_json": str(tracks_render_json_path.resolve()),
            "tracks_overlay_png": str(overlay_path.resolve()),
            "flight_trails_overlay_video": flight_trails_output,
            "track_candidates_csv": (
                str(track_candidates_csv_path.resolve())
                if bool(tracking_cfg.get("export_track_candidates", False))
                else ""
            ),
            "tracking_debug_csv": (
                str(tracking_debug_csv_path.resolve())
                if bool(tracking_cfg.get("export_tracking_debug", False))
                else ""
            ),
            **overlay_smoothing_paths,
            **fast_event_outputs,
            **heatmap_event_outputs,
            "track_clips": track_clip_outputs,
            **valid_region_outputs,
        },
        "postprocess": {
            "auto_merge_enabled": bool(tracking_cfg.get("auto_merge_suggested", False)),
            "fast_merge_overlap_enabled": bool(tracking_cfg.get("merge_fast_overlap_enabled", False)),
            "auto_merges_applied": merges_applied,
            "fast_merges_applied": fast_merges_applied,
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
    }

    meta_path = out_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta_payload, handle, indent=2)

    progress.finish()
    return meta_payload

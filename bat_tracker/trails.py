from __future__ import annotations

from collections import defaultdict, deque
from math import hypot
from pathlib import Path
from typing import Deque, Dict, Iterable, List

import cv2
import numpy as np

from .tracker import TrackPoint
from .video import open_video_capture


def _path_length(points: List[TrackPoint]) -> float:
    if len(points) < 2:
        return 0.0
    return sum(hypot(p1.x - p0.x, p1.y - p0.y) for p0, p1 in zip(points[:-1], points[1:]))


def _history_span(points: List[TrackPoint]) -> float:
    if not points:
        return 0.0
    min_x = min(point.x for point in points)
    max_x = max(point.x for point in points)
    min_y = min(point.y for point in points)
    max_y = max(point.y for point in points)
    return hypot(max_x - min_x, max_y - min_y)


class RealTimeTrailRenderer:
    def __init__(self, frame_shape: tuple[int, int], cfg: dict):
        self.enabled = bool(cfg.get("enabled", False))
        self.history_frames = max(2, int(cfg.get("history_frames", 12)))
        self.max_track_gap_frames = max(1, int(cfg.get("max_track_gap_frames", 2)))
        self.decay = float(np.clip(float(cfg.get("decay", 0.90)), 0.0, 1.0))
        self.segment_thickness = max(1, int(cfg.get("segment_thickness", 3)))
        self.point_radius = max(1, int(cfg.get("point_radius", 2)))
        self.segment_intensity = max(0.0, float(cfg.get("segment_intensity", 1.0)))
        self.point_intensity = max(0.0, float(cfg.get("point_intensity", 1.35)))
        self.overlay_alpha = float(np.clip(float(cfg.get("overlay_alpha", 0.60)), 0.0, 1.0))
        self.min_history_points = max(2, int(cfg.get("min_history_points", 3)))
        self.min_segment_displacement_px = max(0.0, float(cfg.get("min_segment_displacement_px", 4.0)))
        self.min_recent_displacement_px = max(0.0, float(cfg.get("min_recent_displacement_px", 10.0)))
        self.min_recent_path_length_px = max(0.0, float(cfg.get("min_recent_path_length_px", 14.0)))
        self.min_recent_straightness = float(np.clip(float(cfg.get("min_recent_straightness", 0.20)), 0.0, 1.0))
        self.stationary_radius_px = max(0.0, float(cfg.get("stationary_radius_px", 14.0)))
        self.clip_percentile = float(np.clip(float(cfg.get("clip_percentile", 99.0)), 1.0, 100.0))
        self.max_normalization_value = max(0.0, float(cfg.get("max_normalization_value", 0.0)))
        self.colormap_name = str(cfg.get("colormap", "inferno")).strip().lower()

        self._heatmap = np.zeros(frame_shape, dtype=np.float32)
        self._history: Dict[int, Deque[TrackPoint]] = defaultdict(lambda: deque(maxlen=self.history_frames))
        self._current_frame = -1

        colormaps = {
            "inferno": cv2.COLORMAP_INFERNO,
        }
        self._colormap = colormaps.get(self.colormap_name, cv2.COLORMAP_INFERNO)

    def update(self, frame: np.ndarray, frame_points: Iterable[TrackPoint]) -> np.ndarray:
        if not self.enabled:
            if frame.ndim == 2:
                return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            return frame.copy()

        self._current_frame += 1
        self._heatmap *= self.decay
        impulse = np.zeros_like(self._heatmap)
        stale_track_ids = [
            track_id
            for track_id, history in self._history.items()
            if history and self._current_frame - history[-1].frame > self.history_frames + self.max_track_gap_frames
        ]
        for track_id in stale_track_ids:
            del self._history[track_id]

        for point in frame_points:
            history = self._history[point.track_id]
            if history and point.frame - history[-1].frame > self.max_track_gap_frames:
                history.clear()
            history.append(point)

            points = list(history)
            if not self._is_coherent(points):
                continue
            self._accumulate_track_impulse(impulse, points)

        self._heatmap += impulse
        return self._compose_overlay(frame)

    def _is_coherent(self, points: List[TrackPoint]) -> bool:
        if len(points) < self.min_history_points:
            return False

        path_length = _path_length(points)
        if path_length < self.min_recent_path_length_px:
            return False

        displacement = hypot(points[-1].x - points[0].x, points[-1].y - points[0].y)
        if displacement < self.min_recent_displacement_px:
            return False

        if path_length > 1e-6:
            straightness = displacement / path_length
            if straightness < self.min_recent_straightness:
                return False

        if _history_span(points) < self.stationary_radius_px:
            return False

        for p0, p1 in zip(points[:-1], points[1:]):
            if hypot(p1.x - p0.x, p1.y - p0.y) >= self.min_segment_displacement_px:
                return True
        return False

    def _accumulate_track_impulse(self, impulse: np.ndarray, points: List[TrackPoint]) -> None:
        segments = list(zip(points[:-1], points[1:]))
        total_segments = max(1, len(segments))
        for idx, (p0, p1) in enumerate(segments, start=1):
            segment_disp = hypot(p1.x - p0.x, p1.y - p0.y)
            if segment_disp < self.min_segment_displacement_px:
                continue
            weight = self.segment_intensity * (idx / float(total_segments))
            cv2.line(
                impulse,
                (int(round(p0.x)), int(round(p0.y))),
                (int(round(p1.x)), int(round(p1.y))),
                color=float(weight),
                thickness=self.segment_thickness,
                lineType=cv2.LINE_AA,
            )

        end = points[-1]
        cv2.circle(
            impulse,
            (int(round(end.x)), int(round(end.y))),
            radius=self.point_radius,
            color=float(self.segment_intensity * self.point_intensity),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

    def _compose_overlay(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            base = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            base = frame.copy()

        positive = self._heatmap[self._heatmap > 1e-6]
        if positive.size == 0:
            return base

        if self.max_normalization_value > 0.0:
            scale = self.max_normalization_value
        else:
            scale = float(np.percentile(positive, self.clip_percentile))
        scale = max(scale, 1e-6)

        normalized = np.clip(self._heatmap / scale, 0.0, 1.0)
        heatmap_u8 = np.uint8(np.round(normalized * 255.0))
        colored = cv2.applyColorMap(heatmap_u8, self._colormap)
        alpha = (normalized * self.overlay_alpha).astype(np.float32)[..., None]

        out = base.astype(np.float32)
        out *= 1.0 - alpha
        out += colored.astype(np.float32) * alpha
        return np.clip(out, 0.0, 255.0).astype(np.uint8)


def export_realtime_trails_video(
    input_video: str | Path,
    output_path: str | Path,
    points: Iterable[TrackPoint],
    frame_size: tuple[int, int],
    fps: float,
    cfg: dict,
) -> str:
    renderer = RealTimeTrailRenderer((frame_size[1], frame_size[0]), cfg)
    if not renderer.enabled:
        return ""

    points_by_frame: Dict[int, List[TrackPoint]] = defaultdict(list)
    for point in points:
        points_by_frame[int(point.frame)].append(point)

    cap = open_video_capture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for trails export: {input_video}")

    output_path = Path(output_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), frame_size)
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create trails video writer: {output_path}")

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            overlay = renderer.update(frame, points_by_frame.get(frame_idx, ()))
            writer.write(overlay)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    return str(output_path.resolve())

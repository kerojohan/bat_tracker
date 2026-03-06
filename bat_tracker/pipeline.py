from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict, deque
from dataclasses import asdict
from math import hypot
from pathlib import Path
from statistics import mean
from typing import Dict, List

import cv2

from .background import compute_background_median
from .config import load_config
from .detection import detect_foreground_blobs
from .render import render_tracks_overlay
from .tracker import GreedyTracker, TrackPoint
from .video import iter_gray_frames, read_video_meta


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


def _write_tracks_csv(path: Path, points: List[TrackPoint]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for point in sorted(points, key=lambda p: (p.track_id, p.frame)):
            row = asdict(point)
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


def _path_length(track_points: List[TrackPoint]) -> float:
    if len(track_points) < 2:
        return 0.0
    return sum(hypot(p1.x - p0.x, p1.y - p0.y) for p0, p1 in zip(track_points[:-1], track_points[1:]))


def _filter_track_points(points: List[TrackPoint], tracking_cfg: Dict) -> List[TrackPoint]:
    min_track_length = int(tracking_cfg.get("min_track_length", 1))
    min_track_displacement = float(tracking_cfg.get("min_track_displacement", 0.0))
    min_track_path_length = float(tracking_cfg.get("min_track_path_length", 0.0))
    min_track_straightness = float(tracking_cfg.get("min_track_straightness", 0.0))

    by_track: Dict[int, List[TrackPoint]] = defaultdict(list)
    for point in points:
        by_track[point.track_id].append(point)

    filtered: List[TrackPoint] = []
    for track_points in by_track.values():
        track_points = sorted(track_points, key=lambda p: p.frame)
        if len(track_points) < min_track_length:
            continue

        start = track_points[0]
        end = track_points[-1]
        displacement = hypot(end.x - start.x, end.y - start.y)
        if displacement < min_track_displacement:
            continue

        path_length = _path_length(track_points)
        if path_length < min_track_path_length:
            continue

        if min_track_straightness > 0.0 and path_length > 0.0:
            straightness = displacement / path_length
            if straightness < min_track_straightness:
                continue

        filtered.extend(track_points)

    return filtered


def run_pipeline(input_video: str, output_dir: str, config_path: str | None = None) -> Dict:
    cfg = load_config(config_path)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = read_video_meta(input_video)

    background = compute_background_median(
        video_path=input_video,
        meta=meta,
        sample_frames=int(cfg["background"]["sample_frames"]),
        uniform_sampling=bool(cfg["background"]["uniform_sampling"]),
    )
    background_path = out_dir / "background.png"
    cv2.imwrite(str(background_path), background)

    tracker = GreedyTracker(
        max_distance=float(cfg["tracking"]["max_distance"]),
        max_missed=int(cfg["tracking"]["max_missed"]),
        fps=meta.fps,
        video_id=meta.video_id,
    )
    burst_gate = TemporalBurstGate.from_detection_cfg(cfg["detection"])

    all_points: List[TrackPoint] = []
    frame_processed = 0
    suppressed_burst_frames = 0

    for frame_idx, gray in iter_gray_frames(input_video):
        dets = detect_foreground_blobs(gray, background, cfg["detection"])
        if burst_gate is not None and not burst_gate.should_keep(frame_idx, len(dets)):
            dets = []
            suppressed_burst_frames += 1
        frame_points = tracker.step(frame_idx, dets)
        all_points.extend(frame_points)
        frame_processed += 1

    filtered_points = _filter_track_points(all_points, cfg["tracking"])

    tracks_csv_path = out_dir / "tracks.csv"
    _write_tracks_csv(tracks_csv_path, filtered_points)

    overlay = render_tracks_overlay(
        background_gray=background,
        points=filtered_points,
        line_thickness=int(cfg["output"]["overlay_line_thickness"]),
        start_radius=int(cfg["output"]["overlay_start_radius"]),
        alpha=float(cfg["output"].get("overlay_alpha", 1.0)),
    )
    overlay_path = out_dir / "tracks_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    meta_payload = {
        "video": {
            "input_path": str(Path(input_video).resolve()),
            "video_id": meta.video_id,
            "fps": meta.fps,
            "frame_count_reported": meta.frame_count,
            "width": meta.width,
            "height": meta.height,
        },
        "parameters": cfg,
        "metrics": {
            **_build_metrics(filtered_points, frame_processed),
            "frames_suppressed_temporal_burst": suppressed_burst_frames,
        },
        "outputs": {
            "background_png": str(background_path.resolve()),
            "tracks_csv": str(tracks_csv_path.resolve()),
            "tracks_overlay_png": str(overlay_path.resolve()),
        },
    }

    meta_path = out_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta_payload, handle, indent=2)

    return meta_payload

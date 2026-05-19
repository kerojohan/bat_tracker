from __future__ import annotations

import importlib.util
from argparse import Namespace
from collections import defaultdict
from math import hypot
from pathlib import Path
from types import ModuleType
from typing import Dict, Iterable, List

import cv2
import numpy as np

from .tracker import TrackPoint
from .vendor.fast_tracker import bat_tracking2_long_kinetic as bundled_kinetic


KINETIC_DEFAULTS = {
    "output": "",
    "report": "",
    "overlay_output": "",
    "zone_mask": "",
    "zone_require": "off",
    "show": False,
    "profile": False,
    "save_reference_frame": "",
    "skip_seconds": 0.0,
    "max_seconds": 0.0,
    "resize": 1.0,
    "auto_calibrate": False,
    "no_quality": False,
    "draw_rejected_long": False,
    "fg_threshold": 180.0,
    "blur_kernel": 3,
    "temporal_smooth": 0.0,
    "morph_open_iters": 1,
    "morph_close_iters": 1,
    "min_area": 6.0,
    "max_area": 8000.0,
    "max_distance": 120.0,
    "max_segment": 150.0,
    "min_displacement": 120.0,
    "min_path_length": 140.0,
    "max_track_speed": 250.0,
    "min_points": 4,
    "max_missing": 5,
    "quality_percentile": 85.0,
    "quality_threshold": 0.0,
    "area_cost_weight": 15.0,
    "dedupe_overlap_distance": 45.0,
    "dedupe_perp_distance": 18.0,
    "dedupe_direction_cos": 0.92,
    "dedupe_min_overlap_frames": 4,
    "dedupe_min_overlap_ratio": 0.6,
    "dedupe_full_cover_ratio": 0.95,
    "dedupe_polyline_distance": 20.0,
    "dedupe_polyline_cover_ratio": 0.75,
    "dedupe_short_track_cover_ratio": 0.6,
    "dedupe_short_track_max_points": 8,
    "dedupe_parallel_perp_distance": 26.0,
    "dedupe_parallel_direction_cos": 0.98,
    "dedupe_near_full_overlap_ratio": 0.9,
    "dedupe_bbox_contain_ratio": 0.7,
    "dedupe_bbox_contain_frames": 3,
    "dedupe_frame_slack": 1,
}


def run_kinetic_secondary_tracks(
    video_path: str,
    cfg: dict,
    *,
    video_id: str,
    fps: float,
) -> tuple[list[TrackPoint], dict]:
    script_path_raw = str(cfg.get("script_path", "")).strip()
    script_path = Path(script_path_raw).expanduser() if script_path_raw else None
    if script_path is not None:
        if not script_path.exists():
            raise FileNotFoundError(f"secondary_detection.script_path not found: {script_path}")
        mod = _load_kinetic_module(script_path)
        script_source = str(script_path.resolve())
    else:
        mod = bundled_kinetic
        script_source = str(Path(bundled_kinetic.__file__).resolve())
    args = _build_args(video_path, cfg)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for kinetic secondary tracker: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or fps or 30.0
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mod.auto_calibrate(args, source_width * args.resize, source_height * args.resize, source_fps, cap)
    args.blur_kernel = _positive_odd(int(args.blur_kernel))
    args.temporal_smooth = float(np.clip(float(args.temporal_smooth), 0.0, 0.95))

    start_frame = max(0, int(round(args.skip_seconds * source_fps)))
    max_frames = int(round(args.max_seconds * source_fps)) if args.max_seconds > 0 else 0
    max_frame = start_frame + max_frames - 1 if max_frames > 0 else None
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Cannot read initial frame for kinetic secondary tracker.")

    if args.resize != 1.0:
        frame = cv2.resize(frame, None, fx=args.resize, fy=args.resize, interpolation=cv2.INTER_AREA)
    if args.save_reference_frame:
        Path(args.save_reference_frame).expanduser().parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(Path(args.save_reference_frame).expanduser()), frame)

    frame_height, frame_width = frame.shape[:2]
    zone_mask = mod.load_zone_mask(args.zone_mask, (frame_height, frame_width))
    subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
    tracker = mod.BatTracker(args)

    prev_gray = None
    frame_idx = start_frame
    frames_processed = 0
    while ok:
        if max_frame is not None and frame_idx > max_frame:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if args.blur_kernel > 1:
            gray = cv2.GaussianBlur(gray, (args.blur_kernel, args.blur_kernel), 0)
        if prev_gray is not None and args.temporal_smooth > 0:
            gray = cv2.addWeighted(gray, 1.0 - args.temporal_smooth, prev_gray, args.temporal_smooth, 0.0)
        _, detections = mod.detect_blobs(gray, subtractor, args, zone_mask)
        tracker.step(frame_idx, detections)

        prev_gray = gray
        ok, frame = cap.read()
        frame_idx += 1
        frames_processed += 1
        if ok and args.resize != 1.0:
            frame = cv2.resize(frame, None, fx=args.resize, fy=args.resize, interpolation=cv2.INTER_AREA)

    cap.release()
    tracks = tracker.finalize()
    mod.assign_quality(tracks, args)
    accepted_tracks = [track for track in tracks if track.accepted]
    points = _tracks_to_points(accepted_tracks, video_id=video_id, fps=fps)
    meta = {
        "script_path": script_source,
        "script_bundled": script_path is None,
        "frames_processed": frames_processed,
        "tracks_total": len(tracks),
        "tracks_accepted": len(accepted_tracks),
        "points_accepted": len(points),
        "effective_blur_kernel": int(args.blur_kernel),
        "effective_temporal_smooth": float(args.temporal_smooth),
        "effective_morph_open_iters": int(args.morph_open_iters),
        "effective_morph_close_iters": int(args.morph_close_iters),
    }
    return points, meta


def dedupe_secondary_track_points(
    primary_points: Iterable[TrackPoint],
    secondary_points: Iterable[TrackPoint],
    *,
    max_overlap_distance_px: float,
    min_overlap_frames: int,
    min_overlap_ratio: float,
    secondary_track_id_offset: int,
) -> tuple[list[TrackPoint], dict]:
    primary_by_track = _points_by_track(primary_points)
    secondary_by_track = _points_by_track(secondary_points)
    kept: list[TrackPoint] = []
    duplicate_track_ids: list[int] = []
    kept_track_ids: list[int] = []

    next_track_id = int(secondary_track_id_offset)
    for source_track_id in sorted(secondary_by_track):
        track = secondary_by_track[source_track_id]
        if _is_duplicate_of_any_primary(
            track,
            primary_by_track.values(),
            max_overlap_distance_px=max_overlap_distance_px,
            min_overlap_frames=min_overlap_frames,
            min_overlap_ratio=min_overlap_ratio,
        ):
            duplicate_track_ids.append(source_track_id)
            continue

        assigned_track_id = next_track_id
        next_track_id += 1
        kept_track_ids.append(assigned_track_id)
        for point in track:
            kept.append(
                TrackPoint(
                    video_id=point.video_id,
                    track_id=assigned_track_id,
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

    return kept, {
        "secondary_tracks_raw": len(secondary_by_track),
        "secondary_tracks_added": len(kept_track_ids),
        "secondary_tracks_duplicate": len(duplicate_track_ids),
        "secondary_duplicate_source_track_ids": duplicate_track_ids,
        "secondary_added_track_ids": kept_track_ids,
    }


def _build_args(video_path: str, cfg: dict) -> Namespace:
    values = dict(KINETIC_DEFAULTS)
    values["video"] = video_path
    aliases = {
        "morph_close_iters": "morph_close_iters",
        "morph_open_iters": "morph_open_iters",
        "temporal_smooth": "temporal_smooth",
        "auto_calibrate": "auto_calibrate",
    }
    for key, value in cfg.items():
        if key in {"enabled", "inherit_primary", "algorithm", "script_path", "dedupe_max_distance_px", "dedupe_min_iou"}:
            continue
        if key.startswith("kinetic_"):
            values[key.removeprefix("kinetic_")] = value
        elif key in values:
            values[key] = value
        elif key in aliases:
            values[aliases[key]] = value

    values["blur_kernel"] = _positive_odd(int(values["blur_kernel"]))
    values["temporal_smooth"] = float(np.clip(float(values["temporal_smooth"]), 0.0, 0.95))
    return Namespace(**values)


def _load_kinetic_module(script_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("bat_tracker_external_kinetic", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load kinetic tracker script: {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tracks_to_points(tracks: Iterable, *, video_id: str, fps: float) -> list[TrackPoint]:
    points: list[TrackPoint] = []
    for track in tracks:
        prev = None
        for point in sorted(track.points, key=lambda row: row[0]):
            frame, x, y, area, bbox = point
            bx, by, bw, bh = bbox
            if prev is None:
                vx = 0.0
                vy = 0.0
            else:
                prev_frame, prev_x, prev_y, _, _ = prev
                dt = max(1e-6, (frame - prev_frame) / max(1e-6, fps))
                vx = (x - prev_x) / dt
                vy = (y - prev_y) / dt
            points.append(
                TrackPoint(
                    video_id=video_id,
                    track_id=int(track.track_id),
                    frame=int(frame),
                    time_sec=float(frame) / max(1e-6, fps),
                    x=float(x),
                    y=float(y),
                    vx=float(vx),
                    vy=float(vy),
                    bbox_x1=int(bx),
                    bbox_y1=int(by),
                    bbox_x2=int(bx + bw),
                    bbox_y2=int(by + bh),
                    area=float(area),
                )
            )
            prev = point
    return points


def _points_by_track(points: Iterable[TrackPoint]) -> Dict[int, List[TrackPoint]]:
    by_track: Dict[int, List[TrackPoint]] = defaultdict(list)
    for point in points:
        by_track[int(point.track_id)].append(point)
    for track_id in by_track:
        by_track[track_id].sort(key=lambda point: point.frame)
    return by_track


def _is_duplicate_of_any_primary(
    secondary_track: list[TrackPoint],
    primary_tracks: Iterable[list[TrackPoint]],
    *,
    max_overlap_distance_px: float,
    min_overlap_frames: int,
    min_overlap_ratio: float,
) -> bool:
    for primary_track in primary_tracks:
        distances = _same_frame_distances(secondary_track, primary_track)
        if len(distances) < int(min_overlap_frames):
            continue
        overlap_ratio = len(distances) / max(1, min(len(secondary_track), len(primary_track)))
        if overlap_ratio < float(min_overlap_ratio):
            continue
        if float(np.mean(distances)) <= float(max_overlap_distance_px):
            return True
    return False


def _same_frame_distances(a: list[TrackPoint], b: list[TrackPoint]) -> list[float]:
    b_by_frame = {point.frame: point for point in b}
    distances: list[float] = []
    for point in a:
        other = b_by_frame.get(point.frame)
        if other is None:
            continue
        distances.append(hypot(point.x - other.x, point.y - other.y))
    return distances


def _positive_odd(value: int) -> int:
    if value <= 0:
        return 1
    return value if value % 2 == 1 else value + 1

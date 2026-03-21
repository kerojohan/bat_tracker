from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict, deque
from dataclasses import asdict
from math import ceil
from math import hypot
from pathlib import Path
from statistics import mean
from typing import Dict, List

import cv2
import numpy as np

from .background import compute_background_median
from .config import load_config
from .detection import detect_foreground_blobs
from .render import render_tracks_overlay
from .tracker import GreedyTracker, TrackPoint
from .valid_region import load_image as load_valid_region_image
from .valid_region import run_valid_region
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


def _classify_direction(start_inside: bool, end_inside: bool) -> str:
    if start_inside and end_inside:
        return "inside"
    if start_inside and not end_inside:
        return "exits"
    if not start_inside and end_inside:
        return "enters"
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
            direction = _classify_direction(s_in, e_in)
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


def _point_in_mask(point: TrackPoint, mask: np.ndarray) -> bool:
    xi = int(round(point.x))
    yi = int(round(point.y))
    if yi < 0 or yi >= mask.shape[0] or xi < 0 or xi >= mask.shape[1]:
        return False
    return bool(mask[yi, xi] > 0)


def _filter_track_points(
    points: List[TrackPoint],
    tracking_cfg: Dict,
    fps: float,
    valid_mask: np.ndarray | None = None,
) -> List[TrackPoint]:
    min_track_length_cfg = int(tracking_cfg.get("min_track_length", 1))
    min_track_duration_sec = float(tracking_cfg.get("min_track_duration_sec", 0.0))
    min_track_length_from_sec = int(ceil(max(0.0, min_track_duration_sec) * max(1e-6, fps)))
    min_track_length = max(min_track_length_cfg, min_track_length_from_sec)
    min_track_displacement = float(tracking_cfg.get("min_track_displacement", 0.0))
    min_track_path_length = float(tracking_cfg.get("min_track_path_length", 0.0))
    min_track_straightness = float(tracking_cfg.get("min_track_straightness", 0.0))
    require_start_or_end_in_valid_region = bool(tracking_cfg.get("require_start_or_end_in_valid_region", False))
    valid_region_gate_dilate_px = max(0, int(tracking_cfg.get("valid_region_gate_dilate_px", 0)))

    gate_mask = valid_mask
    if gate_mask is not None and valid_region_gate_dilate_px > 0:
        k = 2 * valid_region_gate_dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        gate_mask = cv2.dilate(gate_mask, kernel, iterations=1)

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

        if require_start_or_end_in_valid_region and gate_mask is not None:
            if not (_point_in_mask(start, gate_mask) or _point_in_mask(end, gate_mask)):
                continue

        filtered.extend(track_points)

    return filtered


def _vector_cosine(v0: tuple[float, float], v1: tuple[float, float]) -> float | None:
    n0 = hypot(v0[0], v0[1])
    n1 = hypot(v1[0], v1[1])
    if n0 <= 1e-6 or n1 <= 1e-6:
        return None
    return (v0[0] * v1[0] + v0[1] * v1[1]) / (n0 * n1)


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
    for idx, track_a_id in enumerate(track_ids):
        a_pts = by_track[track_a_id]
        a_start = a_pts[0]
        a_end = a_pts[-1]
        a_start_vec = (a_pts[min(2, len(a_pts) - 1)].x - a_pts[0].x, a_pts[min(2, len(a_pts) - 1)].y - a_pts[0].y)
        a_end_vec = (a_pts[-1].x - a_pts[max(0, len(a_pts) - 3)].x, a_pts[-1].y - a_pts[max(0, len(a_pts) - 3)].y)
        a_frames = {p.frame: p for p in a_pts}

        for track_b_id in track_ids[idx + 1 :]:
            b_pts = by_track[track_b_id]
            b_start = b_pts[0]
            b_end = b_pts[-1]
            b_start_vec = (b_pts[min(2, len(b_pts) - 1)].x - b_pts[0].x, b_pts[min(2, len(b_pts) - 1)].y - b_pts[0].y)
            b_end_vec = (b_pts[-1].x - b_pts[max(0, len(b_pts) - 3)].x, b_pts[-1].y - b_pts[max(0, len(b_pts) - 3)].y)

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
                b_frames = {p.frame: p for p in b_pts}
                common_frames = sorted(set(a_frames.keys()).intersection(b_frames.keys()))
                if len(common_frames) >= min_overlap_common:
                    distances = []
                    cosines = []
                    for frame in common_frames:
                        pa = a_frames[frame]
                        pb = b_frames[frame]
                        distances.append(hypot(pa.x - pb.x, pa.y - pb.y))

                    mean_distance = sum(distances) / len(distances)
                    c0 = _vector_cosine(a_start_vec, b_start_vec)
                    c1 = _vector_cosine(a_end_vec, b_end_vec)
                    if c0 is not None:
                        cosines.append(c0)
                    if c1 is not None:
                        cosines.append(c1)
                    mean_cos = (sum(cosines) / len(cosines)) if cosines else None
                    if mean_distance <= max_overlap_mean_dist and (mean_cos is None or mean_cos >= min_overlap_cos):
                        reason = "overlap"
                        reason_data = {
                            "common_frames": len(common_frames),
                            "mean_distance": mean_distance,
                            "mean_direction_cosine": mean_cos if mean_cos is not None else 1.0,
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
    deduped: List[TrackPoint] = []
    seen = set()
    for point in merged_points:
        key = (point.track_id, point.frame, round(point.x, 3), round(point.y, 3))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(point)

    return deduped, merges_applied


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
    valid_mask: np.ndarray | None = None
    valid_mask_for_detection: np.ndarray | None = None
    valid_region_meta: Dict = {"enabled": False}
    valid_region_outputs: Dict[str, str] = {}

    valid_region_cfg = cfg.get("valid_region", {})
    valid_region_enabled = bool(valid_region_cfg.get("enabled", False))
    if valid_region_enabled:
        valid_input = str(valid_region_cfg.get("input_image", "")).strip()
        if valid_input:
            valid_image = load_valid_region_image(valid_input)
        else:
            valid_image = background

        valid_subdir = str(valid_region_cfg.get("output_subdir", "valid_region"))
        valid_output_dir = out_dir / valid_subdir
        valid_region_meta = run_valid_region(
            image=valid_image,
            output_dir=valid_output_dir,
            config=valid_region_cfg,
        )
        mask_path = valid_output_dir / "mask.png"
        valid_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if valid_mask is None:
            raise RuntimeError(f"Could not load valid-region mask from: {mask_path}")
        if bool(valid_region_cfg.get("apply_to_detection", True)):
            valid_mask_for_detection = valid_mask
        valid_region_outputs = {
            "valid_region_mask_png": str(mask_path.resolve()),
            "valid_region_overlay_png": str((valid_output_dir / "overlay.png").resolve()),
            "valid_region_profile_png": str((valid_output_dir / "profile.png").resolve()),
        }

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
        dets = detect_foreground_blobs(gray, background, cfg["detection"], valid_mask=valid_mask_for_detection)
        if burst_gate is not None and not burst_gate.should_keep(frame_idx, len(dets)):
            dets = []
            suppressed_burst_frames += 1
        frame_points = tracker.step(frame_idx, dets)
        all_points.extend(frame_points)
        frame_processed += 1

    filtered_points = _filter_track_points(all_points, cfg["tracking"], meta.fps, valid_mask=valid_mask)
    filtered_points, merges_applied = _auto_merge_track_points(filtered_points, cfg["tracking"])

    tracks_csv_path = out_dir / "tracks.csv"
    _write_tracks_csv(tracks_csv_path, filtered_points)

    events_csv_path = out_dir / "events.csv"
    _write_events_csv(events_csv_path, filtered_points, valid_mask)

    overlay = render_tracks_overlay(
        background_gray=background,
        points=filtered_points,
        line_thickness=int(cfg["output"]["overlay_line_thickness"]),
        start_radius=int(cfg["output"]["overlay_start_radius"]),
        alpha=float(cfg["output"].get("overlay_alpha", 1.0)),
        draw_track_labels=bool(cfg["output"].get("overlay_draw_track_labels", False)),
        draw_track_labels_at_end=bool(cfg["output"].get("overlay_draw_track_labels_at_end", False)),
        label_font_scale=float(cfg["output"].get("overlay_label_font_scale", 0.5)),
        label_thickness=int(cfg["output"].get("overlay_label_thickness", 1)),
    )
    overlay_path = out_dir / "tracks_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    track_clip_outputs: Dict[str, str] = {}
    if bool(cfg["output"].get("export_track_clips", False)):
        track_clip_outputs = _export_track_clips(
            input_video=input_video,
            output_dir=out_dir,
            points=filtered_points,
            fps=meta.fps,
            frame_size=(meta.width, meta.height),
            clips_subdir=str(cfg["output"].get("track_clips_subdir", "track_clips")),
            pad_frames=max(0, int(cfg["output"].get("track_clips_padding_frames", 0))),
        )

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
        "valid_region": valid_region_meta,
        "metrics": {
            **_build_metrics(filtered_points, frame_processed),
            "frames_suppressed_temporal_burst": suppressed_burst_frames,
            "tracks_merged_auto": len(merges_applied),
        },
        "outputs": {
            "background_png": str(background_path.resolve()),
            "tracks_csv": str(tracks_csv_path.resolve()),
            "events_csv": str(events_csv_path.resolve()),
            "tracks_overlay_png": str(overlay_path.resolve()),
            "track_clips": track_clip_outputs,
            **valid_region_outputs,
        },
        "postprocess": {
            "auto_merge_enabled": bool(cfg["tracking"].get("auto_merge_suggested", False)),
            "auto_merges_applied": merges_applied,
        },
    }

    meta_path = out_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta_payload, handle, indent=2)

    return meta_payload

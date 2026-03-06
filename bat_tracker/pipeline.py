from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict
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

    all_points: List[TrackPoint] = []
    frame_processed = 0

    for frame_idx, gray in iter_gray_frames(input_video):
        dets = detect_foreground_blobs(gray, background, cfg["detection"])
        frame_points = tracker.step(frame_idx, dets)
        all_points.extend(frame_points)
        frame_processed += 1

    min_track_length = int(cfg["tracking"].get("min_track_length", 1))
    track_sizes = Counter(p.track_id for p in all_points)
    filtered_points = [p for p in all_points if track_sizes[p.track_id] >= min_track_length]

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
        "metrics": _build_metrics(filtered_points, frame_processed),
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

from __future__ import annotations

import csv
from pathlib import Path

import cv2

from .video import open_video_capture


def export_fast_track_overlay(
    input_video: str | Path,
    tracks_csv: str | Path,
    events_csv: str | Path,
    output_path: str | Path,
    source_video: str | Path | None = None,
    min_speed: float = 1000.0,
    dot_radius: int = 8,
    dot_color: tuple[int, int, int] = (0, 255, 255),
) -> str:
    fast_ids: set[str] = set()
    with open(events_csv, newline="") as f:
        for row in csv.DictReader(f):
            if float(row.get("mean_speed_px_sec", 0)) >= min_speed:
                fast_ids.add(row["track_id"])

    if not fast_ids:
        return ""

    frames_map: dict[int, list[tuple[int, int, str]]] = {}
    with open(tracks_csv, newline="") as f:
        for row in csv.DictReader(f):
            tid = row["track_id"]
            if tid not in fast_ids:
                continue
            frame = int(row["frame"])
            x = int(float(row["x"]))
            y = int(float(row["y"]))
            frames_map.setdefault(frame, []).append((x, y, tid))

    cap = open_video_capture(str(source_video if source_video else input_video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for x, y, tid in frames_map.get(frame_idx, []):
            cv2.circle(frame, (x, y), dot_radius, dot_color, -1)
            cv2.putText(
                frame, tid,
                (x + dot_radius + 2, y - dot_radius - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, dot_color, 1, cv2.LINE_AA,
            )
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    return str(Path(output_path).resolve())

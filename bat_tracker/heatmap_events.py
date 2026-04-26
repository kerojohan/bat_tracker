from __future__ import annotations

import csv
from dataclasses import dataclass
from math import hypot
from pathlib import Path

import cv2
import numpy as np

from .fast_events import FastEvent


HEATMAP_EVENTS_COLUMNS = [
    "video_id",
    "event_id",
    "source_fast_event_id",
    "source_track_ids",
    "frame_start",
    "frame_end",
    "time_start_sec",
    "time_end_sec",
    "duration_sec",
    "num_points",
    "x_start",
    "y_start",
    "x_end",
    "y_end",
    "displacement_px",
    "path_length_px",
    "straightness",
    "mean_speed_px_sec",
    "direction",
]

HEATMAP_TRACKS_COLUMNS = [
    "video_id",
    "event_id",
    "source_fast_event_id",
    "idx",
    "frame_est",
    "time_sec_est",
    "x",
    "y",
]


@dataclass(frozen=True)
class HeatmapEvent:
    event_id: int
    source_fast_event_id: int
    source_track_ids: list[int]
    video_id: str
    frame_start: int
    frame_end: int
    fps: float
    points: list[tuple[float, float]]
    direction: str


def reconstruct_heatmap_events(
    input_video: str,
    fast_events: list[FastEvent],
    cfg: dict,
    *,
    fps: float,
    frame_count: int,
) -> list[HeatmapEvent]:
    if not bool(cfg.get("enabled", False)) or not fast_events:
        return []

    max_events = int(cfg.get("max_events", 0))
    selected_events = fast_events[:max_events] if max_events > 0 else fast_events
    next_event_id = int(cfg.get("first_event_id", 20001))
    output_events: list[HeatmapEvent] = []

    for fast_event in selected_events:
        if not fast_event.points:
            continue
        frame_start = max(0, min(point.frame for point in fast_event.points) - int(cfg.get("padding_frames", 0)))
        frame_end = min(
            max(0, frame_count - 1),
            max(point.frame for point in fast_event.points) + int(cfg.get("padding_frames", 0)),
        )
        if frame_end <= frame_start:
            continue

        heatmap = _build_motion_heatmap(
            input_video,
            frame_start,
            frame_end,
            threshold=int(cfg.get("threshold", 14)),
            blur_kernel=int(cfg.get("blur_kernel", 5)),
        )
        seed_start, seed_end = _choose_event_axis(
            fast_event,
            seed_y_min=float(cfg.get("seed_y_min", -1.0)),
        )
        try:
            path = _reconstruct_path_from_heatmap(
                heatmap,
                seed_start,
                seed_end,
                percentile=float(cfg.get("percentile", 76.0)),
                corridor_width=float(cfg.get("corridor_width", 180.0)),
                bins=int(cfg.get("bins", 22)),
                y_max=float(cfg.get("y_max", -1.0)),
            )
        except RuntimeError:
            continue

        if len(path) < int(cfg.get("min_points", 6)):
            continue
        displacement = _displacement(path)
        path_length = _path_length(path)
        if displacement < float(cfg.get("min_displacement", 80.0)):
            continue
        if path_length <= 0.0:
            continue

        output_events.append(
            HeatmapEvent(
                event_id=next_event_id,
                source_fast_event_id=fast_event.event_id,
                source_track_ids=fast_event.source_track_ids,
                video_id=fast_event.points[0].video_id,
                frame_start=frame_start,
                frame_end=frame_end,
                fps=fps,
                points=path,
                direction=_infer_direction(path),
            )
        )
        next_event_id += 1

    return output_events


def write_heatmap_events_csv(path: Path, events: list[HeatmapEvent]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEATMAP_EVENTS_COLUMNS)
        writer.writeheader()
        for event in events:
            duration = (event.frame_end - event.frame_start) / event.fps
            displacement = _displacement(event.points)
            path_length = _path_length(event.points)
            writer.writerow(
                {
                    "video_id": event.video_id,
                    "event_id": event.event_id,
                    "source_fast_event_id": event.source_fast_event_id,
                    "source_track_ids": ";".join(str(track_id) for track_id in event.source_track_ids),
                    "frame_start": event.frame_start,
                    "frame_end": event.frame_end,
                    "time_start_sec": round(event.frame_start / event.fps, 4),
                    "time_end_sec": round(event.frame_end / event.fps, 4),
                    "duration_sec": round(duration, 4),
                    "num_points": len(event.points),
                    "x_start": round(event.points[0][0], 2),
                    "y_start": round(event.points[0][1], 2),
                    "x_end": round(event.points[-1][0], 2),
                    "y_end": round(event.points[-1][1], 2),
                    "displacement_px": round(displacement, 2),
                    "path_length_px": round(path_length, 2),
                    "straightness": round(displacement / path_length, 4) if path_length > 0 else 0.0,
                    "mean_speed_px_sec": round(path_length / duration, 2) if duration > 0 else 0.0,
                    "direction": event.direction,
                }
            )


def write_heatmap_tracks_csv(path: Path, events: list[HeatmapEvent]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEATMAP_TRACKS_COLUMNS)
        writer.writeheader()
        for event in events:
            span = max(1, len(event.points) - 1)
            for idx, point in enumerate(event.points):
                frame_est = event.frame_start + (event.frame_end - event.frame_start) * idx / span
                writer.writerow(
                    {
                        "video_id": event.video_id,
                        "event_id": event.event_id,
                        "source_fast_event_id": event.source_fast_event_id,
                        "idx": idx,
                        "frame_est": round(frame_est, 2),
                        "time_sec_est": round(frame_est / event.fps, 4),
                        "x": round(point[0], 2),
                        "y": round(point[1], 2),
                    }
                )


def render_heatmap_events_overlay(
    background_gray: np.ndarray,
    events: list[HeatmapEvent],
    *,
    line_thickness: int = 5,
    alpha: float = 1.0,
) -> np.ndarray:
    if background_gray.ndim == 2:
        base = cv2.cvtColor(background_gray, cv2.COLOR_GRAY2BGR)
    else:
        base = background_gray.copy()
    overlay = base.copy()
    color = (0, 255, 255)
    for event in events:
        pts = [(int(round(x)), int(round(y))) for x, y in event.points]
        for p0, p1 in zip(pts, pts[1:]):
            cv2.line(overlay, p0, p1, color, line_thickness, cv2.LINE_AA)
        for idx, (p0, p1) in enumerate(zip(pts, pts[1:])):
            if idx % 4 == 0:
                cv2.arrowedLine(overlay, p0, p1, color, 2, cv2.LINE_AA, tipLength=0.25)
        if pts:
            cv2.circle(overlay, pts[0], 10, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(overlay, pts[-1], 10, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(
                overlay,
                f"heatmap {event.event_id}",
                (pts[0][0] + 8, max(20, pts[0][1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
    return cv2.addWeighted(overlay, float(alpha), base, 1.0 - float(alpha), 0)


def _build_motion_heatmap(
    input_video: str,
    frame_start: int,
    frame_end: int,
    *,
    threshold: int,
    blur_kernel: int,
) -> np.ndarray:
    if blur_kernel < 1 or blur_kernel % 2 == 0:
        raise ValueError(f"heatmap_events.blur_kernel must be a positive odd integer, got: {blur_kernel}")

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    ok, previous = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Cannot read frame {frame_start} from {input_video}")

    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    heatmap = np.zeros_like(previous_gray, dtype=np.float32)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for _frame_idx in range(frame_start + 1, frame_end + 1):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        previous_blur = cv2.GaussianBlur(previous_gray, (blur_kernel, blur_kernel), 0)
        gray_blur = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        diff = cv2.absdiff(gray_blur, previous_blur)
        _, binary = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
        heatmap += diff.astype(np.float32) * (binary.astype(np.float32) / 255.0)
        previous_gray = gray

    cap.release()
    return cv2.GaussianBlur(heatmap, (9, 9), 0)


def _choose_event_axis(event: FastEvent, *, seed_y_min: float = -1.0) -> tuple[np.ndarray, np.ndarray]:
    points = [point for point in event.points if seed_y_min < 0.0 or point.y >= seed_y_min]
    if len(points) < 2:
        points = event.points
    best_pair = (points[0], points[-1])
    best_score = -1.0
    for start in points:
        for end in points:
            if start is end:
                continue
            distance = hypot(end.x - start.x, end.y - start.y)
            exit_alignment = (start.x + start.y) - (end.x + end.y)
            score = distance + 0.25 * exit_alignment
            if score > best_score:
                best_score = score
                best_pair = (start, end)
    return (
        np.array([best_pair[0].x, best_pair[0].y], dtype=np.float64),
        np.array([best_pair[1].x, best_pair[1].y], dtype=np.float64),
    )


def _reconstruct_path_from_heatmap(
    heatmap: np.ndarray,
    seed_start: np.ndarray,
    seed_end: np.ndarray,
    *,
    percentile: float,
    corridor_width: float,
    bins: int,
    y_max: float,
) -> list[tuple[float, float]]:
    vector = seed_end - seed_start
    length = float(np.linalg.norm(vector))
    if length <= 1e-6:
        raise RuntimeError("heatmap event seed points are identical")
    unit = vector / length

    height, _width = heatmap.shape[:2]
    ys, xs = np.indices((height, heatmap.shape[1]))
    rel_x = xs.astype(np.float64) - seed_start[0]
    rel_y = ys.astype(np.float64) - seed_start[1]
    projection = rel_x * unit[0] + rel_y * unit[1]
    perpendicular = np.abs(rel_x * (-unit[1]) + rel_y * unit[0])

    nonzero = heatmap[heatmap > 0]
    if nonzero.size == 0:
        raise RuntimeError("empty motion heatmap")
    threshold = float(np.percentile(nonzero, percentile))
    corridor = (
        (projection >= -40.0)
        & (projection <= length + 80.0)
        & (perpendicular <= corridor_width)
        & (heatmap >= threshold)
    )
    if y_max > 0.0:
        corridor &= ys < y_max

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(corridor.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel_open)
    route_y, route_x = np.where(mask > 0)
    if route_x.size < 10:
        raise RuntimeError("not enough heatmap support inside corridor")

    route_projection = projection[route_y, route_x]
    route_weights = heatmap[route_y, route_x] ** 1.35
    route_points = np.column_stack([route_x, route_y]).astype(np.float64)

    path: list[tuple[float, float]] = []
    bin_edges = np.linspace(-20.0, length + 50.0, max(3, bins))
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        selected = (route_projection >= lo) & (route_projection < hi)
        if selected.sum() < 5:
            continue
        xy = np.average(route_points[selected], axis=0, weights=route_weights[selected])
        line_xy = seed_start + unit * float(np.dot(xy - seed_start, unit))
        xy = 0.75 * xy + 0.25 * line_xy
        path.append((float(xy[0]), float(xy[1])))

    if len(path) < 2:
        raise RuntimeError("not enough reconstructed path points")
    if path[0][0] + path[0][1] < path[-1][0] + path[-1][1]:
        path.reverse()
    return _smooth_path(path)


def _smooth_path(path: list[tuple[float, float]]) -> list[tuple[float, float]]:
    smoothed: list[tuple[float, float]] = []
    for idx in range(len(path)):
        if idx == 0 or idx == len(path) - 1:
            smoothed.append(path[idx])
            continue
        lo = max(0, idx - 2)
        hi = min(len(path), idx + 3)
        smoothed.append(
            (
                sum(point[0] for point in path[lo:hi]) / float(hi - lo),
                sum(point[1] for point in path[lo:hi]) / float(hi - lo),
            )
        )
    return smoothed


def _path_length(points: list[tuple[float, float]]) -> float:
    return sum(hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))


def _displacement(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    return hypot(points[-1][0] - points[0][0], points[-1][1] - points[0][1])


def _infer_direction(points: list[tuple[float, float]]) -> str:
    if len(points) < 2:
        return "unknown"
    return "exit" if points[-1][1] < points[0][1] else "entry"

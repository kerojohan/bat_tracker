from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from math import hypot
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import numpy as np

from .tracker import TrackPoint


FAST_EVENTS_COLUMNS = [
    "video_id",
    "event_id",
    "source_track_ids",
    "accepted_source_track_ids",
    "rejected_source_track_ids",
    "frame_start",
    "frame_end",
    "time_start_sec",
    "time_end_sec",
    "duration_sec",
    "num_source_tracks",
    "num_points",
    "displacement_px",
    "path_length_px",
    "straightness",
    "mean_speed_px_sec",
    "confidence",
    "direction",
]

FAST_TRACKS_COLUMNS = [
    "video_id",
    "event_id",
    "source_track_id",
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
    "source_track_accepted",
    "source_track_reject_reasons",
]


@dataclass(frozen=True)
class FastEvent:
    event_id: int
    source_track_ids: list[int]
    points: list[TrackPoint]
    accepted_track_ids: list[int]
    rejected_track_ids: list[int]
    confidence: float
    direction: str


def reconstruct_fast_events(
    points: list[TrackPoint],
    assessments: list[dict],
    cfg: dict,
    *,
    frame_shape: tuple[int, int] | None = None,
) -> list[FastEvent]:
    if not bool(cfg.get("enabled", False)):
        return []

    by_track = _points_by_track(points)
    assessment_by_track = {int(row["track_id"]): row for row in assessments}
    candidate_ids = _select_fast_track_ids(by_track, assessment_by_track, cfg)
    if not candidate_ids:
        return []

    groups = _group_fast_track_ids(candidate_ids, by_track, cfg)
    events: list[FastEvent] = []
    next_event_id = int(cfg.get("first_event_id", 10001))
    for group in groups:
        group_points = [
            point
            for track_id in sorted(group)
            for point in by_track[track_id]
        ]
        if not group_points:
            continue

        reconstructed = _reconstruct_event_points(group_points, cfg)
        if len(reconstructed) < int(cfg.get("min_event_points", 4)):
            continue

        displacement = _displacement(reconstructed)
        path_length = _path_length(reconstructed)
        duration = max(0.0, reconstructed[-1].time_sec - reconstructed[0].time_sec)
        mean_speed = path_length / duration if duration > 0 else 0.0
        if displacement < float(cfg.get("min_event_displacement", 80.0)):
            continue
        if path_length < float(cfg.get("min_event_path_length", 120.0)):
            continue
        if mean_speed < float(cfg.get("min_event_mean_speed", 120.0)):
            continue

        accepted_ids = [
            track_id for track_id in sorted(group)
            if str(assessment_by_track.get(track_id, {}).get("accepted", "")).lower() == "true"
        ]
        rejected_ids = [track_id for track_id in sorted(group) if track_id not in accepted_ids]
        confidence = _event_confidence(
            reconstructed,
            accepted_ids=accepted_ids,
            rejected_ids=rejected_ids,
            min_speed=float(cfg.get("min_event_mean_speed", 120.0)),
        )
        direction = _infer_fast_direction(reconstructed, frame_shape)
        events.append(
            FastEvent(
                event_id=next_event_id,
                source_track_ids=sorted(group),
                points=reconstructed,
                accepted_track_ids=accepted_ids,
                rejected_track_ids=rejected_ids,
                confidence=confidence,
                direction=direction,
            )
        )
        next_event_id += 1

    return events


def reconstruct_fast_events_from_candidates(
    assessments: list[dict],
    cfg: dict,
    *,
    fps: float,
    video_id: str,
    frame_shape: tuple[int, int] | None = None,
    first_event_id: int | None = None,
    exclude_source_track_ids: set[int] | None = None,
) -> list[FastEvent]:
    if not bool(cfg.get("enabled", False)):
        return []

    exclude_source_track_ids = exclude_source_track_ids or set()
    rows = [
        row
        for row in assessments
        if int(row["track_id"]) not in exclude_source_track_ids
        and _candidate_row_is_fast(row, cfg)
    ]
    if not rows:
        return []

    groups = _group_candidate_rows(rows, cfg)
    events: list[FastEvent] = []
    next_event_id = first_event_id if first_event_id is not None else int(cfg.get("first_event_id", 10001))
    for group in groups:
        points = _candidate_group_to_points(group, fps=fps, video_id=video_id)
        if len(points) < 2:
            continue
        displacement = _displacement(points)
        path_length = _path_length(points)
        duration = max(0.0, max(p.time_sec for p in points) - min(p.time_sec for p in points))
        mean_speed = path_length / duration if duration > 0 else 0.0
        if displacement < float(cfg.get("min_event_displacement", 80.0)):
            continue
        if path_length < float(cfg.get("min_event_path_length", 120.0)):
            continue
        if mean_speed < float(cfg.get("min_event_mean_speed", 120.0)):
            continue

        source_ids = sorted(int(row["track_id"]) for row in group)
        accepted_ids = sorted(
            int(row["track_id"])
            for row in group
            if str(row.get("accepted", "")).lower() == "true"
        )
        rejected_ids = [track_id for track_id in source_ids if track_id not in accepted_ids]
        events.append(
            FastEvent(
                event_id=next_event_id,
                source_track_ids=source_ids,
                points=points,
                accepted_track_ids=accepted_ids,
                rejected_track_ids=rejected_ids,
                confidence=_event_confidence(
                    points,
                    accepted_ids=accepted_ids,
                    rejected_ids=rejected_ids,
                    min_speed=float(cfg.get("min_event_mean_speed", 120.0)),
                ),
                direction=_infer_fast_direction(points, frame_shape),
            )
        )
        next_event_id += 1
    return events


def write_fast_events_csv(path: Path, events: list[FastEvent]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FAST_EVENTS_COLUMNS)
        writer.writeheader()
        for event in events:
            pts = event.points
            start = min(pts, key=lambda p: p.frame)
            end = max(pts, key=lambda p: p.frame)
            duration = max(0.0, end.time_sec - start.time_sec)
            displacement = _displacement(pts)
            path_length = _path_length(pts)
            writer.writerow(
                {
                    "video_id": start.video_id,
                    "event_id": event.event_id,
                    "source_track_ids": ";".join(str(track_id) for track_id in event.source_track_ids),
                    "accepted_source_track_ids": ";".join(str(track_id) for track_id in event.accepted_track_ids),
                    "rejected_source_track_ids": ";".join(str(track_id) for track_id in event.rejected_track_ids),
                    "frame_start": start.frame,
                    "frame_end": end.frame,
                    "time_start_sec": round(start.time_sec, 4),
                    "time_end_sec": round(end.time_sec, 4),
                    "duration_sec": round(duration, 4),
                    "num_source_tracks": len(event.source_track_ids),
                    "num_points": len(pts),
                    "displacement_px": round(displacement, 2),
                    "path_length_px": round(path_length, 2),
                    "straightness": round(displacement / path_length, 4) if path_length > 0 else 0.0,
                    "mean_speed_px_sec": round(path_length / duration, 2) if duration > 0 else 0.0,
                    "confidence": round(event.confidence, 3),
                    "direction": event.direction,
                }
            )


def write_fast_tracks_csv(
    path: Path,
    events: list[FastEvent],
    assessments: list[dict],
) -> None:
    assessment_by_track = {int(row["track_id"]): row for row in assessments}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FAST_TRACKS_COLUMNS)
        writer.writeheader()
        for event in events:
            source_ids = set(event.source_track_ids)
            for point in event.points:
                source_track_id = point.track_id
                if source_track_id not in source_ids:
                    source_track_id = event.source_track_ids[0]
                assessment = assessment_by_track.get(source_track_id, {})
                writer.writerow(
                    {
                        "video_id": point.video_id,
                        "event_id": event.event_id,
                        "source_track_id": source_track_id,
                        "frame": point.frame,
                        "time_sec": point.time_sec,
                        "x": point.x,
                        "y": point.y,
                        "vx": point.vx,
                        "vy": point.vy,
                        "bbox_x1": point.bbox_x1,
                        "bbox_y1": point.bbox_y1,
                        "bbox_x2": point.bbox_x2,
                        "bbox_y2": point.bbox_y2,
                        "area": point.area,
                        "source_track_accepted": assessment.get("accepted", ""),
                        "source_track_reject_reasons": assessment.get("reject_reasons", ""),
                    }
                )


def render_fast_events_overlay(
    background_gray: np.ndarray,
    events: list[FastEvent],
    *,
    line_thickness: int = 3,
    start_radius: int = 6,
    alpha: float = 1.0,
) -> np.ndarray:
    if background_gray.ndim == 2:
        base = cv2.cvtColor(background_gray, cv2.COLOR_GRAY2BGR)
    else:
        base = background_gray.copy()
    overlay = base.copy()

    for idx, event in enumerate(events):
        color = _event_color(idx)
        pts = [(int(round(p.x)), int(round(p.y))) for p in event.points]
        for p0, p1 in zip(pts, pts[1:]):
            cv2.line(overlay, p0, p1, color, line_thickness, cv2.LINE_AA)
        if pts:
            cv2.circle(overlay, pts[0], start_radius, color, -1, cv2.LINE_AA)
            cv2.circle(overlay, pts[-1], start_radius, color, 2, cv2.LINE_AA)
            cv2.putText(
                overlay,
                f"fast {event.event_id}",
                (pts[0][0] + 8, max(18, pts[0][1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    return cv2.addWeighted(overlay, float(alpha), base, 1.0 - float(alpha), 0)


def _select_fast_track_ids(
    by_track: dict[int, list[TrackPoint]],
    assessment_by_track: dict[int, dict],
    cfg: dict,
) -> list[int]:
    min_track_points = int(cfg.get("min_source_track_points", 3))
    min_displacement = float(cfg.get("min_source_displacement", 35.0))
    min_path = float(cfg.get("min_source_path_length", 60.0))
    min_speed = float(cfg.get("min_source_mean_speed", 80.0))
    min_straightness = float(cfg.get("min_source_straightness", 0.08))
    max_duration = float(cfg.get("max_source_duration_sec", 5.0))
    include_rejected = bool(cfg.get("include_rejected_candidates", True))
    include_accepted = bool(cfg.get("include_accepted_tracks", True))

    selected: list[int] = []
    for track_id, pts in by_track.items():
        assessment = assessment_by_track.get(track_id, {})
        accepted = str(assessment.get("accepted", "")).lower() == "true"
        if accepted and not include_accepted:
            continue
        if not accepted and not include_rejected:
            continue
        if len(pts) < min_track_points:
            continue
        displacement = _displacement(pts)
        path_length = _path_length(pts)
        duration = max(0.0, pts[-1].time_sec - pts[0].time_sec)
        speed = path_length / duration if duration > 0 else 0.0
        straightness = displacement / path_length if path_length > 0 else 0.0
        if max_duration > 0.0 and duration > max_duration:
            continue
        if straightness < min_straightness:
            continue
        if displacement >= min_displacement or path_length >= min_path or speed >= min_speed:
            selected.append(track_id)
    return sorted(selected)


def _group_fast_track_ids(
    track_ids: list[int],
    by_track: dict[int, list[TrackPoint]],
    cfg: dict,
) -> list[list[int]]:
    max_gap_frames = int(cfg.get("max_group_gap_frames", 45))
    max_endpoint_distance = float(cfg.get("max_group_endpoint_distance", 260.0))
    max_overlap_distance = float(cfg.get("max_group_overlap_distance", 180.0))

    parent = {track_id: track_id for track_id in track_ids}

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
        parent[max(ra, rb)] = min(ra, rb)

    for idx, a_id in enumerate(track_ids):
        a = by_track[a_id]
        for b_id in track_ids[idx + 1:]:
            b = by_track[b_id]
            gap = _temporal_gap_frames(a, b)
            if gap > max_gap_frames:
                continue
            if gap == 0:
                overlap_distance = _mean_overlap_distance(a, b)
                if overlap_distance is not None and overlap_distance <= max_overlap_distance:
                    union(a_id, b_id)
            else:
                if _min_endpoint_distance(a, b) <= max_endpoint_distance:
                    gap_cos = _endpoint_gap_cosine(a, b)
                    if gap_cos is None or gap_cos > 0.5:
                        union(a_id, b_id)

    groups_by_root: dict[int, list[int]] = defaultdict(list)
    for track_id in track_ids:
        groups_by_root[find(track_id)].append(track_id)
    return [sorted(group) for group in groups_by_root.values()]


def _reconstruct_event_points(points: list[TrackPoint], cfg: dict) -> list[TrackPoint]:
    by_frame: dict[int, list[TrackPoint]] = defaultdict(list)
    for point in points:
        by_frame[point.frame].append(point)

    frame_items = [(frame, by_frame[frame]) for frame in sorted(by_frame)]
    if not frame_items:
        return []

    selected: list[TrackPoint] = []
    previous: TrackPoint | None = None
    for _, candidates in frame_items:
        if previous is None:
            chosen = _initial_event_point(candidates)
        else:
            chosen = min(
                candidates,
                key=lambda p: (
                    hypot(p.x - previous.x, p.y - previous.y),
                    -p.area,
                ),
            )
        selected.append(chosen)
        previous = chosen

    if bool(cfg.get("order_by_exit_projection", True)) and len(selected) > 2:
        selected = _order_exit_like(selected)
    return selected


def _candidate_row_is_fast(row: dict, cfg: dict) -> bool:
    include_rejected = bool(cfg.get("include_rejected_candidates", True))
    include_accepted = bool(cfg.get("include_accepted_tracks", True))
    accepted = str(row.get("accepted", "")).lower() == "true"
    if accepted and not include_accepted:
        return False
    if not accepted and not include_rejected:
        return False

    duration = float(row.get("duration_sec", 0.0))
    max_duration = float(cfg.get("max_source_duration_sec", 5.0))
    if max_duration > 0.0 and duration > max_duration:
        return False
    if int(row.get("num_detections", 0)) < int(cfg.get("min_source_track_points", 3)):
        return False
    if float(row.get("straightness", 0.0)) < float(cfg.get("min_source_straightness", 0.08)):
        return False

    return (
        float(row.get("displacement_px", 0.0)) >= float(cfg.get("min_source_displacement", 35.0))
        or float(row.get("path_length_px", 0.0)) >= float(cfg.get("min_source_path_length", 60.0))
        or float(row.get("mean_speed_px_sec", 0.0)) >= float(cfg.get("min_source_mean_speed", 80.0))
    )


def _group_candidate_rows(rows: list[dict], cfg: dict) -> list[list[dict]]:
    ids = [int(row["track_id"]) for row in rows]
    by_id = {int(row["track_id"]): row for row in rows}
    parent = {track_id: track_id for track_id in ids}
    max_gap = int(cfg.get("max_group_gap_frames", 45))
    max_dist = float(cfg.get("max_group_endpoint_distance", 260.0))

    def find(track_id: int) -> int:
        while parent[track_id] != track_id:
            parent[track_id] = parent[parent[track_id]]
            track_id = parent[track_id]
        return track_id

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for idx, a_id in enumerate(ids):
        for b_id in ids[idx + 1:]:
            a = by_id[a_id]
            b = by_id[b_id]
            if _candidate_temporal_gap(a, b) > max_gap:
                continue
            if _candidate_min_endpoint_distance(a, b) <= max_dist:
                union(a_id, b_id)

    grouped: dict[int, list[dict]] = defaultdict(list)
    for track_id in ids:
        grouped[find(track_id)].append(by_id[track_id])
    return [sorted(group, key=lambda row: int(row["frame_start"])) for group in grouped.values()]


def _candidate_group_to_points(group: list[dict], *, fps: float, video_id: str) -> list[TrackPoint]:
    points: list[TrackPoint] = []
    for row in sorted(group, key=lambda r: (float(r["x_start"]) + float(r["y_start"]), int(r["frame_start"])), reverse=True):
        track_id = int(row["track_id"])
        frame_start = int(row["frame_start"])
        frame_end = int(row["frame_end"])
        points.append(
            _candidate_endpoint_point(
                video_id,
                track_id,
                frame_start,
                fps,
                float(row["x_start"]),
                float(row["y_start"]),
                float(row.get("mean_area", 1.0) or 1.0),
            )
        )
        points.append(
            _candidate_endpoint_point(
                video_id,
                track_id,
                frame_end,
                fps,
                float(row["x_end"]),
                float(row["y_end"]),
                float(row.get("mean_area", 1.0) or 1.0),
            )
        )
    return points


def _candidate_endpoint_point(
    video_id: str,
    track_id: int,
    frame: int,
    fps: float,
    x: float,
    y: float,
    area: float,
) -> TrackPoint:
    return TrackPoint(
        video_id=video_id,
        track_id=track_id,
        frame=frame,
        time_sec=frame / fps,
        x=x,
        y=y,
        vx=0.0,
        vy=0.0,
        bbox_x1=int(round(x)) - 1,
        bbox_y1=int(round(y)) - 1,
        bbox_x2=int(round(x)) + 1,
        bbox_y2=int(round(y)) + 1,
        area=area,
    )


def _candidate_temporal_gap(a: dict, b: dict) -> int:
    a_start = int(a["frame_start"])
    a_end = int(a["frame_end"])
    b_start = int(b["frame_start"])
    b_end = int(b["frame_end"])
    if a_end < b_start:
        return b_start - a_end
    if b_end < a_start:
        return a_start - b_end
    return 0


def _candidate_min_endpoint_distance(a: dict, b: dict) -> float:
    endpoints_a = (
        (float(a["x_start"]), float(a["y_start"])),
        (float(a["x_end"]), float(a["y_end"])),
    )
    endpoints_b = (
        (float(b["x_start"]), float(b["y_start"])),
        (float(b["x_end"]), float(b["y_end"])),
    )
    return min(hypot(ax - bx, ay - by) for ax, ay in endpoints_a for bx, by in endpoints_b)


def _initial_event_point(candidates: Iterable[TrackPoint]) -> TrackPoint:
    return max(candidates, key=lambda p: (p.area, p.y + p.x))


def _order_exit_like(points: list[TrackPoint]) -> list[TrackPoint]:
    # Fast exits often produce overlapping fragments where the first detected
    # blob is not the physical start.  Projection by x+y gives a stable
    # lower/right to upper/left ordering for these cave-mouth exits while
    # preserving each source point unchanged.
    return sorted(points, key=lambda p: (-(p.x + p.y), p.frame))


def _points_by_track(points: list[TrackPoint]) -> dict[int, list[TrackPoint]]:
    by_track: dict[int, list[TrackPoint]] = defaultdict(list)
    for point in points:
        by_track[point.track_id].append(point)
    for track_id in by_track:
        by_track[track_id] = sorted(by_track[track_id], key=lambda p: p.frame)
    return by_track


def _path_length(points: list[TrackPoint]) -> float:
    return sum(hypot(b.x - a.x, b.y - a.y) for a, b in zip(points, points[1:]))


def _displacement(points: list[TrackPoint]) -> float:
    if len(points) < 2:
        return 0.0
    return hypot(points[-1].x - points[0].x, points[-1].y - points[0].y)


def _temporal_gap_frames(a: list[TrackPoint], b: list[TrackPoint]) -> int:
    if a[-1].frame < b[0].frame:
        return b[0].frame - a[-1].frame
    if b[-1].frame < a[0].frame:
        return a[0].frame - b[-1].frame
    return 0


def _min_endpoint_distance(a: list[TrackPoint], b: list[TrackPoint]) -> float:
    endpoints_a = (a[0], a[-1])
    endpoints_b = (b[0], b[-1])
    return min(hypot(pa.x - pb.x, pa.y - pb.y) for pa in endpoints_a for pb in endpoints_b)


def _endpoint_gap_cosine(a: list[TrackPoint], b: list[TrackPoint]) -> float | None:
    a_end, a_start = a[-1], a[0]
    b_end, b_start = b[-1], b[0]
    if a_end.frame < b_start.frame:
        gap_vec = (b_start.x - a_end.x, b_start.y - a_end.y)
        a_vec = (a_end.x - a_start.x, a_end.y - a_start.y)
        if a_vec[0] == 0 and a_vec[1] == 0:
            return None
        n_gap = (gap_vec[0]**2 + gap_vec[1]**2)**0.5
        n_a = (a_vec[0]**2 + a_vec[1]**2)**0.5
        if n_gap < 1e-6 or n_a < 1e-6:
            return None
        return (gap_vec[0]*a_vec[0] + gap_vec[1]*a_vec[1]) / (n_gap * n_a)
    elif b_end.frame < a_start.frame:
        gap_vec = (a_start.x - b_end.x, a_start.y - b_end.y)
        b_vec = (b_end.x - b_start.x, b_end.y - b_start.y)
        if b_vec[0] == 0 and b_vec[1] == 0:
            return None
        n_gap = (gap_vec[0]**2 + gap_vec[1]**2)**0.5
        n_b = (b_vec[0]**2 + b_vec[1]**2)**0.5
        if n_gap < 1e-6 or n_b < 1e-6:
            return None
        return (gap_vec[0]*b_vec[0] + gap_vec[1]*b_vec[1]) / (n_gap * n_b)
    return None


def _mean_overlap_distance(a: list[TrackPoint], b: list[TrackPoint]) -> float | None:
    by_frame_a = {p.frame: p for p in a}
    by_frame_b = {p.frame: p for p in b}
    common = sorted(set(by_frame_a).intersection(by_frame_b))
    if not common:
        return None
    return sum(hypot(by_frame_a[f].x - by_frame_b[f].x, by_frame_a[f].y - by_frame_b[f].y) for f in common) / len(common)


def _event_confidence(
    points: list[TrackPoint],
    *,
    accepted_ids: list[int],
    rejected_ids: list[int],
    min_speed: float,
) -> float:
    duration = max(0.0, points[-1].time_sec - points[0].time_sec)
    speed = _path_length(points) / duration if duration > 0 else 0.0
    speed_score = min(1.0, speed / max(1.0, min_speed * 2.0))
    accepted_bonus = 0.2 if accepted_ids else 0.0
    rejected_penalty = min(0.25, 0.04 * len(rejected_ids))
    return max(0.0, min(1.0, 0.55 + 0.35 * speed_score + accepted_bonus - rejected_penalty))


def _infer_fast_direction(points: list[TrackPoint], frame_shape: tuple[int, int] | None) -> str:
    if len(points) < 2 or frame_shape is None:
        return "unknown"
    height, _ = frame_shape
    dy = points[-1].y - points[0].y
    if dy < -0.12 * float(height):
        return "exit"
    if dy > 0.12 * float(height):
        return "entry"
    return "inside"


def _event_color(idx: int) -> tuple[int, int, int]:
    palette = [
        (30, 210, 255),
        (80, 255, 80),
        (255, 120, 80),
        (200, 90, 255),
        (255, 220, 60),
    ]
    return palette[idx % len(palette)]

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from math import hypot
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import cv2
import numpy as np

from .tracker import TrackPoint


TRACK_DEDUP_COLUMNS = [
    "track_id_original",
    "track_id_final",
    "duplicate_group_id",
    "duplicate_decision",
    "duplicate_score",
    "reason",
    "source_track_ids",
    "paired_track_ids",
]


@dataclass(frozen=True)
class TrackSummary:
    track_id: int
    frame_start: int
    frame_end: int
    point_count: int
    duration_sec: float
    displacement_px: float
    path_length_px: float
    straightness: float
    mean_speed_px_sec: float
    mean_area: float
    vector_x: float
    vector_y: float
    quality: float


@dataclass(frozen=True)
class DuplicatePair:
    track_a: int
    track_b: int
    duplicate_score: float
    relation: str
    reason: str
    common_frames: int
    overlap_ratio_short: float
    mean_distance: float
    median_distance: float
    p90_distance: float
    gap_frames: int
    endpoint_distance: float
    direction_similarity: float
    speed_similarity: float


@dataclass(frozen=True)
class TrackDuplicateResult:
    points: list[TrackPoint]
    rows: list[dict]
    pairs: list[dict]
    enabled: bool
    groups_total: int
    pairs_total: int
    tracks_discarded: int
    tracks_merged: int


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _path_length(points: Sequence[TrackPoint]) -> float:
    if len(points) < 2:
        return 0.0
    return sum(hypot(p1.x - p0.x, p1.y - p0.y) for p0, p1 in zip(points[:-1], points[1:]))


def _vector_cosine(ax: float, ay: float, bx: float, by: float) -> float | None:
    na = hypot(ax, ay)
    nb = hypot(bx, by)
    if na <= 1e-6 or nb <= 1e-6:
        return None
    return (ax * bx + ay * by) / (na * nb)


def _speed_similarity(speed_a: float, speed_b: float) -> float:
    hi = max(speed_a, speed_b)
    lo = min(speed_a, speed_b)
    if hi <= 1e-6:
        return 1.0
    return _clip01(lo / hi)


def _points_by_track(points: Iterable[TrackPoint]) -> dict[int, list[TrackPoint]]:
    by_track: dict[int, list[TrackPoint]] = defaultdict(list)
    for point in points:
        by_track[int(point.track_id)].append(point)
    for track_id in by_track:
        by_track[track_id].sort(key=lambda point: point.frame)
    return dict(by_track)


def _summarize_track(points: Sequence[TrackPoint]) -> TrackSummary:
    start = points[0]
    end = points[-1]
    displacement = hypot(end.x - start.x, end.y - start.y)
    path_length = _path_length(points)
    duration = max(0.0, end.time_sec - start.time_sec)
    straightness = displacement / path_length if path_length > 1e-6 else 0.0
    mean_speed = path_length / duration if duration > 1e-6 else 0.0
    mean_area = sum(point.area for point in points) / max(1, len(points))
    quality = (
        0.30 * _clip01(len(points) / 12.0)
        + 0.25 * _clip01(displacement / 80.0)
        + 0.25 * _clip01(path_length / 120.0)
        + 0.20 * _clip01(straightness)
    )
    return TrackSummary(
        track_id=int(start.track_id),
        frame_start=int(start.frame),
        frame_end=int(end.frame),
        point_count=len(points),
        duration_sec=float(duration),
        displacement_px=float(displacement),
        path_length_px=float(path_length),
        straightness=float(straightness),
        mean_speed_px_sec=float(mean_speed),
        mean_area=float(mean_area),
        vector_x=float(end.x - start.x),
        vector_y=float(end.y - start.y),
        quality=float(quality),
    )


def _evaluate_pair(
    track_a: Sequence[TrackPoint],
    track_b: Sequence[TrackPoint],
    summary_a: TrackSummary,
    summary_b: TrackSummary,
    cfg: dict,
) -> DuplicatePair | None:
    max_spatial = max(1e-6, float(cfg.get("max_spatial_distance_px", 16.0)))
    max_gap = int(cfg.get("max_temporal_gap_frames", 8))
    min_direction = float(cfg.get("min_direction_similarity", 0.75))
    min_speed = float(cfg.get("min_speed_similarity", 0.50))

    frames_a = {point.frame: point for point in track_a}
    frames_b = {point.frame: point for point in track_b}
    common_frames = sorted(set(frames_a).intersection(frames_b))
    direction = _vector_cosine(summary_a.vector_x, summary_a.vector_y, summary_b.vector_x, summary_b.vector_y)
    direction_similarity = 1.0 if direction is None else float(direction)
    speed_similarity = _speed_similarity(summary_a.mean_speed_px_sec, summary_b.mean_speed_px_sec)

    if common_frames:
        distances = np.array(
            [
                hypot(frames_a[frame].x - frames_b[frame].x, frames_a[frame].y - frames_b[frame].y)
                for frame in common_frames
            ],
            dtype=np.float64,
        )
        mean_distance = float(np.mean(distances))
        median_distance = float(np.median(distances))
        p90_distance = float(np.percentile(distances, 90.0))
        overlap_ratio_short = len(common_frames) / float(max(1, min(len(track_a), len(track_b))))
        spatial_score = _clip01(1.0 - median_distance / max_spatial)
        overlap_score = _clip01(overlap_ratio_short)
        direction_score = _clip01((direction_similarity + 1.0) / 2.0)
        score = 0.45 * spatial_score + 0.25 * overlap_score + 0.20 * direction_score + 0.10 * speed_similarity

        blockers = []
        if median_distance > max_spatial:
            blockers.append("spatial_distance")
        if direction_similarity < min_direction:
            blockers.append("direction")
        if speed_similarity < min_speed:
            blockers.append("speed")
        if blockers:
            score = min(score, 0.49)
            reason = "blocked:" + ",".join(blockers)
        else:
            reason = "overlap_close"

        return DuplicatePair(
            track_a=summary_a.track_id,
            track_b=summary_b.track_id,
            duplicate_score=round(float(score), 4),
            relation="overlap",
            reason=reason,
            common_frames=len(common_frames),
            overlap_ratio_short=round(float(overlap_ratio_short), 4),
            mean_distance=round(mean_distance, 4),
            median_distance=round(median_distance, 4),
            p90_distance=round(p90_distance, 4),
            gap_frames=0,
            endpoint_distance=0.0,
            direction_similarity=round(direction_similarity, 4),
            speed_similarity=round(speed_similarity, 4),
        )

    if summary_a.frame_end < summary_b.frame_start:
        first, second = track_a, track_b
        first_summary, second_summary = summary_a, summary_b
    elif summary_b.frame_end < summary_a.frame_start:
        first, second = track_b, track_a
        first_summary, second_summary = summary_b, summary_a
    else:
        return None

    gap = int(second_summary.frame_start - first_summary.frame_end)
    if gap < 1 or gap > max_gap:
        return None

    endpoint_distance = hypot(second[0].x - first[-1].x, second[0].y - first[-1].y)
    local_a0 = first[max(0, len(first) - min(3, len(first)))]
    local_b1 = second[min(len(second) - 1, 2)]
    first_local_x = first[-1].x - local_a0.x
    first_local_y = first[-1].y - local_a0.y
    second_local_x = local_b1.x - second[0].x
    second_local_y = local_b1.y - second[0].y
    local_direction = _vector_cosine(first_local_x, first_local_y, second_local_x, second_local_y)
    local_direction_similarity = 1.0 if local_direction is None else float(local_direction)
    direction_similarity = min(direction_similarity, local_direction_similarity)

    spatial_score = _clip01(1.0 - endpoint_distance / (2.0 * max_spatial))
    temporal_score = _clip01(1.0 - (gap - 1) / max(1.0, float(max_gap)))
    direction_score = _clip01((direction_similarity + 1.0) / 2.0)
    score = 0.40 * spatial_score + 0.20 * temporal_score + 0.25 * direction_score + 0.15 * speed_similarity

    blockers = []
    if endpoint_distance > max_spatial:
        blockers.append("endpoint_distance")
    if direction_similarity < min_direction:
        blockers.append("direction")
    if speed_similarity < min_speed:
        blockers.append("speed")
    if blockers:
        score = min(score, 0.49)
        reason = "blocked:" + ",".join(blockers)
    else:
        reason = "temporal_continuation"

    return DuplicatePair(
        track_a=summary_a.track_id,
        track_b=summary_b.track_id,
        duplicate_score=round(float(score), 4),
        relation="handoff",
        reason=reason,
        common_frames=0,
        overlap_ratio_short=0.0,
        mean_distance=0.0,
        median_distance=0.0,
        p90_distance=0.0,
        gap_frames=gap,
        endpoint_distance=round(float(endpoint_distance), 4),
        direction_similarity=round(float(direction_similarity), 4),
        speed_similarity=round(float(speed_similarity), 4),
    )


def _best_track_id(track_ids: Sequence[int], summaries: dict[int, TrackSummary]) -> int:
    return max(
        track_ids,
        key=lambda track_id: (
            summaries[track_id].quality,
            summaries[track_id].point_count,
            summaries[track_id].path_length_px,
            -track_id,
        ),
    )


def _copy_point(point: TrackPoint, track_id: int) -> TrackPoint:
    return TrackPoint(
        video_id=point.video_id,
        track_id=track_id,
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


def deduplicate_track_points(
    points: list[TrackPoint],
    cfg: dict,
) -> TrackDuplicateResult:
    if not bool(cfg.get("enable_track_deduplication", False)):
        return TrackDuplicateResult(points=points, rows=[], pairs=[], enabled=False, groups_total=0, pairs_total=0, tracks_discarded=0, tracks_merged=0)

    by_track = _points_by_track(points)
    if len(by_track) < 2:
        rows = [
            {
                "track_id_original": track_id,
                "track_id_final": track_id,
                "duplicate_group_id": "",
                "duplicate_decision": "keep",
                "duplicate_score": "0.0000",
                "reason": "single_track",
                "source_track_ids": str(track_id),
                "paired_track_ids": "",
            }
            for track_id in sorted(by_track)
        ]
        return TrackDuplicateResult(points=points, rows=rows, pairs=[], enabled=True, groups_total=0, pairs_total=0, tracks_discarded=0, tracks_merged=0)

    summaries = {track_id: _summarize_track(track_points) for track_id, track_points in by_track.items()}
    min_score = float(cfg.get("min_duplicate_score", 0.75))
    strategy = str(cfg.get("merge_strategy", "mark")).strip().lower()
    if strategy not in {"mark", "discard", "merge", "auto"}:
        strategy = "mark"

    parent = {track_id: track_id for track_id in by_track}

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
        keep, drop = (ra, rb) if ra < rb else (rb, ra)
        parent[drop] = keep

    duplicate_pairs: list[DuplicatePair] = []
    all_pairs: list[DuplicatePair] = []
    track_ids = sorted(by_track)
    for i, track_a_id in enumerate(track_ids):
        for track_b_id in track_ids[i + 1:]:
            pair = _evaluate_pair(
                by_track[track_a_id],
                by_track[track_b_id],
                summaries[track_a_id],
                summaries[track_b_id],
                cfg,
            )
            if pair is None:
                continue
            all_pairs.append(pair)
            if pair.duplicate_score >= min_score and not pair.reason.startswith("blocked:"):
                duplicate_pairs.append(pair)
                union(track_a_id, track_b_id)

    groups_raw: dict[int, list[int]] = defaultdict(list)
    for track_id in track_ids:
        groups_raw[find(track_id)].append(track_id)
    duplicate_groups = sorted(
        (sorted(group) for group in groups_raw.values() if len(group) > 1),
        key=lambda item: item[0],
    )
    group_by_track: dict[int, int] = {}
    for group_index, group in enumerate(duplicate_groups, start=1):
        for track_id in group:
            group_by_track[track_id] = group_index

    final_track_id_by_original = {track_id: track_id for track_id in track_ids}
    decision_by_original = {track_id: "keep" for track_id in track_ids}
    reason_by_original = {track_id: "not_duplicate" for track_id in track_ids}
    score_by_original = {track_id: 0.0 for track_id in track_ids}
    discarded: set[int] = set()
    merged_sources: set[int] = set()

    pair_score_by_track: dict[int, float] = defaultdict(float)
    pair_reason_by_track: dict[int, list[str]] = defaultdict(list)
    for pair in duplicate_pairs:
        pair_score_by_track[pair.track_a] = max(pair_score_by_track[pair.track_a], pair.duplicate_score)
        pair_score_by_track[pair.track_b] = max(pair_score_by_track[pair.track_b], pair.duplicate_score)
        pair_reason_by_track[pair.track_a].append(pair.reason)
        pair_reason_by_track[pair.track_b].append(pair.reason)

    for group in duplicate_groups:
        best_id = _best_track_id(group, summaries)
        group_pairs = [pair for pair in duplicate_pairs if pair.track_a in group and pair.track_b in group]
        relation_set = {pair.relation for pair in group_pairs}
        effective_strategy = strategy
        if strategy == "auto":
            effective_strategy = "merge" if relation_set == {"handoff"} else "discard"

        if effective_strategy == "merge":
            for track_id in group:
                final_track_id_by_original[track_id] = best_id
                decision_by_original[track_id] = "merge"
                reason_by_original[track_id] = "merge:" + ",".join(sorted(set(pair_reason_by_track[track_id])))
                score_by_original[track_id] = pair_score_by_track[track_id]
                if track_id != best_id:
                    merged_sources.add(track_id)
        elif effective_strategy == "discard":
            for track_id in group:
                final_track_id_by_original[track_id] = best_id
                score_by_original[track_id] = pair_score_by_track[track_id]
                if track_id == best_id:
                    decision_by_original[track_id] = "keep"
                    reason_by_original[track_id] = "best_quality_in_duplicate_group"
                else:
                    decision_by_original[track_id] = "discard"
                    reason_by_original[track_id] = "discard:lower_quality_duplicate"
                    discarded.add(track_id)
        else:
            for track_id in group:
                decision_by_original[track_id] = "uncertain"
                reason_by_original[track_id] = "duplicate_candidate_mark_only"
                score_by_original[track_id] = pair_score_by_track[track_id]

    if strategy in {"merge", "auto"} and merged_sources:
        remapped: list[TrackPoint] = []
        for point in points:
            remapped.append(_copy_point(point, final_track_id_by_original[int(point.track_id)]))
        by_track_frame: dict[tuple[int, int], list[TrackPoint]] = defaultdict(list)
        for point in remapped:
            by_track_frame[(point.track_id, point.frame)].append(point)
        final_points: list[TrackPoint] = []
        for key in sorted(by_track_frame):
            candidates = by_track_frame[key]
            final_points.append(max(candidates, key=lambda point: (point.area, -abs(point.vx) - abs(point.vy), -point.x, -point.y)))
    elif strategy in {"discard", "auto"} and discarded:
        final_points = [point for point in points if int(point.track_id) not in discarded]
    else:
        final_points = list(points)
    final_points = sorted(final_points, key=lambda point: (point.track_id, point.frame))

    paired_by_track: dict[int, set[int]] = defaultdict(set)
    for pair in duplicate_pairs:
        paired_by_track[pair.track_a].add(pair.track_b)
        paired_by_track[pair.track_b].add(pair.track_a)

    rows: list[dict] = []
    for track_id in track_ids:
        group_id = group_by_track.get(track_id, "")
        if group_id == "":
            source_ids = [track_id]
        else:
            source_ids = duplicate_groups[int(group_id) - 1]
        rows.append(
            {
                "track_id_original": int(track_id),
                "track_id_final": int(final_track_id_by_original[track_id]),
                "duplicate_group_id": group_id,
                "duplicate_decision": decision_by_original[track_id],
                "duplicate_score": f"{score_by_original[track_id]:.4f}",
                "reason": reason_by_original[track_id],
                "source_track_ids": ";".join(str(item) for item in source_ids),
                "paired_track_ids": ";".join(str(item) for item in sorted(paired_by_track.get(track_id, set()))),
            }
        )

    return TrackDuplicateResult(
        points=final_points,
        rows=rows,
        pairs=[asdict(pair) for pair in all_pairs],
        enabled=True,
        groups_total=len(duplicate_groups),
        pairs_total=len(duplicate_pairs),
        tracks_discarded=len(discarded),
        tracks_merged=len(merged_sources),
    )


def write_track_deduplication_csv(path: str | Path, rows: Sequence[dict]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACK_DEDUP_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_track_deduplication_json(path: str | Path, result: TrackDuplicateResult) -> None:
    payload = {
        "enabled": result.enabled,
        "groups_total": result.groups_total,
        "pairs_total": result.pairs_total,
        "tracks_discarded": result.tracks_discarded,
        "tracks_merged": result.tracks_merged,
        "tracks": list(result.rows),
        "pairs": list(result.pairs),
    }
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def render_track_deduplication_overlay(
    background_gray: np.ndarray,
    points: Sequence[TrackPoint],
    rows: Sequence[dict],
) -> np.ndarray:
    base = cv2.cvtColor(background_gray, cv2.COLOR_GRAY2BGR)
    if not rows:
        return base

    rows_by_track = {int(row["track_id_original"]): row for row in rows}
    grouped = _points_by_track(points)
    palette = [
        (0, 180, 255),
        (255, 80, 80),
        (80, 220, 120),
        (220, 120, 255),
        (255, 210, 70),
        (80, 180, 255),
    ]
    overlay = base.copy()
    for track_id, track_points in grouped.items():
        row = rows_by_track.get(track_id)
        if row is None or row.get("duplicate_group_id", "") == "":
            continue
        group_id = int(row["duplicate_group_id"])
        color = palette[(group_id - 1) % len(palette)]
        pts = np.array([[int(round(point.x)), int(round(point.y))] for point in track_points], dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(overlay, [pts], False, color, 2, lineType=cv2.LINE_AA)
        for point in (track_points[0], track_points[-1]):
            cv2.circle(overlay, (int(round(point.x)), int(round(point.y))), 5, color, -1, lineType=cv2.LINE_AA)
        label = f"{track_id}:{row['duplicate_decision']}"
        cv2.putText(
            overlay,
            label,
            (int(round(track_points[0].x)) + 6, int(round(track_points[0].y)) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return cv2.addWeighted(base, 0.72, overlay, 0.85, 0)

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from .tracker import TrackPoint


def track_color(track_id: int) -> Tuple[int, int, int]:
    seed = np.random.default_rng(track_id)
    bgr = seed.integers(32, 256, size=3, dtype=np.uint8)
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def render_tracks_overlay(
    background_gray: np.ndarray,
    points: Sequence[TrackPoint],
    line_thickness: int,
    start_radius: int,
    alpha: float = 1.0,
) -> np.ndarray:
    base = cv2.cvtColor(background_gray, cv2.COLOR_GRAY2BGR)
    canvas = base.copy()

    by_track: Dict[int, List[TrackPoint]] = defaultdict(list)
    for point in points:
        by_track[point.track_id].append(point)

    for track_id, track_points in by_track.items():
        track_points = sorted(track_points, key=lambda p: p.frame)
        color = track_color(track_id)

        if len(track_points) >= 2:
            for p0, p1 in zip(track_points[:-1], track_points[1:]):
                cv2.line(
                    canvas,
                    (int(round(p0.x)), int(round(p0.y))),
                    (int(round(p1.x)), int(round(p1.y))),
                    color,
                    thickness=max(1, line_thickness),
                    lineType=cv2.LINE_AA,
                )

        start = track_points[0]
        cv2.circle(
            canvas,
            (int(round(start.x)), int(round(start.y))),
            radius=max(2, start_radius),
            color=color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

    alpha = float(max(0.0, min(1.0, alpha)))
    if alpha < 1.0:
        out = cv2.addWeighted(canvas, alpha, base, 1.0 - alpha, 0)
        return out

    return canvas

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
    draw_track_labels: bool = False,
    draw_track_labels_at_end: bool = False,
    label_font_scale: float = 0.5,
    label_thickness: int = 1,
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
        start_xy = (int(round(start.x)), int(round(start.y)))
        cv2.circle(
            canvas,
            start_xy,
            radius=max(2, start_radius),
            color=color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        if draw_track_labels:
            label = str(track_id)
            label_x = int(start_xy[0] + max(4, start_radius + 2))
            label_y = int(start_xy[1] - max(4, start_radius + 2))
            font_scale = max(0.3, float(label_font_scale))
            text_thickness = max(1, int(label_thickness))
            # Draw an outline first to keep labels readable on bright backgrounds.
            cv2.putText(
                canvas,
                label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                thickness=text_thickness + 2,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness=text_thickness,
                lineType=cv2.LINE_AA,
            )
        if draw_track_labels_at_end and len(track_points) >= 2:
            end = track_points[-1]
            end_xy = (int(round(end.x)), int(round(end.y)))
            label = str(track_id)
            label_x = int(end_xy[0] + max(4, start_radius + 2))
            label_y = int(end_xy[1] - max(4, start_radius + 2))
            font_scale = max(0.3, float(label_font_scale))
            text_thickness = max(1, int(label_thickness))
            cv2.putText(
                canvas,
                label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                thickness=text_thickness + 2,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness=text_thickness,
                lineType=cv2.LINE_AA,
            )

    alpha = float(max(0.0, min(1.0, alpha)))
    if alpha < 1.0:
        out = cv2.addWeighted(canvas, alpha, base, 1.0 - alpha, 0)
        return out

    return canvas

"""
Suavitzat de trajectòries post-tracking (només representació i geometria per events).

No modifica detecció ni associació; genera còpies de TrackPoint amb (x,y) suavitzats.
"""
from __future__ import annotations

from dataclasses import replace
from typing import List

import numpy as np

from .tracker import TrackPoint


def _moving_average_same_length(y: np.ndarray, window: int) -> np.ndarray:
    """Mitjana mòbil amb mateixa longitud que y (padding edge, finestra senar)."""
    w = max(3, window if window % 2 == 1 else window + 1)
    n = len(y)
    if n < w:
        return y.astype(np.float64).copy()
    y = np.asarray(y, dtype=np.float64)
    pad = w // 2
    yp = np.pad(y, (pad, pad), mode="edge")
    k = np.ones(w, dtype=np.float64) / float(w)
    sm = np.convolve(yp, k, mode="valid")
    assert len(sm) == n
    return sm


def smooth_track_points(points: List[TrackPoint], window: int) -> List[TrackPoint]:
    """
    Per cada track_id, aplica mitjana mòbil independent sobre x i y (ordenat per frame).

    Manté frame, bbox, area, video_id, track_id, vx, vy originals (només es canvien x, y).
    """
    if not points:
        return []
    by: dict[int, list[TrackPoint]] = {}
    for p in points:
        by.setdefault(p.track_id, []).append(p)

    out: List[TrackPoint] = []
    for tid in sorted(by.keys()):
        tps = sorted(by[tid], key=lambda p: p.frame)
        xs = np.array([p.x for p in tps], dtype=np.float64)
        ys = np.array([p.y for p in tps], dtype=np.float64)
        xs_s = _moving_average_same_length(xs, window)
        ys_s = _moving_average_same_length(ys, window)
        for p, x, y in zip(tps, xs_s, ys_s):
            out.append(replace(p, x=float(x), y=float(y)))
    return out

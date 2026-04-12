from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .detection import Detection


@dataclass
class TrackPoint:
    video_id: str
    track_id: int
    frame: int
    time_sec: float
    x: float
    y: float
    vx: float
    vy: float
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    area: float


@dataclass
class ActiveTrack:
    track_id: int
    x: float
    y: float
    vx: float
    vy: float
    last_frame: int
    missed: int


class GreedyTracker:
    def __init__(self, max_distance: float, max_missed: int, fps: float, video_id: str):
        self.max_distance = float(max_distance)
        self.max_distance_sq = self.max_distance * self.max_distance
        self.max_missed = int(max_missed)
        self.fps = float(fps)
        self.video_id = video_id
        self._next_track_id = 1
        self._active: Dict[int, ActiveTrack] = {}

    def step(self, frame_idx: int, detections: List[Detection]) -> List[TrackPoint]:
        points: List[TrackPoint] = []

        unmatched_track_ids = set(self._active.keys())
        unmatched_det_idxs = set(range(len(detections)))

        assignments: List[Tuple[int, int]] = []

        if self._active and detections:
            track_ids_list = list(self._active.keys())
            n_tracks = len(track_ids_list)
            n_dets = len(detections)
            max_dist = self.max_distance
            # Sentinel: larger than any valid distance, used to mark invalid pairs
            INF = max_dist * 1e6

            cost = np.full((n_tracks, n_dets), INF, dtype=np.float64)
            for i, track_id in enumerate(track_ids_list):
                track = self._active[track_id]
                dt_pred = max(1, frame_idx - track.last_frame) / self.fps
                pred_x = track.x + track.vx * dt_pred
                pred_y = track.y + track.vy * dt_pred
                for j, det in enumerate(detections):
                    dx = pred_x - det.x
                    dy = pred_y - det.y
                    d = (dx * dx + dy * dy) ** 0.5
                    if d <= max_dist:
                        cost[i, j] = d

            row_ind, col_ind = linear_sum_assignment(cost)
            for i, j in zip(row_ind, col_ind):
                if cost[i, j] < INF:
                    track_id = track_ids_list[i]
                    assignments.append((track_id, j))
                    unmatched_track_ids.discard(track_id)
                    unmatched_det_idxs.discard(j)

        for track_id, det_idx in assignments:
            track = self._active[track_id]
            det = detections[det_idx]
            dt_frames = max(1, frame_idx - track.last_frame)
            dt = dt_frames / self.fps
            new_vx = (det.x - track.x) / dt
            new_vy = (det.y - track.y) / dt
            # Smooth velocity to reduce ID switches when detections are noisy.
            vx = 0.6 * new_vx + 0.4 * track.vx
            vy = 0.6 * new_vy + 0.4 * track.vy

            track.x = det.x
            track.y = det.y
            track.vx = vx
            track.vy = vy
            track.last_frame = frame_idx
            track.missed = 0

            points.append(
                TrackPoint(
                    video_id=self.video_id,
                    track_id=track_id,
                    frame=frame_idx,
                    time_sec=frame_idx / self.fps,
                    x=det.x,
                    y=det.y,
                    vx=vx,
                    vy=vy,
                    bbox_x1=det.bbox_x1,
                    bbox_y1=det.bbox_y1,
                    bbox_x2=det.bbox_x2,
                    bbox_y2=det.bbox_y2,
                    area=det.area,
                )
            )

        to_delete: List[int] = []
        for track_id, track in self._active.items():
            if track_id in unmatched_track_ids:
                track.missed += 1
                track.vx *= 0.9
                track.vy *= 0.9
                if track.missed > self.max_missed:
                    to_delete.append(track_id)

        for track_id in to_delete:
            del self._active[track_id]

        for det_idx in unmatched_det_idxs:
            det = detections[det_idx]
            track_id = self._next_track_id
            self._next_track_id += 1
            self._active[track_id] = ActiveTrack(
                track_id=track_id,
                x=det.x,
                y=det.y,
                vx=0.0,
                vy=0.0,
                last_frame=frame_idx,
                missed=0,
            )
            points.append(
                TrackPoint(
                    video_id=self.video_id,
                    track_id=track_id,
                    frame=frame_idx,
                    time_sec=frame_idx / self.fps,
                    x=det.x,
                    y=det.y,
                    vx=0.0,
                    vy=0.0,
                    bbox_x1=det.bbox_x1,
                    bbox_y1=det.bbox_y1,
                    bbox_x2=det.bbox_x2,
                    bbox_y2=det.bbox_y2,
                    area=det.area,
                )
            )

        return points

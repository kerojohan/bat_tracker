from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Dict, List, Tuple

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
        self.max_missed = int(max_missed)
        self.fps = float(fps)
        self.video_id = video_id
        self._next_track_id = 1
        self._active: Dict[int, ActiveTrack] = {}

    def step(self, frame_idx: int, detections: List[Detection]) -> List[TrackPoint]:
        points: List[TrackPoint] = []

        unmatched_track_ids = set(self._active.keys())
        unmatched_det_idxs = set(range(len(detections)))

        candidate_pairs: List[Tuple[float, int, int]] = []
        for track_id, track in self._active.items():
            dt_pred = max(1, frame_idx - track.last_frame) / self.fps
            pred_x = track.x + track.vx * dt_pred
            pred_y = track.y + track.vy * dt_pred
            for det_idx, det in enumerate(detections):
                d = hypot(pred_x - det.x, pred_y - det.y)
                if d <= self.max_distance:
                    candidate_pairs.append((d, track_id, det_idx))

        candidate_pairs.sort(key=lambda t: t[0])
        assignments: List[Tuple[int, int]] = []

        for _, track_id, det_idx in candidate_pairs:
            if track_id in unmatched_track_ids and det_idx in unmatched_det_idxs:
                assignments.append((track_id, det_idx))
                unmatched_track_ids.remove(track_id)
                unmatched_det_idxs.remove(det_idx)

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

from __future__ import annotations

from dataclasses import dataclass, field
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
    bbox_w: float = 0.0
    bbox_h: float = 0.0
    speed_px_sec: float = 0.0
    last_detection_score: float = 0.0
    consecutive_hits: int = 0
    birth_frame: int = 0


@dataclass
class TrackingDebugRow:
    frame: int
    track_id: int
    state: str
    pred_x: float
    pred_y: float
    gate_px: float
    matched: bool
    det_x: float
    det_y: float
    match_dist: float
    det_score: float
    bbox_w: float
    bbox_h: float
    speed_px_sec: float
    missed_before: int
    missed_after: int
    birth_reason: str
    kill_reason: str


class GreedyTracker:
    def __init__(
        self,
        max_distance: float,
        max_missed: int,
        fps: float,
        video_id: str,
        *,
        adaptive_max_distance_enabled: bool = False,
        adaptive_max_distance_base: float = 120.0,
        adaptive_max_distance_speed_gain: float = 1.25,
        adaptive_max_distance_bbox_gain: float = 0.35,
        adaptive_max_distance_cap: float = 220.0,
        two_stage_association_enabled: bool = False,
        two_stage_association_score_threshold: float = 0.4,
        export_debug: bool = False,
    ):
        self.max_distance = float(max_distance)
        self.max_distance_sq = self.max_distance * self.max_distance
        self.max_missed = int(max_missed)
        self.fps = float(fps)
        self.video_id = video_id
        self._next_track_id = 1
        self._active: Dict[int, ActiveTrack] = {}
        self.adaptive_enabled = bool(adaptive_max_distance_enabled)
        self.adaptive_base = float(adaptive_max_distance_base)
        self.adaptive_speed_gain = float(adaptive_max_distance_speed_gain)
        self.adaptive_bbox_gain = float(adaptive_max_distance_bbox_gain)
        self.adaptive_cap = float(adaptive_max_distance_cap)
        self.two_stage_enabled = bool(two_stage_association_enabled)
        self.two_stage_score_threshold = float(two_stage_association_score_threshold)
        self.export_debug = bool(export_debug)
        self.debug_rows: List[TrackingDebugRow] = []

    def _adaptive_gate(self, track: ActiveTrack) -> float:
        if not self.adaptive_enabled:
            return self.max_distance
        speed_per_frame = track.speed_px_sec / self.fps if track.speed_px_sec > 0.0 else 0.0
        bbox_diag = (track.bbox_w * track.bbox_w + track.bbox_h * track.bbox_h) ** 0.5
        gate = self.adaptive_base + speed_per_frame * self.adaptive_speed_gain + bbox_diag * self.adaptive_bbox_gain
        return max(self.adaptive_base, min(gate, self.adaptive_cap))

    def _match_stage(
        self,
        frame_idx: int,
        detections: List[Detection],
        det_indices: List[int],
        track_ids: List[int],
        gate_map: Dict[int, float],
        unmatched_track_ids: set,
        unmatched_det_idxs: set,
    ) -> List[Tuple[int, int]]:
        assignments: List[Tuple[int, int]] = []
        n_tracks = len(track_ids)
        n_dets = len(det_indices)
        base_max_dist = self.max_distance
        INF = base_max_dist * 1e6

        cost = np.full((n_tracks, n_dets), INF, dtype=np.float64)
        for i, track_id in enumerate(track_ids):
            track = self._active[track_id]
            dt_pred = max(1, frame_idx - track.last_frame) / self.fps
            pred_x = track.x + track.vx * dt_pred
            pred_y = track.y + track.vy * dt_pred
            gate = gate_map.get(track_id, self.max_distance)
            for j_local, j_global in enumerate(det_indices):
                det = detections[j_global]
                dx = pred_x - det.x
                dy = pred_y - det.y
                d = (dx * dx + dy * dy) ** 0.5
                if d <= gate:
                    cost[i, j_local] = d

        row_ind, col_ind = linear_sum_assignment(cost)
        for i, j_local in zip(row_ind, col_ind):
            if cost[i, j_local] < INF:
                track_id = track_ids[i]
                j_global = det_indices[j_local]
                assignments.append((track_id, j_global))
                unmatched_track_ids.discard(track_id)
                unmatched_det_idxs.discard(j_global)

        return assignments

    def step(self, frame_idx: int, detections: List[Detection]) -> List[TrackPoint]:
        points: List[TrackPoint] = []

        unmatched_track_ids = set(self._active.keys())
        unmatched_det_idxs = set(range(len(detections)))

        all_assignments: List[Tuple[int, int]] = []

        if self._active and detections:
            track_ids_list = list(self._active.keys())
            gate_map = {tid: self._adaptive_gate(self._active[tid]) for tid in track_ids_list}

            if self.two_stage_enabled:
                high_det = [j for j, det in enumerate(detections) if det.score >= self.two_stage_score_threshold]
                low_det = [j for j, det in enumerate(detections) if det.score < self.two_stage_score_threshold]

                if high_det:
                    stage1 = self._match_stage(
                        frame_idx, detections, high_det,
                        list(unmatched_track_ids), gate_map,
                        unmatched_track_ids, unmatched_det_idxs,
                    )
                    all_assignments.extend(stage1)

                if low_det and unmatched_track_ids:
                    remaining_tracks = list(unmatched_track_ids)
                    stage2 = self._match_stage(
                        frame_idx, detections, low_det,
                        remaining_tracks, gate_map,
                        unmatched_track_ids, unmatched_det_idxs,
                    )
                    all_assignments.extend(stage2)
            else:
                track_ids_list = list(unmatched_track_ids)
                all_assignments = self._match_stage(
                    frame_idx, detections, list(range(len(detections))),
                    track_ids_list, gate_map,
                    unmatched_track_ids, unmatched_det_idxs,
                )

        for track_id, det_idx in all_assignments:
            track = self._active[track_id]
            det = detections[det_idx]
            missed_before = track.missed
            dt_frames = max(1, frame_idx - track.last_frame)
            dt = dt_frames / self.fps
            new_vx = (det.x - track.x) / dt
            new_vy = (det.y - track.y) / dt
            vx = 0.6 * new_vx + 0.4 * track.vx
            vy = 0.6 * new_vy + 0.4 * track.vy

            track.x = det.x
            track.y = det.y
            track.vx = vx
            track.vy = vy
            track.bbox_w = float(det.bbox_x2 - det.bbox_x1)
            track.bbox_h = float(det.bbox_y2 - det.bbox_y1)
            track.speed_px_sec = (vx * vx + vy * vy) ** 0.5
            track.last_detection_score = det.score
            track.consecutive_hits += 1
            track.last_frame = frame_idx
            track.missed = 0

            if self.export_debug:
                dt_pred_debug = max(1, frame_idx - track.last_frame + dt_frames) / self.fps
                # recalc pred position for debug (before update)
                pred_x_debug = track.x - track.vx * dt
                pred_y_debug = track.y - track.vy * dt
                gate_debug = self._adaptive_gate(track)
                match_dist = ((pred_x_debug - det.x) ** 2 + (pred_y_debug - det.y) ** 2) ** 0.5
                self.debug_rows.append(
                    TrackingDebugRow(
                        frame=frame_idx,
                        track_id=track_id,
                        state="matched",
                        pred_x=round(pred_x_debug, 2),
                        pred_y=round(pred_y_debug, 2),
                        gate_px=round(gate_debug, 2),
                        matched=True,
                        det_x=round(det.x, 2),
                        det_y=round(det.y, 2),
                        match_dist=round(match_dist, 2),
                        det_score=round(det.score, 4),
                        bbox_w=round(track.bbox_w, 2),
                        bbox_h=round(track.bbox_h, 2),
                        speed_px_sec=round(track.speed_px_sec, 2),
                        missed_before=missed_before,
                        missed_after=0,
                        birth_reason="",
                        kill_reason="",
                    )
                )

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
        to_delete_reasons: Dict[int, str] = {}
        for track_id, track in self._active.items():
            if track_id in unmatched_track_ids:
                track.missed += 1
                track.vx *= 0.9
                track.vy *= 0.9
                track.consecutive_hits = 0
                if track.missed > self.max_missed:
                    to_delete.append(track_id)
                    to_delete_reasons[track_id] = f"missed_limit_{track.missed}"

        for track_id in to_delete:
            if self.export_debug:
                track = self._active[track_id]
                self.debug_rows.append(
                    TrackingDebugRow(
                        frame=frame_idx,
                        track_id=track_id,
                        state="killed",
                        pred_x=round(track.x, 2),
                        pred_y=round(track.y, 2),
                        gate_px=round(self._adaptive_gate(track), 2),
                        matched=False,
                        det_x=float("nan"),
                        det_y=float("nan"),
                        match_dist=float("nan"),
                        det_score=float("nan"),
                        bbox_w=round(track.bbox_w, 2),
                        bbox_h=round(track.bbox_h, 2),
                        speed_px_sec=round(track.speed_px_sec, 2),
                        missed_before=track.missed - 1,
                        missed_after=track.missed,
                        birth_reason="",
                        kill_reason=to_delete_reasons.get(track_id, "missed_limit"),
                    )
                )
            del self._active[track_id]

        for det_idx in unmatched_det_idxs:
            det = detections[det_idx]
            track_id = self._next_track_id
            self._next_track_id += 1
            bbox_w = float(det.bbox_x2 - det.bbox_x1)
            bbox_h = float(det.bbox_y2 - det.bbox_y1)
            self._active[track_id] = ActiveTrack(
                track_id=track_id,
                x=det.x,
                y=det.y,
                vx=0.0,
                vy=0.0,
                last_frame=frame_idx,
                missed=0,
                bbox_w=bbox_w,
                bbox_h=bbox_h,
                speed_px_sec=0.0,
                last_detection_score=det.score,
                consecutive_hits=1,
                birth_frame=frame_idx,
            )
            if self.export_debug:
                self.debug_rows.append(
                    TrackingDebugRow(
                        frame=frame_idx,
                        track_id=track_id,
                        state="born",
                        pred_x=round(det.x, 2),
                        pred_y=round(det.y, 2),
                        gate_px=round(self.max_distance, 2),
                        matched=False,
                        det_x=round(det.x, 2),
                        det_y=round(det.y, 2),
                        match_dist=float("nan"),
                        det_score=round(det.score, 4),
                        bbox_w=round(bbox_w, 2),
                        bbox_h=round(bbox_h, 2),
                        speed_px_sec=0.0,
                        missed_before=0,
                        missed_after=0,
                        birth_reason="unmatched_detection",
                        kill_reason="",
                    )
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

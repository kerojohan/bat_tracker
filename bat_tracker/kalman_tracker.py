"""Tracker MOT con filtro de Kalman y asociación en dos fases.

Sustituye a :class:`bat_tracker.tracker.GreedyTracker` manteniendo la
misma interfaz (``step(frame_idx, detections) -> List[TrackPoint]``), de
modo que el resto del pipeline no necesita cambios.

Mejoras frente al tracker greedy:

- Modelo de movimiento de velocidad constante (estado ``[x, y, vx, vy]``)
  con filtro de Kalman, que predice mejor durante huecos/blur y reduce la
  fragmentación.
- Asociación en dos fases estilo ByteTrack: primero las detecciones
  "fuertes" (blobs grandes/claros), luego las "débiles" sobre los tracks
  que quedaron sin emparejar. Esto mantiene la identidad a través de
  frames borrosos sin generar tracks espurios (solo las detecciones
  fuertes no emparejadas crean tracks nuevos).
- Gating por distancia sobre la posición predicha, que evita
  intercambios de identidad en los cruces.

El estado interno de velocidad se mantiene en px/frame; los ``TrackPoint``
exportan ``vx``/``vy`` en px/segundo para mantener paridad con el tracker
anterior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .detection import Detection
from .tracker import TrackPoint


@dataclass
class _PendingDetection:
    """Detectción no emparejada que espera 1 frame para confirmación de velocidad."""

    det: Detection
    frame_idx: int
    track_id: int


def _build_constant_velocity_matrices(
    sigma_acc: float,
    measurement_std: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve (F, H, Q, R) para un modelo de velocidad constante (dt=1)."""
    F = np.array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    H = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    # Ruido de proceso por aceleración blanca discreta (dt=1) por eje.
    q = sigma_acc * sigma_acc
    block = np.array([[0.25, 0.5], [0.5, 1.0]], dtype=np.float64) * q
    Q = np.zeros((4, 4), dtype=np.float64)
    # ejes x:(0,2) e y:(1,3)
    Q[np.ix_([0, 2], [0, 2])] = block
    Q[np.ix_([1, 3], [1, 3])] = block
    r = measurement_std * measurement_std
    R = np.eye(2, dtype=np.float64) * r
    return F, H, Q, R


@dataclass
class _KalmanTrack:
    track_id: int
    mean: np.ndarray  # [x, y, vx, vy]
    cov: np.ndarray  # 4x4
    last_frame: int
    missed: int = 0
    # Última detección asociada (para exportar bbox/area reales).
    last_det: Detection | None = None
    pred_x: float = 0.0
    pred_y: float = 0.0

    def predict(self, F: np.ndarray, Q: np.ndarray) -> None:
        self.mean = F @ self.mean
        self.cov = F @ self.cov @ F.T + Q
        self.pred_x = float(self.mean[0])
        self.pred_y = float(self.mean[1])

    def update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
        residual = z - H @ self.mean
        S = H @ self.cov @ H.T + R
        K = self.cov @ H.T @ np.linalg.inv(S)
        self.mean = self.mean + K @ residual
        identity = np.eye(self.cov.shape[0], dtype=np.float64)
        self.cov = (identity - K @ H) @ self.cov


class KalmanTracker:
    def __init__(
        self,
        max_distance: float,
        max_missed: int,
        fps: float,
        video_id: str,
        sigma_acc: float = 3.0,
        measurement_std: float = 2.0,
        init_velocity_std: float | None = None,
        high_area_threshold: float = 0.0,
        low_match_distance_scale: float = 1.0,
    ):
        self.max_distance = float(max_distance)
        self.max_missed = int(max_missed)
        self.fps = float(fps)
        self.video_id = video_id
        self.high_area_threshold = float(high_area_threshold)
        self.low_match_distance = self.max_distance * float(low_match_distance_scale)

        self.F, self.H, self.Q, self.R = _build_constant_velocity_matrices(
            sigma_acc=sigma_acc,
            measurement_std=measurement_std,
        )
        self._pos_var0 = 2.0 * measurement_std * measurement_std
        if init_velocity_std is None:
            init_velocity_std = max(1.0, self.max_distance / 2.0)
        self._vel_var0 = float(init_velocity_std) * float(init_velocity_std)

        self._next_track_id = 1
        self._active: Dict[int, _KalmanTrack] = {}
        self._pending: List[_PendingDetection] = []
        self._pending_min_distance = self.max_distance * 0.7
        self._blob_split_distance = min(120.0, self.max_distance * 0.8)

    def _new_track(
        self,
        frame_idx: int,
        det: Detection,
        vx: float = 0.0,
        vy: float = 0.0,
    ) -> _KalmanTrack:
        track_id = self._next_track_id
        self._next_track_id += 1
        mean = np.array([det.x, det.y, vx, vy], dtype=np.float64)
        cov = np.diag(
            [self._pos_var0, self._pos_var0, self._vel_var0, self._vel_var0]
        ).astype(np.float64)
        track = _KalmanTrack(
            track_id=track_id,
            mean=mean,
            cov=cov,
            last_frame=frame_idx,
            missed=0,
            last_det=det,
            pred_x=float(det.x),
            pred_y=float(det.y),
        )
        self._active[track_id] = track
        return track

    def _emit_point(self, track: _KalmanTrack, frame_idx: int, det: Detection) -> TrackPoint:
        vx_sec = float(track.mean[2]) * self.fps
        vy_sec = float(track.mean[3]) * self.fps
        return TrackPoint(
            video_id=self.video_id,
            track_id=track.track_id,
            frame=frame_idx,
            time_sec=frame_idx / self.fps,
            x=float(det.x),
            y=float(det.y),
            vx=vx_sec,
            vy=vy_sec,
            bbox_x1=det.bbox_x1,
            bbox_y1=det.bbox_y1,
            bbox_x2=det.bbox_x2,
            bbox_y2=det.bbox_y2,
            area=det.area,
        )

    def _associate(
        self,
        track_ids: Sequence[int],
        detections: Sequence[Detection],
        det_idxs: Sequence[int],
        max_dist: float,
    ) -> List[Tuple[int, int]]:
        """Asocia tracks <-> detecciones por distancia a la posición predicha."""
        if not track_ids or not det_idxs:
            return []

        n_tracks = len(track_ids)
        n_dets = len(det_idxs)
        INF = max_dist * 1e6
        cost = np.full((n_tracks, n_dets), INF, dtype=np.float64)
        for i, track_id in enumerate(track_ids):
            track = self._active[track_id]
            px = track.pred_x
            py = track.pred_y
            for j, det_idx in enumerate(det_idxs):
                det = detections[det_idx]
                d = ((px - det.x) ** 2 + (py - det.y) ** 2) ** 0.5
                if d <= max_dist:
                    cost[i, j] = d

        row_ind, col_ind = linear_sum_assignment(cost)
        matches: List[Tuple[int, int]] = []
        for i, j in zip(row_ind, col_ind):
            if cost[i, j] < INF:
                matches.append((track_ids[i], det_idxs[j]))
        return matches

    def step(self, frame_idx: int, detections: List[Detection]) -> List[TrackPoint]:
        # 1. Predecir todos los tracks activos un frame hacia adelante.
        for track in self._active.values():
            track.predict(self.F, self.Q)

        # 2. Separar detecciones fuertes (alta área) y débiles.
        high_idxs: List[int] = []
        low_idxs: List[int] = []
        for idx, det in enumerate(detections):
            if det.area >= self.high_area_threshold:
                high_idxs.append(idx)
            else:
                low_idxs.append(idx)

        unmatched_track_ids = set(self._active.keys())
        matched: List[Tuple[int, int]] = []

        # 3. Primera fase: detecciones fuertes contra todos los tracks.
        first = self._associate(
            sorted(unmatched_track_ids),
            detections,
            high_idxs,
            self.max_distance,
        )
        matched.extend(first)
        matched_high = set()
        for track_id, det_idx in first:
            unmatched_track_ids.discard(track_id)
            matched_high.add(det_idx)

        # 4. Segunda fase: detecciones débiles contra tracks aún libres.
        remaining_low = [idx for idx in low_idxs]
        if remaining_low and unmatched_track_ids:
            second = self._associate(
                sorted(unmatched_track_ids),
                detections,
                remaining_low,
                self.low_match_distance,
            )
            for track_id, det_idx in second:
                unmatched_track_ids.discard(track_id)
                matched.append((track_id, det_idx))

        # 5. Actualizar tracks emparejados y emitir sus puntos.
        points: List[TrackPoint] = []
        for track_id, det_idx in matched:
            track = self._active[track_id]
            det = detections[det_idx]
            track.update(np.array([det.x, det.y], dtype=np.float64), self.H, self.R)
            track.last_frame = frame_idx
            track.missed = 0
            track.last_det = det
            points.append(self._emit_point(track, frame_idx, det))

        # 6. Tracks sin emparejar: contar fallo y cerrar si superan max_missed.
        to_delete: List[int] = []
        for track_id in unmatched_track_ids:
            track = self._active[track_id]
            track.missed += 1
            if track.missed > self.max_missed:
                to_delete.append(track_id)
        for track_id in to_delete:
            del self._active[track_id]

        # 7. Detecciones fuertes no emparejadas: intentar confirmar con pending.
        unmatched_dets = [
            (det_idx, detections[det_idx])
            for det_idx in high_idxs
            if det_idx not in matched_high
        ]
        # Limpiar pending de frames viejos (más de 2 frames atrás).
        pending_min_frame = frame_idx - 2
        self._pending = [p for p in self._pending if p.frame_idx >= pending_min_frame]

        confirmed_positions: List[Tuple[float, float]] = []
        for track_id, _det_idx in matched:
            track = self._active[track_id]
            confirmed_positions.append((float(track.mean[0]), float(track.mean[1])))

        remaining_unmatched: List[Tuple[int, Detection]] = []
        for det_idx, det in unmatched_dets:
            best_pending: _PendingDetection | None = None
            best_dist = float("inf")
            for p in self._pending:
                d = ((p.det.x - det.x) ** 2 + (p.det.y - det.y) ** 2) ** 0.5
                if d < self._pending_min_distance and d < best_dist:
                    best_dist = d
                    best_pending = p
            if best_pending is not None:
                dt = max(1, frame_idx - best_pending.frame_idx)
                vx = (det.x - best_pending.det.x) / dt
                vy = (det.y - best_pending.det.y) / dt
                track = self._new_track(best_pending.frame_idx, best_pending.det, vx=vx, vy=vy)
                points.append(self._emit_point(track, best_pending.frame_idx, best_pending.det))
                track.update(np.array([det.x, det.y], dtype=np.float64), self.H, self.R)
                track.last_frame = frame_idx
                track.missed = 0
                track.last_det = det
                points.append(self._emit_point(track, frame_idx, det))
                self._pending.remove(best_pending)
                confirmed_positions.append((det.x, det.y))
            else:
                remaining_unmatched.append((det_idx, det))

        # Descartar detecciones sobrantes que estén muy cerca de una confirmada (blob splits).
        filtered_remaining: List[Tuple[int, Detection]] = []
        for det_idx, det in remaining_unmatched:
            too_close = any(
                ((cx - det.x) ** 2 + (cy - det.y) ** 2) ** 0.5 < self._blob_split_distance
                for cx, cy in confirmed_positions
            )
            if not too_close:
                filtered_remaining.append((det_idx, det))

        # Añadir a pending las detecciones que no pudieron emparejarse.
        for det_idx, det in filtered_remaining:
            tid = self._next_track_id
            self._next_track_id += 1
            self._pending.append(_PendingDetection(det=det, frame_idx=frame_idx, track_id=tid))

        return points

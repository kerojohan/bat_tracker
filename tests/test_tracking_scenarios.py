"""Escenarios sintéticos de tracking y fusión.

Verifican el objetivo central del proyecto: una trayectoria limpia e
individualizada por murciélago.

- Dos murciélagos que se cruzan deben producir 2 tracks sin intercambio
  de identidad.
- Un murciélago con frames perdidos debe seguir siendo 1 track.
- El merge conservador no debe fundir tracks que coexisten en el tiempo
  (murciélagos distintos), pero sí debe recomponer fragmentos del mismo
  vuelo.
- No debe haber mega-fusiones transitivas (A~B, B~C => A~C).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from bat_tracker.detection import Detection
from bat_tracker.kalman_tracker import KalmanTracker
from bat_tracker.pipeline import _auto_merge_track_points, _filter_track_points
from bat_tracker.tracker import TrackPoint

FPS = 25.0


def _det(x: float, y: float, size: int = 6) -> Detection:
    half = size // 2
    return Detection(
        x=float(x),
        y=float(y),
        bbox_x1=int(x) - half,
        bbox_y1=int(y) - half,
        bbox_x2=int(x) + half,
        bbox_y2=int(y) + half,
        area=float(size * size),
    )


def _run_tracker(tracker, frames: List[List[Detection]]) -> Dict[int, List[TrackPoint]]:
    by_track: Dict[int, List[TrackPoint]] = {}
    for frame_idx, dets in enumerate(frames):
        for point in tracker.step(frame_idx, dets):
            by_track.setdefault(point.track_id, []).append(point)
    for track_id in by_track:
        by_track[track_id].sort(key=lambda p: p.frame)
    return by_track


def _point(track_id: int, frame: int, x: float, y: float) -> TrackPoint:
    return TrackPoint(
        video_id="clip",
        track_id=track_id,
        frame=frame,
        time_sec=frame / FPS,
        x=float(x),
        y=float(y),
        vx=0.0,
        vy=0.0,
        bbox_x1=int(x) - 2,
        bbox_y1=int(y) - 2,
        bbox_x2=int(x) + 2,
        bbox_y2=int(y) + 2,
        area=16.0,
    )


def test_two_crossing_bats_stay_separate() -> None:
    # Bat A: baja-derecha desde (0,0). Bat B: sube-derecha desde (0,160).
    # Se cruzan cerca del centro pero con velocidades opuestas en Y.
    frames: List[List[Detection]] = []
    for f in range(21):
        a = _det(10 * f, 8 * f)
        b = _det(10 * f, 160 - 8 * f)
        frames.append([a, b])

    tracker = KalmanTracker(max_distance=60.0, max_missed=8, fps=FPS, video_id="clip")
    tracks = _run_tracker(tracker, frames)

    long_tracks = {tid: pts for tid, pts in tracks.items() if len(pts) >= 15}
    assert len(long_tracks) == 2

    # No debe haber intercambio de identidad: el track que empieza arriba
    # (y pequeña) debe acabar abajo (y grande), y viceversa.
    for pts in long_tracks.values():
        start_y = pts[0].y
        end_y = pts[-1].y
        if start_y < 20:
            assert end_y > 120, "el track que baja debió seguir bajando (sin swap)"
        elif start_y > 140:
            assert end_y < 40, "el track que sube debió seguir subiendo (sin swap)"


def test_single_bat_with_gap_stays_one_track() -> None:
    frames: List[List[Detection]] = []
    for f in range(16):
        if 6 <= f <= 8:
            frames.append([])  # 3 frames sin detección (oclusión/blur)
            continue
        frames.append([_det(20 + 12 * f, 100)])

    tracker = KalmanTracker(max_distance=60.0, max_missed=6, fps=FPS, video_id="clip")
    tracks = _run_tracker(tracker, frames)

    long_tracks = {tid: pts for tid, pts in tracks.items() if len(pts) >= 10}
    assert len(long_tracks) == 1


def _merge_cfg(**overrides) -> dict:
    cfg = {
        "auto_merge_suggested": True,
        "merge_max_gap_frames": 8,
        "merge_max_endpoint_distance": 80.0,
        "merge_overlap_min_common_frames": 3,
        "merge_overlap_max_mean_distance": 60.0,
        "merge_overlap_min_direction_cosine": 0.8,
    }
    cfg.update(overrides)
    return cfg


def test_conservative_merge_keeps_coexisting_tracks_separate() -> None:
    # Tres murciélagos paralelos que coexisten en el tiempo (mismas frames),
    # separados verticalmente. No deben fusionarse entre sí.
    points: List[TrackPoint] = []
    for tid, y0 in ((1, 100.0), (2, 200.0), (3, 300.0)):
        for f in range(20):
            points.append(_point(tid, f, 50.0 + 12.0 * f, y0))

    merged, merges = _auto_merge_track_points(points, _merge_cfg())
    track_ids = {p.track_id for p in merged}
    assert len(track_ids) == 3
    assert merges == []


def test_conservative_merge_repairs_fragmentation() -> None:
    # Un único vuelo partido en dos: el fragmento B empieza justo donde
    # acabó A, pocos frames después y muy cerca, misma dirección.
    points: List[TrackPoint] = []
    for f in range(0, 10):
        points.append(_point(1, f, 50.0 + 12.0 * f, 100.0 + 6.0 * f))
    # gap de 2 frames; B continúa la misma recta
    for i, f in enumerate(range(12, 22)):
        x = 50.0 + 12.0 * (10 + i + 2)
        y = 100.0 + 6.0 * (10 + i + 2)
        points.append(_point(2, f, x, y))

    merged, merges = _auto_merge_track_points(points, _merge_cfg())
    track_ids = {p.track_id for p in merged}
    assert len(track_ids) == 1
    assert len(merges) == 1


def test_no_transitive_megamerge() -> None:
    # A y B son el mismo vuelo (handoff válido). C coexiste con B pero es un
    # murciélago distinto y paralelo. No debe colapsarse A+B+C en uno solo.
    points: List[TrackPoint] = []
    for f in range(0, 10):
        points.append(_point(1, f, 50.0 + 12.0 * f, 100.0))
    for i, f in enumerate(range(12, 22)):
        points.append(_point(2, f, 50.0 + 12.0 * (10 + i + 2), 100.0))
    # C coexiste temporalmente con B, paralelo y separado en Y.
    for i, f in enumerate(range(12, 22)):
        points.append(_point(3, f, 50.0 + 12.0 * (10 + i + 2), 260.0))

    merged, merges = _auto_merge_track_points(points, _merge_cfg())
    track_ids = {p.track_id for p in merged}
    assert len(track_ids) == 2, "A+B deben unirse; C debe quedar aparte"


def _filter_cfg(mode: str) -> dict:
    return {
        "min_track_length": 1,
        "min_track_duration_sec": 0.0,
        "min_track_displacement": 0.0,
        "min_track_path_length": 0.0,
        "min_track_straightness": 0.0,
        "require_start_or_end_in_valid_region": True,
        "valid_region_mode": mode,
        "valid_region_gate_dilate_px": 0,
    }


def test_valid_region_require_start_or_end_rejects_out_of_gate_track_in_any_mode() -> None:
    # Mascara valida en una banda; un track que vive completamente fuera.
    mask = np.zeros((400, 400), dtype=np.uint8)
    mask[:, 150:250] = 255  # banda valida central
    points: List[TrackPoint] = []
    for f in range(10):
        points.append(_point(1, f, 10.0 + 5.0 * f, 350.0))  # siempre fuera de la banda

    kept_annotate, assess_annotate = _filter_track_points(points, _filter_cfg("annotate"), FPS, valid_mask=mask)
    kept_gate, assess_gate = _filter_track_points(points, _filter_cfg("gate"), FPS, valid_mask=mask)

    # Con require_start_or_end=true (compat v1.1.11), annotate también
    # descarta tracks que no tocan la gate.
    assert kept_annotate == []
    assert assess_annotate[0]["accepted"] is False
    assert assess_annotate[0]["start_in_valid_region"] is False
    assert "valid_region_gate" in assess_annotate[0]["reject_reasons"]

    # gate lo descarta por valid_region_gate.
    assert kept_gate == []
    assert assess_gate[0]["accepted"] is False
    assert "valid_region_gate" in assess_gate[0]["reject_reasons"]


def test_valid_region_annotate_keeps_out_of_gate_track_when_not_required() -> None:
    mask = np.zeros((400, 400), dtype=np.uint8)
    mask[:, 150:250] = 255
    points: List[TrackPoint] = []
    for f in range(10):
        points.append(_point(1, f, 10.0 + 5.0 * f, 350.0))

    cfg = _filter_cfg("annotate")
    cfg["require_start_or_end_in_valid_region"] = False
    kept, assess = _filter_track_points(points, cfg, FPS, valid_mask=mask)

    assert {p.track_id for p in kept} == {1}
    assert assess[0]["accepted"] is True
    assert assess[0]["start_in_valid_region"] is False

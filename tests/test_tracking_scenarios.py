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

import math
from typing import Dict, List

import numpy as np

from bat_tracker.detection import Detection
from bat_tracker.kalman_tracker import KalmanTracker
from bat_tracker.pipeline import _auto_merge_track_points, _filter_track_points
from bat_tracker.track_deduplication import deduplicate_track_points
from bat_tracker.track_deduplication import render_track_deduplication_overlay
from bat_tracker.track_deduplication import write_track_deduplication_csv
from bat_tracker.track_deduplication import write_track_deduplication_json
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


def _dedupe_cfg(**overrides) -> dict:
    cfg = {
        "enable_track_deduplication": True,
        "max_spatial_distance_px": 12.0,
        "max_temporal_gap_frames": 3,
        "min_direction_similarity": 0.75,
        "min_speed_similarity": 0.50,
        "min_duplicate_score": 0.75,
        "merge_strategy": "mark",
    }
    cfg.update(overrides)
    return cfg


def test_track_deduplication_discards_clear_duplicate_with_traceability() -> None:
    points: List[TrackPoint] = []
    for frame in range(10):
        points.append(_point(1, frame, 40.0 + 10.0 * frame, 100.0 + 4.0 * frame))
        points.append(_point(2, frame, 42.0 + 10.0 * frame, 102.0 + 4.0 * frame))

    result = deduplicate_track_points(points, _dedupe_cfg(merge_strategy="discard"))

    assert {point.track_id for point in result.points} == {1}
    rows = {int(row["track_id_original"]): row for row in result.rows}
    assert rows[1]["duplicate_decision"] == "keep"
    assert rows[2]["duplicate_decision"] == "discard"
    assert rows[2]["duplicate_group_id"] == rows[1]["duplicate_group_id"]
    assert float(rows[2]["duplicate_score"]) >= 0.75
    assert rows[2]["reason"]


def test_track_deduplication_keeps_crossing_bats_separate() -> None:
    points: List[TrackPoint] = []
    for frame in range(21):
        points.append(_point(1, frame, 40.0 + 8.0 * frame, 40.0 + 5.0 * frame))
        points.append(_point(2, frame, 40.0 + 8.0 * frame, 140.0 - 5.0 * frame))

    result = deduplicate_track_points(points, _dedupe_cfg(merge_strategy="auto"))

    assert {point.track_id for point in result.points} == {1, 2}
    assert result.groups_total == 0
    assert {row["duplicate_decision"] for row in result.rows} == {"keep"}


def test_track_deduplication_merges_consecutive_same_bat() -> None:
    points: List[TrackPoint] = []
    for frame in range(10):
        points.append(_point(1, frame, 30.0 + 10.0 * frame, 80.0))
    for idx, frame in enumerate(range(10, 20)):
        points.append(_point(2, frame, 130.0 + 10.0 * idx, 80.0))

    result = deduplicate_track_points(points, _dedupe_cfg(merge_strategy="merge", max_spatial_distance_px=12.0))

    assert {point.track_id for point in result.points} == {1}
    assert len(result.points) == 20
    rows = {int(row["track_id_original"]): row for row in result.rows}
    assert rows[1]["duplicate_decision"] == "merge"
    assert rows[2]["duplicate_decision"] == "merge"
    assert rows[2]["track_id_final"] == 1
    assert rows[1]["source_track_ids"] == "1;2"


def test_track_deduplication_keeps_parallel_tracks_separate() -> None:
    points: List[TrackPoint] = []
    for frame in range(12):
        points.append(_point(1, frame, 20.0 + 9.0 * frame, 80.0))
        points.append(_point(2, frame, 20.0 + 9.0 * frame, 100.0))

    result = deduplicate_track_points(points, _dedupe_cfg(merge_strategy="auto", max_spatial_distance_px=8.0))

    assert {point.track_id for point in result.points} == {1, 2}
    assert result.groups_total == 0
    assert {row["duplicate_decision"] for row in result.rows} == {"keep"}


def test_track_deduplication_writes_debug_artifacts(tmp_path) -> None:
    points: List[TrackPoint] = []
    for frame in range(6):
        points.append(_point(1, frame, 20.0 + 8.0 * frame, 40.0))
        points.append(_point(2, frame, 21.0 + 8.0 * frame, 41.0))

    result = deduplicate_track_points(points, _dedupe_cfg(merge_strategy="mark"))
    csv_path = tmp_path / "track_deduplication.csv"
    json_path = tmp_path / "track_deduplication.json"
    write_track_deduplication_csv(csv_path, result.rows)
    write_track_deduplication_json(json_path, result)
    overlay = render_track_deduplication_overlay(np.zeros((120, 120), dtype=np.uint8), points, result.rows)

    assert csv_path.read_text(encoding="utf-8").startswith("track_id_original,track_id_final")
    assert "\"duplicate_decision\": \"uncertain\"" in json_path.read_text(encoding="utf-8")
    assert overlay.shape == (120, 120, 3)
    assert int(np.count_nonzero(overlay)) > 0


def _filter_cfg(mode: str) -> dict:
    return {
        "min_track_length": 1,
        "min_track_duration_sec": 0.0,
        "min_track_displacement": 0.0,
        "min_track_path_length": 0.0,
        "min_track_straightness": 0.0,
        "require_start_or_end_in_valid_region": True,
        "entry_exit_zone_source": "valid_region",
        "valid_region_mode": mode,
        "valid_region_gate_dilate_px": 0,
    }


def _static_noise_cfg() -> dict:
    cfg = _filter_cfg("annotate")
    cfg.update(
        {
            "static_noise_filter_enabled": True,
            "static_noise_min_duration_sec": 3.0,
            "static_noise_max_mean_speed_ratio_per_sec": 0.025,
            "static_noise_max_displacement_ratio_per_sec": 0.020,
        }
    )
    return cfg


def test_static_noise_filter_rejects_sustained_slow_blob_but_keeps_active_track() -> None:
    valid_mask = np.ones((720, 1280), dtype=np.uint8) * 255

    slow_blob = [_point(1, f, 500.0 + 0.2 * f, 360.0) for f in range(101)]
    kept_blob, assess_blob = _filter_track_points(slow_blob, _static_noise_cfg(), FPS, valid_mask=valid_mask)

    assert kept_blob == []
    assert assess_blob[0]["accepted"] is False
    assert "static_noise" in assess_blob[0]["reject_reasons"]

    active_track = [_point(2, f, 500.0 + (80.0 if f % 2 else 0.0), 360.0) for f in range(101)]
    kept_active, assess_active = _filter_track_points(active_track, _static_noise_cfg(), FPS, valid_mask=valid_mask)

    assert {p.track_id for p in kept_active} == {2}
    assert assess_active[0]["accepted"] is True
    assert "static_noise" not in assess_active[0]["reject_reasons"]


def test_static_noise_filter_rejects_persistent_blob_with_teleport_jumps() -> None:
    valid_mask = np.ones((720, 1280), dtype=np.uint8) * 255
    cfg = _filter_cfg("annotate")
    cfg.update(
        {
            "min_track_displacement": 0.0,
            "min_track_path_length": 0.0,
            "static_noise_filter_enabled": True,
            "static_noise_min_duration_sec": 3.0,
            # Velocidad/avance desactivados: forzamos que solo dispare la fraccion estatica.
            "static_noise_max_mean_speed_ratio_per_sec": 0.0,
            "static_noise_max_displacement_ratio_per_sec": 0.0,
            "static_noise_min_static_fraction": 0.80,
            "static_noise_static_step_ratio_per_frame": 0.0005,
        }
    )

    # Blob fijo en (700,400) durante 16 s con saltos grandes esporadicos (teletransporte).
    blob: List[TrackPoint] = []
    for f in range(480):
        if f % 60 == 0 and f > 0:
            blob.append(_point(1, f, 700.0 + 180.0 * ((f // 60) % 2), 400.0))
        else:
            blob.append(_point(1, f, 700.0, 400.0))
    kept_blob, assess_blob = _filter_track_points(blob, cfg, FPS, valid_mask=valid_mask)

    assert kept_blob == []
    assert assess_blob[0]["accepted"] is False
    assert "static_noise" in assess_blob[0]["reject_reasons"]

    # Murcielago real: se mueve en todos los frames (fraccion estatica baja).
    flier = [_point(2, f, 200.0 + 8.0 * f, 360.0 + 4.0 * f) for f in range(120)]
    kept_flier, assess_flier = _filter_track_points(flier, cfg, FPS, valid_mask=valid_mask)

    assert {p.track_id for p in kept_flier} == {2}
    assert "static_noise" not in assess_flier[0]["reject_reasons"]


def test_temporal_gap_filter_rejects_stitched_fragments_but_keeps_dense_track() -> None:
    valid_mask = np.ones((720, 1280), dtype=np.uint8) * 255
    cfg = _filter_cfg("annotate")
    cfg.update({"min_track_displacement": 0.0, "max_track_internal_gap_frames": 45})

    # Fragmentos cosidos: vuelo corto + rafaga lejana separados por un hueco enorme.
    stitched = [_point(1, f, 1050.0 + 12.0 * f, 120.0) for f in range(13)]
    stitched += [_point(1, 1176 + f, 800.0 + 20.0 * f, 110.0) for f in range(10)]
    kept_stitched, assess_stitched = _filter_track_points(stitched, cfg, FPS, valid_mask=valid_mask)

    assert kept_stitched == []
    assert assess_stitched[0]["accepted"] is False
    assert "temporal_gap" in assess_stitched[0]["reject_reasons"]

    # Track denso (huecos pequenos, dentro de la tolerancia): se conserva.
    dense = [_point(2, f, 200.0 + 15.0 * f, 360.0) for f in range(40)]
    kept_dense, assess_dense = _filter_track_points(dense, cfg, FPS, valid_mask=valid_mask)

    assert {p.track_id for p in kept_dense} == {2}
    assert "temporal_gap" not in assess_dense[0]["reject_reasons"]


def test_loiter_filter_rejects_long_confined_track_but_keeps_long_transit() -> None:
    valid_mask = np.ones((720, 1280), dtype=np.uint8) * 255
    cfg = _filter_cfg("annotate")
    cfg.update(
        {
            "min_track_displacement": 0.0,
            "min_track_path_length": 0.0,
            "loiter_filter_enabled": True,
            "loiter_min_duration_sec": 10.0,
            "loiter_min_displacement_ratio": 0.20,
        }
    )

    # Merodeo: 12 s dando vueltas en una zona pequena (mucho recorrido, avance neto minimo).
    loiter: List[TrackPoint] = []
    for f in range(300):
        angle = f * 0.5
        loiter.append(_point(1, f, 640.0 + 30.0 * math.cos(angle), 360.0 + 30.0 * math.sin(angle)))
    kept_loiter, assess_loiter = _filter_track_points(loiter, cfg, FPS, valid_mask=valid_mask)

    assert kept_loiter == []
    assert assess_loiter[0]["accepted"] is False
    assert "loiter" in assess_loiter[0]["reject_reasons"]

    # Transito largo real: cruza casi toda la escena en 12 s (avance neto grande).
    transit = [_point(2, f, 100.0 + 3.6 * f, 200.0 + 1.5 * f) for f in range(300)]
    kept_transit, assess_transit = _filter_track_points(transit, cfg, FPS, valid_mask=valid_mask)

    assert {p.track_id for p in kept_transit} == {2}
    assert "loiter" not in assess_transit[0]["reject_reasons"]


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


def test_entry_exit_mask_classifies_outside_to_inside_as_entry() -> None:
    valid_mask = np.ones((120, 120), dtype=np.uint8) * 255
    entry_exit_mask = np.zeros((120, 120), dtype=np.uint8)
    entry_exit_mask[45:85, 55:95] = 255
    points = [_point(1, 0, 15.0, 60.0), _point(1, 1, 40.0, 60.0), _point(1, 2, 65.0, 60.0)]

    kept, assess = _filter_track_points(
        points,
        _filter_cfg("gate"),
        FPS,
        valid_mask=valid_mask,
        entry_exit_mask=entry_exit_mask,
    )

    assert {point.track_id for point in kept} == {1}
    assert assess[0]["accepted"] is True
    assert assess[0]["direction"] == "entry"


def test_entry_exit_mask_classifies_inside_to_outside_as_exit() -> None:
    valid_mask = np.ones((120, 120), dtype=np.uint8) * 255
    entry_exit_mask = np.zeros((120, 120), dtype=np.uint8)
    entry_exit_mask[45:85, 25:65] = 255
    points = [_point(1, 0, 45.0, 60.0), _point(1, 1, 70.0, 60.0), _point(1, 2, 100.0, 60.0)]

    kept, assess = _filter_track_points(
        points,
        _filter_cfg("gate"),
        FPS,
        valid_mask=valid_mask,
        entry_exit_mask=entry_exit_mask,
    )

    assert {point.track_id for point in kept} == {1}
    assert assess[0]["accepted"] is True
    assert assess[0]["direction"] == "exit"


def test_entry_exit_mask_keeps_track_crossing_zone_in_intermediate_points() -> None:
    valid_mask = np.ones((120, 120), dtype=np.uint8) * 255
    entry_exit_mask = np.zeros((120, 120), dtype=np.uint8)
    entry_exit_mask[45:85, 45:75] = 255
    points = [
        _point(1, 0, 15.0, 60.0),
        _point(1, 1, 55.0, 60.0),
        _point(1, 2, 95.0, 60.0),
    ]
    cfg = _filter_cfg("gate")
    cfg["require_start_or_end_in_valid_region"] = False

    kept, assess = _filter_track_points(
        points,
        cfg,
        FPS,
        valid_mask=valid_mask,
        entry_exit_mask=entry_exit_mask,
    )

    assert {point.track_id for point in kept} == {1}
    assert assess[0]["accepted"] is True
    assert assess[0]["direction"] in {"entry", "exit", "inside"}

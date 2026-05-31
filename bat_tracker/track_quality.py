"""Métricas de calidad de trayectorias (sin ground truth).

Este módulo evalúa, de forma puramente interna, lo "limpias e
individualizadas" que son las trayectorias producidas por el pipeline.
No necesita un conteo manual de referencia: trabaja sobre los propios
``TrackPoint`` resultantes y, opcionalmente, sobre la lista de fusiones
aplicadas en el post-proceso.

Señales clave que expone:

- Distribuciones de longitud, duración, desplazamiento y rectitud.
- ``over_merge_suspect_tracks``: tracks largos en el tiempo pero con
  rectitud muy baja, que típicamente delatan varios murciélagos fundidos
  en un único track (el síntoma principal del over-merge transitivo).
- Resumen de fusiones por motivo y el mayor grupo de tracks fusionados
  (``max_merge_group_size``), que es un indicador directo de over-merge.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from math import hypot
from typing import Dict, Iterable, List, Sequence

from .tracker import TrackPoint


def _path_length(points: Sequence[TrackPoint]) -> float:
    if len(points) < 2:
        return 0.0
    return sum(
        hypot(p1.x - p0.x, p1.y - p0.y)
        for p0, p1 in zip(points[:-1], points[1:])
    )


def _distribution(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "p90": 0.0}
    ordered = sorted(values)
    n = len(ordered)

    def _percentile(p: float) -> float:
        if n == 1:
            return float(ordered[0])
        idx = p * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)

    return {
        "count": n,
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "mean": float(sum(ordered) / n),
        "median": _percentile(0.5),
        "p90": _percentile(0.9),
    }


def _group_by_track(points: Iterable[TrackPoint]) -> Dict[int, List[TrackPoint]]:
    by_track: Dict[int, List[TrackPoint]] = defaultdict(list)
    for point in points:
        by_track[point.track_id].append(point)
    for track_id in by_track:
        by_track[track_id] = sorted(by_track[track_id], key=lambda p: p.frame)
    return by_track


def summarize_merges(merges_applied: Sequence[Dict] | None) -> Dict:
    """Resume las fusiones aplicadas en el post-proceso.

    ``max_merge_group_size`` cuenta cuántos tracks distintos acabaron
    colapsados en un mismo destino (``merged_to``); valores altos indican
    cadenas de fusión transitivas, que es justo lo que queremos evitar.
    """
    if not merges_applied:
        return {
            "merges_total": 0,
            "merges_by_reason": {},
            "overlap_merges": 0,
            "handoff_merges": 0,
            "max_merge_group_size": 1,
        }

    reasons = Counter(str(m.get("reason", "")) for m in merges_applied)
    groups: Dict[int, set] = defaultdict(set)
    for merge in merges_applied:
        merged_to = int(merge.get("merged_to"))
        groups[merged_to].add(int(merge.get("track_a")))
        groups[merged_to].add(int(merge.get("track_b")))

    max_group = max((len(members) for members in groups.values()), default=1)
    overlap = sum(count for reason, count in reasons.items() if reason.startswith("overlap"))
    handoff = sum(count for reason, count in reasons.items() if reason.startswith("handoff"))

    return {
        "merges_total": len(merges_applied),
        "merges_by_reason": dict(reasons),
        "overlap_merges": int(overlap),
        "handoff_merges": int(handoff),
        "max_merge_group_size": int(max_group),
    }


def compute_track_quality(
    points: Sequence[TrackPoint],
    fps: float,
    merges_applied: Sequence[Dict] | None = None,
    over_merge_straightness: float = 0.2,
    over_merge_min_detections: int = 40,
    over_merge_min_duration_sec: float = 1.5,
) -> Dict:
    """Calcula el bloque de métricas de calidad de trayectorias.

    Args:
        points: ``TrackPoint`` finales (ya filtrados/fusionados).
        fps: frames por segundo del vídeo.
        merges_applied: fusiones del auto-merge (opcional).
        over_merge_straightness: rectitud por debajo de la cual un track
            largo se considera sospechoso de contener varios murciélagos.
        over_merge_min_detections: detecciones mínimas para considerar el
            track "largo" a efectos del proxy de over-merge.
        over_merge_min_duration_sec: duración mínima para el proxy.
    """
    by_track = _group_by_track(points)

    lengths: List[float] = []
    durations: List[float] = []
    displacements: List[float] = []
    path_lengths: List[float] = []
    straightnesses: List[float] = []
    over_merge_suspects: List[int] = []

    for track_id, tps in by_track.items():
        start = tps[0]
        end = tps[-1]
        n = len(tps)
        duration = end.time_sec - start.time_sec
        displacement = hypot(end.x - start.x, end.y - start.y)
        pl = _path_length(tps)
        straightness = (displacement / pl) if pl > 0 else 0.0

        lengths.append(float(n))
        durations.append(float(duration))
        displacements.append(float(displacement))
        path_lengths.append(float(pl))
        straightnesses.append(float(straightness))

        if (
            n >= over_merge_min_detections
            and duration >= over_merge_min_duration_sec
            and straightness < over_merge_straightness
        ):
            over_merge_suspects.append(int(track_id))

    merge_summary = summarize_merges(merges_applied)

    return {
        "tracks_total": len(by_track),
        "track_length": _distribution(lengths),
        "duration_sec": _distribution(durations),
        "displacement_px": _distribution(displacements),
        "path_length_px": _distribution(path_lengths),
        "straightness": _distribution(straightnesses),
        "over_merge_suspect_tracks": sorted(over_merge_suspects),
        "over_merge_suspect_count": len(over_merge_suspects),
        "over_merge_criteria": {
            "max_straightness": float(over_merge_straightness),
            "min_detections": int(over_merge_min_detections),
            "min_duration_sec": float(over_merge_min_duration_sec),
        },
        **merge_summary,
    }

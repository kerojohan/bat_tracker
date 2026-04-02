"""
Anàlisi causal de la classificació d'esdeveniments (p.ex. 'exits') respecte a valid_region.

events.csv classifica només amb els extrems del track sobre la màscara raw (sense dilatar),
amb la regla: exit <=> start dins AND end fora. No considera punts intermedis.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import ceil, hypot
from typing import Dict, List, Literal, Optional

import cv2
import numpy as np

from .tracker import TrackPoint

COLOR_EXIT = (0, 220, 0)
COLOR_NO_EXIT_GEOMETRY_OUTSIDE = (0, 140, 255)
COLOR_NO_EXIT_GEOMETRY_ENTERS = (255, 100, 0)
COLOR_NO_EXIT_GEOMETRY_INSIDE = (180, 180, 180)
COLOR_NO_EXIT_GEOMETRY_CROSS_ONLY = (255, 0, 255)
COLOR_NO_EXIT_GATE = (60, 60, 255)
COLOR_NO_EXIT_LENGTH = (0, 255, 255)
COLOR_NO_EXIT_FRAGMENT = (150, 255, 150)
COLOR_UNKNOWN = (128, 128, 128)

OverlayCategory = Literal[
    "exit",
    "no_exit_geometry_outside",
    "no_exit_geometry_enters",
    "no_exit_geometry_inside",
    "no_exit_geometry_cross_only",
    "no_exit_gate_final",
    "no_exit_length_or_shape",
    "no_exit_fragmentation",
    "unknown",
]


def _point_tp_in_mask(p: TrackPoint, mask: np.ndarray) -> bool:
    xi = int(round(p.x))
    yi = int(round(p.y))
    if yi < 0 or yi >= mask.shape[0] or xi < 0 or xi >= mask.shape[1]:
        return False
    return bool(mask[yi, xi] > 0)


def _classify_direction(start_inside: bool, end_inside: bool) -> str:
    if start_inside and end_inside:
        return "inside"
    if start_inside and not end_inside:
        return "exits"
    if not start_inside and end_inside:
        return "enters"
    return "outside"


def classify_event_direction(
    tps: List[TrackPoint],
    valid_mask: np.ndarray | None,
    direction_mode: str,
) -> str:
    """
    Mateixa regla que s'escriu a events.csv (veure pipeline._write_events_csv).
    """
    if valid_mask is None:
        return "unknown"
    tps = sorted(tps, key=lambda p: p.frame)
    start, end = tps[0], tps[-1]
    s_in = _point_tp_in_mask(start, valid_mask)
    e_in = _point_tp_in_mask(end, valid_mask)
    mode = str(direction_mode or "endpoint").strip().lower()
    if mode == "outward_crossing":
        any_in = any_sampled_point_in_mask(tps, valid_mask)
        if s_in and not e_in:
            return "exits"
        if any_in and not e_in:
            return "exits"
    return _classify_direction(s_in, e_in)


def build_gate_mask(valid_mask: np.ndarray | None, dilate_px: int) -> np.ndarray | None:
    if valid_mask is None:
        return None
    if dilate_px <= 0:
        return valid_mask
    k = 2 * dilate_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(valid_mask, kernel, iterations=1)


def _path_length(track_points: List[TrackPoint]) -> float:
    if len(track_points) < 2:
        return 0.0
    return sum(hypot(p1.x - p0.x, p1.y - p0.y) for p0, p1 in zip(track_points[:-1], track_points[1:]))


def _sample_track_indices(n: int, max_samples: int) -> List[int]:
    if n <= 0:
        return []
    if n <= max_samples:
        return list(range(n))
    return list(np.linspace(0, n - 1, max_samples).astype(int))


def any_sampled_point_in_mask(track_points: List[TrackPoint], mask: np.ndarray | None, max_samples: int = 48) -> bool:
    if mask is None or not track_points:
        return False
    for i in _sample_track_indices(len(track_points), max_samples):
        if _point_tp_in_mask(track_points[i], mask):
            return True
    return False


def filter_fail_reason(
    track_points: List[TrackPoint],
    tracking_cfg: Dict,
    fps: float,
    gate_mask: np.ndarray | None,
) -> Optional[str]:
    min_track_length_cfg = int(tracking_cfg.get("min_track_length", 1))
    min_track_duration_sec = float(tracking_cfg.get("min_track_duration_sec", 0.0))
    min_track_length_from_sec = int(ceil(max(0.0, min_track_duration_sec) * max(1e-6, fps)))
    min_track_length = max(min_track_length_cfg, min_track_length_from_sec)
    min_track_displacement = float(tracking_cfg.get("min_track_displacement", 0.0))
    min_track_path_length = float(tracking_cfg.get("min_track_path_length", 0.0))
    min_track_straightness = float(tracking_cfg.get("min_track_straightness", 0.0))
    require_start_or_end = bool(tracking_cfg.get("require_start_or_end_in_valid_region", False))

    tps = sorted(track_points, key=lambda p: p.frame)
    n = len(tps)
    if n < min_track_length:
        return "min_track_length"
    start, end = tps[0], tps[-1]
    displacement = hypot(end.x - start.x, end.y - start.y)
    if displacement < min_track_displacement:
        return "min_track_displacement"
    pl = _path_length(tps)
    if pl < min_track_path_length:
        return "min_track_path_length"
    if min_track_straightness > 0.0 and pl > 0.0:
        straightness = displacement / pl
        if straightness < min_track_straightness:
            return "min_track_straightness"
    if require_start_or_end and gate_mask is not None:
        if not (_point_tp_in_mask(start, gate_mask) or _point_tp_in_mask(end, gate_mask)):
            return "require_start_or_end_in_valid_region"
    return None


def _geometry_fields(
    tps: List[TrackPoint],
    valid_mask: np.ndarray | None,
    gate_mask: np.ndarray | None,
    events_direction_mode: str = "endpoint",
) -> Dict:
    start, end = tps[0], tps[-1]
    if valid_mask is None:
        return {
            "start_in_valid_raw": "",
            "end_in_valid_raw": "",
            "start_in_valid_dilated": "",
            "end_in_valid_dilated": "",
            "any_sample_inside_raw": "",
            "direction_endpoint_rule": "unknown",
            "direction_as_events_csv": "unknown",
            "is_exit_as_events_csv": False,
            "not_exit_reason_primary": "no_valid_mask",
            "relaxed_exit_outward": False,
        }

    s_raw = _point_tp_in_mask(start, valid_mask)
    e_raw = _point_tp_in_mask(end, valid_mask)
    s_dil = _point_tp_in_mask(start, gate_mask) if gate_mask is not None else s_raw
    e_dil = _point_tp_in_mask(end, gate_mask) if gate_mask is not None else e_raw
    direction_endpoint = _classify_direction(s_raw, e_raw)
    direction_file = classify_event_direction(tps, valid_mask, events_direction_mode)
    any_inside = any_sampled_point_in_mask(tps, valid_mask)
    is_exit = direction_file == "exits"
    if is_exit:
        not_exit_primary = ""
    elif direction_endpoint == "outside":
        if any_inside:
            not_exit_primary = "strict_endpoints_outside_but_path_crosses_valid_region"
        else:
            not_exit_primary = "strict_endpoints_both_outside_never_inside_mask"
    elif direction_endpoint == "enters":
        not_exit_primary = "strict_endpoints_enters_start_outside_end_inside"
    elif direction_endpoint == "inside":
        not_exit_primary = "strict_endpoints_both_inside"
    else:
        not_exit_primary = "strict_endpoints_unknown"

    relaxed_exit = bool(any_inside and not e_raw)

    return {
        "start_in_valid_raw": s_raw,
        "end_in_valid_raw": e_raw,
        "start_in_valid_dilated": s_dil,
        "end_in_valid_dilated": e_dil,
        "any_sample_inside_raw": any_inside,
        "direction_endpoint_rule": direction_endpoint,
        "direction_as_events_csv": direction_file,
        "is_exit_as_events_csv": is_exit,
        "not_exit_reason_primary": not_exit_primary,
        "relaxed_exit_outward": relaxed_exit,
    }


def _overlay_category_final_track(
    direction_endpoint: str,
    any_inside: bool,
    is_exit_as_file: bool,
    internal_gap: int,
    frag_threshold: int,
) -> OverlayCategory:
    if is_exit_as_file:
        return "exit"
    if direction_endpoint == "outside" and any_inside:
        cat: OverlayCategory = "no_exit_geometry_cross_only"
    elif direction_endpoint == "outside":
        cat = "no_exit_geometry_outside"
    elif direction_endpoint == "enters":
        cat = "no_exit_geometry_enters"
    elif direction_endpoint == "inside":
        cat = "no_exit_geometry_inside"
    else:
        cat = "unknown"

    if internal_gap >= frag_threshold and cat != "exit":
        if cat in (
            "no_exit_geometry_outside",
            "no_exit_geometry_cross_only",
            "no_exit_geometry_enters",
            "no_exit_geometry_inside",
        ):
            return "no_exit_fragmentation"
    return cat


def build_raw_track_event_rows(
    all_points: List[TrackPoint],
    filtered_pre_merge_points: List[TrackPoint],
    valid_mask: np.ndarray | None,
    tracking_cfg: Dict,
    fps: float,
    fragmentation_gap_threshold: int = 6,
    events_direction_mode: str = "endpoint",
) -> List[Dict]:
    """Una fila per cada track_id del tracker (abans del merge)."""
    gate_mask = build_gate_mask(valid_mask, max(0, int(tracking_cfg.get("valid_region_gate_dilate_px", 0))))
    passed_ids = {p.track_id for p in filtered_pre_merge_points}

    by_track: Dict[int, List[TrackPoint]] = defaultdict(list)
    for p in all_points:
        by_track[p.track_id].append(p)

    rows: List[Dict] = []
    for track_id in sorted(by_track.keys()):
        tps = sorted(by_track[track_id], key=lambda p: p.frame)
        start, end = tps[0], tps[-1]
        frame_span = end.frame - start.frame + 1
        n = len(tps)
        internal_gap = max(0, frame_span - n)
        fail = filter_fail_reason(tps, tracking_cfg, fps, gate_mask)
        passed_filter = track_id in passed_ids
        geom = _geometry_fields(tps, valid_mask, gate_mask, events_direction_mode)

        if fail is not None:
            if fail == "require_start_or_end_in_valid_region":
                ocat: OverlayCategory = "no_exit_gate_final"
            else:
                ocat = "no_exit_length_or_shape"
        elif not passed_filter:
            ocat = "unknown"
        else:
            ocat = _overlay_category_final_track(
                geom["direction_endpoint_rule"],
                bool(geom["any_sample_inside_raw"]),
                bool(geom["is_exit_as_events_csv"]),
                internal_gap,
                fragmentation_gap_threshold,
            )

        displacement = hypot(end.x - start.x, end.y - start.y)
        pl = _path_length(tps)

        rows.append(
            {
                "track_id": track_id,
                "passed_post_filter_pre_merge": passed_filter,
                "final_gate_fail_reason": fail or "",
                "frame_start": start.frame,
                "frame_end": end.frame,
                "frame_span": frame_span,
                "num_detections": n,
                "internal_gap_frames": internal_gap,
                "displacement_px": round(displacement, 2),
                "path_length_px": round(pl, 2),
                "x_start": round(start.x, 2),
                "y_start": round(start.y, 2),
                "x_end": round(end.x, 2),
                "y_end": round(end.y, 2),
                **geom,
                "overlay_category": ocat,
            }
        )
    return rows


def build_merged_track_event_rows(
    filtered_post_merge_points: List[TrackPoint],
    valid_mask: np.ndarray | None,
    tracking_cfg: Dict,
    fps: float,
    fragmentation_gap_threshold: int = 6,
    events_direction_mode: str = "endpoint",
) -> List[Dict]:
    """Una fila per cada track_id igual que events.csv (després del merge)."""
    gate_mask = build_gate_mask(valid_mask, max(0, int(tracking_cfg.get("valid_region_gate_dilate_px", 0))))

    by_track: Dict[int, List[TrackPoint]] = defaultdict(list)
    for p in filtered_post_merge_points:
        by_track[p.track_id].append(p)

    rows: List[Dict] = []
    for track_id in sorted(by_track.keys()):
        tps = sorted(by_track[track_id], key=lambda p: p.frame)
        start, end = tps[0], tps[-1]
        frame_span = end.frame - start.frame + 1
        n = len(tps)
        internal_gap = max(0, frame_span - n)
        geom = _geometry_fields(tps, valid_mask, gate_mask, events_direction_mode)
        ocat = _overlay_category_final_track(
            geom["direction_endpoint_rule"],
            bool(geom["any_sample_inside_raw"]),
            bool(geom["is_exit_as_events_csv"]),
            internal_gap,
            fragmentation_gap_threshold,
        )
        displacement = hypot(end.x - start.x, end.y - start.y)
        pl = _path_length(tps)

        rows.append(
            {
                "track_id": track_id,
                "frame_start": start.frame,
                "frame_end": end.frame,
                "frame_span": frame_span,
                "num_detections": n,
                "internal_gap_frames": internal_gap,
                "displacement_px": round(displacement, 2),
                "path_length_px": round(pl, 2),
                "x_start": round(start.x, 2),
                "y_start": round(start.y, 2),
                "x_end": round(end.x, 2),
                "y_end": round(end.y, 2),
                **geom,
                "overlay_category": ocat,
            }
        )
    return rows


def overlay_color_for_category(category: str) -> tuple[int, int, int]:
    return {
        "exit": COLOR_EXIT,
        "no_exit_geometry_outside": COLOR_NO_EXIT_GEOMETRY_OUTSIDE,
        "no_exit_geometry_enters": COLOR_NO_EXIT_GEOMETRY_ENTERS,
        "no_exit_geometry_inside": COLOR_NO_EXIT_GEOMETRY_INSIDE,
        "no_exit_geometry_cross_only": COLOR_NO_EXIT_GEOMETRY_CROSS_ONLY,
        "no_exit_gate_final": COLOR_NO_EXIT_GATE,
        "no_exit_length_or_shape": COLOR_NO_EXIT_LENGTH,
        "no_exit_fragmentation": COLOR_NO_EXIT_FRAGMENT,
        "unknown": COLOR_UNKNOWN,
    }.get(category, COLOR_UNKNOWN)


def summarize_exit_blockers(
    raw_rows: List[Dict],
    merged_rows: List[Dict],
) -> Dict[str, object]:
    """Comptadors per meta.json."""
    merged_exits = sum(1 for r in merged_rows if r.get("is_exit_as_events_csv"))
    merged_not_exit = len(merged_rows) - merged_exits

    raw_passed = [r for r in raw_rows if r.get("passed_post_filter_pre_merge")]
    cross_only_passed = [
        r
        for r in raw_passed
        if r.get("direction_endpoint_rule") == "outside" and r.get("any_sample_inside_raw") is True
    ]

    return {
        "merged_tracks_total": len(merged_rows),
        "merged_exits_as_events_csv": merged_exits,
        "merged_non_exits": merged_not_exit,
        "raw_tracks_total": len(raw_rows),
        "raw_passed_post_filter": len(raw_passed),
        "raw_relaxed_exit_outward_among_passed": sum(1 for r in raw_passed if r.get("relaxed_exit_outward")),
        "raw_would_cross_opening_but_endpoints_outside_among_passed": len(cross_only_passed),
        "direction_endpoint_counts_merged": _count_key(merged_rows, "direction_endpoint_rule"),
        "direction_as_events_file_counts_merged": _count_key(merged_rows, "direction_as_events_csv"),
        "not_exit_primary_counts_merged": _count_key(
            [r for r in merged_rows if not r.get("is_exit_as_events_csv")], "not_exit_reason_primary"
        ),
    }


def _count_key(rows: List[Dict], key: str) -> Dict[str, int]:
    out: Dict[str, int] = defaultdict(int)
    for r in rows:
        k = str(r.get(key, ""))
        out[k] += 1
    return dict(out)

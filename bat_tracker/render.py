from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
from xml.etree import ElementTree as ET

import cv2
import numpy as np

from .detection import Detection
from .tracker import TrackPoint


def track_color(track_id: int) -> Tuple[int, int, int]:
    seed = np.random.default_rng(track_id)
    bgr = seed.integers(32, 256, size=3, dtype=np.uint8)
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def render_detections_overlay(
    background_gray: np.ndarray,
    detections: Sequence[Detection],
    *,
    color: tuple[int, int, int],
    alpha: float = 0.85,
    line_thickness: int = 1,
    point_radius: int = 2,
) -> np.ndarray:
    base = cv2.cvtColor(background_gray, cv2.COLOR_GRAY2BGR)
    marks = np.zeros_like(base)
    for det in detections:
        cv2.rectangle(
            marks,
            (int(det.bbox_x1), int(det.bbox_y1)),
            (int(det.bbox_x2), int(det.bbox_y2)),
            color,
            max(1, int(line_thickness)),
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            marks,
            (int(round(det.x)), int(round(det.y))),
            max(1, int(point_radius)),
            color,
            -1,
            lineType=cv2.LINE_AA,
        )
    return cv2.addWeighted(base, 1.0, marks, max(0.0, min(1.0, float(alpha))), 0)


def _track_color_hex(track_id: int) -> str:
    blue, green, red = track_color(track_id)
    return f"#{red:02x}{green:02x}{blue:02x}"


def _svg_number(value: float) -> str:
    return format(float(value), ".15g")


def _svg_stroke_width(line_thickness: int) -> str:
    # OpenCV anti-aliased lines render visually thicker than an SVG stroke with
    # the same numeric width. Apply a small compensation so the vector export
    # better matches the PNG overlay.
    return _svg_number(max(1.0, float(line_thickness) * 1.5))


def _svg_label_font_size(label_font_scale: float) -> str:
    # cv2.putText with FONT_HERSHEY_SIMPLEX renders larger than a same-number
    # SVG font-size. This compensation keeps SVG labels visually aligned with
    # the PNG overlay labels.
    return _svg_number(max(0.3, float(label_font_scale)) * 28.0)


def _point_in_mask_xy(x: float, y: float, mask: np.ndarray) -> bool:
    xi = int(round(x))
    yi = int(round(y))
    if yi < 0 or yi >= mask.shape[0] or xi < 0 or xi >= mask.shape[1]:
        return False
    return bool(mask[yi, xi] > 0)


def _classify_track_direction(start_inside: bool | None, end_inside: bool | None) -> str:
    if start_inside is None or end_inside is None:
        return "unknown"
    if start_inside and end_inside:
        return "inside"
    if start_inside and not end_inside:
        return "exit"
    if not start_inside and end_inside:
        return "entry"
    return "outside"


def _infer_outside_direction_from_motion(start: TrackPoint, end: TrackPoint, height: int) -> str:
    dy = end.y - start.y
    min_vertical_move = max(40.0, 0.15 * float(height))
    top_band = 0.20 * float(height)
    if dy <= -min_vertical_move and end.y <= top_band:
        return "exit"
    if dy >= min_vertical_move and start.y <= top_band:
        return "entry"
    return "outside"


def _classify_track_direction_full(
    track_points: Sequence[TrackPoint],
    mask: np.ndarray | None,
    height: int,
) -> str:
    if mask is None or not track_points:
        return "unknown"
    start = track_points[0]
    end = track_points[-1]
    direction = _classify_track_direction(
        _point_in_mask_xy(start.x, start.y, mask),
        _point_in_mask_xy(end.x, end.y, mask),
    )
    if direction == "inside" and len(track_points) > 2:
        if any(not _point_in_mask_xy(point.x, point.y, mask) for point in track_points[1:-1]):
            return "exit"
    if direction == "outside":
        inside_midpoints = [point for point in track_points[1:-1] if _point_in_mask_xy(point.x, point.y, mask)]
        if inside_midpoints:
            ys, xs = np.nonzero(mask > 0)
            if xs.size > 0:
                cx = float(np.mean(xs))
                cy = float(np.mean(ys))
                start_vec = (start.x - cx, start.y - cy)
                end_vec = (end.x - cx, end.y - cy)
                if start_vec[0] * end_vec[0] + start_vec[1] * end_vec[1] > 0.0:
                    return "outside"
                start_dist = float(np.hypot(start.x - cx, start.y - cy))
                end_dist = float(np.hypot(end.x - cx, end.y - cy))
                if end_dist < start_dist * 0.92:
                    return "entry"
                if start_dist < end_dist * 0.92:
                    return "exit"
            return "inside"
        return _infer_outside_direction_from_motion(start, end, height)
    return direction


def _point_payload(point: TrackPoint) -> Dict[str, Any]:
    return {
        "x": float(point.x),
        "y": float(point.y),
        "frame": int(point.frame),
        "time_sec": float(point.time_sec),
    }


def _mask_contours_payload(mask: np.ndarray) -> List[List[Dict[str, int]]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    payload: List[List[Dict[str, int]]] = []
    for contour in contours:
        coords = contour.reshape(-1, 2)
        payload.append([{"x": int(x), "y": int(y)} for x, y in coords])
    return payload


def build_tracks_render_payload(
    width: int,
    height: int,
    points: Sequence[TrackPoint],
    *,
    valid_region_mask: np.ndarray | None = None,
    direction_mask: np.ndarray | None = None,
) -> Dict[str, Any]:
    by_track: Dict[int, List[TrackPoint]] = defaultdict(list)
    for point in points:
        by_track[point.track_id].append(point)

    payload: Dict[str, Any] = {
        "width": int(width),
        "height": int(height),
        "tracks": [],
    }

    if valid_region_mask is not None:
        payload["valid_region"] = {
            "contours": _mask_contours_payload(valid_region_mask),
        }

    effective_direction_mask = direction_mask if direction_mask is not None else valid_region_mask
    if direction_mask is not None:
        payload["entry_exit_region"] = {
            "contours": _mask_contours_payload(direction_mask),
        }
    tracks_payload: List[Dict[str, Any]] = []
    for track_id in sorted(by_track):
        track_points = sorted(by_track[track_id], key=lambda p: p.frame)
        start = track_points[0]
        end = track_points[-1]
        start_inside = None
        end_inside = None
        direction = "unknown"
        if effective_direction_mask is not None:
            start_inside = _point_in_mask_xy(start.x, start.y, effective_direction_mask)
            end_inside = _point_in_mask_xy(end.x, end.y, effective_direction_mask)
            direction = _classify_track_direction_full(track_points, effective_direction_mask, int(height))

        tracks_payload.append(
            {
                "track_id": int(track_id),
                "color": _track_color_hex(track_id),
                "frame_start": int(start.frame),
                "frame_end": int(end.frame),
                "duration_sec": float(end.time_sec - start.time_sec),
                "direction": direction if effective_direction_mask is not None else _classify_track_direction(start_inside, end_inside),
                "point_start": _point_payload(start),
                "point_end": _point_payload(end),
                "points": [_point_payload(point) for point in track_points],
            }
        )

    payload["tracks"] = tracks_payload
    return payload


def export_tracks_render_json(
    path: str | Path,
    width: int,
    height: int,
    points: Sequence[TrackPoint],
    *,
    valid_region_mask: np.ndarray | None = None,
    direction_mask: np.ndarray | None = None,
) -> Dict[str, Any]:
    payload = build_tracks_render_payload(
        width=width,
        height=height,
        points=points,
        valid_region_mask=valid_region_mask,
        direction_mask=direction_mask,
    )
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def export_tracks_svg(
    path: str | Path,
    width: int,
    height: int,
    points: Sequence[TrackPoint],
    *,
    line_thickness: int,
    start_radius: int,
    alpha: float = 1.0,
    draw_track_labels: bool = False,
    draw_track_labels_at_end: bool = False,
    label_font_scale: float = 0.5,
    label_thickness: int = 1,
    valid_region_mask: np.ndarray | None = None,
    direction_mask: np.ndarray | None = None,
) -> Dict[str, Any]:
    payload = build_tracks_render_payload(
        width=width,
        height=height,
        points=points,
        valid_region_mask=valid_region_mask,
        direction_mask=direction_mask,
    )

    svg = ET.Element(
        "svg",
        {
            "xmlns": "http://www.w3.org/2000/svg",
            "viewBox": f"0 0 {int(width)} {int(height)}",
            "width": str(int(width)),
            "height": str(int(height)),
            "role": "img",
            "aria-label": "bat_tracker trajectories",
        },
    )
    title = ET.SubElement(svg, "title")
    title.text = "bat_tracker trajectories"
    desc = ET.SubElement(svg, "desc")
    desc.text = "Vector export of tracked trajectories in original video coordinates."
    style = ET.SubElement(svg, "style")
    style.text = (
        ".track polyline { fill: none; stroke: var(--track-color); stroke-linecap: round; "
        "stroke-linejoin: round; vector-effect: non-scaling-stroke; }\n"
        ".track .track-start { fill: var(--track-color); }\n"
        ".track .track-end { fill: var(--track-color); opacity: 0.9; }\n"
        ".track text { fill: var(--track-color); stroke: #000; paint-order: stroke fill; "
        "stroke-linejoin: round; dominant-baseline: alphabetic; }\n"
        ".valid-region path { fill: rgba(0, 255, 0, 0.12); stroke: #00ffaa; stroke-width: 1.5; "
        "vector-effect: non-scaling-stroke; }\n"
        ".entry-exit-region path { fill: rgba(255, 210, 0, 0.14); stroke: #ffd200; stroke-width: 1.8; "
        "vector-effect: non-scaling-stroke; }\n"
    )

    valid_region = payload.get("valid_region", {})
    contours = valid_region.get("contours", []) if isinstance(valid_region, dict) else []
    if contours:
        valid_group = ET.SubElement(svg, "g", {"id": "valid-region", "class": "valid-region"})
        valid_title = ET.SubElement(valid_group, "title")
        valid_title.text = "Valid region"
        for idx, contour in enumerate(contours):
            if not contour:
                continue
            commands = [f"M {_svg_number(contour[0]['x'])} {_svg_number(contour[0]['y'])}"]
            for point in contour[1:]:
                commands.append(f"L {_svg_number(point['x'])} {_svg_number(point['y'])}")
            commands.append("Z")
            ET.SubElement(
                valid_group,
                "path",
                {
                    "id": f"valid-region-contour-{idx}",
                    "d": " ".join(commands),
                },
            )

    entry_exit_region = payload.get("entry_exit_region", {})
    entry_exit_contours = entry_exit_region.get("contours", []) if isinstance(entry_exit_region, dict) else []
    if entry_exit_contours:
        entry_exit_group = ET.SubElement(svg, "g", {"id": "entry-exit-region", "class": "entry-exit-region"})
        entry_exit_title = ET.SubElement(entry_exit_group, "title")
        entry_exit_title.text = "Entry/exit region"
        for idx, contour in enumerate(entry_exit_contours):
            if not contour:
                continue
            commands = [f"M {_svg_number(contour[0]['x'])} {_svg_number(contour[0]['y'])}"]
            for point in contour[1:]:
                commands.append(f"L {_svg_number(point['x'])} {_svg_number(point['y'])}")
            commands.append("Z")
            ET.SubElement(
                entry_exit_group,
                "path",
                {
                    "id": f"entry-exit-region-contour-{idx}",
                    "d": " ".join(commands),
                },
            )

    polyline_width = _svg_stroke_width(line_thickness)
    start_radius_px = str(max(2, int(start_radius)))
    end_radius_px = str(max(1, int(round(max(2, start_radius) * 0.6))))
    group_opacity = _svg_number(max(0.0, min(1.0, alpha)))
    label_offset = max(4, int(start_radius) + 2)
    label_font_size = _svg_label_font_size(label_font_scale)
    label_stroke_width = _svg_number(max(1, int(label_thickness)) + 2)
    for track in payload["tracks"]:
        track_id = int(track["track_id"])
        group = ET.SubElement(
            svg,
            "g",
            {
                "id": f"track-{track_id}",
                "class": "track",
                "style": f"--track-color: {track['color']}",
                "data-track-id": str(track_id),
                "data-frame-start": str(track["frame_start"]),
                "data-frame-end": str(track["frame_end"]),
                "data-direction": str(track["direction"]),
                "opacity": group_opacity,
            },
        )
        group_title = ET.SubElement(group, "title")
        group_title.text = (
            f"Track {track_id} | frames {track['frame_start']}-{track['frame_end']} | "
            f"duration {track['duration_sec']:.4f}s | direction {track['direction']}"
        )

        points_attr = " ".join(
            f"{_svg_number(point['x'])},{_svg_number(point['y'])}" for point in track["points"]
        )
        ET.SubElement(
            group,
            "polyline",
            {
                "points": points_attr,
                "stroke-width": polyline_width,
            },
        )

        start = track["point_start"]
        end = track["point_end"]
        ET.SubElement(
            group,
            "circle",
            {
                "class": "track-start",
                "cx": _svg_number(start["x"]),
                "cy": _svg_number(start["y"]),
                "r": start_radius_px,
            },
        )
        ET.SubElement(
            group,
            "circle",
            {
                "class": "track-end",
                "cx": _svg_number(end["x"]),
                "cy": _svg_number(end["y"]),
                "r": end_radius_px,
            },
        )
        if draw_track_labels:
            ET.SubElement(
                group,
                "text",
                {
                    "class": "track-label track-label-start",
                    "x": _svg_number(start["x"] + label_offset),
                    "y": _svg_number(start["y"] - label_offset),
                    "font-size": label_font_size,
                    "font-family": "sans-serif",
                    "stroke-width": label_stroke_width,
                },
            ).text = str(track_id)
        if draw_track_labels_at_end and track["points"]:
            ET.SubElement(
                group,
                "text",
                {
                    "class": "track-label track-label-end",
                    "x": _svg_number(end["x"] + label_offset),
                    "y": _svg_number(end["y"] - label_offset),
                    "font-size": label_font_size,
                    "font-family": "sans-serif",
                    "stroke-width": label_stroke_width,
                },
            ).text = str(track_id)

    tree = ET.ElementTree(svg)
    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ")
    tree.write(Path(path), encoding="utf-8", xml_declaration=True)
    return payload


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

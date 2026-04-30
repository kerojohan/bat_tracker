from __future__ import annotations

import csv
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np
import yaml

from bat_tracker.heatmap_events import HeatmapEvent
from bat_tracker.pipeline import (
    _auto_merge_track_points,
    _build_tracks_overlay_points_with_heatmap,
    run_pipeline,
)
from bat_tracker.render import export_tracks_render_json, export_tracks_svg
from bat_tracker.tracker import TrackPoint


SVG_NS = {"svg": "http://www.w3.org/2000/svg"}


def _write_video(path: Path, frames: list[np.ndarray], fps: int = 10) -> None:
    height, width = frames[0].shape
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    assert writer.isOpened(), f"could not open writer for {path}"
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()


def _read_tracks(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _base_config() -> dict:
    return {
        "background": {
            "sample_frames": 12,
            "uniform_sampling": True,
        },
        "detection": {
            "blur_kernel": 1,
            "threshold_mode": "fixed",
            "diff_threshold": 10,
            "morph_open": 1,
            "morph_close": 1,
            "min_area": 8,
            "max_area": 5000,
            "max_global_intensity_shift": -1.0,
            "max_foreground_ratio": -1.0,
            "max_detections_per_frame": 0,
            "temporal_burst_min_detections": 0,
            "temporal_burst_window_frames": 0,
            "temporal_burst_trigger_frames": 0,
            "temporal_burst_cooldown_frames": 0,
        },
        "tracking": {
            "max_distance": 18,
            "max_missed": 2,
            "min_track_length": 1,
            "min_track_displacement": 0.0,
            "min_track_path_length": 0.0,
            "min_track_straightness": 0.0,
            "min_track_duration_sec": 0.0,
            "auto_merge_suggested": False,
            "require_start_or_end_in_valid_region": False,
            "valid_region_gate_dilate_px": 0,
        },
        "valid_region": {
            "enabled": False,
        },
        "output": {
            "progress_enabled": False,
            "overlay_line_thickness": 2,
            "overlay_start_radius": 5,
            "overlay_alpha": 1.0,
            "overlay_draw_track_labels": True,
            "overlay_draw_track_labels_at_end": True,
            "overlay_label_font_scale": 0.5,
            "overlay_label_thickness": 1,
            "export_track_clips": False,
        },
    }


def _make_single_track_video(tmp_path: Path) -> Path:
    frames: list[np.ndarray] = []
    for idx in range(10):
        frame = np.zeros((48, 64), dtype=np.uint8)
        if idx < 5:
            x0 = 8 + idx * 5
            cv2.rectangle(frame, (x0, 22), (x0 + 6, 28), 220, -1)
        frames.append(frame)

    video_path = tmp_path / "single_track.mp4"
    _write_video(video_path, frames)
    return video_path


def test_pipeline_exports_svg_and_render_json_from_in_memory_tracks(tmp_path: Path) -> None:
    video_path = _make_single_track_video(tmp_path)

    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[:, 20:60] = 255
    mask_path = tmp_path / "valid_mask.png"
    cv2.imwrite(str(mask_path), mask)

    cfg = _base_config()
    cfg["valid_region"] = {
        "enabled": True,
        "input_mask": str(mask_path),
        "apply_to_detection": False,
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    out_dir = tmp_path / "out"
    meta = run_pipeline(str(video_path), str(out_dir), str(cfg_path))

    tracks_rows = _read_tracks(out_dir / "tracks.csv")
    assert tracks_rows

    with (out_dir / "tracks_render.json").open(encoding="utf-8") as handle:
        render_payload = json.load(handle)

    assert render_payload["width"] == 64
    assert render_payload["height"] == 48
    assert len(render_payload["tracks"]) == 1
    track_payload = render_payload["tracks"][0]
    assert track_payload["track_id"] == 1
    assert track_payload["direction"] == "entry"
    assert track_payload["frame_start"] == 0
    assert track_payload["frame_end"] == 4
    assert track_payload["point_start"]["frame"] == 0
    assert track_payload["point_end"]["frame"] == 4
    assert "valid_region" in render_payload
    assert render_payload["valid_region"]["contours"]

    csv_points = [
        {
            "frame": int(row["frame"]),
            "time_sec": float(row["time_sec"]),
            "x": float(row["x"]),
            "y": float(row["y"]),
        }
        for row in tracks_rows
    ]
    assert track_payload["points"] == csv_points

    svg_root = ET.parse(out_dir / "tracks.svg").getroot()
    assert svg_root.attrib["viewBox"] == "0 0 64 48"
    valid_region_group = svg_root.find("svg:g[@id='valid-region']", SVG_NS)
    assert valid_region_group is not None
    track_group = svg_root.find("svg:g[@id='track-1']", SVG_NS)
    assert track_group is not None
    assert track_group.attrib["data-track-id"] == "1"
    assert track_group.attrib["data-frame-start"] == "0"
    assert track_group.attrib["data-frame-end"] == "4"
    assert track_group.attrib["data-direction"] == "entry"
    title = track_group.find("svg:title", SVG_NS)
    assert title is not None
    assert "Track 1" in (title.text or "")

    polyline = track_group.find("svg:polyline", SVG_NS)
    assert polyline is not None
    expected_points = " ".join(f"{point['x']},{point['y']}" for point in csv_points)
    assert polyline.attrib["points"] == expected_points

    circles = track_group.findall("svg:circle", SVG_NS)
    assert len(circles) == 2
    labels = track_group.findall("svg:text", SVG_NS)
    assert len(labels) == 2
    assert {label.attrib["class"] for label in labels} == {
        "track-label track-label-start",
        "track-label track-label-end",
    }
    assert {label.text for label in labels} == {"1"}
    assert meta["outputs"]["tracks_svg"] == str((out_dir / "tracks.svg").resolve())
    assert meta["outputs"]["tracks_render_json"] == str((out_dir / "tracks_render.json").resolve())


def test_svg_and_render_json_export_empty_tracks_as_valid_empty_documents(tmp_path: Path) -> None:
    svg_path = tmp_path / "tracks.svg"
    json_path = tmp_path / "tracks_render.json"

    export_tracks_svg(svg_path, width=64, height=48, points=[], line_thickness=2, start_radius=5)
    payload = export_tracks_render_json(json_path, width=64, height=48, points=[])

    assert payload["tracks"] == []
    with json_path.open(encoding="utf-8") as handle:
        assert json.load(handle)["tracks"] == []

    svg_root = ET.parse(svg_path).getroot()
    assert svg_root.attrib["viewBox"] == "0 0 64 48"
    assert svg_root.findall("svg:g[@class='track']", SVG_NS) == []


def test_tracks_overlay_heatmap_events_complete_existing_tracks() -> None:
    points = [
        _make_track_point(7, 10, 15.0, 20.0),
        _make_track_point(7, 20, 25.0, 20.0),
    ]
    event = HeatmapEvent(
        event_id=20001,
        source_fast_event_id=10001,
        source_track_ids=[7, 99],
        video_id="video",
        frame_start=0,
        frame_end=30,
        fps=30.0,
        points=[(5.0, 20.0), (15.0, 20.0), (25.0, 20.0), (35.0, 20.0)],
        direction="exit",
    )

    overlay_points, summary = _build_tracks_overlay_points_with_heatmap(
        points,
        [event],
        {"tracks_overlay_include_heatmap_events": True},
    )

    assert {point.track_id for point in overlay_points} == {7}
    assert [point.frame for point in overlay_points] == [0, 10, 20, 30]
    assert summary["heatmap_events_completed"] == 1
    assert summary["heatmap_events_added"] == 0
    assert summary["heatmap_completed_track_ids"] == [7]


def test_tracks_overlay_heatmap_events_add_synthetic_tracks() -> None:
    points = [_make_track_point(7, 10, 15.0, 20.0)]
    event = HeatmapEvent(
        event_id=20001,
        source_fast_event_id=10001,
        source_track_ids=[99],
        video_id="video",
        frame_start=0,
        frame_end=10,
        fps=30.0,
        points=[(305.0, 320.0), (335.0, 320.0)],
        direction="exit",
    )

    overlay_points, summary = _build_tracks_overlay_points_with_heatmap(
        points,
        [event],
        {"tracks_overlay_include_heatmap_events": True},
    )

    assert {point.track_id for point in overlay_points} == {7, 20001}
    assert [point.frame for point in overlay_points if point.track_id == 20001] == [0, 10]
    assert summary["heatmap_events_completed"] == 0
    assert summary["heatmap_events_added"] == 1
    assert summary["heatmap_added_track_ids"] == [20001]


def test_tracks_overlay_heatmap_events_match_by_time_and_distance() -> None:
    points = [_make_track_point(7, 10, 15.0, 20.0)]
    event = HeatmapEvent(
        event_id=20001,
        source_fast_event_id=10001,
        source_track_ids=[99],
        video_id="video",
        frame_start=0,
        frame_end=10,
        fps=30.0,
        points=[(5.0, 20.0), (35.0, 20.0)],
        direction="exit",
    )

    overlay_points, summary = _build_tracks_overlay_points_with_heatmap(
        points,
        [event],
        {
            "tracks_overlay_include_heatmap_events": True,
            "tracks_overlay_heatmap_match_max_gap_frames": 12,
            "tracks_overlay_heatmap_match_max_distance": 120.0,
        },
    )

    assert {point.track_id for point in overlay_points} == {7}
    assert [point.frame for point in overlay_points] == [0, 10]
    assert summary["heatmap_events_completed"] == 1
    assert summary["heatmap_events_added"] == 0


def test_tracks_overlay_heatmap_events_respect_valid_region_directions() -> None:
    mask = np.zeros((50, 60), dtype=np.uint8)
    mask[10:30, 10:30] = 255
    entry_event = HeatmapEvent(
        event_id=20001,
        source_fast_event_id=10001,
        source_track_ids=[99],
        video_id="video",
        frame_start=0,
        frame_end=10,
        fps=30.0,
        points=[(5.0, 20.0), (20.0, 20.0)],
        direction="entry",
    )
    exit_event = HeatmapEvent(
        event_id=20002,
        source_fast_event_id=10002,
        source_track_ids=[100],
        video_id="video",
        frame_start=20,
        frame_end=30,
        fps=30.0,
        points=[(20.0, 20.0), (45.0, 20.0)],
        direction="exit",
    )

    overlay_points, summary = _build_tracks_overlay_points_with_heatmap(
        [],
        [entry_event, exit_event],
        {
            "tracks_overlay_include_heatmap_events": True,
            "tracks_overlay_heatmap_allowed_directions": ["inside", "entry"],
        },
        valid_mask=mask,
    )

    assert {point.track_id for point in overlay_points} == {20001}
    assert summary["heatmap_events_added"] == 1
    assert summary["heatmap_events_rejected_valid_region"] == 1


def test_tracks_overlay_heatmap_events_respect_tracking_conditions() -> None:
    event = HeatmapEvent(
        event_id=20001,
        source_fast_event_id=10001,
        source_track_ids=[],
        video_id="video",
        frame_start=0,
        frame_end=10,
        fps=30.0,
        points=[(10.0, 20.0), (11.0, 20.0)],
        direction="inside",
    )

    overlay_points, summary = _build_tracks_overlay_points_with_heatmap(
        [],
        [event],
        {"tracks_overlay_include_heatmap_events": True},
        tracking_cfg={
            "min_track_length": 1,
            "min_track_duration_sec": 0.0,
            "min_track_displacement": 20.0,
            "min_track_path_length": 20.0,
            "min_track_straightness": 0.0,
            "require_start_or_end_in_valid_region": False,
            "valid_region_gate_dilate_px": 0,
            "spatial_thresholds_ref_width_px": 0,
        },
        frame_width=60,
    )

    assert overlay_points == []
    assert summary["heatmap_events_rejected_track_conditions"] == 1


def test_tracks_overlay_heatmap_events_reject_winding_geometry() -> None:
    event = HeatmapEvent(
        event_id=20001,
        source_fast_event_id=10001,
        source_track_ids=[],
        video_id="video",
        frame_start=0,
        frame_end=30,
        fps=30.0,
        points=[(0.0, 0.0), (20.0, 35.0), (40.0, 0.0), (60.0, 35.0)],
        direction="inside",
    )

    overlay_points, summary = _build_tracks_overlay_points_with_heatmap(
        [],
        [event],
        {
            "tracks_overlay_include_heatmap_events": True,
            "tracks_overlay_heatmap_min_straightness": 0.85,
            "tracks_overlay_heatmap_max_turn_deg": 90.0,
            "tracks_overlay_heatmap_max_mean_turn_deg": 35.0,
        },
    )

    assert overlay_points == []
    assert summary["heatmap_events_rejected_geometry"] == 1


def test_tracks_overlay_heatmap_events_simplify_to_line() -> None:
    event = HeatmapEvent(
        event_id=20001,
        source_fast_event_id=10001,
        source_track_ids=[],
        video_id="video",
        frame_start=0,
        frame_end=20,
        fps=30.0,
        points=[(0.0, 0.0), (10.0, 1.0), (20.0, 0.0)],
        direction="inside",
    )

    overlay_points, summary = _build_tracks_overlay_points_with_heatmap(
        [],
        [event],
        {
            "tracks_overlay_include_heatmap_events": True,
            "tracks_overlay_heatmap_min_straightness": 0.85,
            "tracks_overlay_heatmap_simplify_to_line": True,
        },
    )

    assert summary["heatmap_events_added"] == 1
    assert [point.y for point in overlay_points] == [0.0, 0.0, 0.0]


def _make_track_point(track_id: int, frame: int, x: float, y: float) -> TrackPoint:
    return TrackPoint(
        video_id="video",
        track_id=track_id,
        frame=frame,
        time_sec=frame / 30.0,
        x=x,
        y=y,
        vx=0.0,
        vy=0.0,
        bbox_x1=int(round(x)) - 1,
        bbox_y1=int(round(y)) - 1,
        bbox_x2=int(round(x)) + 1,
        bbox_y2=int(round(y)) + 1,
        area=20.0,
    )


def test_auto_merge_uses_local_overlap_continuity_for_short_shared_window() -> None:
    points = [
        _make_track_point(202, 32671, 715.5, 621.0),
        _make_track_point(202, 32672, 739.5, 631.5),
        _make_track_point(202, 32673, 760.5, 635.5),
        _make_track_point(202, 32674, 785.0, 644.0),
        _make_track_point(202, 32675, 810.5, 641.5),
        _make_track_point(202, 32676, 837.5, 625.5),
        _make_track_point(202, 32677, 860.5, 609.0),
        _make_track_point(202, 32678, 895.0, 570.0),
        _make_track_point(202, 32679, 922.0, 531.0),
        _make_track_point(202, 32680, 917.5, 465.5),
        _make_track_point(203, 32679, 882.0, 494.5),
        _make_track_point(203, 32680, 979.0, 449.0),
        _make_track_point(203, 32681, 1027.0, 404.0),
        _make_track_point(203, 32682, 1035.5, 285.0),
        _make_track_point(203, 32683, 1078.0, 172.0),
        _make_track_point(203, 32684, 1128.5, 54.0),
    ]
    cfg = {
        "auto_merge_suggested": True,
        "merge_max_gap_frames": 12,
        "merge_max_endpoint_distance": 100.0,
        "merge_overlap_min_common_frames": 3,
        "merge_overlap_max_mean_distance": 60.0,
        "merge_overlap_min_direction_cosine": 0.8,
    }

    merged_points, merges = _auto_merge_track_points(points, cfg)

    assert any(
        merge["track_a"] == 202 and merge["track_b"] == 203 and merge["reason"] == "overlap_local"
        for merge in merges
    )
    assert {point.track_id for point in merged_points} == {202}


def test_auto_merge_uses_connector_direction_for_overlapping_fragments() -> None:
    points = [
        _make_track_point(180, 13432, 708.0, 667.0),
        _make_track_point(180, 13433, 738.0, 666.5),
        _make_track_point(180, 13434, 769.0, 661.0),
        _make_track_point(180, 13435, 797.5, 650.5),
        _make_track_point(180, 13436, 830.0, 634.5),
        _make_track_point(180, 13437, 859.0, 606.0),
        _make_track_point(180, 13438, 881.5, 588.0),
        _make_track_point(180, 13439, 916.0, 549.5),
        _make_track_point(180, 13440, 956.0, 486.5),
        _make_track_point(180, 13441, 1012.0, 461.0),
        _make_track_point(180, 13442, 1029.0, 363.5),
        _make_track_point(180, 13443, 1081.0, 303.0),
        _make_track_point(181, 13439, 937.5, 550.0),
        _make_track_point(181, 13440, 910.5, 504.5),
        _make_track_point(181, 13441, 962.5, 477.5),
        _make_track_point(181, 13442, 984.5, 393.0),
        _make_track_point(181, 13443, 1002.5, 335.0),
    ]
    cfg = {
        "auto_merge_suggested": True,
        "merge_max_gap_frames": 12,
        "merge_max_endpoint_distance": 100.0,
        "merge_overlap_min_common_frames": 3,
        "merge_overlap_max_mean_distance": 60.0,
        "merge_overlap_min_direction_cosine": 0.8,
    }

    merged_points, merges = _auto_merge_track_points(points, cfg)

    assert any(
        merge["track_a"] == 180 and merge["track_b"] == 181 and merge["reason"] == "overlap"
        for merge in merges
    )
    assert {point.track_id for point in merged_points} == {180}


def test_auto_merge_keeps_nearby_tracks_separate_when_connector_direction_breaks() -> None:
    points = [
        _make_track_point(86, 1091, 1703.5, 36.0),
        _make_track_point(86, 1092, 1703.5, 36.0),
        _make_track_point(86, 1093, 1722.0, 47.5),
        _make_track_point(86, 1094, 1693.5, 16.5),
        _make_track_point(86, 1095, 1663.0, 24.5),
        _make_track_point(86, 1096, 1671.5, 2.5),
        _make_track_point(90, 1092, 1640.0, 71.0),
        _make_track_point(90, 1093, 1651.0, 59.5),
        _make_track_point(90, 1094, 1671.5, 46.5),
        _make_track_point(90, 1095, 1707.5, 50.0),
    ]
    cfg = {
        "auto_merge_suggested": True,
        "merge_max_gap_frames": 12,
        "merge_max_endpoint_distance": 100.0,
        "merge_overlap_min_common_frames": 3,
        "merge_overlap_max_mean_distance": 60.0,
        "merge_overlap_min_direction_cosine": 0.8,
    }

    merged_points, merges = _auto_merge_track_points(points, cfg)

    assert merges == []
    assert {point.track_id for point in merged_points} == {86, 90}

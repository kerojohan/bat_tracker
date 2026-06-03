from __future__ import annotations

import csv
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np
import yaml

from bat_tracker.pipeline import (
    _auto_merge_track_points,
    _dedupe_coexisting_track_points,
    _filter_points_start_or_end_in_mask,
    _rescue_crossing_continuation_points,
    run_pipeline,
)
from bat_tracker.render import export_tracks_render_json, export_tracks_svg
from bat_tracker.tracker import TrackPoint
from bat_tracker.trails import RealTimeTrailRenderer


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


def _read_video_frame(path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), f"could not open video {path}"
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
    ok, frame = cap.read()
    cap.release()
    assert ok, f"could not read frame {frame_idx} from {path}"
    return frame


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


def _make_bright_and_dim_tracks_video(tmp_path: Path) -> Path:
    frames: list[np.ndarray] = []
    for idx in range(10):
        frame = np.zeros((64, 96), dtype=np.uint8)
        if idx < 6:
            cv2.rectangle(frame, (8 + idx * 5, 16), (14 + idx * 5, 22), 230, -1)
            cv2.rectangle(frame, (12 + idx * 5, 42), (18 + idx * 5, 48), 55, -1)
        frames.append(frame)

    video_path = tmp_path / "bright_dim_tracks.mp4"
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
    assert meta["outputs"]["flight_trails_overlay_video"] == ""


def test_pipeline_secondary_detection_adds_missing_dim_track(tmp_path: Path) -> None:
    video_path = _make_bright_and_dim_tracks_video(tmp_path)

    cfg = _base_config()
    cfg["detection"]["diff_threshold"] = 120
    cfg["secondary_detection"] = {
        "enabled": True,
        "inherit_primary": True,
        "diff_threshold": 20,
        "dedupe_max_distance_px": 8.0,
        "dedupe_min_iou": 0.10,
    }
    cfg_path = tmp_path / "cfg_secondary.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    out_dir = tmp_path / "out_secondary"
    meta = run_pipeline(str(video_path), str(out_dir), str(cfg_path))
    tracks_rows = _read_tracks(out_dir / "tracks.csv")

    assert len({row["track_id"] for row in tracks_rows}) == 2
    assert meta["metrics"]["secondary_detection_raw_detections"] > 0
    assert meta["metrics"]["secondary_detection_added_detections"] > 0
    assert meta["metrics"]["secondary_detection_duplicate_detections"] > 0
    primary_overlay = out_dir / "primary_detections_overlay.png"
    secondary_overlay = out_dir / "secondary_detections_overlay.png"
    assert primary_overlay.exists()
    assert secondary_overlay.exists()
    assert meta["outputs"]["primary_detections_overlay_png"] == str(primary_overlay.resolve())
    assert meta["outputs"]["secondary_detections_overlay_png"] == str(secondary_overlay.resolve())
    assert int(np.count_nonzero(cv2.imread(str(primary_overlay), cv2.IMREAD_COLOR))) > 0
    assert int(np.count_nonzero(cv2.imread(str(secondary_overlay), cv2.IMREAD_COLOR))) > 0


def test_secondary_points_are_filtered_by_start_or_end_mask() -> None:
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[10:30, 10:30] = 255
    points = [
        _make_track_point(1, 0, 12, 12),
        _make_track_point(1, 1, 60, 60),
        _make_track_point(2, 0, 50, 50),
        _make_track_point(2, 1, 55, 55),
        _make_track_point(3, 0, 55, 55),
        _make_track_point(3, 1, 20, 20),
    ]

    kept, meta = _filter_points_start_or_end_in_mask(points, mask)

    assert {point.track_id for point in kept} == {1, 3}
    assert meta["mask_filter_enabled"] is True
    assert meta["tracks_before_mask_filter"] == 3
    assert meta["tracks_after_mask_filter"] == 2
    assert meta["tracks_rejected_by_mask_filter"] == 1


def test_realtime_trails_keep_coherent_motion_and_reject_local_jitter() -> None:
    renderer = RealTimeTrailRenderer(
        (80, 80),
        {
            "enabled": True,
            "history_frames": 6,
            "decay": 0.92,
            "segment_thickness": 3,
            "overlay_alpha": 0.75,
            "min_history_points": 3,
            "min_segment_displacement_px": 4.0,
            "min_recent_displacement_px": 10.0,
            "min_recent_path_length_px": 12.0,
            "min_recent_straightness": 0.4,
            "stationary_radius_px": 10.0,
        },
    )

    jitter_xy = [(55, 55), (56, 54), (55, 56), (54, 55), (56, 55)]
    out = np.zeros((80, 80, 3), dtype=np.uint8)
    for frame in range(5):
        out = renderer.update(
            np.zeros((80, 80, 3), dtype=np.uint8),
            [
                _make_track_point(1, frame, 8 + frame * 12, 20 + frame * 6),
                _make_track_point(2, frame, jitter_xy[frame][0], jitter_xy[frame][1]),
            ],
        )

    moving_energy = int(out[10:50, 5:65].sum())
    jitter_energy = int(out[48:62, 48:62].sum())
    assert moving_energy > 0
    assert jitter_energy * 4 < moving_energy


def test_pipeline_exports_realtime_trails_video_when_enabled(tmp_path: Path) -> None:
    video_path = _make_single_track_video(tmp_path)

    cfg = _base_config()
    cfg["flight_trails"] = {
        "enabled": True,
        "video_filename": "flight_trails_overlay.mp4",
        "history_frames": 6,
        "decay": 0.92,
        "segment_thickness": 3,
        "point_radius": 2,
        "overlay_alpha": 0.75,
        "min_history_points": 3,
        "min_segment_displacement_px": 3.0,
        "min_recent_displacement_px": 8.0,
        "min_recent_path_length_px": 10.0,
        "min_recent_straightness": 0.4,
        "stationary_radius_px": 8.0,
    }
    cfg_path = tmp_path / "cfg_trails.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    out_dir = tmp_path / "out_trails"
    meta = run_pipeline(str(video_path), str(out_dir), str(cfg_path))

    trail_video = out_dir / "flight_trails_overlay.mp4"
    assert trail_video.exists()
    assert meta["outputs"]["flight_trails_overlay_video"] == str(trail_video.resolve())

    source_frame = _read_video_frame(video_path, 4)
    overlay_frame = _read_video_frame(trail_video, 4)
    assert overlay_frame.shape == source_frame.shape
    assert int(np.abs(overlay_frame.astype(np.int16) - source_frame.astype(np.int16)).sum()) > 0


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


def test_dedupe_coexisting_tracks_removes_short_embedded_duplicate() -> None:
    points = []
    for frame in range(10):
        points.append(_make_track_point(1, frame, 100.0 + frame * 20.0, 400.0 - frame * 15.0))
    for frame in range(3, 7):
        points.append(_make_track_point(2, frame, 102.0 + frame * 20.0, 402.0 - frame * 15.0))
    for frame in range(10):
        points.append(_make_track_point(3, frame, 100.0 + frame * 20.0, 520.0 - frame * 15.0))

    deduped, duplicate_ids = _dedupe_coexisting_track_points(
        points,
        {"dedupe_coexisting_tracks": True},
    )

    assert duplicate_ids == [2]
    assert {point.track_id for point in deduped} == {1, 3}


def test_dedupe_coexisting_tracks_scales_distance_with_resolution() -> None:
    points = []
    for frame in range(10):
        points.append(_make_track_point(1, frame, 200.0 + frame * 40.0, 800.0 - frame * 30.0))
    for frame in range(3, 7):
        points.append(_make_track_point(2, frame, 204.0 + frame * 40.0, 804.0 - frame * 30.0))

    cfg = {
        "dedupe_coexisting_tracks": True,
        "dedupe_coexisting_max_mean_distance": 3.0,
        "dedupe_coexisting_reference_width": 1024,
        "dedupe_coexisting_reference_height": 576,
    }

    _, duplicate_ids_without_scaling = _dedupe_coexisting_track_points(
        points,
        {**cfg, "auto_scale_with_resolution": False},
        frame_size=(2048, 1152),
    )
    _, duplicate_ids_with_scaling = _dedupe_coexisting_track_points(
        points,
        cfg,
        frame_size=(2048, 1152),
    )

    assert duplicate_ids_without_scaling == []
    assert duplicate_ids_with_scaling == [2]


def test_rescue_crossing_continuation_extends_accepted_track() -> None:
    accepted = [_make_track_point(10, frame, 100.0 + frame * 10.0, 200.0) for frame in range(5)]
    fragment = [_make_track_point(20, frame, 144.0 + (frame - 4) * 9.0, 202.0) for frame in range(4, 9)]
    crossing = [_make_track_point(30, frame, 280.0 + frame * 8.0, 120.0) for frame in range(4, 9)]
    assessments = [
        {"track_id": 10, "accepted": True, "reject_reasons": ""},
        {"track_id": 20, "accepted": False, "reject_reasons": "valid_region_gate"},
        {"track_id": 30, "accepted": True, "reject_reasons": ""},
    ]

    rescued, rescues = _rescue_crossing_continuation_points(
        [*accepted, *fragment, *crossing],
        [*accepted, *crossing],
        assessments,
        {"rescue_crossing_continuations": True},
        frame_size=(1024, 576),
    )

    by_track = {}
    for point in rescued:
        by_track.setdefault(point.track_id, []).append(point)

    assert rescues == [
        {
            "track_id": 10,
            "source_track_id": 20,
            "points_added": 4,
            "gap_frames": 0,
            "start_distance": 4.472,
            "score": -3.528,
        }
    ]
    assert [point.frame for point in by_track[10]] == list(range(9))
    assert [point.frame for point in by_track[30]] == list(range(4, 9))

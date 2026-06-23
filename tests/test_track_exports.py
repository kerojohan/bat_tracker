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
    _exclude_mask_from_vegetation,
    _filter_points_excluding_directions,
    _filter_points_start_or_end_in_mask,
    _rescue_crossing_continuation_points,
    _rescue_motion_candidate_points,
    _select_entry_exit_mask,
    _write_events_csv,
    _write_tracks_csv,
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
        "cave_zones": {
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
            "cleanup_intermediate_outputs": False,
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


def test_pipeline_uses_cave_zones_mask_for_entry_exit_contract(tmp_path: Path) -> None:
    video_path = _make_single_track_video(tmp_path)
    cave_mask = np.zeros((48, 64), dtype=np.uint8)
    cave_mask[18:34, 24:44] = 255
    cave_mask_path = tmp_path / "cave_mask.png"
    cv2.imwrite(str(cave_mask_path), cave_mask)

    cfg = _base_config()
    cfg["tracking"]["require_start_or_end_in_valid_region"] = True
    cfg["tracking"]["entry_exit_zone_source"] = "cave_zones"
    cfg["cave_zones"] = {
        "enabled": True,
        "method": "annotation",
        "input_mask": str(cave_mask_path),
        "input_annotation": "",
        "min_component_area_ratio": 0.001,
        "max_components": 1,
        "dilate_px": 0,
        "output_subdir": "cave_zones",
    }
    cfg["output"]["cleanup_intermediate_outputs"] = False
    cfg_path = tmp_path / "cfg_cave_zones.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    out_dir = tmp_path / "out_cave_zones"
    meta = run_pipeline(str(video_path), str(out_dir), str(cfg_path))

    assert meta["metrics"]["entry_exit_zone_source"] == "cave_zones"
    assert meta["cave_zones"]["zones_total"] == 1
    assert Path(meta["outputs"]["cave_zones_mask_png"]).exists()
    assert Path(meta["outputs"]["cave_zones_overlay_png"]).exists()
    assert Path(meta["outputs"]["cave_zones_zones_json"]).exists()
    diagnostics = json.loads(Path(meta["outputs"]["cave_zones_diagnostics_json"]).read_text(encoding="utf-8"))
    assert diagnostics["track_endpoint_diagnostics"]["final_tracks_vs_entry_exit_zone"]["tracks_total"] == 1
    assert meta["cave_zones"]["track_endpoint_diagnostics"]["final_tracks_vs_entry_exit_zone"]["tracks_total"] == 1
    events = list(csv.DictReader((out_dir / "events.csv").open(newline="", encoding="utf-8")))
    assert events
    assert {row["direction"] for row in events} == {"entry"}
    render_payload = json.loads((out_dir / "tracks_render.json").read_text(encoding="utf-8"))
    assert render_payload["entry_exit_region"]["contours"]
    assert {track["direction"] for track in render_payload["tracks"]} == {"entry"}


def test_pipeline_exports_track_deduplication_debug_outputs(tmp_path: Path) -> None:
    video_path = _make_single_track_video(tmp_path)

    cfg = _base_config()
    cfg["tracking"]["enable_track_deduplication"] = True
    cfg["tracking"]["merge_strategy"] = "mark"
    cfg_path = tmp_path / "cfg_track_dedup.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    out_dir = tmp_path / "out_track_dedup"
    meta = run_pipeline(str(video_path), str(out_dir), str(cfg_path))

    csv_path = out_dir / "track_deduplication.csv"
    json_path = out_dir / "track_deduplication.json"
    overlay_path = out_dir / "track_deduplication_overlay.png"
    assert csv_path.exists()
    assert json_path.exists()
    assert overlay_path.exists()
    assert meta["outputs"]["track_deduplication_csv"] == str(csv_path.resolve())
    assert meta["outputs"]["track_deduplication_json"] == str(json_path.resolve())
    assert meta["outputs"]["track_deduplication_overlay_png"] == str(overlay_path.resolve())
    rows = _read_tracks(csv_path)
    assert rows
    assert {
        "track_id_original",
        "duplicate_group_id",
        "duplicate_decision",
        "duplicate_score",
        "reason",
    }.issubset(rows[0])
    with json_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["enabled"] is True
    assert cv2.imread(str(overlay_path), cv2.IMREAD_COLOR) is not None


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


def test_vegetation_mask_excludes_entry_exit_zone() -> None:
    vegetation = np.zeros((80, 80), dtype=np.uint8)
    vegetation[10:30, 10:30] = 255
    vegetation[45:65, 45:65] = 255
    entry_exit = np.zeros((80, 80), dtype=np.uint8)
    entry_exit[8:32, 8:32] = 255
    background = np.full((80, 80), 40, dtype=np.uint8)
    background[45:65, 45:65] = 160

    cleaned, meta = _exclude_mask_from_vegetation(background, vegetation, entry_exit, dilate_px=0)

    assert cleaned is not None
    assert int(np.count_nonzero(cleaned[10:30, 10:30])) == 0
    assert int(np.count_nonzero(cleaned[45:65, 45:65])) > 0
    assert meta["vegetation_exclusion_enabled"] is True
    assert meta["vegetation_pixels_removed_by_exclusion"] == 400


def test_vegetation_mask_keeps_textured_entry_exit_evidence() -> None:
    vegetation = np.zeros((80, 80), dtype=np.uint8)
    vegetation[10:30, 10:30] = 255
    entry_exit = np.zeros((80, 80), dtype=np.uint8)
    entry_exit[8:32, 8:32] = 255
    background = np.full((80, 80), 90, dtype=np.uint8)
    background[10:30, 10:30] = 155
    for x in range(10, 30, 4):
        background[10:30, x : x + 2] = 230

    cleaned, meta = _exclude_mask_from_vegetation(
        background,
        vegetation,
        entry_exit,
        dilate_px=0,
        mode="weak_evidence",
        keep_texture_percentile=70.0,
        keep_min_intensity_percentile=20.0,
    )

    assert cleaned is not None
    assert int(np.count_nonzero(cleaned[10:30, 10:30])) > 0
    assert meta["vegetation_pixels_kept_in_entry_exit_zone"] > 0


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


def test_auto_entry_exit_selection_penalizes_vegetation_overlap() -> None:
    background = np.full((80, 100), 180, dtype=np.uint8)
    background[20:45, 12:35] = 20
    background[20:45, 62:88] = 35

    cavemark = np.zeros_like(background)
    cavemark[20:45, 12:35] = 255
    cave_zones = np.zeros_like(background)
    cave_zones[20:45, 62:88] = 255
    vegetation = np.zeros_like(background)
    vegetation[18:48, 60:90] = 255
    motion = np.zeros_like(background, dtype=np.float32)
    motion[cave_zones > 0] = 10.0
    motion[cavemark > 0] = 7.0
    points = [
        _make_track_point(1, 0, 5, 30),
        _make_track_point(1, 1, 20, 30),
        _make_track_point(2, 0, 20, 30),
        _make_track_point(2, 1, 45, 30),
    ]

    selected, selected_source, meta = _select_entry_exit_mask(
        source_cfg="auto",
        candidates={"cavemark": cavemark, "cave_zones": cave_zones},
        background=background,
        motion_heatmap=motion,
        vegetation_mask=vegetation,
        raw_points=points,
        selection_cfg={
            "vegetation_overlap_penalty": 0.8,
            "motion_weight": 0.25,
            "dark_weight": 0.25,
            "endpoint_weight": 0.30,
            "area_weight": 0.20,
            "cavemark_bias": 0.12,
            "cave_zones_bias": 0.0,
            "dark_percentile": 20.0,
        },
    )

    assert selected is cavemark
    assert selected_source == "cavemark"
    assert meta["selected_source"] == "cavemark"
    scores = {row["source"]: row for row in meta["scores"]}
    assert scores["cave_zones"]["vegetation_overlap_ratio"] > 0.9
    assert scores["cavemark"]["vegetation_overlap_ratio"] == 0.0
    assert scores["cavemark"]["score"] > scores["cave_zones"]["score"]


def test_explicit_entry_exit_selection_keeps_requested_source() -> None:
    background = np.full((40, 40), 100, dtype=np.uint8)
    cave_zones = np.zeros_like(background)
    cave_zones[10:20, 10:20] = 255
    valid_region = np.zeros_like(background)
    valid_region[20:35, 20:35] = 255

    selected, selected_source, meta = _select_entry_exit_mask(
        source_cfg="cave_zones",
        candidates={"cave_zones": cave_zones, "valid_region": valid_region},
        background=background,
        motion_heatmap=None,
        vegetation_mask=None,
        raw_points=[],
        selection_cfg={},
    )

    assert selected is cave_zones
    assert selected_source == "cave_zones"
    assert meta["reason"] == "explicit_source"


def test_final_exports_drop_outside_tracks_consistently(tmp_path: Path) -> None:
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[8:32, 8:32] = 255
    points = [
        _make_track_point(1, 0, 45, 45),
        _make_track_point(1, 1, 50, 50),
        _make_track_point(2, 0, 50, 50),
        _make_track_point(2, 1, 20, 20),
        _make_track_point(3, 0, 20, 20),
        _make_track_point(3, 1, 50, 50),
        _make_track_point(3, 2, 25, 25),
    ]

    kept, meta = _filter_points_excluding_directions(points, mask, {"outside"})

    assert {point.track_id for point in kept} == {2, 3}
    assert meta["tracks_rejected_by_direction_filter"] == 1

    tracks_csv = tmp_path / "tracks.csv"
    events_csv = tmp_path / "events.csv"
    render_json = tmp_path / "tracks_render.json"
    _write_tracks_csv(tracks_csv, kept)
    _write_events_csv(events_csv, kept, mask)
    export_tracks_render_json(render_json, 80, 80, kept, direction_mask=mask)

    assert {row["track_id"] for row in _read_tracks(tracks_csv)} == {"2", "3"}
    event_rows = _read_tracks(events_csv)
    assert {row["track_id"] for row in event_rows} == {"2", "3"}
    assert {row["track_id"]: row["direction"] for row in event_rows} == {"2": "entry", "3": "exit"}
    with render_json.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    assert [track["track_id"] for track in payload["tracks"]] == [2, 3]
    assert {str(track["track_id"]): track["direction"] for track in payload["tracks"]} == {
        "2": "entry",
        "3": "exit",
    }


def test_motion_rescue_kept_but_final_exports_still_drop_outside_tracks() -> None:
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[8:32, 8:32] = 255
    accepted = [
        _make_track_point(2, 0, 50, 50),
        _make_track_point(2, 1, 20, 20),
    ]
    rejected_outside = [
        _make_track_point(1, 0, 45, 45),
        _make_track_point(1, 1, 20, 20),
        _make_track_point(1, 2, 50, 50),
    ]
    assessments = [
        {
            "track_id": 1,
            "accepted": False,
            "reject_reasons": "valid_region_gate",
            "num_detections": 3,
            "displacement_px": 60.0,
            "path_length_px": 90.0,
            "mean_speed_px_sec": 300.0,
            "straightness": 0.6,
        }
    ]
    rescued, rescues = _rescue_motion_candidate_points(
        [*accepted, *rejected_outside],
        accepted,
        assessments,
        {
            "rescue_motion_candidates": True,
            "rescue_motion_reject_reasons": "valid_region_gate;vegetation_mask",
            "rescue_motion_min_points": 3,
            "rescue_motion_min_displacement": 18.0,
            "rescue_motion_min_path_length": 24.0,
            "rescue_motion_min_mean_speed": 120.0,
        },
        interaction_mask=mask,
    )

    assert rescues
    assert {point.track_id for point in rescued} == {1, 2}

    final_points, meta = _filter_points_excluding_directions(rescued, mask, {"outside"})

    assert {point.track_id for point in final_points} == {2}
    assert meta["tracks_rejected_by_direction_filter"] == 1


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


def test_auto_merge_uses_single_frame_overlap_for_continuation() -> None:
    points = [
        _make_track_point(210, 100, 100.0, 300.0),
        _make_track_point(210, 101, 120.0, 285.0),
        _make_track_point(210, 102, 140.0, 270.0),
        _make_track_point(210, 103, 160.0, 255.0),
        _make_track_point(211, 103, 161.5, 254.0),
        _make_track_point(211, 104, 181.5, 239.0),
        _make_track_point(211, 105, 201.5, 224.0),
        _make_track_point(211, 106, 221.5, 209.0),
    ]
    cfg = {
        "auto_merge_suggested": True,
        "merge_max_gap_frames": 8,
        "merge_max_endpoint_distance": 80.0,
        "merge_overlap_min_common_frames": 3,
        "merge_overlap_max_mean_distance": 60.0,
        "merge_overlap_min_direction_cosine": 0.8,
    }

    merged_points, merges = _auto_merge_track_points(points, cfg)

    assert any(
        merge["track_a"] == 210 and merge["track_b"] == 211 and merge["reason"] == "overlap_local"
        for merge in merges
    )
    assert {point.track_id for point in merged_points} == {210}


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


def test_dedupe_coexisting_tracks_removes_same_start_duplicate_even_if_lengths_are_close() -> None:
    points = []
    for frame in range(4):
        points.append(_make_track_point(1, frame, 200.0 + frame * 18.0, 320.0 - frame * 11.0))
    for frame in range(3):
        points.append(_make_track_point(2, frame, 202.0 + frame * 18.0, 322.0 - frame * 11.0))

    deduped, duplicate_ids = _dedupe_coexisting_track_points(
        points,
        {"dedupe_coexisting_tracks": True},
    )

    assert duplicate_ids == [2]
    assert {point.track_id for point in deduped} == {1}


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

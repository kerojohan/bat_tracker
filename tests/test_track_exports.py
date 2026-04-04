from __future__ import annotations

import csv
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np
import yaml

from bat_tracker.pipeline import run_pipeline
from bat_tracker.render import export_tracks_render_json, export_tracks_svg


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

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import yaml

from bat_tracker.pipeline import run_pipeline


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


def _normalize_tracks(rows: list[dict[str, str]]) -> list[tuple[str, ...]]:
    keys = [
        "track_id",
        "frame",
        "time_sec",
        "x",
        "y",
        "vx",
        "vy",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "area",
    ]
    return [tuple(row[key] for key in keys) for row in rows]


def _filter_rows_to_frame_range(
    rows: list[dict[str, str]],
    *,
    start_frame: int,
    end_frame: int,
) -> list[dict[str, str]]:
    return [row for row in rows if start_frame <= int(row["frame"]) <= end_frame]


def _base_config() -> dict:
    return {
        "background": {
            "sample_frames": 20,
            "uniform_sampling": True,
        },
        "detection": {
            "blur_kernel": 1,
            "threshold_mode": "fixed",
            "diff_threshold": 10,
            "morph_open": 1,
            "morph_close": 1,
            "min_area": 20,
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
            "max_distance": 30,
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
            "export_track_clips": False,
        },
    }


def _make_prefix_clip_case(tmp_path: Path) -> tuple[Path, Path]:
    full_frames: list[np.ndarray] = []
    clip_frames: list[np.ndarray] = []
    for idx in range(40):
        frame = np.zeros((48, 64), dtype=np.uint8)
        if idx < 12:
            x0 = 18 + idx
            cv2.rectangle(frame, (x0, 18), (x0 + 18, 32), 220, -1)
        full_frames.append(frame)
        if idx < 12:
            clip_frames.append(frame.copy())

    full_path = tmp_path / "full.mp4"
    clip_path = tmp_path / "clip.mp4"
    _write_video(full_path, full_frames)
    _write_video(clip_path, clip_frames)
    return full_path, clip_path


def test_clip_can_reuse_full_background_for_prefix_reproducibility(tmp_path: Path) -> None:
    full_path, clip_path = _make_prefix_clip_case(tmp_path)

    cfg = _base_config()
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    out_full = tmp_path / "out_full"
    out_clip = tmp_path / "out_clip"
    out_clip_reused = tmp_path / "out_clip_reused"

    run_pipeline(str(full_path), str(out_full), str(cfg_path))
    run_pipeline(str(clip_path), str(out_clip), str(cfg_path))

    cfg_reused = _base_config()
    cfg_reused["background"]["input_image"] = str(out_full / "background.png")
    cfg_reused_path = tmp_path / "cfg_reused.yaml"
    cfg_reused_path.write_text(yaml.safe_dump(cfg_reused), encoding="utf-8")
    run_pipeline(str(clip_path), str(out_clip_reused), str(cfg_reused_path))

    full_tracks = _read_tracks(out_full / "tracks.csv")
    clip_tracks = _read_tracks(out_clip / "tracks.csv")
    reused_tracks = _read_tracks(out_clip_reused / "tracks.csv")

    assert _normalize_tracks(clip_tracks) != _normalize_tracks(full_tracks)
    assert _normalize_tracks(reused_tracks) == _normalize_tracks(full_tracks)


def test_precomputed_valid_region_mask_is_loaded_verbatim(tmp_path: Path) -> None:
    _, clip_path = _make_prefix_clip_case(tmp_path)

    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[:, 20:44] = 255
    mask_path = tmp_path / "mask.png"
    cv2.imwrite(str(mask_path), mask)

    cfg = _base_config()
    cfg["valid_region"] = {
        "enabled": True,
        "input_mask": str(mask_path),
        "apply_to_detection": False,
    }
    cfg_path = tmp_path / "cfg_mask.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    out_dir = tmp_path / "out_mask"
    meta = run_pipeline(str(clip_path), str(out_dir), str(cfg_path))

    exported_mask = cv2.imread(str(out_dir / "valid_region" / "mask.png"), cv2.IMREAD_GRAYSCALE)
    assert exported_mask is not None
    assert np.array_equal(exported_mask, mask)
    assert meta["valid_region"]["method"] == "input_mask"
    assert meta["valid_region"]["input_mask"] == str(mask_path.resolve())


def test_prefix_context_window_makes_full_and_clip_match(tmp_path: Path) -> None:
    full_path, clip_path = _make_prefix_clip_case(tmp_path)

    cfg = _base_config()
    cfg["background"]["context_start_sec"] = 0.0
    cfg["background"]["context_duration_sec"] = 1.2
    cfg_path = tmp_path / "cfg_prefix.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    out_full = tmp_path / "out_full_prefix"
    out_clip = tmp_path / "out_clip_prefix"
    run_pipeline(str(full_path), str(out_full), str(cfg_path))
    run_pipeline(str(clip_path), str(out_clip), str(cfg_path))

    full_tracks = _read_tracks(out_full / "tracks.csv")
    clip_tracks = _read_tracks(out_clip / "tracks.csv")
    full_prefix_tracks = _filter_rows_to_frame_range(full_tracks, start_frame=0, end_frame=11)
    assert _normalize_tracks(full_prefix_tracks) == _normalize_tracks(clip_tracks)


def test_valid_region_can_use_dedicated_context_without_changing_detection_background(tmp_path: Path) -> None:
    full_path, _ = _make_prefix_clip_case(tmp_path)

    cfg = _base_config()
    cfg["background"]["context_start_sec"] = 0.0
    cfg["background"]["context_duration_sec"] = -1.0
    cfg["valid_region"] = {
        "enabled": True,
        "method": "horizontal_illumination_profile",
        "apply_to_detection": False,
        "context_start_sec": 0.0,
        "context_duration_sec": 1.2,
        "blur_kernel_size": 31,
        "profile_smooth_window": 9,
        "threshold_ratio": 0.4,
        "safety_margin": 0,
        "min_region_width_ratio": 0.2,
    }
    cfg_path = tmp_path / "cfg_vr_context.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    out_dir = tmp_path / "out_vr_context"
    meta = run_pipeline(str(full_path), str(out_dir), str(cfg_path))

    assert meta["background"]["context_duration_sec"] == -1.0
    assert meta["valid_region"]["enabled"] is True
    mask = cv2.imread(str(out_dir / "valid_region" / "mask.png"), cv2.IMREAD_GRAYSCALE)
    assert mask is not None
    assert int(np.count_nonzero(mask)) > 0
    gate_overlay = out_dir / "valid_region" / "gate_overlay.png"
    assert gate_overlay.exists()

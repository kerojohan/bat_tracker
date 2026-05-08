from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULT_CONFIG: Dict[str, Any] = {
    "execution": {
        "device": "auto",
        "strict_parity": True,
    },
    "background": {
        "sample_frames": 200,
        "uniform_sampling": True,
        "input_image": "",
        "context_start_sec": 0.0,
        "context_duration_sec": -1.0,
    },
    "detection": {
        "blur_kernel": 5,
        "threshold_mode": "otsu",
        "diff_threshold": 18,
        "otsu_offset": -4,
        "morph_open": 1,
        "morph_close": 3,
        "min_area": 6,
        "max_area": 5000,
        "max_global_intensity_shift": -1.0,
        "max_foreground_ratio": -1.0,
        "max_detections_per_frame": 0,
        "roi_x_min": -1.0,
        "roi_x_max": -1.0,
        "roi_y_min": -1.0,
        "roi_y_max": -1.0,
        "temporal_burst_min_detections": 0,
        "temporal_burst_window_frames": 0,
        "temporal_burst_trigger_frames": 0,
        "temporal_burst_cooldown_frames": 0,
        "noise_mask_path": "",
        "clahe_enabled": False,
        "ff_diff_enabled": False,
        "opticalflow_enabled": False,
        "dedup_proximity_px": 30.0,
    },
    "tracking": {
        "max_distance": 60.0,
        "max_missed": 12,
        "min_track_length": 1,
        "min_track_duration_sec": 0.0,
        "min_track_displacement": 12.0,
        "min_track_path_length": 18.0,
        "min_track_straightness": 0.0,
        "require_start_or_end_in_valid_region": False,
        "valid_region_gate_dilate_px": 0,
        "auto_merge_suggested": False,
        "merge_max_gap_frames": 8,
        "merge_max_endpoint_distance": 80.0,
        "merge_overlap_min_common_frames": 3,
        "merge_overlap_max_mean_distance": 60.0,
        "merge_overlap_min_direction_cosine": 0.8,
        "export_track_candidates": False,
        "export_tracking_debug": False,
        "fast_track_enabled": False,
        "fast_track_speed_px_sec": 500,
        "fast_track_min_length": 3,
        "fast_track_min_duration_sec": 0.08,
        "fast_track_min_displacement": 40,
        "fast_track_min_path_length": 60,
        "fast_track_min_straightness": 0.65,
        "fast_track_require_gate_touch": True,
        "fast_track_gate_crossing_margin_px": 35,
        "adaptive_max_distance_enabled": False,
        "adaptive_max_distance_base": 120,
        "adaptive_max_distance_speed_gain": 1.25,
        "adaptive_max_distance_bbox_gain": 0.35,
        "adaptive_max_distance_cap": 220,
        "merge_fast_overlap_enabled": False,
        "merge_fast_overlap_max_mean_distance": 35,
        "merge_fast_overlap_min_common_frames": 2,
        "merge_fast_handoff_max_gap_frames": 3,
        "merge_fast_handoff_projection_tolerance_px": 45,
        "two_stage_association_enabled": False,
        "two_stage_association_score_threshold": 0.4,
        "kalman_enabled": False,
        "kalman_gate": 3.0,
    },
    "fast_events": {
        "enabled": False,
        "include_rejected_candidates": True,
        "include_accepted_tracks": True,
        "first_event_id": 10001,
        "min_source_track_points": 3,
        "min_source_displacement": 35.0,
        "min_source_path_length": 60.0,
        "min_source_mean_speed": 80.0,
        "min_source_straightness": 0.08,
        "max_source_duration_sec": 5.0,
        "max_group_gap_frames": 45,
        "max_group_endpoint_distance": 260.0,
        "max_group_overlap_distance": 180.0,
        "min_event_points": 4,
        "min_event_displacement": 80.0,
        "min_event_path_length": 120.0,
        "min_event_mean_speed": 120.0,
        "order_by_exit_projection": False,
        "overlay_line_thickness": 3,
        "overlay_start_radius": 6,
        "overlay_alpha": 1.0,
    },
    "heatmap_events": {
        "enabled": False,
        "first_event_id": 20001,
        "max_events": 0,
        "padding_frames": 0,
        "threshold": 14,
        "blur_kernel": 5,
        "percentile": 76.0,
        "corridor_width": 180.0,
        "bins": 22,
        "seed_y_min": -1.0,
        "y_max": -1.0,
        "min_points": 6,
        "min_displacement": 80.0,
        "overlay_line_thickness": 5,
        "overlay_alpha": 1.0,
    },
    "temporal_heatmap_events": {
        "enabled": False,
        "first_event_id": 30001,
        "window_sec": 3.0,
        "stride_sec": 1.0,
        "max_events": 0,
        "threshold": 14,
        "blur_kernel": 5,
        "percentile": 84.0,
        "morph_close": 9,
        "bins": 18,
        "min_component_area": 24,
        "min_displacement": 80.0,
        "min_path_length": 90.0,
        "min_straightness": 0.80,
        "min_elongation": 8.0,
        "y_max": -1.0,
        "dedupe_max_time_overlap_sec": 2.0,
        "dedupe_max_endpoint_distance": 140.0,
    },
    "valid_region": {
        "enabled": True,
        "method": "horizontal_illumination_profile",
        "apply_to_detection": True,
        "hybrid_combine_mode": "and",
        "input_image": "",
        "input_mask": "",
        "context_start_sec": -1.0,
        "context_duration_sec": -1.0,
        "blur_kernel_size": 151,
        "profile_smooth_window": 31,
        "threshold_ratio": 0.45,
        "safety_margin": 10,
        "min_region_width_ratio": 0.35,
        "depth_percentile": 85.0,
        "depth_morph_kernel": 9,
        "depth_min_area_ratio": 0.02,
        "depth_layer_percentiles": [],
        "depth_layer_dilate_px": [],
        "bottom_contour_snap_enabled": False,
        "bottom_contour_search_up_px": 18,
        "bottom_contour_search_down_px": 48,
        "bottom_contour_smooth_window": 31,
        "bottom_contour_gradient_quantile": 55.0,
        "bottom_contour_regularization": 0.90,
        "bottom_contour_max_step_px": 10,
        "bottom_contour_downward_bias": 0.10,
        "bottom_contour_regularization_mix": 0.75,
        "bottom_contour_deepest_strong_ratio": 0.70,
        "output_subdir": "valid_region",
    },
    "noise_mask": {
        "enabled": False,
        "sample_frames": 200,
        "noise_std_threshold": 8.0,
        "dilate_kernel": 15,
        "dilate_iterations": 2,
        "blur_kernel": 0,
        "save_mask_png": True,
    },
    "temporal_accumulation": {
        "enabled": False,
        "window_frames": 5,
        "diff_threshold": 12,
        "min_area": 20,
        "max_area": 50000,
        "morph_open": 3,
        "morph_close": 5,
        "min_trail_aspect": 3.0,
        "min_trail_length": 60,
    },
    "flight_trails": {
        "enabled": False,
        "video_filename": "flight_trails_overlay.mp4",
        "history_frames": 12,
        "max_track_gap_frames": 2,
        "decay": 0.90,
        "segment_thickness": 3,
        "point_radius": 2,
        "segment_intensity": 1.0,
        "point_intensity": 1.35,
        "overlay_alpha": 0.60,
        "colormap": "inferno",
        "clip_percentile": 99.0,
        "max_normalization_value": 0.0,
        "min_history_points": 3,
        "min_segment_displacement_px": 4.0,
        "min_recent_displacement_px": 10.0,
        "min_recent_path_length_px": 14.0,
        "min_recent_straightness": 0.20,
        "stationary_radius_px": 14.0,
    },
    "output": {
        "overlay_line_thickness": 2,
        "overlay_start_radius": 5,
        "overlay_alpha": 1.0,
        "overlay_draw_track_labels": False,
        "overlay_draw_track_labels_at_end": False,
        "overlay_label_font_scale": 0.5,
        "overlay_label_thickness": 1,
        "progress_enabled": True,
        "progress_step_percent": 5,
        "export_track_clips": False,
        "track_clips_subdir": "track_clips",
        "track_clips_padding_frames": 0,
        "trajectory_smoothing_enabled": False,
        "trajectory_smoothing_window": 5,
    },
}


def _deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        cfg = deepcopy(DEFAULT_CONFIG)
        _validate_config(cfg)
        return cfg

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as handle:
        user_cfg = yaml.safe_load(handle) or {}

    if not isinstance(user_cfg, dict):
        raise ValueError("Config must be a YAML mapping/dictionary")

    merged = _deep_update(DEFAULT_CONFIG, user_cfg)
    _validate_config(merged)
    return merged


def _validate_config(cfg: Dict[str, Any]) -> None:
    execution = cfg.get("execution", {})
    if not isinstance(execution, dict):
        raise ValueError("execution config must be a mapping/dictionary")
    device = str(execution.get("device", "auto")).strip().lower()
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"execution.device must be one of auto/cpu/cuda, got: {device}")

    output = cfg.get("output", {})
    if not isinstance(output, dict):
        raise ValueError("output config must be a mapping/dictionary")
    ts_win = int(output.get("trajectory_smoothing_window", 5))
    if ts_win < 3 or ts_win % 2 == 0:
        raise ValueError(
            f"output.trajectory_smoothing_window must be an odd integer >= 3, got: {ts_win}"
        )
    progress_step_percent = int(output.get("progress_step_percent", 5))
    if progress_step_percent < 1 or progress_step_percent > 100:
        raise ValueError(
            f"output.progress_step_percent must be between 1 and 100, got: {progress_step_percent}"
        )

    valid_region = cfg.get("valid_region", {})
    if not isinstance(valid_region, dict):
        raise ValueError("valid_region config must be a mapping/dictionary")

    background = cfg.get("background", {})
    if not isinstance(background, dict):
        raise ValueError("background config must be a mapping/dictionary")
    context_start_sec = float(background.get("context_start_sec", 0.0))
    context_duration_sec = float(background.get("context_duration_sec", -1.0))
    if context_start_sec < 0.0:
        raise ValueError(f"background.context_start_sec must be >= 0, got: {context_start_sec}")
    if context_duration_sec == 0.0 or context_duration_sec < -1.0:
        raise ValueError(
            "background.context_duration_sec must be > 0 or -1 to use the whole video, "
            f"got: {context_duration_sec}"
        )

    valid_context_start_sec = float(valid_region.get("context_start_sec", -1.0))
    valid_context_duration_sec = float(valid_region.get("context_duration_sec", -1.0))
    if valid_context_start_sec < -1.0:
        raise ValueError(
            f"valid_region.context_start_sec must be >= 0 or -1 to inherit background context, got: {valid_context_start_sec}"
        )
    if valid_context_duration_sec == 0.0 or valid_context_duration_sec < -1.0:
        raise ValueError(
            "valid_region.context_duration_sec must be > 0 or -1 to inherit/use whole source, "
            f"got: {valid_context_duration_sec}"
        )

    for field in ("blur_kernel_size", "profile_smooth_window"):
        value = int(valid_region.get(field, 0))
        if value < 1 or value % 2 == 0:
            raise ValueError(f"valid_region.{field} must be a positive odd integer, got: {value}")

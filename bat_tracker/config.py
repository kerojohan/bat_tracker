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
        "adaptive_block_size": 51,
        "adaptive_c": -5.0,
        "centroid_mode": "bbox",
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
    },
    "secondary_detection": {
        "enabled": False,
        "algorithm": "foreground",
        "inherit_primary": True,
        "dedupe_max_distance_px": 8.0,
        "dedupe_min_iou": 0.10,
        "kinetic_dedupe_max_overlap_distance_px": 45.0,
        "kinetic_dedupe_min_overlap_frames": 4,
        "kinetic_dedupe_min_overlap_ratio": 0.6,
    },
    "tracking": {
        "tracker": "kalman",
        "kalman_sigma_acc": 3.0,
        "kalman_measurement_std": 2.0,
        "kalman_high_area_threshold": 0.0,
        "max_distance": 60.0,
        "max_missed": 12,
        "min_track_length": 1,
        "min_track_duration_sec": 0.0,
        "min_track_displacement": 12.0,
        "min_track_path_length": 18.0,
        "min_track_straightness": 0.0,
        "static_noise_filter_enabled": True,
        "static_noise_min_duration_sec": 3.0,
        "static_noise_max_mean_speed_ratio_per_sec": 0.025,
        "static_noise_max_displacement_ratio_per_sec": 0.020,
        "require_start_or_end_in_valid_region": False,
        "entry_exit_zone_source": "auto",
        "valid_region_mode": "annotate",
        "valid_region_gate_dilate_px": 0,
        "auto_merge_suggested": False,
        "merge_max_gap_frames": 8,
        "merge_max_endpoint_distance": 80.0,
        "merge_overlap_min_common_frames": 3,
        "merge_overlap_max_mean_distance": 60.0,
        "merge_overlap_min_direction_cosine": 0.8,
        "merge_max_group_overlap_frames": 6,
        "merge_duplicate_max_distance": 12.0,
        "merge_max_group_size": 6,
        "enable_track_deduplication": False,
        "max_spatial_distance_px": 16.0,
        "max_temporal_gap_frames": 8,
        "min_direction_similarity": 0.75,
        "min_speed_similarity": 0.50,
        "min_duplicate_score": 0.75,
        "merge_strategy": "mark",
        "rescue_motion_candidates": False,
        "rescue_motion_reject_reasons": "valid_region_gate;vegetation_mask",
        "rescue_motion_min_points": 3,
        "rescue_motion_min_displacement": 18.0,
        "rescue_motion_min_path_length": 24.0,
        "rescue_motion_min_mean_speed": 120.0,
        "rescue_motion_min_straightness": 0.0,
        "rescue_motion_interaction_dilate_px": 0,
        "export_track_candidates": False,
    },
    "fast_events": {
        "enabled": False,
        "require_entry_or_exit": False,
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
    "cave_zones": {
        "enabled": True,
        "method": "hybrid",
        "input_mask": "",
        "input_annotation": "",
        "use_motion_heatmap": True,
        "use_dark_regions": True,
        "min_component_area_ratio": 0.002,
        "max_components": 3,
        "dilate_px": 8,
        "motion_percentile": 94.0,
        "dark_percentile": 18.0,
        "motion_dark_connect_dilate_px": 18,
        "output_subdir": "cave_zones",
    },
    "cavemark": {
        "enabled": False,
        "input_mask": "",
        "input_annotation": "",
        "dilate_px": 0,
        "output_subdir": "cavemark",
    },
    "entry_exit_zone_selection": {
        "vegetation_overlap_penalty": 0.45,
        "motion_weight": 0.25,
        "dark_weight": 0.25,
        "endpoint_weight": 0.30,
        "area_weight": 0.20,
        "cavemark_bias": 0.12,
        "cave_zones_bias": 0.0,
        "valid_region_bias": -0.15,
        "dark_percentile": 18.0,
        "ideal_area_ratio": 0.04,
        "max_reasonable_area_ratio": 0.18,
    },
    "vegetation_noise": {
        "enabled": False,
        "input_mask": "",
        # Scale pixel-based vegetation parameters by resolution using this reference size.
        "auto_scale_with_resolution": True,
        "reference_width": 1024,
        "reference_height": 576,
        "mask_dilate_px": 0,
        # Prevent cave entrances/exits from being treated as vegetation noise.
        "exclude_entry_exit_zones": True,
        "exclude_entry_exit_dilate_px": 12,
        "entry_exit_exclusion_mode": "weak_evidence",
        "entry_exit_keep_texture_percentile": 88.0,
        "entry_exit_keep_min_intensity_percentile": 35.0,
        "entry_exit_keep_min_gradient": 4.0,
        # If True, remove every track point that falls inside vegetation mask.
        "drop_all_points_in_mask": False,
        "auto_sample_frames": 220,
        "auto_max_frame_for_sampling": 1250,
        "auto_percentile": 85.0,
        "auto_min_component_area": 24,
        # Points inside vegetation mask with very low normalized speed are treated as jitter noise.
        "min_motion_ratio_per_sec": 0.25,
        # Require at least this many consecutive noisy points before removing the run.
        "min_consecutive_points": 3,
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
        "cleanup_intermediate_outputs": True,
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
    cleanup_intermediate_outputs = output.get("cleanup_intermediate_outputs", True)
    if not isinstance(cleanup_intermediate_outputs, bool):
        raise ValueError(
            "output.cleanup_intermediate_outputs must be a boolean, "
            f"got: {cleanup_intermediate_outputs!r}"
        )

    detection = cfg.get("detection", {})
    if not isinstance(detection, dict):
        raise ValueError("detection config must be a mapping/dictionary")
    threshold_mode = str(detection.get("threshold_mode", "fixed")).strip().lower()
    if threshold_mode not in {"fixed", "otsu", "adaptive"}:
        raise ValueError(
            f"detection.threshold_mode must be one of fixed/otsu/adaptive, got: {threshold_mode}"
        )
    centroid_mode = str(detection.get("centroid_mode", "moments")).strip().lower()
    if centroid_mode not in {"bbox", "moments"}:
        raise ValueError(
            f"detection.centroid_mode must be one of bbox/moments, got: {centroid_mode}"
        )

    tracking = cfg.get("tracking", {})
    if not isinstance(tracking, dict):
        raise ValueError("tracking config must be a mapping/dictionary")
    tracker_kind = str(tracking.get("tracker", "kalman")).strip().lower()
    if tracker_kind not in {"greedy", "kalman"}:
        raise ValueError(f"tracking.tracker must be one of greedy/kalman, got: {tracker_kind}")
    valid_region_mode = str(tracking.get("valid_region_mode", "annotate")).strip().lower()
    if valid_region_mode not in {"gate", "annotate"}:
        raise ValueError(
            f"tracking.valid_region_mode must be one of gate/annotate, got: {valid_region_mode}"
        )
    entry_exit_zone_source = str(tracking.get("entry_exit_zone_source", "auto")).strip().lower()
    if entry_exit_zone_source not in {"auto", "cave_zones", "cavemark", "valid_region"}:
        raise ValueError(
            "tracking.entry_exit_zone_source must be one of auto/cave_zones/cavemark/valid_region, "
            f"got: {entry_exit_zone_source}"
        )
    merge_strategy = str(tracking.get("merge_strategy", "mark")).strip().lower()
    if merge_strategy not in {"mark", "discard", "merge", "auto"}:
        raise ValueError(
            f"tracking.merge_strategy must be one of mark/discard/merge/auto, got: {merge_strategy}"
        )
    for field in (
        "max_spatial_distance_px",
        "min_direction_similarity",
        "min_speed_similarity",
        "min_duplicate_score",
    ):
        value = float(tracking.get(field, 0.0))
        if field.startswith("min_") and (value < 0.0 or value > 1.0):
            raise ValueError(f"tracking.{field} must be between 0 and 1, got: {value}")
        if field == "max_spatial_distance_px" and value <= 0.0:
            raise ValueError(f"tracking.{field} must be > 0, got: {value}")
    max_temporal_gap_frames = int(tracking.get("max_temporal_gap_frames", 0))
    if max_temporal_gap_frames < 0:
        raise ValueError(
            f"tracking.max_temporal_gap_frames must be >= 0, got: {max_temporal_gap_frames}"
        )

    valid_region = cfg.get("valid_region", {})
    if not isinstance(valid_region, dict):
        raise ValueError("valid_region config must be a mapping/dictionary")

    cave_zones = cfg.get("cave_zones", {})
    if not isinstance(cave_zones, dict):
        raise ValueError("cave_zones config must be a mapping/dictionary")
    cave_zones_method = str(cave_zones.get("method", "hybrid")).strip().lower()
    if cave_zones_method not in {"hybrid", "annotation", "motion", "dark"}:
        raise ValueError(
            f"cave_zones.method must be one of hybrid/annotation/motion/dark, got: {cave_zones_method}"
        )
    max_components = int(cave_zones.get("max_components", 3))
    if max_components < 1:
        raise ValueError(f"cave_zones.max_components must be >= 1, got: {max_components}")
    min_component_area_ratio = float(cave_zones.get("min_component_area_ratio", 0.002))
    if min_component_area_ratio < 0.0 or min_component_area_ratio > 1.0:
        raise ValueError(
            "cave_zones.min_component_area_ratio must be between 0 and 1, "
            f"got: {min_component_area_ratio}"
        )

    cavemark = cfg.get("cavemark", {})
    if not isinstance(cavemark, dict):
        raise ValueError("cavemark config must be a mapping/dictionary")

    entry_exit_selection = cfg.get("entry_exit_zone_selection", {})
    if not isinstance(entry_exit_selection, dict):
        raise ValueError("entry_exit_zone_selection config must be a mapping/dictionary")
    for field in (
        "vegetation_overlap_penalty",
        "motion_weight",
        "dark_weight",
        "endpoint_weight",
        "area_weight",
        "cavemark_bias",
        "cave_zones_bias",
        "valid_region_bias",
    ):
        float(entry_exit_selection.get(field, DEFAULT_CONFIG["entry_exit_zone_selection"][field]))

    secondary_detection = cfg.get("secondary_detection", {})
    if not isinstance(secondary_detection, dict):
        raise ValueError("secondary_detection config must be a mapping/dictionary")
    secondary_algorithm = str(secondary_detection.get("algorithm", "foreground")).strip().lower()
    if secondary_algorithm not in {"foreground", "kinetic"}:
        raise ValueError(
            f"secondary_detection.algorithm must be one of foreground/kinetic, got: {secondary_algorithm}"
        )
    dedupe_max_distance_px = float(secondary_detection.get("dedupe_max_distance_px", 8.0))
    if dedupe_max_distance_px < 0.0:
        raise ValueError(
            f"secondary_detection.dedupe_max_distance_px must be >= 0, got: {dedupe_max_distance_px}"
        )
    dedupe_min_iou = float(secondary_detection.get("dedupe_min_iou", 0.10))
    if dedupe_min_iou < 0.0 or dedupe_min_iou > 1.0:
        raise ValueError(f"secondary_detection.dedupe_min_iou must be between 0 and 1, got: {dedupe_min_iou}")

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

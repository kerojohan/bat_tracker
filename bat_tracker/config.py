from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Set

import yaml

DEFAULT_CONFIG: Dict[str, Any] = {
    "execution": {
        "device": "auto",
        "strict_parity": True,
    },
    "background": {
        "sample_frames": 200,
        "uniform_sampling": True,
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
    },
    "valid_region": {
        "enabled": True,
        "method": "horizontal_illumination_profile",
        "apply_to_detection": True,
        "hybrid_combine_mode": "and",
        "input_image": "",
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
    "events": {
        "direction_mode": "endpoint",
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
        "event_analysis_enabled": False,
        "event_analysis_fragmentation_gap": 6,
        "trajectory_smoothing_enabled": False,
        "trajectory_smoothing_window": 5,
    },
    "diagnostics": {
        "enabled": False,
        "frame_metrics_csv": "diagnostics_frame_metrics.csv",
        "track_filter_csv": "diagnostics_track_filter.csv",
        "intermediate_subdir": "diagnostics_intermediate",
        "save_binary_stride": 0,
        "sample_miss_stride": 0,
        "miss_diff_threshold": 18,
    },
    "scene_auto_tune": {
        "enabled": False,
        "write_profile": True,
        "use_recommended_values": True,
        # Si és true, l'autotune pot sobreescriure claus ja definides al YAML d'usuari.
        "overrides_allowed": False,
        "profile_subdir": "scene_profile",
        "sample_frames": 48,
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


def collect_user_explicit_paths(user_cfg: Dict[str, Any], prefix: str = "") -> Set[str]:
    """
    Claus fulla presents només al YAML d'usuari (per saber què està fixat explícitament).
    """
    paths: Set[str] = set()
    for key, value in user_cfg.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict) and value:
            paths |= collect_user_explicit_paths(value, path)
        else:
            paths.add(path)
    return paths


def load_config(path: str | Path | None) -> Dict[str, Any]:
    cfg, _ = load_config_with_user_paths(path)
    return cfg


def load_config_with_user_paths(path: str | Path | None) -> tuple[Dict[str, Any], Set[str]]:
    if path is None:
        cfg = deepcopy(DEFAULT_CONFIG)
        _validate_config(cfg)
        return cfg, set()

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as handle:
        user_cfg = yaml.safe_load(handle) or {}

    if not isinstance(user_cfg, dict):
        raise ValueError("Config must be a YAML mapping/dictionary")

    explicit = collect_user_explicit_paths(user_cfg)
    merged = _deep_update(deepcopy(DEFAULT_CONFIG), user_cfg)
    _validate_config(merged)
    return merged, explicit


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

    events = cfg.get("events", {})
    if not isinstance(events, dict):
        raise ValueError("events config must be a mapping/dictionary")
    direction_mode = str(events.get("direction_mode", "endpoint")).strip().lower()
    if direction_mode not in {"endpoint", "outward_crossing"}:
        raise ValueError(
            f"events.direction_mode must be 'endpoint' or 'outward_crossing', got: {direction_mode}"
        )

    mg = valid_region.get("mask_geometry")
    if mg is not None:
        if not isinstance(mg, dict):
            raise ValueError("valid_region.mask_geometry must be a mapping/dictionary or omitted")
        mmode = str(mg.get("mode", "baseline")).strip().lower()
        allowed_geom = {
            "baseline",
            "none",
            "dilate",
            "convex_hull",
            "convexhull",
            "min_area_rect",
            "oriented_bbox",
            "minarearect",
            "gradient_band_union",
            "gradient_band",
        }
        if mmode not in allowed_geom:
            raise ValueError(
                f"valid_region.mask_geometry.mode must be one of {sorted(allowed_geom)}, got: {mmode}"
            )

    for field in ("blur_kernel_size", "profile_smooth_window"):
        value = int(valid_region.get(field, 0))
        if value < 1 or value % 2 == 0:
            raise ValueError(f"valid_region.{field} must be a positive odd integer, got: {value}")

    sat = cfg.get("scene_auto_tune", {})
    if not isinstance(sat, dict):
        raise ValueError("scene_auto_tune must be a mapping/dictionary")
    for key in ("enabled", "write_profile", "use_recommended_values", "overrides_allowed"):
        if key in sat and not isinstance(sat[key], bool):
            raise ValueError(f"scene_auto_tune.{key} must be a boolean")
    if "sample_frames" in sat:
        sf = int(sat["sample_frames"])
        if sf < 1:
            raise ValueError(f"scene_auto_tune.sample_frames must be >= 1, got: {sf}")
    if "profile_subdir" in sat and not isinstance(sat["profile_subdir"], str):
        raise ValueError("scene_auto_tune.profile_subdir must be a string")

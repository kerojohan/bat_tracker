from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULT_CONFIG: Dict[str, Any] = {
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
    },
    "tracking": {
        "max_distance": 60.0,
        "max_missed": 12,
        "min_track_length": 1,
    },
    "output": {
        "overlay_line_thickness": 2,
        "overlay_start_radius": 5,
        "overlay_alpha": 1.0,
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
        return deepcopy(DEFAULT_CONFIG)

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as handle:
        user_cfg = yaml.safe_load(handle) or {}

    if not isinstance(user_cfg, dict):
        raise ValueError("Config must be a YAML mapping/dictionary")

    return _deep_update(DEFAULT_CONFIG, user_cfg)

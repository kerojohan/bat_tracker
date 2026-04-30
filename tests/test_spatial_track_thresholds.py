from __future__ import annotations

from bat_tracker.pipeline import _effective_min_track_spatial_thresholds, _filter_track_points
from bat_tracker.tracker import TrackPoint


def _minimal_tracking_cfg(**overrides: object) -> dict:
    base = {
        "min_track_length": 1,
        "min_track_duration_sec": 0.0,
        "min_track_displacement": 100.0,
        "min_track_path_length": 100.0,
        "min_track_straightness": 0.0,
        "require_start_or_end_in_valid_region": False,
        "valid_region_gate_dilate_px": 0,
        "spatial_thresholds_ref_width_px": 0,
    }
    base.update(overrides)
    return base


def test_effective_spatial_thresholds_disabled_when_ref_zero() -> None:
    cfg = _minimal_tracking_cfg(
        min_track_displacement=40.0,
        min_track_path_length=50.0,
        spatial_thresholds_ref_width_px=0,
    )
    scale, d, p = _effective_min_track_spatial_thresholds(cfg, 960)
    assert scale == 1.0
    assert d == 40.0
    assert p == 50.0


def test_effective_spatial_thresholds_scales_with_frame_width() -> None:
    cfg = _minimal_tracking_cfg(
        min_track_displacement=40.0,
        min_track_path_length=50.0,
        spatial_thresholds_ref_width_px=1920,
    )
    scale, d, p = _effective_min_track_spatial_thresholds(cfg, 960)
    assert scale == 0.5
    assert d == 20.0
    assert p == 25.0


def test_filter_track_points_respects_spatial_scaling() -> None:
    """Mitad de resolución → umbrales mitad → un track que fallaría en ref pasa."""
    cfg = _minimal_tracking_cfg(
        min_track_displacement=100.0,
        min_track_path_length=100.0,
        spatial_thresholds_ref_width_px=1920,
    )
    pts = [
        TrackPoint("v", 1, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 1.0),
        TrackPoint("v", 1, 1, 0.1, 60.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 1.0),
    ]
    rejected, _ = _filter_track_points(pts, cfg, fps=30.0, valid_mask=None, frame_width=1920)
    accepted, _ = _filter_track_points(pts, cfg, fps=30.0, valid_mask=None, frame_width=960)
    assert rejected == []
    assert len(accepted) == 2

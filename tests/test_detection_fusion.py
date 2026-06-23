from __future__ import annotations

from bat_tracker.detection import Detection
from bat_tracker.detection_fusion import build_secondary_detection_config
from bat_tracker.detection_fusion import fuse_detections
from bat_tracker.kinetic_secondary import suppress_temporal_burst_track_points
from bat_tracker.tracker import TrackPoint


def _det(x: float, y: float, size: int = 6) -> Detection:
    half = size // 2
    return Detection(
        x=x,
        y=y,
        bbox_x1=int(x) - half,
        bbox_y1=int(y) - half,
        bbox_x2=int(x) + half,
        bbox_y2=int(y) + half,
        area=float(size * size),
    )


def _point(track_id: int, frame: int, x: float = 0.0, y: float = 0.0) -> TrackPoint:
    return TrackPoint(
        video_id="clip",
        track_id=track_id,
        frame=frame,
        time_sec=frame / 25.0,
        x=x,
        y=y,
        vx=0.0,
        vy=0.0,
        bbox_x1=int(x) - 2,
        bbox_y1=int(y) - 2,
        bbox_x2=int(x) + 2,
        bbox_y2=int(y) + 2,
        area=16.0,
    )


def test_fuse_detections_keeps_primary_and_adds_secondary_missing() -> None:
    primary = [_det(20, 20), _det(60, 60)]
    secondary = [_det(22, 21), _det(95, 95)]

    fused, stats = fuse_detections(
        primary,
        secondary,
        dedupe_max_distance_px=5.0,
        dedupe_min_iou=0.10,
    )

    assert fused == [primary[0], primary[1], secondary[1]]
    assert stats.primary_count == 2
    assert stats.secondary_count == 2
    assert stats.secondary_added == 1
    assert stats.secondary_duplicates == 1


def test_suppress_temporal_burst_track_points_removes_synchronized_secondary_burst() -> None:
    points = []
    for frame in range(10, 14):
        for track_id in range(1, 6):
            points.append(_point(track_id, frame, x=frame * 5.0, y=track_id * 10.0))
    for frame in range(30, 34):
        points.append(_point(99, frame, x=frame * 4.0, y=50.0))

    filtered, meta = suppress_temporal_burst_track_points(
        points,
        min_points_per_frame=5,
        window_frames=4,
        trigger_frames=3,
        cooldown_frames=2,
    )

    assert {point.track_id for point in filtered} == {99}
    assert meta["temporal_burst_tracks_removed"] == 5
    assert meta["temporal_burst_points_removed"] == 20
    assert meta["temporal_burst_frame_start"] == 10


def test_secondary_detection_config_inherits_primary_and_applies_overrides() -> None:
    cfg = build_secondary_detection_config(
        {"threshold_mode": "otsu", "diff_threshold": 18, "min_area": 6},
        {
            "enabled": True,
            "inherit_primary": True,
            "dedupe_max_distance_px": 8.0,
            "threshold_mode": "fixed",
            "diff_threshold": 10,
        },
    )

    assert cfg == {"threshold_mode": "fixed", "diff_threshold": 10, "min_area": 6}

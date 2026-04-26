from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from bat_tracker.fast_events import reconstruct_fast_events
from bat_tracker.fast_events import render_fast_events_overlay
from bat_tracker.fast_events import write_fast_events_csv
from bat_tracker.fast_events import write_fast_tracks_csv
from bat_tracker.heatmap_events import reconstruct_heatmap_events
from bat_tracker.tracker import TrackPoint


def _point(track_id: int, frame: int, x: float, y: float) -> TrackPoint:
    return TrackPoint(
        video_id="clip",
        track_id=track_id,
        frame=frame,
        time_sec=frame / 30.0,
        x=x,
        y=y,
        vx=0.0,
        vy=0.0,
        bbox_x1=int(x) - 2,
        bbox_y1=int(y) - 2,
        bbox_x2=int(x) + 2,
        bbox_y2=int(y) + 2,
        area=20.0,
    )


def _cfg(enabled: bool = True) -> dict:
    return {
        "enabled": enabled,
        "include_rejected_candidates": True,
        "include_accepted_tracks": True,
        "first_event_id": 900,
        "min_source_track_points": 3,
        "min_source_displacement": 15.0,
        "min_source_path_length": 15.0,
        "min_source_mean_speed": 10.0,
        "min_source_straightness": 0.0,
        "max_source_duration_sec": 5.0,
        "max_group_gap_frames": 20,
        "max_group_endpoint_distance": 80.0,
        "max_group_overlap_distance": 80.0,
        "min_event_points": 4,
        "min_event_displacement": 40.0,
        "min_event_path_length": 40.0,
        "min_event_mean_speed": 10.0,
        "order_by_exit_projection": False,
    }


def test_fast_events_group_accepted_and_rejected_fragments() -> None:
    points = [
        _point(24, 100, 300, 250),
        _point(24, 101, 292, 246),
        _point(24, 102, 284, 242),
        _point(23, 101, 270, 230),
        _point(23, 102, 245, 210),
        _point(23, 103, 220, 190),
        _point(27, 105, 185, 165),
        _point(27, 106, 150, 135),
        _point(27, 107, 115, 105),
    ]
    assessments = [
        {"track_id": "24", "accepted": "True", "reject_reasons": ""},
        {"track_id": "23", "accepted": "False", "reject_reasons": "valid_region_gate"},
        {"track_id": "27", "accepted": "False", "reject_reasons": "valid_region_gate"},
    ]

    events = reconstruct_fast_events(points, assessments, _cfg(), frame_shape=(300, 400))

    assert len(events) == 1
    event = events[0]
    assert event.event_id == 900
    assert event.source_track_ids == [23, 24, 27]
    assert event.accepted_track_ids == [24]
    assert event.rejected_track_ids == [23, 27]
    assert event.direction == "exit"
    assert event.confidence > 0.5


def test_fast_events_exports_csv_and_overlay(tmp_path: Path) -> None:
    points = [
        _point(1, 0, 90, 90),
        _point(1, 1, 70, 70),
        _point(1, 2, 50, 50),
        _point(1, 3, 30, 30),
    ]
    assessments = [{"track_id": "1", "accepted": "True", "reject_reasons": ""}]
    events = reconstruct_fast_events(points, assessments, _cfg(), frame_shape=(120, 120))

    events_path = tmp_path / "fast_events.csv"
    tracks_path = tmp_path / "fast_tracks.csv"
    write_fast_events_csv(events_path, events)
    write_fast_tracks_csv(tracks_path, events, assessments)

    with events_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["event_id"] == "900"
    assert rows[0]["source_track_ids"] == "1"

    with tracks_path.open(newline="", encoding="utf-8") as handle:
        track_rows = list(csv.DictReader(handle))
    assert len(track_rows) == 4
    assert {row["event_id"] for row in track_rows} == {"900"}

    overlay = render_fast_events_overlay(np.zeros((120, 120), dtype=np.uint8), events)
    assert overlay.shape == (120, 120, 3)
    assert int(np.count_nonzero(overlay)) > 0


def test_fast_events_disabled_returns_no_events() -> None:
    points = [_point(1, frame, 100 - frame * 10, 100 - frame * 10) for frame in range(5)]
    assessments = [{"track_id": "1", "accepted": "True", "reject_reasons": ""}]

    assert reconstruct_fast_events(points, assessments, _cfg(enabled=False), frame_shape=(120, 120)) == []


def test_heatmap_events_reconstruct_motion_corridor(tmp_path: Path) -> None:
    frames: list[np.ndarray] = []
    for idx in range(12):
        frame = np.zeros((90, 140), dtype=np.uint8)
        x = 112 - idx * 8
        y = 70 - idx * 4
        cv2.circle(frame, (x, y), 4, 230, -1)
        frames.append(frame)

    video_path = tmp_path / "fast_line.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (140, 90),
    )
    assert writer.isOpened()
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()

    fast_event = reconstruct_fast_events(
        [
            _point(1, 0, 112, 70),
            _point(1, 5, 72, 50),
            _point(1, 11, 24, 26),
        ],
        [{"track_id": "1", "accepted": "True", "reject_reasons": ""}],
        {
            **_cfg(),
            "min_event_points": 2,
            "min_event_displacement": 20.0,
            "min_event_path_length": 20.0,
        },
        frame_shape=(90, 140),
    )[0]

    events = reconstruct_heatmap_events(
        str(video_path),
        [fast_event],
        {
            "enabled": True,
            "threshold": 8,
            "blur_kernel": 5,
            "percentile": 60.0,
            "corridor_width": 30.0,
            "bins": 10,
            "y_max": -1.0,
            "min_points": 3,
            "min_displacement": 20.0,
        },
        fps=10.0,
        frame_count=12,
    )

    assert len(events) == 1
    assert events[0].direction == "exit"
    assert len(events[0].points) >= 3

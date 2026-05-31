from __future__ import annotations

from bat_tracker.track_quality import compute_track_quality, summarize_merges
from bat_tracker.tracker import TrackPoint


def _point(track_id: int, frame: int, x: float, y: float, fps: float = 25.0) -> TrackPoint:
    return TrackPoint(
        video_id="clip",
        track_id=track_id,
        frame=frame,
        time_sec=frame / fps,
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


def test_quality_counts_tracks_and_distributions() -> None:
    points = [_point(1, f, float(f * 10), 100.0) for f in range(10)]
    points += [_point(2, f, 100.0, float(f * 10)) for f in range(8)]

    quality = compute_track_quality(points, fps=25.0)

    assert quality["tracks_total"] == 2
    assert quality["track_length"]["max"] == 10
    assert quality["straightness"]["mean"] > 0.9


def test_over_merge_suspect_flags_long_low_straightness_track() -> None:
    # Track largo en el tiempo pero que vuelve sobre sí mismo: rectitud baja.
    fps = 25.0
    points = []
    for f in range(60):
        # zig-zag de amplitud pequeña: muchas detecciones, desplazamiento neto bajo
        x = 500.0 + (10.0 if f % 2 == 0 else -10.0)
        y = 400.0 + (10.0 if f % 4 < 2 else -10.0)
        points.append(_point(1, f, x, y, fps))

    quality = compute_track_quality(
        points,
        fps=fps,
        over_merge_min_detections=40,
        over_merge_min_duration_sec=1.0,
    )

    assert 1 in quality["over_merge_suspect_tracks"]
    assert quality["over_merge_suspect_count"] == 1


def test_summarize_merges_reports_chain_size() -> None:
    merges = [
        {"track_a": 1, "track_b": 2, "merged_to": 1, "reason": "overlap"},
        {"track_a": 1, "track_b": 3, "merged_to": 1, "reason": "overlap"},
        {"track_a": 1, "track_b": 4, "merged_to": 1, "reason": "handoff"},
    ]
    summary = summarize_merges(merges)

    assert summary["merges_total"] == 3
    assert summary["overlap_merges"] == 2
    assert summary["handoff_merges"] == 1
    # tracks 1,2,3,4 colapsados en un único destino
    assert summary["max_merge_group_size"] == 4


def test_summarize_merges_empty() -> None:
    summary = summarize_merges(None)
    assert summary["merges_total"] == 0
    assert summary["max_merge_group_size"] == 1

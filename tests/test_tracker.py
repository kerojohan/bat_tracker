from bat_tracker.detection import Detection
from bat_tracker.tracker import GreedyTracker


def test_tracker_keeps_id_and_velocity():
    tracker = GreedyTracker(max_distance=10.0, max_missed=2, fps=10.0, video_id="vid")

    p0 = tracker.step(0, [Detection(10, 10, 8, 8, 12, 12, 16)])
    p1 = tracker.step(1, [Detection(12, 10, 10, 8, 14, 12, 16)])

    assert len(p0) == 1
    assert len(p1) == 1
    assert p0[0].track_id == p1[0].track_id
    assert p1[0].vx > 0
    assert abs(p1[0].vy) < 1e-6

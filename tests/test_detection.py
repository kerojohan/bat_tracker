import numpy as np
import cv2

from bat_tracker.detection import detect_foreground_blobs


def test_detect_foreground_single_blob():
    background = np.zeros((80, 80), dtype=np.uint8)
    frame = background.copy()
    cv2.circle(frame, (40, 30), 5, 255, -1)

    cfg = {
        "blur_kernel": 3,
        "diff_threshold": 10,
        "morph_open": 1,
        "morph_close": 1,
        "min_area": 20,
        "max_area": 200,
    }

    detections = detect_foreground_blobs(frame, background, cfg)
    assert len(detections) == 1
    det = detections[0]
    assert abs(det.x - 40) < 2
    assert abs(det.y - 30) < 2
    assert det.area >= 20


def test_detect_foreground_rejects_global_intensity_shift():
    background = np.zeros((80, 80), dtype=np.uint8)
    frame = np.full((80, 80), 35, dtype=np.uint8)
    cv2.circle(frame, (40, 30), 5, 255, -1)

    cfg = {
        "blur_kernel": 3,
        "diff_threshold": 10,
        "morph_open": 1,
        "morph_close": 1,
        "min_area": 5,
        "max_area": 500,
        "max_global_intensity_shift": 5.0,
    }

    detections = detect_foreground_blobs(frame, background, cfg)
    assert detections == []


def test_detect_foreground_rejects_frame_with_too_many_blobs():
    background = np.zeros((100, 100), dtype=np.uint8)
    frame = background.copy()
    for y in (20, 50, 80):
        for x in (20, 50, 80):
            cv2.circle(frame, (x, y), 3, 255, -1)

    cfg = {
        "blur_kernel": 1,
        "diff_threshold": 10,
        "morph_open": 1,
        "morph_close": 1,
        "min_area": 5,
        "max_area": 200,
        "max_detections_per_frame": 4,
    }

    detections = detect_foreground_blobs(frame, background, cfg)
    assert detections == []


def test_detect_foreground_applies_roi_limits():
    background = np.zeros((80, 80), dtype=np.uint8)
    frame = background.copy()
    cv2.circle(frame, (15, 40), 4, 255, -1)
    cv2.circle(frame, (60, 40), 4, 255, -1)

    cfg = {
        "blur_kernel": 1,
        "diff_threshold": 10,
        "morph_open": 1,
        "morph_close": 1,
        "min_area": 10,
        "max_area": 200,
        "roi_x_min": 40.0,
        "roi_x_max": 75.0,
    }

    detections = detect_foreground_blobs(frame, background, cfg)
    assert len(detections) == 1
    assert detections[0].x > 40.0

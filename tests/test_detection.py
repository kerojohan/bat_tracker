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

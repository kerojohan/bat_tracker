from __future__ import annotations

import cv2
import numpy as np

from bat_tracker.detection import build_detection_context, detect_foreground_blobs


def _scene(blob) -> tuple[np.ndarray, np.ndarray]:
    background = np.zeros((80, 80), dtype=np.uint8)
    frame = background.copy()
    blob(frame)
    return frame, background


def _cfg(**overrides) -> dict:
    cfg = {
        "blur_kernel": 1,
        "threshold_mode": "fixed",
        "diff_threshold": 30,
        "otsu_offset": 0,
        "morph_open": 1,
        "morph_close": 1,
        "min_area": 4,
        "max_area": 5000,
        "centroid_mode": "moments",
    }
    cfg.update(overrides)
    return cfg


def test_adaptive_threshold_detects_blob() -> None:
    frame, background = _scene(lambda f: cv2.circle(f, (40, 40), 6, 255, -1))
    cfg = _cfg(threshold_mode="adaptive", adaptive_block_size=25, adaptive_c=-5.0)
    ctx = build_detection_context(background, cfg)

    dets = detect_foreground_blobs(frame, background, cfg, context=ctx)

    assert len(dets) == 1
    assert abs(dets[0].x - 40) < 2.5
    assert abs(dets[0].y - 40) < 2.5


def test_moment_centroid_follows_mass_for_asymmetric_blob() -> None:
    # Triángulo relleno: el centroide de masa NO coincide con el centro del
    # bounding box; el modo moments debe seguir la masa.
    def _triangle(f: np.ndarray) -> None:
        pts = np.array([[20, 20], [60, 20], [20, 60]], dtype=np.int32)
        cv2.fillPoly(f, [pts], 255)

    frame, background = _scene(_triangle)

    ctx_moments = build_detection_context(background, _cfg(centroid_mode="moments"))
    ctx_bbox = build_detection_context(background, _cfg(centroid_mode="bbox"))

    det_m = detect_foreground_blobs(frame, background, _cfg(centroid_mode="moments"), context=ctx_moments)[0]
    det_b = detect_foreground_blobs(frame, background, _cfg(centroid_mode="bbox"), context=ctx_bbox)[0]

    # bbox center ~ (40, 40); centroide de masa del triángulo ~ (33, 33).
    assert abs(det_b.x - 40) < 1.5 and abs(det_b.y - 40) < 1.5
    assert det_m.x < det_b.x - 3
    assert det_m.y < det_b.y - 3

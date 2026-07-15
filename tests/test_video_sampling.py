from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from bat_tracker.pipeline import _same_effective_context
from bat_tracker.video import frame_to_gray, read_gray_frames_at_indices


def test_sequential_index_sampling_returns_exact_decoded_frames(tmp_path: Path) -> None:
    path = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24))
    for frame_idx in range(12):
        image = np.full((24, 32, 3), frame_idx * 17, dtype=np.uint8)
        writer.write(image)
    writer.release()

    expected = {}
    capture = cv2.VideoCapture(str(path))
    frame_idx = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_idx in {0, 3, 7, 11}:
            expected[frame_idx] = frame_to_gray(frame)
        frame_idx += 1
    capture.release()

    sampled = read_gray_frames_at_indices(path, [11, 3, 3, 0, 7])
    assert sampled.keys() == expected.keys()
    assert all(np.array_equal(sampled[index], expected[index]) for index in expected)

    sought = read_gray_frames_at_indices(path, [11, 3, 0, 7], sequential=False)
    assert sought.keys() == expected.keys()
    assert all(np.array_equal(sought[index], expected[index]) for index in expected)

    hybrid = read_gray_frames_at_indices(path, [11, 3, 0, 7], seek_from_index=6)
    assert hybrid.keys() == expected.keys()
    assert all(np.array_equal(hybrid[index], expected[index]) for index in expected)


def test_context_ranges_are_compared_after_clamping_to_video() -> None:
    assert _same_effective_context((0, None), (0, 1874), frame_count=1504)
    assert _same_effective_context((0, 1503), (0, None), frame_count=1504)
    assert not _same_effective_context((0, 1000), (0, None), frame_count=1504)

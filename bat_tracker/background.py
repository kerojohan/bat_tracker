from __future__ import annotations

from typing import List

import cv2
import numpy as np

from .video import VideoMeta


def _sample_indices(frame_count: int, sample_frames: int, uniform: bool) -> np.ndarray:
    if frame_count <= 0:
        return np.array([], dtype=np.int32)

    sample_frames = max(1, min(sample_frames, frame_count))
    if uniform:
        return np.linspace(0, frame_count - 1, sample_frames).astype(np.int32)

    return np.arange(sample_frames, dtype=np.int32)


def compute_background_median(
    video_path: str,
    meta: VideoMeta,
    sample_frames: int,
    uniform_sampling: bool,
) -> np.ndarray:
    indices = _sample_indices(meta.frame_count, sample_frames, uniform_sampling)
    if indices.size == 0:
        raise RuntimeError("Video has zero frames")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for background: {video_path}")

    sampled: List[np.ndarray] = []
    try:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            if frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            sampled.append(gray)
    finally:
        cap.release()

    if not sampled:
        raise RuntimeError("Could not sample frames to compute background")

    stack = np.stack(sampled, axis=0)
    median = np.median(stack, axis=0)
    return median.astype(np.uint8)

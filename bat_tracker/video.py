from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Tuple

import cv2
import numpy as np


@dataclass
class VideoMeta:
    path: Path
    video_id: str
    fps: float
    frame_count: int
    width: int
    height: int


def read_video_meta(path: str | Path) -> VideoMeta:
    video_path = Path(path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0:
        fps = 25.0

    return VideoMeta(
        path=video_path,
        video_id=video_path.stem,
        fps=fps,
        frame_count=max(frame_count, 0),
        width=width,
        height=height,
    )


def iter_gray_frames(path: str | Path) -> Generator[Tuple[int, np.ndarray], None, None]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            yield frame_idx, gray
            frame_idx += 1
    finally:
        cap.release()

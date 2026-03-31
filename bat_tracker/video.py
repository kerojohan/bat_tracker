from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Thread
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


def open_video_capture(path: str | Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    return cap


def frame_to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def read_video_meta(path: str | Path) -> VideoMeta:
    video_path = Path(path)
    cap = open_video_capture(video_path)
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
    cap = open_video_capture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    queue: Queue[tuple[int, np.ndarray] | BaseException | None] = Queue(maxsize=8)

    def _reader() -> None:
        frame_idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                queue.put((frame_idx, frame_to_gray(frame)))
                frame_idx += 1
        except BaseException as exc:
            queue.put(exc)
        finally:
            queue.put(None)

    reader = Thread(target=_reader, name="bat-tracker-frame-reader", daemon=True)
    reader.start()
    try:
        while True:
            item = queue.get()
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        reader.join(timeout=1.0)
        cap.release()

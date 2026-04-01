from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Thread
from time import perf_counter
from typing import Generator, Tuple

import cv2
import numpy as np

from .perf import PerformanceCollector

FRAME_PREFETCH_QUEUE_SIZE = 64


@dataclass
class VideoMeta:
    path: Path
    video_id: str
    fps: float
    frame_count: int
    width: int
    height: int


def open_video_capture(path: str | Path) -> cv2.VideoCapture:
    return cv2.VideoCapture(str(path))


def frame_to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    # Keep the stable BGR -> gray path. Direct backend gray output was faster
    # but changed detections and broke exact functional equivalence.
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


def iter_gray_frames(
    path: str | Path,
    perf: PerformanceCollector | None = None,
) -> Generator[Tuple[int, np.ndarray], None, None]:
    cap = open_video_capture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    # A queue depth of 64 was the best stable point found in end-to-end tests.
    queue: Queue[tuple[int, np.ndarray] | BaseException | None] = Queue(maxsize=FRAME_PREFETCH_QUEUE_SIZE)

    def _reader() -> None:
        frame_idx = 0
        try:
            while True:
                read_started = perf_counter()
                ok, frame = cap.read()
                read_dt = perf_counter() - read_started
                if not ok:
                    break
                if perf is not None:
                    perf.record("video_read", read_dt, frame_idx=frame_idx)
                gray_started = perf_counter()
                gray = frame_to_gray(frame)
                if perf is not None:
                    perf.record("cvt_color", perf_counter() - gray_started, frame_idx=frame_idx)
                queue.put((frame_idx, gray))
                frame_idx += 1
        except BaseException as exc:
            queue.put(exc)
        finally:
            queue.put(None)

    reader = Thread(target=_reader, name="bat-tracker-frame-reader", daemon=True)
    reader.start()
    try:
        while True:
            wait_started = perf_counter()
            item = queue.get()
            wait_dt = perf_counter() - wait_started
            if item is None:
                break
            if perf is not None:
                perf.record(
                    "prefetch_queue_wait",
                    wait_dt,
                    frame_idx=item[0] if not isinstance(item, BaseException) else None,
                )
            if isinstance(item, BaseException):
                raise item
            frame_idx, gray = item
            yield frame_idx, gray
    finally:
        reader.join(timeout=1.0)
        cap.release()

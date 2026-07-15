from __future__ import annotations

from typing import List

import cv2
import numpy as np

from .video import VideoMeta
from .video import read_gray_frames_at_indices


def _sample_indices(
    frame_count: int,
    sample_frames: int,
    uniform: bool,
    *,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> np.ndarray:
    if frame_count <= 0:
        return np.array([], dtype=np.int32)

    start = max(0, min(int(start_frame), frame_count - 1))
    end = frame_count - 1 if end_frame is None else max(start, min(int(end_frame), frame_count - 1))
    available = end - start + 1
    sample_frames = max(1, min(sample_frames, available))
    if uniform:
        return np.linspace(start, end, sample_frames).astype(np.int32)

    return np.arange(start, start + sample_frames, dtype=np.int32)


def compute_background_median(
    video_path: str,
    meta: VideoMeta,
    sample_frames: int,
    uniform_sampling: bool,
    *,
    compute_device: str = "cpu",
    strict_parity: bool = True,
    runtime_stats: dict | None = None,
    context_start_frame: int = 0,
    context_end_frame: int | None = None,
) -> np.ndarray:
    indices = _sample_indices(
        meta.frame_count,
        sample_frames,
        uniform_sampling,
        start_frame=context_start_frame,
        end_frame=context_end_frame,
    )
    if indices.size == 0:
        raise RuntimeError("Video has zero frames")

    constant_rate = abs(float(meta.fps) - round(float(meta.fps))) < 1e-6
    frames = read_gray_frames_at_indices(
        video_path,
        indices,
        sequential=True,
        seek_from_index=None if constant_rate else max(1, int(meta.frame_count) // 2),
    )
    sampled: List[np.ndarray] = [frames[int(idx)] for idx in indices if int(idx) in frames]

    if not sampled:
        raise RuntimeError("Could not sample frames to compute background")

    def _median_cpu() -> np.ndarray:
        stack = np.stack(sampled, axis=0)
        median = np.median(stack, axis=0)
        return median.astype(np.uint8)

    if compute_device != "cuda":
        return _median_cpu()

    try:
        import cupy as cp  # type: ignore
    except Exception:
        if runtime_stats is not None:
            runtime_stats["background_gpu_unavailable"] = runtime_stats.get("background_gpu_unavailable", 0) + 1
        return _median_cpu()

    try:
        stack_gpu = cp.asarray(np.stack(sampled, axis=0))
        median_gpu = cp.median(stack_gpu, axis=0)
        background_gpu = cp.asnumpy(median_gpu).astype(np.uint8)

        if strict_parity:
            background_cpu = _median_cpu()
            if runtime_stats is not None:
                runtime_stats["background_gpu_parity_checked"] = runtime_stats.get(
                    "background_gpu_parity_checked", 0
                ) + 1
            if np.array_equal(background_cpu, background_gpu):
                if runtime_stats is not None:
                    runtime_stats["background_gpu_used"] = runtime_stats.get("background_gpu_used", 0) + 1
                return background_cpu
            if runtime_stats is not None:
                runtime_stats["background_gpu_parity_mismatch"] = runtime_stats.get(
                    "background_gpu_parity_mismatch", 0
                ) + 1
            return background_cpu

        if runtime_stats is not None:
            runtime_stats["background_gpu_used"] = runtime_stats.get("background_gpu_used", 0) + 1
        return background_gpu
    except Exception:
        if runtime_stats is not None:
            runtime_stats["background_gpu_failures"] = runtime_stats.get("background_gpu_failures", 0) + 1
        return _median_cpu()

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class Detection:
    x: float
    y: float
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    area: float


# ---------------------------------------------------------------------------
# CPU path (original, unchanged)
# ---------------------------------------------------------------------------

def _prepare_frame_and_background(
    frame_gray: np.ndarray,
    background: np.ndarray,
    blur_kernel: int,
) -> tuple[np.ndarray, np.ndarray]:
    if blur_kernel > 1 and blur_kernel % 2 == 1:
        frame_proc = cv2.GaussianBlur(frame_gray, (blur_kernel, blur_kernel), 0)
        bg_proc = cv2.GaussianBlur(background, (blur_kernel, blur_kernel), 0)
    else:
        frame_proc = frame_gray
        bg_proc = background
    return frame_proc, bg_proc


def _binary_cpu(
    frame_gray: np.ndarray,
    background: np.ndarray,
    *,
    blur_kernel: int,
    threshold_mode: str,
    diff_threshold: int,
    otsu_offset: int,
    morph_open: int,
    morph_close: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_proc, bg_proc = _prepare_frame_and_background(frame_gray, background, blur_kernel)

    diff = cv2.absdiff(frame_proc, bg_proc)
    if threshold_mode == "otsu":
        otsu_thr, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = max(1, min(255, int(otsu_thr + otsu_offset)))
        _, binary = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
    else:
        _, binary = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)

    if morph_open > 1:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open, morph_open))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)

    if morph_close > 1:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close, morph_close))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)

    return binary, frame_proc, bg_proc


# ---------------------------------------------------------------------------
# GPU path via CuPy
# ---------------------------------------------------------------------------

def _otsu_threshold_from_histogram(hist: np.ndarray) -> int:
    """Compute Otsu threshold from a 256-bin histogram (runs on CPU, O(256))."""
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0

    sum_total = np.dot(np.arange(256, dtype=np.float64), hist)
    sum_bg = 0.0
    weight_bg = 0.0
    max_variance = 0.0
    best_thr = 0

    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if variance > max_variance:
            max_variance = variance
            best_thr = t

    return best_thr


def _binary_cupy(
    frame_gray: np.ndarray,
    background: np.ndarray,
    *,
    blur_kernel: int,
    threshold_mode: str,
    diff_threshold: int,
    otsu_offset: int,
    morph_open: int,
    morph_close: int,
    bg_gpu=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GPU-accelerated binary mask computation using CuPy.

    Strategy: blur and morphology on CPU (OpenCV, SIMD-fast, no CUDA JIT needed),
    absdiff + threshold on GPU (CuPy pre-compiled kernels, no JIT needed).

    Parameters
    ----------
    bg_gpu : cupy.ndarray or None
        Pre-uploaded *blurred* background on GPU. If None, will upload from *background*.
    """
    import cupy as cp  # type: ignore

    # --- Blur on CPU (OpenCV uses SIMD, very fast, avoids CUDA JIT) ---
    frame_proc, bg_proc = _prepare_frame_and_background(frame_gray, background, blur_kernel)

    # --- Upload blurred images to GPU ---
    frame_proc_gpu = cp.asarray(frame_proc)
    if bg_gpu is not None:
        bg_proc_gpu = bg_gpu
    else:
        bg_proc_gpu = cp.asarray(bg_proc)

    # --- Absdiff on GPU (pre-compiled kernel, no JIT) ---
    diff_gpu = cp.abs(
        frame_proc_gpu.astype(cp.int16) - bg_proc_gpu.astype(cp.int16)
    ).astype(cp.uint8)

    # --- Threshold on GPU ---
    if threshold_mode == "otsu":
        # Histogram on GPU, Otsu computation on CPU (O(256), instant)
        hist_gpu = cp.histogram(diff_gpu, bins=256, range=(0, 256))[0]
        hist_cpu = cp.asnumpy(hist_gpu)
        otsu_thr = _otsu_threshold_from_histogram(hist_cpu)
        thr = max(1, min(255, int(otsu_thr + otsu_offset)))
    else:
        thr = diff_threshold
    binary_gpu = cp.where(diff_gpu > thr, cp.uint8(255), cp.uint8(0))

    # --- Download binary mask to CPU ---
    binary = cp.asnumpy(binary_gpu)

    # --- Morphology on CPU (OpenCV, operates on small binary mask) ---
    if morph_open > 1:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open, morph_open))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)

    if morph_close > 1:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close, morph_close))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)

    return binary, frame_proc, bg_proc


# ---------------------------------------------------------------------------
# Contour extraction (always CPU — no good GPU equivalent)
# ---------------------------------------------------------------------------

def _extract_detections_from_binary(
    binary: np.ndarray,
    frame_proc: np.ndarray,
    bg_proc: np.ndarray,
    *,
    min_area: float,
    max_area: float,
    max_global_intensity_shift: float,
    max_foreground_ratio: float,
    max_detections_per_frame: int,
    roi_x_min: float,
    roi_x_max: float,
    roi_y_min: float,
    roi_y_max: float,
    valid_mask: np.ndarray | None,
) -> List[Detection]:
    if max_global_intensity_shift >= 0:
        frame_mean = float(np.mean(frame_proc))
        bg_mean = float(np.mean(bg_proc))
        if abs(frame_mean - bg_mean) > max_global_intensity_shift:
            return []

    if max_foreground_ratio > 0:
        fg_ratio = float(np.count_nonzero(binary)) / float(binary.size)
        if fg_ratio > max_foreground_ratio:
            return []

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections: List[Detection] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cx = float(x + w / 2.0)
        cy = float(y + h / 2.0)
        if valid_mask is not None:
            xi = int(round(cx))
            yi = int(round(cy))
            if (
                yi < 0
                or yi >= valid_mask.shape[0]
                or xi < 0
                or xi >= valid_mask.shape[1]
                or valid_mask[yi, xi] == 0
            ):
                continue
        if roi_x_min >= 0 and cx < roi_x_min:
            continue
        if roi_x_max >= 0 and cx > roi_x_max:
            continue
        if roi_y_min >= 0 and cy < roi_y_min:
            continue
        if roi_y_max >= 0 and cy > roi_y_max:
            continue
        detections.append(
            Detection(
                x=cx,
                y=cy,
                bbox_x1=int(x),
                bbox_y1=int(y),
                bbox_x2=int(x + w),
                bbox_y2=int(y + h),
                area=area,
            )
        )

    if max_detections_per_frame > 0 and len(detections) > max_detections_per_frame:
        return []

    return detections


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_foreground_blobs(
    frame_gray: np.ndarray,
    background: np.ndarray,
    cfg: dict,
    valid_mask: np.ndarray | None = None,
    *,
    compute_device: str = "cpu",
    strict_parity: bool = False,
    runtime_stats: dict | None = None,
    bg_gpu=None,
) -> List[Detection]:
    blur_kernel = int(cfg.get("blur_kernel", 5))
    threshold_mode = str(cfg.get("threshold_mode", "fixed")).lower()
    diff_threshold = int(cfg.get("diff_threshold", 25))
    otsu_offset = int(cfg.get("otsu_offset", 0))
    morph_open = int(cfg.get("morph_open", 3))
    morph_close = int(cfg.get("morph_close", 5))
    min_area = float(cfg.get("min_area", 10))
    max_area = float(cfg.get("max_area", 5000))
    max_global_intensity_shift = float(cfg.get("max_global_intensity_shift", -1))
    max_foreground_ratio = float(cfg.get("max_foreground_ratio", -1))
    max_detections_per_frame = int(cfg.get("max_detections_per_frame", 0))
    roi_x_min = float(cfg.get("roi_x_min", -1))
    roi_x_max = float(cfg.get("roi_x_max", -1))
    roi_y_min = float(cfg.get("roi_y_min", -1))
    roi_y_max = float(cfg.get("roi_y_max", -1))

    binary: np.ndarray
    frame_proc: np.ndarray
    bg_proc: np.ndarray

    if compute_device == "cuda":
        try:
            binary_gpu, frame_proc_gpu, bg_proc_gpu = _binary_cupy(
                frame_gray,
                background,
                blur_kernel=blur_kernel,
                threshold_mode=threshold_mode,
                diff_threshold=diff_threshold,
                otsu_offset=otsu_offset,
                morph_open=morph_open,
                morph_close=morph_close,
                bg_gpu=bg_gpu,
            )

            if strict_parity:
                binary_cpu, frame_proc, bg_proc = _binary_cpu(
                    frame_gray,
                    background,
                    blur_kernel=blur_kernel,
                    threshold_mode=threshold_mode,
                    diff_threshold=diff_threshold,
                    otsu_offset=otsu_offset,
                    morph_open=morph_open,
                    morph_close=morph_close,
                )
                if runtime_stats is not None:
                    runtime_stats["cuda_parity_checked_frames"] = runtime_stats.get("cuda_parity_checked_frames", 0) + 1
                if np.array_equal(binary_cpu, binary_gpu):
                    binary = binary_cpu
                else:
                    if runtime_stats is not None:
                        runtime_stats["cuda_parity_mismatch_frames"] = runtime_stats.get("cuda_parity_mismatch_frames", 0) + 1
                    binary = binary_cpu
            else:
                binary = binary_gpu
                frame_proc = frame_proc_gpu
                bg_proc = bg_proc_gpu
                if runtime_stats is not None:
                    runtime_stats["cuda_frames_used"] = runtime_stats.get("cuda_frames_used", 0) + 1
        except Exception as exc:
            if runtime_stats is not None:
                runtime_stats["cuda_runtime_failures"] = runtime_stats.get("cuda_runtime_failures", 0) + 1
            # Log only first failure to avoid flooding
            if runtime_stats is not None and runtime_stats.get("cuda_runtime_failures", 0) <= 1:
                print(f"[detection] GPU fallback to CPU: {exc}", file=sys.stderr, flush=True)
            binary, frame_proc, bg_proc = _binary_cpu(
                frame_gray,
                background,
                blur_kernel=blur_kernel,
                threshold_mode=threshold_mode,
                diff_threshold=diff_threshold,
                otsu_offset=otsu_offset,
                morph_open=morph_open,
                morph_close=morph_close,
            )
    else:
        binary, frame_proc, bg_proc = _binary_cpu(
            frame_gray,
            background,
            blur_kernel=blur_kernel,
            threshold_mode=threshold_mode,
            diff_threshold=diff_threshold,
            otsu_offset=otsu_offset,
            morph_open=morph_open,
            morph_close=morph_close,
        )

    return _extract_detections_from_binary(
        binary,
        frame_proc,
        bg_proc,
        min_area=min_area,
        max_area=max_area,
        max_global_intensity_shift=max_global_intensity_shift,
        max_foreground_ratio=max_foreground_ratio,
        max_detections_per_frame=max_detections_per_frame,
        roi_x_min=roi_x_min,
        roi_x_max=roi_x_max,
        roi_y_min=roi_y_min,
        roi_y_max=roi_y_max,
        valid_mask=valid_mask,
    )

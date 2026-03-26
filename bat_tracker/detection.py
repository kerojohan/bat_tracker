from __future__ import annotations

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


def _binary_cuda(
    frame_gray: np.ndarray,
    background: np.ndarray,
    *,
    blur_kernel: int,
    threshold_mode: str,
    diff_threshold: int,
    otsu_offset: int,
    morph_open: int,
    morph_close: int,
) -> np.ndarray:
    gpu_frame = cv2.cuda_GpuMat()
    gpu_bg = cv2.cuda_GpuMat()
    gpu_frame.upload(frame_gray)
    gpu_bg.upload(background)

    gpu_frame_proc = gpu_frame
    gpu_bg_proc = gpu_bg

    if blur_kernel > 1 and blur_kernel % 2 == 1:
        blur = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (blur_kernel, blur_kernel), 0)
        gpu_frame_proc = blur.apply(gpu_frame)
        gpu_bg_proc = blur.apply(gpu_bg)

    gpu_diff = cv2.cuda.absdiff(gpu_frame_proc, gpu_bg_proc)

    if threshold_mode == "otsu":
        otsu_thr, _ = cv2.cuda.threshold(gpu_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = max(1, min(255, int(otsu_thr + otsu_offset)))
        _, gpu_binary = cv2.cuda.threshold(gpu_diff, thr, 255, cv2.THRESH_BINARY)
    else:
        _, gpu_binary = cv2.cuda.threshold(gpu_diff, diff_threshold, 255, cv2.THRESH_BINARY)

    if morph_open > 1:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open, morph_open))
        morph_open_op = cv2.cuda.createMorphologyFilter(cv2.MORPH_OPEN, cv2.CV_8UC1, k_open)
        gpu_binary = morph_open_op.apply(gpu_binary)

    if morph_close > 1:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close, morph_close))
        morph_close_op = cv2.cuda.createMorphologyFilter(cv2.MORPH_CLOSE, cv2.CV_8UC1, k_close)
        gpu_binary = morph_close_op.apply(gpu_binary)

    return gpu_binary.download()


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


def detect_foreground_blobs(
    frame_gray: np.ndarray,
    background: np.ndarray,
    cfg: dict,
    valid_mask: np.ndarray | None = None,
    *,
    compute_device: str = "cpu",
    strict_parity: bool = True,
    runtime_stats: dict | None = None,
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
            binary_cuda = _binary_cuda(
                frame_gray,
                background,
                blur_kernel=blur_kernel,
                threshold_mode=threshold_mode,
                diff_threshold=diff_threshold,
                otsu_offset=otsu_offset,
                morph_open=morph_open,
                morph_close=morph_close,
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
                if np.array_equal(binary_cpu, binary_cuda):
                    binary = binary_cpu
                else:
                    if runtime_stats is not None:
                        runtime_stats["cuda_parity_mismatch_frames"] = runtime_stats.get("cuda_parity_mismatch_frames", 0) + 1
                    binary = binary_cpu
            else:
                binary = binary_cuda
                frame_proc, bg_proc = _prepare_frame_and_background(frame_gray, background, blur_kernel)
                if runtime_stats is not None:
                    runtime_stats["cuda_frames_used"] = runtime_stats.get("cuda_frames_used", 0) + 1
        except Exception:
            if runtime_stats is not None:
                runtime_stats["cuda_runtime_failures"] = runtime_stats.get("cuda_runtime_failures", 0) + 1
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

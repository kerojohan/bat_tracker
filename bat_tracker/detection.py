from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import List

import cv2
import numpy as np

from .perf import PerformanceCollector


@dataclass
class Detection:
    x: float
    y: float
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    area: float


@dataclass(frozen=True)
class DetectionContext:
    background: np.ndarray
    bg_proc: np.ndarray
    bg_proc_view: np.ndarray
    frame_blur_buf: np.ndarray | None
    diff_buf: np.ndarray
    blur_kernel_1d: np.ndarray | None
    blur_kernel: int
    threshold_mode: str
    diff_threshold: int
    otsu_offset: int
    morph_open: int
    morph_close: int
    min_area: float
    max_area: float
    max_global_intensity_shift: float
    max_foreground_ratio: float
    max_detections_per_frame: int
    roi_x_min: float
    roi_x_max: float
    roi_y_min: float
    roi_y_max: float
    k_open: np.ndarray | None
    k_close: np.ndarray | None
    process_x0: int
    process_x1: int
    process_y0: int
    process_y1: int
    process_area: int


def build_detection_context(background: np.ndarray, cfg: dict) -> DetectionContext:
    height, width = background.shape[:2]
    blur_kernel = int(cfg.get("blur_kernel", 5))
    threshold_mode = str(cfg.get("threshold_mode", "fixed")).lower()
    diff_threshold = int(cfg.get("diff_threshold", 25))
    otsu_offset = int(cfg.get("otsu_offset", 0))
    morph_open = int(cfg.get("morph_open", 3))
    morph_close = int(cfg.get("morph_close", 5))
    roi_x_min = float(cfg.get("roi_x_min", -1))
    roi_x_max = float(cfg.get("roi_x_max", -1))
    roi_y_min = float(cfg.get("roi_y_min", -1))
    roi_y_max = float(cfg.get("roi_y_max", -1))

    pad = max(16, blur_kernel + morph_open + morph_close)
    process_x0 = 0
    process_x1 = width
    process_y0 = 0
    process_y1 = height
    if roi_x_min >= 0:
        process_x0 = max(0, int(np.floor(roi_x_min)) - pad)
    if roi_x_max >= 0:
        process_x1 = min(width, int(np.ceil(roi_x_max)) + pad + 1)
    if roi_y_min >= 0:
        process_y0 = max(0, int(np.floor(roi_y_min)) - pad)
    if roi_y_max >= 0:
        process_y1 = min(height, int(np.ceil(roi_y_max)) + pad + 1)

    if blur_kernel > 1 and blur_kernel % 2 == 1:
        bg_proc = cv2.GaussianBlur(background, (blur_kernel, blur_kernel), 0)
    else:
        bg_proc = background

    bg_proc_view = bg_proc[process_y0:process_y1, process_x0:process_x1]
    frame_blur_buf = None
    blur_kernel_1d = None
    if blur_kernel > 1 and blur_kernel % 2 == 1:
        frame_blur_buf = np.empty_like(bg_proc_view)
        blur_kernel_1d = cv2.getGaussianKernel(blur_kernel, 0)
    diff_buf = np.empty_like(bg_proc_view)

    k_open = None
    if morph_open > 1:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open, morph_open))

    k_close = None
    if morph_close > 1:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close, morph_close))

    return DetectionContext(
        background=background,
        bg_proc=bg_proc,
        bg_proc_view=bg_proc_view,
        frame_blur_buf=frame_blur_buf,
        diff_buf=diff_buf,
        blur_kernel_1d=blur_kernel_1d,
        blur_kernel=blur_kernel,
        threshold_mode=threshold_mode,
        diff_threshold=diff_threshold,
        otsu_offset=otsu_offset,
        morph_open=morph_open,
        morph_close=morph_close,
        min_area=float(cfg.get("min_area", 10)),
        max_area=float(cfg.get("max_area", 5000)),
        max_global_intensity_shift=float(cfg.get("max_global_intensity_shift", -1)),
        max_foreground_ratio=float(cfg.get("max_foreground_ratio", -1)),
        max_detections_per_frame=int(cfg.get("max_detections_per_frame", 0)),
        roi_x_min=roi_x_min,
        roi_x_max=roi_x_max,
        roi_y_min=roi_y_min,
        roi_y_max=roi_y_max,
        k_open=k_open,
        k_close=k_close,
        process_x0=process_x0,
        process_x1=process_x1,
        process_y0=process_y0,
        process_y1=process_y1,
        process_area=max(1, (process_x1 - process_x0) * (process_y1 - process_y0)),
    )

def _prepare_frame(
    frame_gray: np.ndarray,
    blur_kernel: int,
    blur_kernel_1d: np.ndarray | None = None,
    out: np.ndarray | None = None,
    perf: PerformanceCollector | None = None,
    frame_idx: int | None = None,
) -> np.ndarray:
    if blur_kernel > 1 and blur_kernel % 2 == 1:
        started = perf_counter()
        if blur_kernel_1d is not None:
            out = cv2.sepFilter2D(frame_gray, -1, blur_kernel_1d, blur_kernel_1d, dst=out)
        else:
            out = cv2.GaussianBlur(frame_gray, (blur_kernel, blur_kernel), 0, dst=out)
        if perf is not None:
            perf.record("gaussian_blur", perf_counter() - started, frame_idx=frame_idx)
        return out
    return frame_gray


def _prepare_frame_and_background(
    frame_gray: np.ndarray,
    background: np.ndarray,
    blur_kernel: int,
) -> tuple[np.ndarray, np.ndarray]:
    frame_proc = _prepare_frame(frame_gray, blur_kernel)
    if blur_kernel > 1 and blur_kernel % 2 == 1:
        bg_proc = cv2.GaussianBlur(background, (blur_kernel, blur_kernel), 0)
    else:
        bg_proc = background
    return frame_proc, bg_proc


def _binary_cpu(
    frame_gray: np.ndarray,
    ctx: DetectionContext,
    perf: PerformanceCollector | None = None,
    frame_idx: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    blur_kernel = ctx.blur_kernel
    threshold_mode = ctx.threshold_mode
    diff_threshold = ctx.diff_threshold
    otsu_offset = ctx.otsu_offset
    k_open = ctx.k_open
    k_close = ctx.k_close
    process_x0 = ctx.process_x0
    process_x1 = ctx.process_x1
    process_y0 = ctx.process_y0
    process_y1 = ctx.process_y1

    frame_view = frame_gray[process_y0:process_y1, process_x0:process_x1]
    frame_proc = _prepare_frame(
        frame_view,
        blur_kernel,
        blur_kernel_1d=ctx.blur_kernel_1d,
        out=ctx.frame_blur_buf,
        perf=perf,
        frame_idx=frame_idx,
    )
    bg_proc = ctx.bg_proc_view

    absdiff_started = perf_counter()
    diff = cv2.absdiff(frame_proc, bg_proc, dst=ctx.diff_buf)
    if perf is not None:
        perf.record("background_absdiff", perf_counter() - absdiff_started, frame_idx=frame_idx)

    threshold_started = perf_counter()
    if threshold_mode == "otsu":
        otsu_thr, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = max(1, min(255, int(otsu_thr + otsu_offset)))
        _, binary = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
    else:
        _, binary = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY, dst=ctx.diff_buf)
    if perf is not None:
        perf.record("threshold", perf_counter() - threshold_started, frame_idx=frame_idx)

    fg_count = int(cv2.countNonZero(binary))
    if fg_count == 0:
        return binary, frame_proc, bg_proc, 0

    morph_total = 0.0
    morph_exec = 0
    if k_open is not None:
        morph_started = perf_counter()
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)
        morph_total += perf_counter() - morph_started
        morph_exec += 1

    if k_close is not None:
        morph_started = perf_counter()
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)
        morph_total += perf_counter() - morph_started
        morph_exec += 1
    if perf is not None and morph_exec > 0:
        perf.record("morphology", morph_total, frame_idx=frame_idx, executions=morph_exec)

    fg_count = int(cv2.countNonZero(binary))
    return binary, frame_proc, bg_proc, fg_count


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
    ctx: DetectionContext,
    valid_mask: np.ndarray | None,
    fg_count: int | None = None,
    perf: PerformanceCollector | None = None,
    frame_idx: int | None = None,
) -> List[Detection]:
    max_global_intensity_shift = ctx.max_global_intensity_shift
    max_foreground_ratio = ctx.max_foreground_ratio
    min_area = ctx.min_area
    max_area = ctx.max_area
    roi_x_min = ctx.roi_x_min
    roi_x_max = ctx.roi_x_max
    roi_y_min = ctx.roi_y_min
    roi_y_max = ctx.roi_y_max
    max_detections_per_frame = ctx.max_detections_per_frame
    process_x0 = ctx.process_x0
    process_y0 = ctx.process_y0

    if max_global_intensity_shift >= 0:
        frame_mean = float(np.mean(frame_proc))
        bg_mean = float(np.mean(bg_proc))
        if abs(frame_mean - bg_mean) > max_global_intensity_shift:
            return []

    if fg_count is None:
        fg_count = int(cv2.countNonZero(binary))
    if fg_count == 0:
        return []

    if max_foreground_ratio > 0:
        fg_ratio = float(fg_count) / float(ctx.process_area)
        if fg_ratio > max_foreground_ratio:
            return []

    contours_started = perf_counter()
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if perf is not None:
        perf.record("find_contours", perf_counter() - contours_started, frame_idx=frame_idx)

    detections: List[Detection] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        x += process_x0
        y += process_y0
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
    context: DetectionContext | None = None,
    frame_idx: int | None = None,
    perf: PerformanceCollector | None = None,
) -> List[Detection]:
    ctx = context if context is not None else build_detection_context(background, cfg)
    bg = ctx.background
    blur_kernel = ctx.blur_kernel
    threshold_mode = ctx.threshold_mode
    diff_threshold = ctx.diff_threshold
    otsu_offset = ctx.otsu_offset
    morph_open = ctx.morph_open
    morph_close = ctx.morph_close

    binary: np.ndarray
    frame_proc: np.ndarray
    bg_proc: np.ndarray
    fg_count: int | None = None

    if compute_device == "cuda":
        try:
            binary_cuda = _binary_cuda(
                frame_gray,
                bg,
                blur_kernel=blur_kernel,
                threshold_mode=threshold_mode,
                diff_threshold=diff_threshold,
                otsu_offset=otsu_offset,
                morph_open=morph_open,
                morph_close=morph_close,
            )

            if strict_parity:
                binary_cpu, frame_proc, bg_proc, fg_count = _binary_cpu(
                    frame_gray,
                    ctx,
                    perf=perf,
                    frame_idx=frame_idx,
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
                frame_view = frame_gray[ctx.process_y0:ctx.process_y1, ctx.process_x0:ctx.process_x1]
                frame_proc = _prepare_frame(
                    frame_view,
                    blur_kernel,
                    blur_kernel_1d=ctx.blur_kernel_1d,
                    perf=perf,
                    frame_idx=frame_idx,
                )
                bg_proc = ctx.bg_proc_view
                if runtime_stats is not None:
                    runtime_stats["cuda_frames_used"] = runtime_stats.get("cuda_frames_used", 0) + 1
        except Exception:
            if runtime_stats is not None:
                runtime_stats["cuda_runtime_failures"] = runtime_stats.get("cuda_runtime_failures", 0) + 1
            binary, frame_proc, bg_proc, fg_count = _binary_cpu(
                frame_gray,
                ctx,
                perf=perf,
                frame_idx=frame_idx,
            )
    else:
        binary, frame_proc, bg_proc, fg_count = _binary_cpu(
            frame_gray,
            ctx,
            perf=perf,
            frame_idx=frame_idx,
        )

    return _extract_detections_from_binary(
        binary,
        frame_proc,
        bg_proc,
        ctx=ctx,
        valid_mask=valid_mask,
        fg_count=fg_count,
        perf=perf,
        frame_idx=frame_idx,
    )

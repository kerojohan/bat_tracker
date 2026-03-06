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


def detect_foreground_blobs(
    frame_gray: np.ndarray,
    background: np.ndarray,
    cfg: dict,
    valid_mask: np.ndarray | None = None,
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

    if blur_kernel > 1 and blur_kernel % 2 == 1:
        frame_proc = cv2.GaussianBlur(frame_gray, (blur_kernel, blur_kernel), 0)
        bg_proc = cv2.GaussianBlur(background, (blur_kernel, blur_kernel), 0)
    else:
        frame_proc = frame_gray
        bg_proc = background

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

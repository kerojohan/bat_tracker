from __future__ import annotations

import cv2
import numpy as np


def enhance_motion(
    gray: np.ndarray,
    background: np.ndarray,
    diff_threshold: int = 25,
    morph_open: int = 3,
    morph_close: int = 5,
    merge_distance: int = 0,
) -> np.ndarray:
    diff = cv2.absdiff(gray, background)
    _, binary = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)

    if morph_open > 0:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open, morph_open))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)
    if morph_close > 0:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close, morph_close))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)

    if merge_distance > 0:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        merged_mask = np.zeros_like(binary)
        merged = set()
        for i in range(1, num_labels):
            if i in merged:
                continue
            x1 = stats[i, cv2.CC_STAT_LEFT]
            y1 = stats[i, cv2.CC_STAT_TOP]
            x2 = x1 + stats[i, cv2.CC_STAT_WIDTH]
            y2 = y1 + stats[i, cv2.CC_STAT_HEIGHT]
            for j in range(i + 1, num_labels):
                if j in merged:
                    continue
                jx1 = stats[j, cv2.CC_STAT_LEFT]
                jy1 = stats[j, cv2.CC_STAT_TOP]
                jx2 = jx1 + stats[j, cv2.CC_STAT_WIDTH]
                jy2 = jy1 + stats[j, cv2.CC_STAT_HEIGHT]
                gap_x = max(0, max(x1, jx1) - min(x2, jx2))
                gap_y = max(0, max(y1, jy1) - min(y2, jy2))
                if gap_x <= merge_distance and gap_y <= merge_distance:
                    x1 = min(x1, jx1)
                    y1 = min(y1, jy1)
                    x2 = max(x2, jx2)
                    y2 = max(y2, jy2)
                    merged.add(j)
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            merged_mask[y1:y1 + h, x1:x1 + w] = 255
        binary = merged_mask

    enhanced = gray.copy()
    enhanced[binary == 0] = background[binary == 0]
    return enhanced


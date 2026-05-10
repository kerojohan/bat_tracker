from __future__ import annotations

import cv2
import numpy as np


def enhance_motion(
    gray: np.ndarray,
    background: np.ndarray,
    diff_threshold: int = 25,
    morph_open: int = 3,
    morph_close: int = 5,
) -> np.ndarray:
    diff = cv2.absdiff(gray, background)
    _, binary = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)

    if morph_open > 0:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open, morph_open))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)
    if morph_close > 0:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close, morph_close))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)

    enhanced = gray.copy()
    enhanced[binary == 0] = background[binary == 0]
    return enhanced


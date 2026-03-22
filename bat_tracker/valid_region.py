from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


def _ensure_odd(value: int, field_name: str) -> int:
    ivalue = int(value)
    if ivalue < 1 or ivalue % 2 == 0:
        raise ValueError(f"{field_name} must be a positive odd integer, got: {value}")
    return ivalue


def load_image(path: str | Path) -> np.ndarray:
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not read grayscale image: {image_path}")
    return image


def estimate_illumination(image: np.ndarray, blur_kernel_size: int) -> np.ndarray:
    k = _ensure_odd(blur_kernel_size, "blur_kernel_size")
    return cv2.GaussianBlur(image, (k, k), 0)


def horizontal_profile(
    illumination_image: np.ndarray, profile_smooth_window: int
) -> Tuple[np.ndarray, np.ndarray]:
    window = _ensure_odd(profile_smooth_window, "profile_smooth_window")
    raw_profile = illumination_image.mean(axis=0).astype(np.float32)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    pad = window // 2
    padded = np.pad(raw_profile, (pad, pad), mode="edge")
    smoothed_profile = np.convolve(padded, kernel, mode="valid")
    return raw_profile, smoothed_profile


def _fallback_centered_region(width: int, span_ratio: float = 0.70) -> Tuple[int, int]:
    span = max(2, int(round(width * span_ratio)))
    x_start = max(0, (width - span) // 2)
    x_end = min(width - 1, x_start + span - 1)
    if x_end <= x_start:
        x_end = min(width - 1, x_start + 1)
    return x_start, x_end


def detect_valid_region(
    smoothed_profile: np.ndarray,
    threshold_ratio: float,
    safety_margin: int,
    min_region_width_ratio: float,
) -> Tuple[int, int]:
    width = int(smoothed_profile.shape[0])
    if width < 2:
        return 0, 0

    threshold_ratio = float(max(0.0, min(1.0, threshold_ratio)))
    safety_margin = max(0, int(safety_margin))
    min_region_width = max(2, int(round(width * float(min_region_width_ratio))))

    pmax = float(np.max(smoothed_profile))
    pmin = float(np.min(smoothed_profile))
    contrast = pmax - pmin
    contrast_ratio = contrast / max(1.0, abs(pmax))

    # In fixed IR cave scenes, cave interiors can be truly dark.
    # The invalid area is usually side vignette from IR falloff, so we only
    # analyze left-to-right illumination trend and keep a vertical band.
    # If profile contrast is too low, avoid overfitting and use centered fallback.
    if contrast < 1e-6 or contrast_ratio < 0.08:
        detect_valid_region.last_threshold = pmax  # type: ignore[attr-defined]
        return _fallback_centered_region(width)

    threshold = threshold_ratio * pmax
    detect_valid_region.last_threshold = float(threshold)  # type: ignore[attr-defined]
    above = smoothed_profile >= threshold

    segments: list[Tuple[int, int]] = []
    in_segment = False
    start = 0
    for i, valid in enumerate(above):
        if valid and not in_segment:
            in_segment = True
            start = i
        elif not valid and in_segment:
            segments.append((start, i - 1))
            in_segment = False
    if in_segment:
        segments.append((start, width - 1))

    if not segments:
        return _fallback_centered_region(width)

    wide_segments = [(s, e) for s, e in segments if (e - s + 1) >= min_region_width]
    candidates = wide_segments if wide_segments else segments
    x_start, x_end = max(candidates, key=lambda t: t[1] - t[0] + 1)

    x_start += safety_margin
    x_end -= safety_margin
    x_start = max(0, min(width - 2, x_start))
    x_end = max(1, min(width - 1, x_end))

    if x_end <= x_start:
        return _fallback_centered_region(width)

    if (x_end - x_start + 1) < min_region_width:
        return _fallback_centered_region(width)

    return x_start, x_end


def build_mask(image_shape: Tuple[int, ...], x_start: int, x_end: int) -> np.ndarray:
    height, width = int(image_shape[0]), int(image_shape[1])
    x_start = int(max(0, min(width - 2, x_start)))
    x_end = int(max(1, min(width - 1, x_end)))
    if x_end <= x_start:
        raise ValueError(f"Invalid region bounds: x_start={x_start}, x_end={x_end}")
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[:, x_start : x_end + 1] = 255
    return mask


def save_debug_outputs(
    original_image: np.ndarray,
    mask: np.ndarray,
    x_start: int,
    x_end: int,
    raw_profile: np.ndarray,
    smoothed_profile: np.ndarray,
    output_dir: str | Path,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_path = out_dir / "mask.png"
    overlay_path = out_dir / "overlay.png"
    profile_path = out_dir / "profile.png"

    cv2.imwrite(str(mask_path), mask)

    overlay = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    tint = np.zeros_like(overlay)
    tint[:, x_start : x_end + 1] = (0, 255, 0)
    overlay = cv2.addWeighted(overlay, 1.0, tint, 0.22, 0)
    cv2.line(overlay, (x_start, 0), (x_start, overlay.shape[0] - 1), (0, 255, 255), 2)
    cv2.line(overlay, (x_end, 0), (x_end, overlay.shape[0] - 1), (0, 255, 255), 2)
    cv2.imwrite(str(overlay_path), overlay)

    threshold = float(
        getattr(detect_valid_region, "last_threshold", np.min(smoothed_profile[x_start : x_end + 1]))
    )
    xs = np.arange(raw_profile.shape[0])
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(11, 4))
        plt.plot(xs, raw_profile, label="raw_profile", linewidth=1.1)
        plt.plot(xs, smoothed_profile, label="smoothed_profile", linewidth=1.8)
        plt.axhline(threshold, linestyle="--", linewidth=1.0, label="threshold")
        plt.axvline(x_start, color="green", linestyle="--", linewidth=1.3, label="x_start")
        plt.axvline(x_end, color="red", linestyle="--", linewidth=1.3, label="x_end")
        plt.xlabel("x (column)")
        plt.ylabel("intensity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(profile_path)
        plt.close()
    except ModuleNotFoundError:
        # Keep pipeline usable in restricted environments without matplotlib.
        canvas = np.zeros((320, max(320, raw_profile.shape[0]), 3), dtype=np.uint8)
        norm = float(max(1e-6, np.max(raw_profile)))
        raw_y = ((1.0 - (raw_profile / norm)) * 280 + 20).astype(np.int32)
        smooth_y = ((1.0 - (smoothed_profile / norm)) * 280 + 20).astype(np.int32)
        for i in range(1, len(xs)):
            cv2.line(canvas, (i - 1, raw_y[i - 1]), (i, raw_y[i]), (255, 150, 150), 1)
            cv2.line(canvas, (i - 1, smooth_y[i - 1]), (i, smooth_y[i]), (80, 255, 80), 1)
        cv2.line(canvas, (x_start, 0), (x_start, canvas.shape[0] - 1), (0, 255, 255), 1)
        cv2.line(canvas, (x_end, 0), (x_end, canvas.shape[0] - 1), (0, 255, 255), 1)
        cv2.imwrite(str(profile_path), canvas)


def _save_depth_debug_outputs(
    original_image: np.ndarray,
    depth_map: np.ndarray,
    mask: np.ndarray,
    output_dir: str | Path,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_path = out_dir / "mask.png"
    overlay_path = out_dir / "overlay.png"
    profile_path = out_dir / "profile.png"

    cv2.imwrite(str(mask_path), mask)

    overlay = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    tint = np.zeros_like(overlay)
    tint[mask > 0] = (0, 255, 0)
    overlay = cv2.addWeighted(overlay, 0.78, tint, 0.55, 0)

    # Draw explicit contour so the valid region is visible even when tint contrast is low.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2, lineType=cv2.LINE_AA)
    cv2.imwrite(str(overlay_path), overlay)

    depth_u8 = np.clip(depth_map, 0, 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    cv2.imwrite(str(profile_path), depth_vis)


def _bbox_from_mask(mask: np.ndarray) -> Tuple[int, int]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return 0, max(1, mask.shape[1] - 1)
    return int(xs.min()), int(xs.max())


def _central_deep_layer_mask(
    image: np.ndarray,
    config: Dict,
    constraint_mask: np.ndarray | None = None,
) -> np.ndarray:
    illumination = estimate_illumination(image, int(config.get("blur_kernel_size", 151)))
    depth_map = 255.0 - illumination.astype(np.float32)
    percentile = float(config.get("depth_percentile", 85.0))
    percentile = float(max(50.0, min(99.9, percentile)))
    thr = float(np.percentile(depth_map, percentile))
    binary = np.where(depth_map >= thr, 255, 0).astype(np.uint8)
    if constraint_mask is not None:
        binary = np.where((binary > 0) & (constraint_mask > 0), 255, 0).astype(np.uint8)

    morph_k = _ensure_odd(int(config.get("depth_morph_kernel", 9)), "depth_morph_kernel")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary

    h, w = binary.shape[:2]
    cx = w // 2
    cy = h // 2
    center_label = int(labels[cy, cx])
    min_area_ratio = float(max(0.0, min(1.0, config.get("depth_min_area_ratio", 0.02))))
    min_area = int(min_area_ratio * h * w)

    selected_label = 0
    if center_label > 0 and int(stats[center_label, cv2.CC_STAT_AREA]) >= min_area:
        selected_label = center_label
    else:
        best_score = float("inf")
        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < max(1, min_area):
                continue
            x = int(stats[label_id, cv2.CC_STAT_LEFT])
            ww = int(stats[label_id, cv2.CC_STAT_WIDTH])
            comp_cx = x + ww / 2.0
            score = abs(comp_cx - cx)
            if score < best_score:
                best_score = score
                selected_label = label_id

    if selected_label <= 0:
        selected_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

    mask = np.where(labels == selected_label, 255, 0).astype(np.uint8)
    mask = _expand_mask_by_depth_layers(mask, depth_map, config, constraint_mask=constraint_mask)
    return mask


def _as_float_list(value: object) -> list[float]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[float] = []
    for item in value:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            continue
    return out


def _as_int_list(value: object) -> list[int]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[int] = []
    for item in value:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def _expand_mask_by_depth_layers(
    mask: np.ndarray,
    depth_map: np.ndarray,
    config: Dict,
    constraint_mask: np.ndarray | None = None,
) -> np.ndarray:
    percentiles = _as_float_list(config.get("depth_layer_percentiles", []))
    dilate_px = _as_int_list(config.get("depth_layer_dilate_px", []))
    if not percentiles or not dilate_px:
        return mask

    n = min(len(percentiles), len(dilate_px))
    out_mask = mask.copy()
    for percentile, px in zip(percentiles[:n], dilate_px[:n]):
        p = float(max(0.0, min(99.9, percentile)))
        radius = max(0, int(px))
        thr = float(np.percentile(depth_map, p))
        depth_layer = np.where(depth_map >= thr, 255, 0).astype(np.uint8)
        if constraint_mask is not None:
            depth_layer = np.where((depth_layer > 0) & (constraint_mask > 0), 255, 0).astype(np.uint8)

        if radius > 0:
            k = 2 * radius + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            expanded = cv2.dilate(out_mask, kernel, iterations=1)
        else:
            expanded = out_mask

        if constraint_mask is not None:
            expanded = np.where((expanded > 0) & (constraint_mask > 0), 255, 0).astype(np.uint8)
        keep = np.logical_and(expanded > 0, depth_layer > 0)
        out_mask[keep] = 255

    return out_mask


def _moving_average_1d(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values
    w = max(1, int(window))
    if w % 2 == 0:
        w += 1
    if w <= 1:
        return values.astype(np.float32)
    pad = w // 2
    padded = np.pad(values.astype(np.float32), (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=np.float32) / float(w)
    return np.convolve(padded, kernel, mode="valid")


def _regularized_bottom_path(
    abs_grad_y: np.ndarray,
    xs: list[int],
    tops: list[int],
    bottoms: list[int],
    search_up: int,
    search_down: int,
    config: Dict,
) -> np.ndarray:
    n = len(xs)
    if n == 0:
        return np.zeros((0,), dtype=np.int32)

    offsets = np.arange(-search_up, search_down + 1, dtype=np.int32)
    m = int(offsets.size)
    if m <= 1:
        return np.asarray(bottoms, dtype=np.int32)

    h = abs_grad_y.shape[0]
    scores = np.full((n, m), -1e9, dtype=np.float32)
    valid = np.zeros((n, m), dtype=bool)

    for i, x in enumerate(xs):
        top = int(tops[i])
        bottom = int(bottoms[i])
        y_raw = bottom + offsets
        ok = np.logical_and(y_raw >= (top + 1), y_raw <= (h - 1))
        if not np.any(ok):
            continue
        yy = y_raw[ok]
        vals = abs_grad_y[yy, x].astype(np.float32)
        scores[i, ok] = vals
        valid[i, ok] = True

    if not np.any(valid):
        return np.asarray(bottoms, dtype=np.int32)

    grad_vals = scores[valid]
    grad_scale = float(np.percentile(grad_vals, 90)) if grad_vals.size else 1.0
    grad_scale = max(1e-6, grad_scale)
    scores = scores / grad_scale

    # Soft preference for slightly lower boundaries when gradient evidence is similar.
    down_bias = float(config.get("bottom_contour_downward_bias", 0.10))
    scores = scores + down_bias * (offsets[np.newaxis, :] / float(max(1, search_down)))

    smooth_penalty = float(config.get("bottom_contour_regularization", 0.90))
    max_step = max(1, int(config.get("bottom_contour_max_step_px", 10)))

    dp = np.full((n, m), -1e9, dtype=np.float32)
    back = np.full((n, m), -1, dtype=np.int32)

    dp[0, valid[0]] = scores[0, valid[0]]
    for i in range(1, n):
        valid_j = np.where(valid[i])[0]
        if valid_j.size == 0:
            continue
        prev_valid = np.where(valid[i - 1])[0]
        if prev_valid.size == 0:
            dp[i, valid_j] = scores[i, valid_j]
            continue

        for j in valid_j:
            k0 = max(int(prev_valid.min()), j - max_step)
            k1 = min(int(prev_valid.max()), j + max_step)
            k_idx = prev_valid[(prev_valid >= k0) & (prev_valid <= k1)]
            if k_idx.size == 0:
                continue
            trans_pen = smooth_penalty * np.abs(offsets[k_idx] - offsets[j]).astype(np.float32)
            cand = dp[i - 1, k_idx] - trans_pen
            best_local = int(np.argmax(cand))
            best_k = int(k_idx[best_local])
            dp[i, j] = scores[i, j] + float(cand[best_local])
            back[i, j] = best_k

    last_valid = np.where(valid[-1])[0]
    if last_valid.size == 0:
        return np.asarray(bottoms, dtype=np.int32)
    tie_break = 1e-3 * offsets[last_valid].astype(np.float32)
    j_best = int(last_valid[int(np.argmax(dp[-1, last_valid] + tie_break))])

    path_idx = np.full((n,), j_best, dtype=np.int32)
    for i in range(n - 1, 0, -1):
        prev = int(back[i, path_idx[i]])
        if prev >= 0:
            path_idx[i - 1] = prev
        else:
            # If transition is missing, keep nearest feasible state.
            candidates = np.where(valid[i - 1])[0]
            if candidates.size:
                nearest = int(np.argmin(np.abs(candidates - path_idx[i])))
                path_idx[i - 1] = int(candidates[nearest])
            else:
                path_idx[i - 1] = path_idx[i]

    ys = np.asarray(bottoms, dtype=np.int32) + offsets[path_idx]
    ys = np.clip(ys, np.asarray(tops, dtype=np.int32) + 1, h - 1)
    return ys.astype(np.int32)


def _snap_lower_contour_to_depth_gradient(
    mask: np.ndarray,
    depth_map: np.ndarray,
    config: Dict,
) -> np.ndarray:
    if not bool(config.get("bottom_contour_snap_enabled", False)):
        return mask

    h, w = mask.shape[:2]
    search_up = max(0, int(config.get("bottom_contour_search_up_px", 18)))
    search_down = max(0, int(config.get("bottom_contour_search_down_px", 48)))
    smooth_window = int(config.get("bottom_contour_smooth_window", 31))
    gradient_quantile = float(config.get("bottom_contour_gradient_quantile", 55.0))
    gradient_quantile = float(max(0.0, min(100.0, gradient_quantile)))

    grad_y = cv2.Sobel(depth_map.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    abs_grad_y = np.abs(grad_y)
    global_thr = float(np.percentile(abs_grad_y, gradient_quantile))

    xs: list[int] = []
    tops: list[int] = []
    bottoms: list[int] = []
    snapped: list[int] = []
    for x in range(w):
        ys = np.where(mask[:, x] > 0)[0]
        if ys.size == 0:
            continue
        top = int(ys.min())
        bottom = int(ys.max())
        y0 = max(top, bottom - search_up)
        y1 = min(h - 1, bottom + search_down)
        if y1 <= y0:
            continue
        col = abs_grad_y[y0 : y1 + 1, x]
        local_idx = int(np.argmax(col))
        local_best = float(col[local_idx])
        strong_ratio = float(config.get("bottom_contour_deepest_strong_ratio", 0.70))
        strong_ratio = float(max(0.0, min(1.0, strong_ratio)))
        strong_thr = max(global_thr, local_best * strong_ratio)
        strong_idx = np.where(col >= strong_thr)[0]
        if strong_idx.size > 0:
            # Prefer the deepest strong edge to avoid selecting intermediate ridges.
            best_y = int(y0 + strong_idx[-1])
        else:
            best_y = int(y0 + local_idx)
        if local_best < global_thr:
            best_y = bottom
        if best_y <= top:
            best_y = min(h - 1, top + 1)
        xs.append(x)
        tops.append(top)
        bottoms.append(bottom)
        snapped.append(best_y)

    if not xs:
        return mask

    # Global regularization across x: reduce notches by solving a smooth path.
    regularized = _regularized_bottom_path(
        abs_grad_y=abs_grad_y,
        xs=xs,
        tops=tops,
        bottoms=bottoms,
        search_up=search_up,
        search_down=search_down,
        config=config,
    )
    base = np.asarray(snapped, dtype=np.float32)
    mix = float(max(0.0, min(1.0, config.get("bottom_contour_regularization_mix", 0.75))))
    mixed = (1.0 - mix) * base + mix * regularized.astype(np.float32)
    smoothed = _moving_average_1d(mixed, smooth_window)
    smoothed = np.rint(smoothed).astype(np.int32)

    refined = np.zeros_like(mask, dtype=np.uint8)
    for i, x in enumerate(xs):
        top = tops[i]
        new_bottom = int(np.clip(smoothed[i], top + 1, h - 1))
        refined[top : new_bottom + 1, x] = 255

    return refined


def _combine_masks(mask_a: np.ndarray, mask_b: np.ndarray, mode: str) -> np.ndarray:
    mode = mode.strip().lower()
    if mode == "or":
        return np.where((mask_a > 0) | (mask_b > 0), 255, 0).astype(np.uint8)
    # Default to AND for conservative valid-region selection.
    return np.where((mask_a > 0) & (mask_b > 0), 255, 0).astype(np.uint8)


def _horizontal_profile_mask(
    image: np.ndarray,
    config: Dict,
) -> tuple[np.ndarray, int, int, np.ndarray, np.ndarray]:
    illumination = estimate_illumination(image, int(config.get("blur_kernel_size", 151)))
    raw_profile, smoothed_profile = horizontal_profile(illumination, int(config.get("profile_smooth_window", 31)))
    x_start, x_end = detect_valid_region(
        smoothed_profile=smoothed_profile,
        threshold_ratio=float(config.get("threshold_ratio", 0.45)),
        safety_margin=int(config.get("safety_margin", 10)),
        min_region_width_ratio=float(config.get("min_region_width_ratio", 0.35)),
    )
    mask = build_mask(image.shape, x_start, x_end)
    return mask, x_start, x_end, raw_profile, smoothed_profile


def generate_valid_region_mask(image: np.ndarray, config: Dict) -> Tuple[np.ndarray, int, int]:
    illumination = estimate_illumination(image, int(config.get("blur_kernel_size", 151)))
    raw_profile, smoothed_profile = horizontal_profile(illumination, int(config.get("profile_smooth_window", 31)))
    x_start, x_end = detect_valid_region(
        smoothed_profile=smoothed_profile,
        threshold_ratio=float(config.get("threshold_ratio", 0.45)),
        safety_margin=int(config.get("safety_margin", 10)),
        min_region_width_ratio=float(config.get("min_region_width_ratio", 0.35)),
    )
    mask = build_mask(image.shape, x_start, x_end)
    return mask, x_start, x_end


def run_valid_region(
    image: np.ndarray,
    output_dir: str | Path,
    config: Dict,
) -> Dict:
    method = str(config.get("method", "horizontal_illumination_profile")).strip().lower()
    if method == "central_deep_layer":
        mask = _central_deep_layer_mask(image, config)
        illumination = estimate_illumination(image, int(config.get("blur_kernel_size", 151)))
        depth_map = 255.0 - illumination.astype(np.float32)
        mask = _snap_lower_contour_to_depth_gradient(mask, depth_map, config)
        x_start, x_end = _bbox_from_mask(mask)
        _save_depth_debug_outputs(
            original_image=image,
            depth_map=depth_map,
            mask=mask,
            output_dir=output_dir,
        )
        return {
            "enabled": True,
            "x_start": int(x_start),
            "x_end": int(x_end),
            "width": int(max(1, x_end - x_start + 1)),
            "method": "central_deep_layer",
            "output_dir": str(Path(output_dir).resolve()),
        }

    if method == "hybrid_deep_layer_profile":
        profile_mask, _, _, raw_profile, smoothed_profile = _horizontal_profile_mask(image, config)
        # Hybrid strategy: first constrain by profile-based valid area, then refine by depth.
        depth_mask = _central_deep_layer_mask(image, config, constraint_mask=profile_mask)
        combine_mode = str(config.get("hybrid_combine_mode", "and"))
        mask = _combine_masks(depth_mask, profile_mask, combine_mode)
        if int(np.count_nonzero(mask)) == 0:
            depth_nz = int(np.count_nonzero(depth_mask))
            profile_nz = int(np.count_nonzero(profile_mask))
            # Fail-safe: avoid empty valid-region masks by falling back to the richer source mask.
            mask = depth_mask if depth_nz >= profile_nz else profile_mask
        illumination = estimate_illumination(image, int(config.get("blur_kernel_size", 151)))
        depth_map = 255.0 - illumination.astype(np.float32)
        mask = _snap_lower_contour_to_depth_gradient(mask, depth_map, config)
        x_start, x_end = _bbox_from_mask(mask)
        _save_depth_debug_outputs(
            original_image=image,
            depth_map=depth_map,
            mask=mask,
            output_dir=output_dir,
        )
        return {
            "enabled": True,
            "x_start": int(x_start),
            "x_end": int(x_end),
            "width": int(max(1, x_end - x_start + 1)),
            "method": "hybrid_deep_layer_profile",
            "hybrid_combine_mode": combine_mode.lower(),
            "output_dir": str(Path(output_dir).resolve()),
        }

    mask, x_start, x_end, raw_profile, smoothed_profile = _horizontal_profile_mask(image, config)
    save_debug_outputs(image, mask, x_start, x_end, raw_profile, smoothed_profile, output_dir)
    return {
        "enabled": True,
        "x_start": int(x_start),
        "x_end": int(x_end),
        "width": int(x_end - x_start + 1),
        "method": "horizontal_illumination_profile",
        "output_dir": str(Path(output_dir).resolve()),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bat_tracker.valid_region",
        description="Generate vertical valid-area mask from IR illumination profile.",
    )
    parser.add_argument("--input", required=True, help="Path to grayscale input image (e.g., background.png).")
    parser.add_argument("--output", required=True, help="Output directory for mask/overlay/profile.")
    parser.add_argument("--blur-kernel-size", type=int, default=151)
    parser.add_argument("--profile-smooth-window", type=int, default=31)
    parser.add_argument("--threshold-ratio", type=float, default=0.45)
    parser.add_argument("--safety-margin", type=int, default=10)
    parser.add_argument("--min-region-width-ratio", type=float, default=0.35)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    cfg = {
        "blur_kernel_size": args.blur_kernel_size,
        "profile_smooth_window": args.profile_smooth_window,
        "threshold_ratio": args.threshold_ratio,
        "safety_margin": args.safety_margin,
        "min_region_width_ratio": args.min_region_width_ratio,
    }

    image = load_image(args.input)
    result = run_valid_region(image=image, output_dir=args.output, config=cfg)
    print(f"Detected valid region: x_start={result['x_start']}, x_end={result['x_end']}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

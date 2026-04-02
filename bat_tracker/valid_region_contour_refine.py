"""
valid_region_contour_refine.py
──────────────────────────────
Optional contour-refinement phase for valid_region masks.

After the base mask is produced by run_valid_region (including the existing
bottom-contour snap), this module snaps ALL four edges of the mask towards
strong gradients in a combined image+depth signal, constrained by a ROI and
a maximum displacement budget.

Configuration block (under valid_region in YAML)
────────────────────────────────────────────────
contour_refine:
  enabled: false            # set true to activate
  method: gradient_snap     # only method in Iteration 1
  roi_expand_px: 40         # ROI = base mask dilated by this many px
  edge_source: combined     # 'image' | 'depth' | 'combined'
  image_gradient_weight: 1.0
  depth_gradient_weight: 1.0
  smoothness_weight: 0.8    # DP transition penalty per px jump (≈ regularization)
  max_snap_px: 35           # max displacement allowed per edge
  regularization_weight: 0.6  # mix ratio: 0 = pure grad snap, 1 = pure smooth
  smooth_window: 35         # moving-average window applied after DP

Public API
──────────
  apply_contour_refine(base_mask, image, depth_map, refine_cfg)
      → (refined_mask, metrics_dict)

  compute_refine_metrics(base_mask, refined_mask)
      → dict

  save_refine_artifacts(base_mask, refined_mask, image, depth_map,
                        grad_combined, out_dir)
      → None
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Gradient helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_combined_gradient(
    image: np.ndarray,
    depth_map: np.ndarray,
    image_weight: float,
    depth_weight: float,
) -> np.ndarray:
    """Return a float32 gradient-magnitude map from image and/or depth."""
    def _gmag(arr: np.ndarray) -> np.ndarray:
        f = arr.astype(np.float32)
        gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gx, gy)

    out = np.zeros(image.shape[:2], dtype=np.float32)
    if image_weight > 0:
        out += float(image_weight) * _gmag(image)
    if depth_weight > 0:
        out += float(depth_weight) * _gmag(depth_map)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ROI
# ─────────────────────────────────────────────────────────────────────────────

def _build_roi(mask: np.ndarray, expand_px: int) -> np.ndarray:
    if expand_px <= 0:
        return mask.copy()
    k = 2 * expand_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask, kernel, iterations=1)


# ─────────────────────────────────────────────────────────────────────────────
# 1-D moving-average that respects inactive (−1) positions
# ─────────────────────────────────────────────────────────────────────────────

def _smooth_1d_active(arr: np.ndarray, window: int) -> np.ndarray:
    active = arr >= 0
    if not np.any(active):
        return arr.copy()

    w = max(3, int(window))
    if w % 2 == 0:
        w += 1

    # Fill inactive positions with interpolated values before smoothing
    filled = arr.astype(np.float32)
    active_idxs = np.where(active)[0]
    inactive_idxs = np.where(~active)[0]

    if inactive_idxs.size > 0:
        left_pos = np.searchsorted(active_idxs, inactive_idxs, side="left") - 1
        right_pos = np.searchsorted(active_idxs, inactive_idxs, side="right")
        left_valid = left_pos >= 0
        right_valid = right_pos < active_idxs.size
        both = left_valid & right_valid
        left_only = left_valid & ~right_valid
        right_only = ~left_valid & right_valid
        vals = np.zeros(inactive_idxs.size, dtype=np.float32)
        if np.any(both):
            vals[both] = (arr[active_idxs[left_pos[both]]] + arr[active_idxs[right_pos[both]]]) / 2.0
        if np.any(left_only):
            vals[left_only] = arr[active_idxs[left_pos[left_only]]].astype(np.float32)
        if np.any(right_only):
            vals[right_only] = arr[active_idxs[right_pos[right_only]]].astype(np.float32)
        filled[inactive_idxs] = vals

    pad = w // 2
    kernel = np.ones(w, dtype=np.float32) / float(w)
    padded = np.pad(filled, (pad, pad), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")

    result = np.where(active, smoothed, -1.0)
    return np.rint(result).astype(np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Core DP snap (1-D)
# ─────────────────────────────────────────────────────────────────────────────

def _dp_snap_1d(
    scores: np.ndarray,      # (n, m) float32 – already normalised & biased
    valid: np.ndarray,       # (n, m) bool
    offsets: np.ndarray,     # (m,) int32
    smoothness_w: float,
    max_snap_px: int,
) -> np.ndarray:             # (n,) int32 – optimal offset value per position
    """DP that selects, for each index i, the offset index j maximising
    score[i,j] + sum_t transition_penalty, subject to a max-step constraint."""
    n, m = scores.shape
    max_step = max(1, max_snap_px // 3)

    dp = np.full((n, m), -1e9, dtype=np.float32)
    back = np.full((n, m), -1, dtype=np.int32)

    v0 = np.where(valid[0])[0]
    if v0.size == 0:
        return np.zeros(n, dtype=np.int32)
    dp[0, v0] = scores[0, v0]

    for i in range(1, n):
        vi = np.where(valid[i])[0]
        if vi.size == 0:
            continue
        prev_vi = np.where(valid[i - 1])[0]
        if prev_vi.size == 0:
            dp[i, vi] = scores[i, vi]
            continue

        pv_min = int(prev_vi.min())
        pv_max = int(prev_vi.max())

        for j in vi:
            lo = max(pv_min, j - max_step)
            hi = min(pv_max, j + max_step)
            k_idx = prev_vi[(prev_vi >= lo) & (prev_vi <= hi)]
            if k_idx.size == 0:
                continue
            pen = smoothness_w * np.abs(offsets[k_idx] - offsets[j]).astype(np.float32)
            cand = dp[i - 1, k_idx] - pen
            best = int(np.argmax(cand))
            dp[i, j] = scores[i, j] + float(cand[best])
            back[i, j] = int(k_idx[best])

    lv = np.where(valid[-1])[0]
    if lv.size == 0:
        return np.zeros(n, dtype=np.int32)

    j_best = int(lv[int(np.argmax(dp[-1, lv]))])
    path = np.full(n, j_best, dtype=np.int32)
    for i in range(n - 1, 0, -1):
        prev = int(back[i, path[i]])
        if prev >= 0:
            path[i - 1] = prev
        else:
            cands = np.where(valid[i - 1])[0]
            if cands.size:
                path[i - 1] = int(cands[int(np.argmin(np.abs(cands - path[i])))])
            else:
                path[i - 1] = path[i]

    return offsets[path]


# ─────────────────────────────────────────────────────────────────────────────
# Per-edge snap passes
# ─────────────────────────────────────────────────────────────────────────────

def _snap_column_edge(
    mask: np.ndarray,
    grad: np.ndarray,
    edge: str,           # 'top' | 'bottom'
    max_snap_px: int,
    smoothness_w: float,
    reg_w: float,
    smooth_window: int,
) -> np.ndarray:
    """Snap either the top or the bottom edge of the mask (per-column DP)."""
    h, w = mask.shape[:2]
    offsets = np.arange(-max_snap_px, max_snap_px + 1, dtype=np.int32)
    m = offsets.size

    # Extract per-column top/bottom and current edge position
    tops_full = np.full(w, -1, dtype=np.int32)
    bots_full = np.full(w, -1, dtype=np.int32)
    pos_full = np.full(w, -1, dtype=np.int32)

    for x in range(w):
        ys = np.where(mask[:, x] > 0)[0]
        if ys.size == 0:
            continue
        tops_full[x] = int(ys.min())
        bots_full[x] = int(ys.max())
        pos_full[x] = int(ys.min()) if edge == "top" else int(ys.max())

    active = np.where(pos_full >= 0)[0]
    if active.size == 0:
        return mask.copy()

    n = active.size
    positions = pos_full[active]  # (n,)

    # Build score array (vectorised)
    y_cands = positions[:, np.newaxis] + offsets[np.newaxis, :]   # (n, m)
    ok = (y_cands >= 0) & (y_cands < h)
    y_safe = np.clip(y_cands, 0, h - 1)
    x_rep = np.tile(active[:, np.newaxis], (1, m))                 # (n, m)
    raw_scores = grad[y_safe, x_rep].astype(np.float32)
    raw_scores[~ok] = -1e9

    # Normalise
    valid_vals = raw_scores[ok]
    if valid_vals.size > 0:
        scale = float(np.percentile(valid_vals, 90))
        scale = max(1e-6, scale)
        raw_scores = raw_scores / scale
        raw_scores[~ok] = -1e9

    # Directional bias: pull top edge upward, bottom downward
    bias = -0.05 if edge == "top" else 0.05
    raw_scores = raw_scores + bias * (offsets[np.newaxis, :] / float(max(1, max_snap_px)))

    # DP snap
    snap_offsets = _dp_snap_1d(raw_scores, ok, offsets, smoothness_w, max_snap_px)
    snapped = np.clip(positions + snap_offsets, 0, h - 1)

    # Store in full-width array and smooth
    result_full = np.full(w, -1, dtype=np.int32)
    result_full[active] = snapped
    smoothed_full = _smooth_1d_active(result_full, smooth_window)

    # Mix pure gradient snap with smoothed version
    mixed_full = result_full.copy()
    if reg_w > 0 and np.any(smoothed_full >= 0):
        for x in active:
            s = int(result_full[x])
            sm = int(smoothed_full[x]) if smoothed_full[x] >= 0 else s
            mixed_full[x] = int(np.clip(round((1.0 - reg_w) * s + reg_w * sm), 0, h - 1))

    # Rebuild mask for this edge only
    new_mask = np.zeros_like(mask)
    for x in range(w):
        if tops_full[x] < 0:
            continue
        t = int(tops_full[x])
        b = int(bots_full[x])
        if edge == "top":
            new_t = int(mixed_full[x]) if mixed_full[x] >= 0 else t
            new_t = int(np.clip(new_t, 0, b - 1))
            new_mask[new_t : b + 1, x] = 255
        else:
            new_b = int(mixed_full[x]) if mixed_full[x] >= 0 else b
            new_b = int(np.clip(new_b, t + 1, h - 1))
            new_mask[t : new_b + 1, x] = 255

    return new_mask


def _snap_row_edge(
    mask: np.ndarray,
    grad: np.ndarray,
    edge: str,           # 'left' | 'right'
    max_snap_px: int,
    smoothness_w: float,
    reg_w: float,
    smooth_window: int,
) -> np.ndarray:
    """Snap either the left or the right edge of the mask (per-row DP)."""
    h, w = mask.shape[:2]
    offsets = np.arange(-max_snap_px, max_snap_px + 1, dtype=np.int32)
    m = offsets.size

    lefts_full = np.full(h, -1, dtype=np.int32)
    rights_full = np.full(h, -1, dtype=np.int32)
    pos_full = np.full(h, -1, dtype=np.int32)

    for y in range(h):
        xs = np.where(mask[y, :] > 0)[0]
        if xs.size == 0:
            continue
        lefts_full[y] = int(xs.min())
        rights_full[y] = int(xs.max())
        pos_full[y] = int(xs.min()) if edge == "left" else int(xs.max())

    active = np.where(pos_full >= 0)[0]
    if active.size == 0:
        return mask.copy()

    n = active.size
    positions = pos_full[active]  # (n,)

    # Build score array (vectorised)
    x_cands = positions[:, np.newaxis] + offsets[np.newaxis, :]   # (n, m)
    ok = (x_cands >= 0) & (x_cands < w)
    x_safe = np.clip(x_cands, 0, w - 1)
    y_rep = np.tile(active[:, np.newaxis], (1, m))                 # (n, m)
    raw_scores = grad[y_rep, x_safe].astype(np.float32)
    raw_scores[~ok] = -1e9

    valid_vals = raw_scores[ok]
    if valid_vals.size > 0:
        scale = float(np.percentile(valid_vals, 90))
        scale = max(1e-6, scale)
        raw_scores = raw_scores / scale
        raw_scores[~ok] = -1e9

    bias = -0.05 if edge == "left" else 0.05
    raw_scores = raw_scores + bias * (offsets[np.newaxis, :] / float(max(1, max_snap_px)))

    snap_offsets = _dp_snap_1d(raw_scores, ok, offsets, smoothness_w, max_snap_px)
    snapped = np.clip(positions + snap_offsets, 0, w - 1)

    result_full = np.full(h, -1, dtype=np.int32)
    result_full[active] = snapped
    smoothed_full = _smooth_1d_active(result_full, smooth_window)

    mixed_full = result_full.copy()
    if reg_w > 0 and np.any(smoothed_full >= 0):
        for y in active:
            s = int(result_full[y])
            sm = int(smoothed_full[y]) if smoothed_full[y] >= 0 else s
            mixed_full[y] = int(np.clip(round((1.0 - reg_w) * s + reg_w * sm), 0, w - 1))

    new_mask = np.zeros_like(mask)
    for y in range(h):
        if lefts_full[y] < 0:
            continue
        l = int(lefts_full[y])
        r = int(rights_full[y])
        if edge == "left":
            new_l = int(mixed_full[y]) if mixed_full[y] >= 0 else l
            new_l = int(np.clip(new_l, 0, r - 1))
            new_mask[y, new_l : r + 1] = 255
        else:
            new_r = int(mixed_full[y]) if mixed_full[y] >= 0 else r
            new_r = int(np.clip(new_r, l + 1, w - 1))
            new_mask[y, l : new_r + 1] = 255

    return new_mask


# ─────────────────────────────────────────────────────────────────────────────
# Intensity-guided expansion (Iteration 2)
# ─────────────────────────────────────────────────────────────────────────────

def _intensity_expansion(
    base_mask: np.ndarray,
    depth_map: np.ndarray,
    roi_mask: np.ndarray,
    cfg: Dict,
    grad: np.ndarray | None = None,
    image: np.ndarray | None = None,
) -> np.ndarray:
    """
    Expand base mask to include adjacent dark (= high-depth) regions within
    the ROI that are connected to the base mask.

    depth_map = 255 - illumination  →  cave interior = HIGH, lit rock = LOW.

    The threshold is placed between the interior median and the corona median:
      T = d_inside - ratio * (d_inside - d_corona)
    ratio=0 → T=d_inside (strict, only deepest)
    ratio=1 → T=d_corona (loose, includes rock)

    When ``expansion_blur_kernel`` is set (and *image* is provided), a sharper
    depth map is computed with a smaller Gaussian blur.  This prevents the
    large (151 px) pipeline blur from smearing the cave/rock boundary and
    inflating depth values of nearby rock pixels.

    Optional gradient barrier: prevents expansion across strong gradient
    ridges (rock edges), which stops the dark-region flood from leaking into
    adjacent shadows that aren't part of the cave opening.
    """
    # ── Optionally build a sharper depth map for thresholding ───────────
    exp_blur = int(cfg.get("expansion_blur_kernel", 0))
    if exp_blur > 0 and image is not None:
        if exp_blur % 2 == 0:
            exp_blur += 1
        exp_blur = max(3, exp_blur)
        sharp_illum = cv2.GaussianBlur(image, (exp_blur, exp_blur), 0)
        exp_depth = 255.0 - sharp_illum.astype(np.float32)
    else:
        exp_depth = depth_map  # fallback: use the pipeline depth map

    inside_vals = exp_depth[base_mask > 0]
    if inside_vals.size == 0:
        return base_mask.copy()

    corona = (roi_mask > 0) & (base_mask == 0)
    corona_vals = exp_depth[corona]
    if corona_vals.size == 0:
        return base_mask.copy()

    d_inside = float(np.median(inside_vals))
    d_corona = float(np.median(corona_vals))

    ratio = float(cfg.get("expansion_intensity_ratio", 0.30))
    threshold = d_inside - ratio * (d_inside - d_corona)

    # All pixels dark enough to be "cave-like" (using sharp depth if available)
    dark = (exp_depth >= threshold).astype(np.uint8) * 255

    # ── Gradient barrier ────────────────────────────────────────────────
    # Strong gradient ridges act as walls: expansion can't cross them.
    # This prevents the flood from leaking into adjacent dark areas (shadows,
    # dark ground) that share a boundary with the cave opening.
    if grad is not None and bool(cfg.get("gradient_barrier_enabled", True)):
        barrier_pct = float(cfg.get("gradient_barrier_percentile", 75.0))
        barrier_thr = float(np.percentile(grad, barrier_pct))
        barrier = (grad >= barrier_thr).astype(np.uint8) * 255

        # Close the barrier to make edge fragments more continuous
        barrier_close = max(3, int(cfg.get("gradient_barrier_close_px", 7)))
        if barrier_close % 2 == 0:
            barrier_close += 1
        bk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (barrier_close, barrier_close))
        barrier = cv2.morphologyEx(barrier, cv2.MORPH_CLOSE, bk)

        # Remove barrier pixels from the dark mask
        # (but base mask pixels are NEVER removed — they seed the flood)
        dark = np.where((dark > 0) & ((barrier == 0) | (base_mask > 0)), 255, 0).astype(np.uint8)

    # Union with base mask to ensure connectivity seed
    merged = np.where((dark > 0) | (base_mask > 0), 255, 0).astype(np.uint8)

    # Morphological closing to bridge small gaps (joints between rocks, etc.)
    close_k = max(3, int(cfg.get("expansion_close_px", 15)))
    if close_k % 2 == 0:
        close_k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, kernel)

    # Keep only the connected component touching the base mask centre
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    if n_labels <= 1:
        return merged

    ys, xs = np.where(base_mask > 0)
    if ys.size == 0:
        return merged
    cy, cx = int(ys.mean()), int(xs.mean())
    center_label = int(labels[cy, cx])
    if center_label > 0:
        expanded = np.where(labels == center_label, 255, 0).astype(np.uint8)
    else:
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        expanded = np.where(labels == best, 255, 0).astype(np.uint8)

    # Clip to ROI
    return np.where((expanded > 0) & (roi_mask > 0), 255, 0).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# gradient_snap method
# ─────────────────────────────────────────────────────────────────────────────

def _refine_gradient_snap(
    base_mask: np.ndarray,
    image: np.ndarray,
    depth_map: np.ndarray,
    cfg: Dict,
) -> Tuple[np.ndarray, np.ndarray | None]:
    """Apply intensity expansion + 4-edge gradient snap.

    Returns (refined_mask, expanded_mask_or_None).
    """
    roi_expand = max(0, int(cfg.get("roi_expand_px", 120)))
    max_snap = max(1, int(cfg.get("max_snap_px", 35)))
    smooth_w = max(1, int(cfg.get("smooth_window", 35)))
    smooth_penalty = float(cfg.get("smoothness_weight", 0.8))
    reg_w = float(cfg.get("regularization_weight", 0.6))
    img_w = float(cfg.get("image_gradient_weight", 1.0))
    dep_w = float(cfg.get("depth_gradient_weight", 1.0))
    edge_src = str(cfg.get("edge_source", "combined")).strip().lower()

    if edge_src == "image":
        img_w_use, dep_w_use = img_w, 0.0
    elif edge_src == "depth":
        img_w_use, dep_w_use = 0.0, dep_w
    else:
        img_w_use, dep_w_use = img_w, dep_w

    grad = compute_combined_gradient(image, depth_map, img_w_use, dep_w_use)
    roi = _build_roi(base_mask, roi_expand)

    # Phase 1: Intensity-guided expansion (grow mask to full dark opening)
    expanded_mask: np.ndarray | None = None
    if bool(cfg.get("intensity_expansion_enabled", True)):
        refined = _intensity_expansion(base_mask, depth_map, roi, cfg, grad=grad, image=image)
        expanded_mask = refined.copy()
    else:
        refined = base_mask.copy()

    # Phase 2: 4-edge gradient snap (fine-tune edges to strong gradients)
    refined = _snap_column_edge(refined, grad, "bottom", max_snap, smooth_penalty, reg_w, smooth_w)
    refined = _snap_column_edge(refined, grad, "top",    max_snap, smooth_penalty, reg_w, smooth_w)
    refined = _snap_row_edge(   refined, grad, "left",   max_snap, smooth_penalty, reg_w, smooth_w)
    refined = _snap_row_edge(   refined, grad, "right",  max_snap, smooth_penalty, reg_w, smooth_w)

    # Clip to ROI (safety: never expand beyond roi_expand_px from base)
    refined = np.where((refined > 0) & (roi > 0), 255, 0).astype(np.uint8)

    # Keep only the largest connected component (avoids fragmentation)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined, connectivity=8)
    if n_labels > 1:
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        refined = np.where(labels == best, 255, 0).astype(np.uint8)

    return refined, expanded_mask


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def apply_contour_refine(
    base_mask: np.ndarray,
    image: np.ndarray,
    depth_map: np.ndarray,
    refine_cfg: Dict,
) -> Tuple[np.ndarray, Dict, np.ndarray | None]:
    """
    Entry point called by run_valid_region when contour_refine.enabled = true.

    Returns (refined_mask, metrics_dict, expanded_mask_or_None).
    If disabled, returns (base_mask, {}, None).
    """
    if not bool(refine_cfg.get("enabled", False)):
        return base_mask, {}, None

    expanded_mask: np.ndarray | None = None
    method = str(refine_cfg.get("method", "gradient_snap")).strip().lower()
    if method == "gradient_snap":
        refined, expanded_mask = _refine_gradient_snap(base_mask, image, depth_map, refine_cfg)
    else:
        refined = base_mask.copy()

    metrics = compute_refine_metrics(base_mask, refined)
    return refined, metrics, expanded_mask


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_refine_metrics(
    base_mask: np.ndarray,
    refined_mask: np.ndarray,
) -> Dict:
    """Compute area, perimeter, bbox, IoU and contour displacement metrics."""

    def _area(m: np.ndarray) -> int:
        return int(np.count_nonzero(m))

    def _perimeter(m: np.ndarray) -> float:
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return float(sum(len(c) for c in cnts))

    def _bbox(m: np.ndarray) -> Tuple[int, int, int, int]:
        ys, xs = np.where(m > 0)
        if xs.size == 0:
            return 0, 0, 0, 0
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        inter = int(np.count_nonzero((a > 0) & (b > 0)))
        union = int(np.count_nonzero((a > 0) | (b > 0)))
        return float(inter) / max(1, union)

    def _contour_disp(m_a: np.ndarray, m_b: np.ndarray) -> Tuple[float, float]:
        """Mean/max distance from contour-A points to nearest contour-B point."""
        cnt_a, _ = cv2.findContours(m_a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt_b, _ = cv2.findContours(m_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnt_a or not cnt_b:
            return 0.0, 0.0
        # Distance transform of contour-B canvas
        canvas_b = np.zeros_like(m_b)
        cv2.drawContours(canvas_b, cnt_b, -1, 255, 1)
        dt = cv2.distanceTransform(255 - canvas_b, cv2.DIST_L2, 3)
        pts = np.concatenate(cnt_a).reshape(-1, 2)
        pts[:, 0] = np.clip(pts[:, 0], 0, m_a.shape[1] - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, m_a.shape[0] - 1)
        dists = dt[pts[:, 1], pts[:, 0]]
        return float(dists.mean()), float(dists.max())

    a_b = _area(base_mask)
    a_r = _area(refined_mask)
    p_b = _perimeter(base_mask)
    p_r = _perimeter(refined_mask)
    bb_b = _bbox(base_mask)
    bb_r = _bbox(refined_mask)
    iou = _iou(base_mask, refined_mask)
    mean_d, max_d = _contour_disp(base_mask, refined_mask)

    return {
        "area_base": a_b,
        "area_refined": a_r,
        "area_change_pct": float(100.0 * (a_r - a_b) / max(1, a_b)),
        "perimeter_base": p_b,
        "perimeter_refined": p_r,
        "perimeter_change_pct": float(100.0 * (p_r - p_b) / max(1, p_b)),
        "bbox_base": {"x1": bb_b[0], "y1": bb_b[1], "x2": bb_b[2], "y2": bb_b[3]},
        "bbox_refined": {"x1": bb_r[0], "y1": bb_r[1], "x2": bb_r[2], "y2": bb_r[3]},
        "iou_base_refined": iou,
        "contour_mean_displacement_px": mean_d,
        "contour_max_displacement_px": max_d,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Artifact saving
# ─────────────────────────────────────────────────────────────────────────────

def _draw_contour_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> np.ndarray:
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image.copy()
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(bgr, cnts, -1, color, thickness, lineType=cv2.LINE_AA)
    return bgr


def _normalize_to_u8(arr: np.ndarray) -> np.ndarray:
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-6:
        return np.zeros_like(arr, dtype=np.uint8)
    return np.clip((arr - mn) / (mx - mn) * 255, 0, 255).astype(np.uint8)


def save_refine_artifacts(
    base_mask: np.ndarray,
    refined_mask: np.ndarray,
    image: np.ndarray,
    depth_map: np.ndarray,
    grad_combined: np.ndarray,
    out_dir: Path,
    *,
    expanded_mask: np.ndarray | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. mask_base.png
    cv2.imwrite(str(out_dir / "mask_base.png"), base_mask)

    # 2. contour_base.png  (cyan contour on image)
    cv2.imwrite(
        str(out_dir / "contour_base.png"),
        _draw_contour_on_image(image, base_mask, (0, 255, 255)),
    )

    # 3. gradient_combined.png
    cv2.imwrite(
        str(out_dir / "gradient_combined.png"),
        cv2.applyColorMap(_normalize_to_u8(grad_combined), cv2.COLORMAP_HOT),
    )

    # 3b. mask_expanded.png (intensity expansion, before gradient snap)
    if expanded_mask is not None:
        cv2.imwrite(str(out_dir / "mask_expanded.png"), expanded_mask)
        cv2.imwrite(
            str(out_dir / "contour_expanded.png"),
            _draw_contour_on_image(image, expanded_mask, (255, 128, 0)),  # orange
        )

    # 4. depth_map.png
    depth_u8 = np.clip(depth_map, 0, 255).astype(np.uint8)
    cv2.imwrite(
        str(out_dir / "depth_map.png"),
        cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO),
    )

    # 5. mask_refined.png
    cv2.imwrite(str(out_dir / "mask_refined.png"), refined_mask)

    # 6. contour_refined.png  (green contour on image)
    cv2.imwrite(
        str(out_dir / "contour_refined.png"),
        _draw_contour_on_image(image, refined_mask, (0, 255, 0)),
    )

    # 7. comparison.png – left: base (cyan), right: refined (green), side by side
    img_base = _draw_contour_on_image(image, base_mask, (0, 255, 255))
    img_ref = _draw_contour_on_image(image, refined_mask, (0, 255, 0))
    h = max(img_base.shape[0], img_ref.shape[0])
    sep = np.full((h, 4, 3), 80, dtype=np.uint8)  # thin grey separator

    def _pad_h(img: np.ndarray, target_h: int) -> np.ndarray:
        if img.shape[0] < target_h:
            pad = np.zeros((target_h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
            return np.vstack([img, pad])
        return img

    comparison = np.hstack([_pad_h(img_base, h), sep, _pad_h(img_ref, h)])

    # Add labels
    cv2.putText(comparison, "BASE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(comparison, "REFINED", (img_base.shape[1] + 14, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imwrite(str(out_dir / "comparison.png"), comparison)

    # 8. overlay_both.png – both contours on the same image
    both = _draw_contour_on_image(image, base_mask, (0, 255, 255))  # cyan = base
    cnts_r, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(both, cnts_r, -1, (0, 255, 0), 2, cv2.LINE_AA)  # green = refined
    cv2.putText(both, "CYAN=base  GREEN=refined", (10, both.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "overlay_both.png"), both)

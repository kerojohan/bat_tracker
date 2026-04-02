"""
valid_region_refine_study.py
────────────────────────────
Reproducible study tool for evaluating contour_refine on one or more videos.

Usage
─────
  python -m bat_tracker.valid_region_refine_study \\
      --video /path/to/video.mp4 [/path/to/video2.mp4 ...] \\
      --config config.out3_clean.yaml \\
      --out-dir runs/valid_region_refine_study

Per-video output (inside <out-dir>/<video_stem>/)
──────────────────────────────────────────────────
  background.png            – median background frame
  mask_base.png             – base mask (no contour_refine)
  contour_base.png          – base contour overlaid (cyan)
  gradient_combined.png     – combined gradient map
  depth_map.png             – depth map (TURBO colormap)
  mask_refined.png          – refined mask
  contour_refined.png       – refined contour overlaid (green)
  comparison.png            – side-by-side base vs refined
  overlay_both.png          – both contours on one image
  metrics.json              – numerical comparison

Top-level
─────────
  summary.json              – metrics for all videos
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import yaml

from .valid_region import (
    _apply_mask_geometry,
    _central_deep_layer_mask,
    _combine_masks,
    _horizontal_profile_mask,
    _snap_lower_contour_to_depth_gradient,
    estimate_illumination,
)
from .valid_region_contour_refine import (
    apply_contour_refine,
    compute_combined_gradient,
    compute_refine_metrics,
    save_refine_artifacts,
)


# ─────────────────────────────────────────────────────────────────────────────
# Background extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_background(
    video_path: str,
    sample_frames: int = 120,
) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = max(1, total)
    n = min(sample_frames, total)
    indices = np.linspace(0, total - 1, n).astype(np.int32)

    frames = []
    try:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            if frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            frames.append(gray)
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"Could not read frames from: {video_path}")

    stack = np.stack(frames, axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Base mask computation (mirrors run_valid_region without I/O)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_base_mask(
    image: np.ndarray,
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the full valid_region pipeline (minus contour_refine) and return
    (base_mask, depth_map).  Supports all three methods.
    """
    method = str(config.get("method", "horizontal_illumination_profile")).strip().lower()

    illumination = estimate_illumination(image, int(config.get("blur_kernel_size", 151)))
    depth_map = (255.0 - illumination.astype(np.float32))

    if method == "central_deep_layer":
        mask = _central_deep_layer_mask(image, config)
        mask = _snap_lower_contour_to_depth_gradient(mask, depth_map, config)
        mask = _apply_mask_geometry(mask, depth_map, None, config)
        return mask, depth_map

    if method == "hybrid_deep_layer_profile":
        profile_mask, _, _, _, _ = _horizontal_profile_mask(image, config)
        depth_mask = _central_deep_layer_mask(image, config, constraint_mask=profile_mask)
        combine_mode = str(config.get("hybrid_combine_mode", "and"))
        mask = _combine_masks(depth_mask, profile_mask, combine_mode)
        if int(np.count_nonzero(mask)) == 0:
            d_nz = int(np.count_nonzero(depth_mask))
            p_nz = int(np.count_nonzero(profile_mask))
            mask = depth_mask if d_nz >= p_nz else profile_mask
        mask = _snap_lower_contour_to_depth_gradient(mask, depth_map, config)
        mask = _apply_mask_geometry(mask, depth_map, profile_mask, config)
        return mask, depth_map

    # horizontal_illumination_profile (fallback)
    profile_mask, _, _, _, _ = _horizontal_profile_mask(image, config)
    return profile_mask, depth_map


# ─────────────────────────────────────────────────────────────────────────────
# Per-video study
# ─────────────────────────────────────────────────────────────────────────────

def _study_one_video(
    video_path: str,
    vr_config: dict,
    refine_cfg: dict,
    out_dir: Path,
    sample_frames: int,
) -> dict:
    stem = Path(video_path).stem
    vid_dir = out_dir / stem
    vid_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [{stem}] extracting background ({sample_frames} frames)…")
    bg = _extract_background(video_path, sample_frames)
    cv2.imwrite(str(vid_dir / "background.png"), bg)

    print(f"  [{stem}] computing base mask…")
    base_mask, depth_map = _compute_base_mask(bg, vr_config)

    print(f"  [{stem}] applying contour_refine…")
    # Force enabled=True for the study (regardless of config)
    refine_cfg_study = dict(refine_cfg)
    refine_cfg_study["enabled"] = True
    refined_mask, metrics, expanded_mask = apply_contour_refine(base_mask, bg, depth_map, refine_cfg_study)

    # Compute gradient for artifact saving
    img_w = float(refine_cfg_study.get("image_gradient_weight", 1.0))
    dep_w = float(refine_cfg_study.get("depth_gradient_weight", 1.0))
    edge_src = str(refine_cfg_study.get("edge_source", "combined")).lower()
    if edge_src == "image":
        img_w_use, dep_w_use = img_w, 0.0
    elif edge_src == "depth":
        img_w_use, dep_w_use = 0.0, dep_w
    else:
        img_w_use, dep_w_use = img_w, dep_w
    grad = compute_combined_gradient(bg, depth_map, img_w_use, dep_w_use)

    save_refine_artifacts(base_mask, refined_mask, bg, depth_map, grad, vid_dir,
                          expanded_mask=expanded_mask)

    # Write metrics
    (vid_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    return {"video": video_path, "stem": stem, **metrics}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m bat_tracker.valid_region_refine_study",
        description="Evaluate contour_refine on one or more videos.",
    )
    p.add_argument(
        "--video",
        nargs="+",
        required=True,
        metavar="VIDEO",
        help="Path(s) to input video(s).",
    )
    p.add_argument(
        "--config",
        default=None,
        metavar="YAML",
        help="Config YAML (default: built-in defaults).",
    )
    p.add_argument(
        "--out-dir",
        default="runs/valid_region_refine_study",
        metavar="DIR",
        help="Output directory.",
    )
    p.add_argument(
        "--sample-frames",
        type=int,
        default=120,
        help="Number of frames to sample for background (default: 120).",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    # Load config
    vr_config: dict = {}
    refine_cfg: dict = {}
    if args.config:
        with open(args.config) as f:
            full_cfg = yaml.safe_load(f) or {}
        vr_config = full_cfg.get("valid_region", {})
        refine_cfg = vr_config.get("contour_refine", {})

    # Default contour_refine params if not in config
    refine_defaults = {
        "enabled": True,
        "method": "gradient_snap",
        "roi_expand_px": 40,
        "edge_source": "combined",
        "image_gradient_weight": 1.0,
        "depth_gradient_weight": 1.0,
        "smoothness_weight": 0.8,
        "max_snap_px": 35,
        "regularization_weight": 0.6,
        "smooth_window": 35,
    }
    for k, v in refine_defaults.items():
        refine_cfg.setdefault(k, v)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for video_path in args.video:
        print(f"\n{'='*60}")
        print(f"Video: {video_path}")
        print(f"{'='*60}")
        try:
            result = _study_one_video(
                video_path=video_path,
                vr_config=vr_config,
                refine_cfg=refine_cfg,
                out_dir=out_dir,
                sample_frames=args.sample_frames,
            )
            all_results.append(result)
            _print_metrics(result)
        except Exception:
            print(f"  ERROR processing {video_path}:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            all_results.append({"video": video_path, "error": traceback.format_exc()})

    # Write summary
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSummary written to: {summary_path}")


def _print_metrics(r: dict) -> None:
    print(f"\n  ── Metrics for {r.get('stem', r.get('video', '?'))} ──")
    if "error" in r:
        print(f"  ERROR: {r['error'][:200]}")
        return
    print(f"  Area:       base={r['area_base']:>8,}  refined={r['area_refined']:>8,}  Δ={r['area_change_pct']:+.1f}%")
    print(f"  Perimeter:  base={r['perimeter_base']:>8.0f}  refined={r['perimeter_refined']:>8.0f}  Δ={r['perimeter_change_pct']:+.1f}%")
    b, rf = r["bbox_base"], r["bbox_refined"]
    print(f"  BBox base:    x=[{b['x1']},{b['x2']}]  y=[{b['y1']},{b['y2']}]")
    print(f"  BBox refined: x=[{rf['x1']},{rf['x2']}]  y=[{rf['y1']},{rf['y2']}]")
    print(f"  IoU(base,refined):         {r['iou_base_refined']:.4f}")
    print(f"  Contour displacement mean: {r['contour_mean_displacement_px']:.2f} px")
    print(f"  Contour displacement max:  {r['contour_max_displacement_px']:.2f} px")


if __name__ == "__main__":
    main()

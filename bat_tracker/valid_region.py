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
    illumination = estimate_illumination(image, int(config.get("blur_kernel_size", 151)))
    raw_profile, smoothed_profile = horizontal_profile(illumination, int(config.get("profile_smooth_window", 31)))
    x_start, x_end = detect_valid_region(
        smoothed_profile=smoothed_profile,
        threshold_ratio=float(config.get("threshold_ratio", 0.45)),
        safety_margin=int(config.get("safety_margin", 10)),
        min_region_width_ratio=float(config.get("min_region_width_ratio", 0.35)),
    )
    mask = build_mask(image.shape, x_start, x_end)
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

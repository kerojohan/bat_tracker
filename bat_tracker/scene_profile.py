"""
Perfilatge de escena: mètriques geomètriques, fotomètriques i temporals per recomanar paràmetres.

Dissenyat per ser determinista, reproduïble i sense dependències pesades.
"""
from __future__ import annotations

import csv
import json
import logging
import tempfile
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from .scene_auto_tune import compute_recommendations
from .scene_auto_tune import write_decisions_summary
from .valid_region import _apply_mask_geometry
from .valid_region import _horizontal_profile_mask
from .valid_region import estimate_illumination
from .valid_region import run_valid_region
from .video import frame_to_gray
from .video import open_video_capture
from .video import read_video_meta

logger = logging.getLogger(__name__)


@dataclass
class VideoMetaSnapshot:
    input_path: str
    video_id: str
    fps: float
    frame_count: int
    width: int
    height: int


@dataclass
class OpeningGeometry:
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    width_px: int
    height_px: int
    area_px: int
    centroid_x: float
    centroid_y: float
    orientation_deg: float
    solidity: float
    extent: float
    frame_area_fraction: float
    horizontal_profile_mean: List[float] = field(default_factory=list)
    vertical_profile_mean: List[float] = field(default_factory=list)


@dataclass
class ContrastStats:
    mean_inside: float
    mean_corona: float
    delta_mean: float
    p10_inside: float
    p50_inside: float
    p90_inside: float
    p10_corona: float
    p50_corona: float
    p90_corona: float
    mean_gradient_mag_on_contour: float
    depth_mean_inside: float
    depth_mean_border: float
    depth_p90_inside: float


@dataclass
class TemporalNoiseStats:
    sample_frames: List[int]
    per_frame_median_absdiff: List[float]
    per_frame_mean_absdiff: List[float]
    per_frame_p95_absdiff: List[float]
    median_of_frame_mean_absdiff: float
    median_of_frame_p95_absdiff: float
    mean_absdiff_roi: float
    std_absdiff_roi: float
    p50_absdiff_roi: float
    p90_absdiff_roi: float
    p95_absdiff_roi: float
    p99_absdiff_roi: float
    global_intensity_shift_mean: float
    global_intensity_shift_std: float
    foreground_ratio_by_threshold: Dict[str, float]
    absdiff_temporal_stability_std: float


@dataclass
class BlobStats:
    thresholds_tested: List[int]
    blobs_per_frame_median: Dict[str, float]
    area_samples: List[float]
    area_p10: float
    area_p25: float
    area_p50: float
    area_p75: float
    area_p90: float
    dust_proxy_high_frequency_small: float


def _strip_mask_geometry(vr_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in vr_cfg.items() if k != "mask_geometry"}


def _deterministic_area_samples(areas_arr: np.ndarray, max_n: int = 2000) -> List[float]:
    if areas_arr.size == 0:
        return []
    s = np.sort(areas_arr)
    if len(s) <= max_n:
        return [float(x) for x in s.tolist()]
    step = max(1, len(s) // max_n)
    return [float(s[i]) for i in range(0, len(s), step)]


def _largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _opening_geometry(mask: np.ndarray, frame_shape: Tuple[int, int]) -> OpeningGeometry:
    h, w = frame_shape
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return OpeningGeometry(0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [], [])

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    bw, bh = x2 - x1 + 1, y2 - y1 + 1
    area = int(np.sum(mask > 0))
    m = cv2.moments(mask, binaryImage=True)
    cx = float(m["m10"] / m["m00"]) if m["m00"] else float(x1 + bw / 2)
    cy = float(m["m01"] / m["m00"]) if m["m00"] else float(y1 + bh / 2)

    cnt = _largest_contour(mask)
    orient_deg = 0.0
    solidity = 0.0
    extent = 0.0
    if cnt is not None and len(cnt) >= 5:
        hull = cv2.convexHull(cnt)
        area_c = cv2.contourArea(cnt)
        hull_a = cv2.contourArea(hull)
        solidity = float(area_c / hull_a) if hull_a > 1e-6 else 0.0
        rect_x, rect_y, rw, rh = cv2.boundingRect(cnt)
        extent = float(area_c / max(1, rw * rh))
        pts = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
        pts = pts - np.array([cx, cy], dtype=np.float32)
        if pts.shape[0] >= 3:
            _, _, vt = cv2.SVDecomp(pts.T @ pts)
            vx, vy = float(vt[0, 0]), float(vt[0, 1])
            orient_deg = float(np.degrees(np.arctan2(vy, vx)))

    roi = mask[y1 : y2 + 1, x1 : x2 + 1]
    h_prof = roi.mean(axis=1).astype(float).tolist() if roi.size else []
    v_prof = roi.mean(axis=0).astype(float).tolist() if roi.size else []

    return OpeningGeometry(
        bbox_x1=x1,
        bbox_y1=y1,
        bbox_x2=x2,
        bbox_y2=y2,
        width_px=bw,
        height_px=bh,
        area_px=area,
        centroid_x=cx,
        centroid_y=cy,
        orientation_deg=orient_deg,
        solidity=float(solidity),
        extent=float(extent),
        frame_area_fraction=float(area / max(1, h * w)),
        horizontal_profile_mean=h_prof[:256],
        vertical_profile_mean=v_prof[:256],
    )


def _corona_mask(interior: np.ndarray, corona_px: int = 12) -> np.ndarray:
    k = 2 * corona_px + 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dil = cv2.dilate(interior, ker)
    return np.where((dil > 0) & (interior == 0), 255, 0).astype(np.uint8)


def _contrast_and_depth(
    background: np.ndarray,
    mask_interior: np.ndarray,
    blur_k: int,
) -> Tuple[ContrastStats, np.ndarray]:
    ill = estimate_illumination(background, blur_k)
    depth = 255.0 - ill.astype(np.float32)
    corona = _corona_mask(mask_interior, 12)
    ins = background[mask_interior > 0]
    co = background[corona > 0]
    if ins.size == 0:
        ins = np.array([0.0], dtype=np.float32)
    if co.size == 0:
        co = np.array([0.0], dtype=np.float32)

    def pct(a: np.ndarray, p: float) -> float:
        return float(np.percentile(a, p)) if a.size else 0.0

    cnt = _largest_contour(mask_interior)
    grad_mag_border = 0.0
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    gmag = cv2.magnitude(gx, gy)
    if cnt is not None:
        border = np.zeros_like(mask_interior)
        cv2.drawContours(border, [cnt], -1, 255, 2)
        vals = gmag[border > 0]
        grad_mag_border = float(np.mean(vals)) if vals.size else 0.0

    d_in = depth[mask_interior > 0]
    border_mask = np.zeros_like(mask_interior)
    if cnt is not None:
        cv2.drawContours(border_mask, [cnt], -1, 255, 3)
    d_bd = depth[border_mask > 0]

    return (
        ContrastStats(
            mean_inside=float(np.mean(ins)),
            mean_corona=float(np.mean(co)),
            delta_mean=float(np.mean(ins) - np.mean(co)),
            p10_inside=pct(ins, 10),
            p50_inside=pct(ins, 50),
            p90_inside=pct(ins, 90),
            p10_corona=pct(co, 10),
            p50_corona=pct(co, 50),
            p90_corona=pct(co, 90),
            mean_gradient_mag_on_contour=grad_mag_border,
            depth_mean_inside=float(np.mean(d_in)) if d_in.size else 0.0,
            depth_mean_border=float(np.mean(d_bd)) if d_bd.size else 0.0,
            depth_p90_inside=float(np.percentile(d_in, 90)) if d_in.size else 0.0,
        ),
        depth,
    )


def _sample_frame_indices(frame_count: int, n: int, uniform: bool) -> List[int]:
    if frame_count <= 0:
        return []
    n = max(1, min(n, frame_count))
    if uniform:
        return list(np.linspace(0, frame_count - 1, n).astype(int))
    return list(range(0, frame_count, max(1, frame_count // n)))[:n]


def _temporal_and_blobs(
    video_path: str,
    background: np.ndarray,
    mask_roi: np.ndarray,
    meta,
    sample_n: int,
    blur_for_diff: int = 5,
) -> Tuple[TemporalNoiseStats, BlobStats]:
    indices = _sample_frame_indices(meta.frame_count, sample_n, uniform=True)
    cap = open_video_capture(video_path)
    absdiff_samples: List[float] = []
    per_frame_medians: List[float] = []
    per_frame_means: List[float] = []
    per_frame_p95: List[float] = []
    shifts: List[float] = []
    thresh_list = [8, 12, 16, 20, 24, 28]
    fg_ratios: Dict[int, List[float]] = {t: [] for t in thresh_list}
    all_areas: List[float] = []
    blobs_per_frame: Dict[int, List[int]] = {t: [] for t in thresh_list}

    if blur_for_diff % 2 == 0:
        blur_for_diff += 1
    bg_blur = (
        cv2.GaussianBlur(background, (blur_for_diff, blur_for_diff), 0)
        if blur_for_diff > 1
        else background
    )

    used_indices: List[int] = []
    try:
        for fi in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok:
                continue
            used_indices.append(int(fi))
            gray = frame_to_gray(frame)
            gb = (
                cv2.GaussianBlur(gray, (blur_for_diff, blur_for_diff), 0)
                if blur_for_diff > 1
                else gray
            )
            diff = cv2.absdiff(gb, bg_blur)
            roi = diff[mask_roi > 0] if np.any(mask_roi > 0) else diff.ravel()
            if roi.size:
                absdiff_samples.extend(roi.tolist())
                per_frame_medians.append(float(np.median(roi)))
                per_frame_means.append(float(np.mean(roi)))
                per_frame_p95.append(float(np.percentile(roi, 95)))
            else:
                per_frame_medians.append(0.0)
                per_frame_means.append(0.0)
                per_frame_p95.append(0.0)
            shifts.append(float(np.mean(gray.astype(np.float32)) - np.mean(background.astype(np.float32))))

            for t in thresh_list:
                _, binary = cv2.threshold(diff, t, 255, cv2.THRESH_BINARY)
                binary = np.where(mask_roi > 0, binary, 0).astype(np.uint8)
                fg_ratios[t].append(float(np.mean(binary > 0)))
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                blobs_per_frame[t].append(len(contours))
                for c in contours:
                    all_areas.append(float(cv2.contourArea(c)))
    finally:
        cap.release()

    arr = np.array(absdiff_samples, dtype=np.float32) if absdiff_samples else np.zeros(1, dtype=np.float32)
    medians_arr = np.array(per_frame_medians, dtype=np.float32) if per_frame_medians else np.zeros(1)
    means_pf = np.array(per_frame_means, dtype=np.float32) if per_frame_means else np.zeros(1)
    p95_pf = np.array(per_frame_p95, dtype=np.float32) if per_frame_p95 else np.zeros(1)

    fg_by_thr = {str(t): float(np.median(fg_ratios[t])) if fg_ratios[t] else 0.0 for t in thresh_list}

    small = sum(1 for a in all_areas if 0 < a < 15)
    bigish = sum(1 for a in all_areas if a >= 15)
    dust_proxy = float(small / max(1, small + bigish))

    blob_median_count = {
        str(t): float(np.median(blobs_per_frame[t])) if blobs_per_frame[t] else 0.0 for t in thresh_list
    }
    areas_arr = np.array(all_areas, dtype=np.float32) if all_areas else np.zeros(1)
    area_samples = _deterministic_area_samples(areas_arr, 2000)

    return (
        TemporalNoiseStats(
            sample_frames=used_indices,
            per_frame_median_absdiff=per_frame_medians,
            per_frame_mean_absdiff=per_frame_means,
            per_frame_p95_absdiff=per_frame_p95,
            median_of_frame_mean_absdiff=float(np.median(means_pf)) if means_pf.size else 0.0,
            median_of_frame_p95_absdiff=float(np.median(p95_pf)) if p95_pf.size else 0.0,
            mean_absdiff_roi=float(np.mean(arr)),
            std_absdiff_roi=float(np.std(arr)),
            p50_absdiff_roi=float(np.percentile(arr, 50)),
            p90_absdiff_roi=float(np.percentile(arr, 90)),
            p95_absdiff_roi=float(np.percentile(arr, 95)),
            p99_absdiff_roi=float(np.percentile(arr, 99)),
            global_intensity_shift_mean=float(np.mean(shifts)) if shifts else 0.0,
            global_intensity_shift_std=float(np.std(shifts)) if shifts else 0.0,
            foreground_ratio_by_threshold=fg_by_thr,
            absdiff_temporal_stability_std=float(np.std(means_pf)) if means_pf.size else 0.0,
        ),
        BlobStats(
            thresholds_tested=thresh_list,
            blobs_per_frame_median=blob_median_count,
            area_samples=area_samples,
            area_p10=float(np.percentile(areas_arr, 10)) if areas_arr.size else 0.0,
            area_p25=float(np.percentile(areas_arr, 25)) if areas_arr.size else 0.0,
            area_p50=float(np.percentile(areas_arr, 50)) if areas_arr.size else 0.0,
            area_p75=float(np.percentile(areas_arr, 75)) if areas_arr.size else 0.0,
            area_p90=float(np.percentile(areas_arr, 90)) if areas_arr.size else 0.0,
            dust_proxy_high_frequency_small=dust_proxy,
        ),
    )


def build_base_valid_mask(background: np.ndarray, valid_region_cfg: Dict[str, Any], tmp_dir: Path) -> np.ndarray:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cfg_clean = _strip_mask_geometry(deepcopy(valid_region_cfg))
    run_valid_region(image=background, output_dir=tmp_dir, config=cfg_clean)
    mask = cv2.imread(str(tmp_dir / "mask.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError("scene_profile: no s'ha pogut generar mask.png base")
    return mask


def _save_blob_histogram(area_samples: List[float], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt

        path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        if area_samples:
            ax.hist(area_samples, bins=min(60, max(10, len(set(area_samples)))), color="#2a6f97", edgecolor="white")
        ax.set_xlabel("Àrea (px²)")
        ax.set_title("Distribució d'àrees de candidats (mostra)")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
    except Exception as exc:
        logger.warning("No s'ha pogut escriure histograma de blobs: %s", exc)


def build_scene_profile(
    *,
    video_path: str,
    background: np.ndarray,
    merged_config: Dict[str, Any],
    out_dir: Path,
    sample_frames: int = 48,
    write_artifacts: bool = True,
) -> Dict[str, Any]:
    """
    Genera mètriques i recomanacions. Si write_artifacts és cert, escriu PNG/CSV/JSON sota out_dir.
    """
    out_dir = Path(out_dir)
    if write_artifacts:
        out_dir.mkdir(parents=True, exist_ok=True)
    meta = read_video_meta(video_path)
    vr = merged_config.get("valid_region", {})

    if write_artifacts:
        base_dir = out_dir / "valid_region_base"
    else:
        base_dir = Path(tempfile.mkdtemp(prefix="bat_scene_vr_"))
    mask_base = build_base_valid_mask(background, vr, base_dir)
    geo = _opening_geometry(mask_base, background.shape[:2])
    blur_k = int(vr.get("blur_kernel_size", 151))
    contrast, depth = _contrast_and_depth(background, mask_base, blur_k)
    det_blur = int(merged_config.get("detection", {}).get("blur_kernel", 5))
    if det_blur % 2 == 0:
        det_blur += 1
    temporal, blobs = _temporal_and_blobs(
        video_path, background, mask_base, meta, sample_frames, blur_for_diff=det_blur
    )

    video_snap = VideoMetaSnapshot(
        input_path=str(Path(video_path).resolve()),
        video_id=meta.video_id,
        fps=meta.fps,
        frame_count=meta.frame_count,
        width=meta.width,
        height=meta.height,
    )

    profile_core: Dict[str, Any] = {
        "video": asdict(video_snap),
        "opening_geometry": asdict(geo),
        "contrast": asdict(contrast),
        "temporal_noise": asdict(temporal),
        "blob_stats": asdict(blobs),
    }

    recommended, rationale, decision_inputs = compute_recommendations(profile_core)
    profile_core["recommended"] = recommended
    profile_core["rationale"] = rationale
    profile_core["decision_inputs"] = decision_inputs

    vr_geom = deepcopy(vr)
    mg = {
        "mode": recommended.get("valid_region.mask_geometry.mode", "dilate"),
        "dilate_px": int(recommended.get("valid_region.mask_geometry.dilate_px", 12)),
        "iterations": int(recommended.get("valid_region.mask_geometry.iterations", 1)),
        "clip_to_profile_mask": bool(
            recommended.get("valid_region.mask_geometry.clip_to_profile_mask", True)
        ),
    }
    vr_geom["mask_geometry"] = mg
    profile_mask, _, _, _, _ = _horizontal_profile_mask(background, _strip_mask_geometry(deepcopy(vr)))
    mask_dilated = _apply_mask_geometry(mask_base, depth, profile_mask, vr_geom)

    if write_artifacts:
        bg_bgr = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
        vis_base = bg_bgr.copy()
        vis_base[mask_base > 0] = (vis_base[mask_base > 0] * 0.55 + np.array([0, 200, 255]) * 0.45).astype(
            np.uint8
        )
        cv2.imwrite(str(out_dir / "opening_mask_base_overlay.png"), vis_base)
        vis_dil = bg_bgr.copy()
        vis_dil[mask_dilated > 0] = (vis_dil[mask_dilated > 0] * 0.55 + np.array([0, 255, 100]) * 0.45).astype(
            np.uint8
        )
        cv2.imwrite(str(out_dir / "opening_mask_recommended_dilate_overlay.png"), vis_dil)
        cmp_vis = np.hstack([vis_base, vis_dil])
        cv2.imwrite(str(out_dir / "opening_base_vs_dilated_compare.png"), cmp_vis)

        with (out_dir / "blob_area_samples.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["area_px"])
            for a in blobs.area_samples[:5000]:
                w.writerow([a])

        _save_blob_histogram(blobs.area_samples, out_dir / "blob_area_histogram.png")

        with (out_dir / "temporal_per_frame_medians.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["frame_index", "median_absdiff_roi", "mean_absdiff_roi", "p95_absdiff_roi"])
            for fi, med, mn, p95 in zip(
                temporal.sample_frames,
                temporal.per_frame_median_absdiff,
                temporal.per_frame_mean_absdiff,
                temporal.per_frame_p95_absdiff,
            ):
                w.writerow([fi, med, mn, p95])

        with (out_dir / "temporal_sample_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["metric", "value"])
            w.writerow(["mean_absdiff_roi", temporal.mean_absdiff_roi])
            w.writerow(["std_absdiff_roi", temporal.std_absdiff_roi])
            w.writerow(["p50_absdiff_roi", temporal.p50_absdiff_roi])
            w.writerow(["p90_absdiff_roi", temporal.p90_absdiff_roi])
            w.writerow(["p95_absdiff_roi", temporal.p95_absdiff_roi])
            w.writerow(["p99_absdiff_roi", temporal.p99_absdiff_roi])
            w.writerow(["median_of_frame_mean_absdiff", temporal.median_of_frame_mean_absdiff])
            w.writerow(["median_of_frame_p95_absdiff", temporal.median_of_frame_p95_absdiff])
            w.writerow(["global_shift_mean", temporal.global_intensity_shift_mean])
            w.writerow(["global_shift_std", temporal.global_intensity_shift_std])
            w.writerow(["median_stability_std", temporal.absdiff_temporal_stability_std])

        profile_core["debug_paths"] = {
            "valid_region_base_dir": str(base_dir.resolve()),
            "mask_base_overlay": str((out_dir / "opening_mask_base_overlay.png").resolve()),
            "mask_dilated_overlay": str((out_dir / "opening_mask_recommended_dilate_overlay.png").resolve()),
            "compare": str((out_dir / "opening_base_vs_dilated_compare.png").resolve()),
            "blob_histogram": str((out_dir / "blob_area_histogram.png").resolve()),
            "temporal_per_frame_medians_csv": str((out_dir / "temporal_per_frame_medians.csv").resolve()),
        }

        decisions_payload = {
            "recommended": recommended,
            "rationale": rationale,
            "decision_inputs": decision_inputs,
        }
        write_decisions_summary(out_dir / "autotune_decisions.json", decisions_payload)

        with (out_dir / "scene_profile.json").open("w", encoding="utf-8") as fh:
            json.dump(profile_core, fh, indent=2)
    else:
        profile_core["debug_paths"] = {"valid_region_base_dir": str(base_dir.resolve()), "artifacts": "disabled"}

    return profile_core


def profile_to_serializable(profile: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(profile, default=str))


def main() -> None:
    import argparse

    from .background import compute_background_median
    from .compute import build_execution_plan
    from .config import load_config

    parser = argparse.ArgumentParser(
        prog="python -m bat_tracker.scene_profile",
        description="Perfilatge de escena i recomanacions de paràmetres (sense executar el pipeline sencer).",
    )
    parser.add_argument("--video", required=True, help="Camí al vídeo d'entrada.")
    parser.add_argument("--config", default=None, help="YAML de configuració (fusionat amb defaults).")
    parser.add_argument("--out-dir", required=True, help="Directori de sortida (scene_profile.json i artefactes).")
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=48,
        help="Nombre de frames uniformes per mètriques temporals i blobs.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    meta = read_video_meta(args.video)
    plan = build_execution_plan(cfg)
    bg = compute_background_median(
        video_path=args.video,
        meta=meta,
        sample_frames=int(cfg["background"]["sample_frames"]),
        uniform_sampling=bool(cfg["background"]["uniform_sampling"]),
        compute_device=plan.selected_device,
        strict_parity=bool(cfg.get("execution", {}).get("strict_parity", True)),
    )
    out = build_scene_profile(
        video_path=args.video,
        background=bg,
        merged_config=cfg,
        out_dir=Path(args.out_dir),
        sample_frames=args.sample_frames,
    )
    print(json.dumps({"recommended": out.get("recommended"), "rationale": out.get("rationale")}, indent=2))


if __name__ == "__main__":
    main()

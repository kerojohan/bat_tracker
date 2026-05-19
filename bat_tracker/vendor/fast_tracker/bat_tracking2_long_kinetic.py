#!/usr/bin/env python3
import argparse
import csv
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


Point = Tuple[int, float, float, float, Tuple[int, int, int, int]]


def positive_odd(value: int) -> int:
    if value <= 0:
        return 1
    return value if value % 2 == 1 else value + 1


def safe_float(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return float(value)


def ensure_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def color_for_track(track_id: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(track_id * 9973 + 17)
    return tuple(int(v) for v in rng.integers(64, 255, size=3))


def sigmoid_score(value: float, center: float, scale: float) -> float:
    if scale <= 0:
        return 1.0 if value >= center else 0.0
    return 1.0 / (1.0 + math.exp(-(value - center) / scale))


def path_length(points: Sequence[Point]) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    for prev, cur in zip(points, points[1:]):
        total += math.hypot(cur[1] - prev[1], cur[2] - prev[2])
    return total


def displacement(points: Sequence[Point]) -> float:
    if len(points) < 2:
        return 0.0
    start = points[0]
    end = points[-1]
    return math.hypot(end[1] - start[1], end[2] - start[2])


@dataclass
class Detection:
    centroid: Tuple[float, float]
    area: float
    bbox: Tuple[int, int, int, int]


@dataclass
class Track:
    track_id: int
    points: List[Point] = field(default_factory=list)
    missing: int = 0
    vx: float = 0.0
    vy: float = 0.0
    accepted: bool = False
    quality_score: float = 0.0
    suppressed: bool = False
    suppress_reason: str = ""
    merged_track_ids: List[int] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    def last_point(self) -> Point:
        return self.points[-1]

    def predict(self) -> Tuple[float, float]:
        frame_idx, x, y, _, _ = self.last_point()
        return (x + self.vx, y + self.vy)

    def update(self, frame_idx: int, detection: Detection) -> None:
        _, last_x, last_y, _, _ = self.last_point()
        new_x, new_y = detection.centroid
        self.vx = new_x - last_x
        self.vy = new_y - last_y
        self.points.append((frame_idx, new_x, new_y, detection.area, detection.bbox))
        self.missing = 0

    def mark_missing(self) -> None:
        self.missing += 1


class BatTracker:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.active_tracks: List[Track] = []
        self.finished_tracks: List[Track] = []
        self.next_track_id = 1

    def step(self, frame_idx: int, detections: List[Detection]) -> None:
        if not self.active_tracks:
            for det in detections:
                self._start_track(frame_idx, det)
            return

        if not detections:
            self._age_tracks()
            return

        cost_matrix = np.full((len(self.active_tracks), len(detections)), 1e9, dtype=np.float32)
        for i, track in enumerate(self.active_tracks):
            pred_x, pred_y = track.predict()
            _, last_x, last_y, last_area, _ = track.last_point()
            for j, det in enumerate(detections):
                dx = det.centroid[0] - pred_x
                dy = det.centroid[1] - pred_y
                dist = math.hypot(dx, dy)
                actual_jump = math.hypot(det.centroid[0] - last_x, det.centroid[1] - last_y)
                if dist > self.args.max_distance or actual_jump > self.args.max_track_speed:
                    continue
                area_ratio = max(det.area, last_area) / max(1.0, min(det.area, last_area))
                area_penalty = min(area_ratio - 1.0, 4.0) * self.args.area_cost_weight
                direction_penalty = self._direction_penalty(track, det)
                cost_matrix[i, j] = dist + area_penalty + direction_penalty

        rows, cols = linear_sum_assignment(cost_matrix)
        matched_tracks = set()
        matched_detections = set()

        for row, col in zip(rows, cols):
            cost = cost_matrix[row, col]
            if not np.isfinite(cost) or cost >= 1e8:
                continue
            track = self.active_tracks[row]
            det = detections[col]
            track.update(frame_idx, det)
            matched_tracks.add(row)
            matched_detections.add(col)

        survivors: List[Track] = []
        for idx, track in enumerate(self.active_tracks):
            if idx in matched_tracks:
                survivors.append(track)
                continue
            track.mark_missing()
            if track.missing > self.args.max_missing:
                self.finished_tracks.append(track)
            else:
                survivors.append(track)
        self.active_tracks = survivors

        for idx, det in enumerate(detections):
            if idx not in matched_detections:
                self._start_track(frame_idx, det)

    def finalize(self) -> List[Track]:
        self.finished_tracks.extend(self.active_tracks)
        self.active_tracks = []
        return self.finished_tracks

    def _start_track(self, frame_idx: int, detection: Detection) -> None:
        track = Track(track_id=self.next_track_id)
        track.points.append((frame_idx, detection.centroid[0], detection.centroid[1], detection.area, detection.bbox))
        self.next_track_id += 1
        self.active_tracks.append(track)

    def _age_tracks(self) -> None:
        survivors: List[Track] = []
        for track in self.active_tracks:
            track.mark_missing()
            if track.missing > self.args.max_missing:
                self.finished_tracks.append(track)
            else:
                survivors.append(track)
        self.active_tracks = survivors

    @staticmethod
    def _direction_penalty(track: Track, det: Detection) -> float:
        if len(track.points) < 2:
            return 0.0
        _, last_x, last_y, _, _ = track.points[-1]
        step = np.array([det.centroid[0] - last_x, det.centroid[1] - last_y], dtype=np.float32)
        velocity = np.array([track.vx, track.vy], dtype=np.float32)
        step_norm = np.linalg.norm(step)
        vel_norm = np.linalg.norm(velocity)
        if step_norm < 1e-6 or vel_norm < 1e-6:
            return 0.0
        cos_sim = float(np.clip(np.dot(step, velocity) / (step_norm * vel_norm), -1.0, 1.0))
        return (1.0 - cos_sim) * 20.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bat flight tracking for IR cave footage.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="outputs/tracking.avi")
    parser.add_argument("--report", default="outputs/tracks.csv")
    parser.add_argument("--overlay-output", default="")
    parser.add_argument("--zone-mask", default="")
    parser.add_argument("--zone-require", choices=["off", "center", "either", "full"], default="off")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--save-reference-frame", default="")
    parser.add_argument("--skip-seconds", type=float, default=0.0)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--resize", type=float, default=1.0)
    parser.add_argument("--auto-calibrate", action="store_true")
    parser.add_argument("--no-quality", action="store_true")
    parser.add_argument("--draw-rejected-long", action="store_true")
    parser.add_argument("--fg-threshold", type=float, default=180.0)
    parser.add_argument("--blur-kernel", type=int, default=3)
    parser.add_argument("--temporal-smooth", type=float, default=0.0)
    parser.add_argument("--morph-open-iters", type=int, default=1)
    parser.add_argument("--morph-close-iters", type=int, default=1)
    parser.add_argument("--min-area", type=float, default=6.0)
    parser.add_argument("--max-area", type=float, default=8000.0)
    parser.add_argument("--max-distance", type=float, default=120.0)
    parser.add_argument("--max-segment", type=float, default=150.0)
    parser.add_argument("--min-displacement", type=float, default=120.0)
    parser.add_argument("--min-path-length", type=float, default=140.0)
    parser.add_argument("--max-track-speed", type=float, default=250.0)
    parser.add_argument("--min-points", type=int, default=4)
    parser.add_argument("--max-missing", type=int, default=5)
    parser.add_argument("--quality-percentile", type=float, default=85.0)
    parser.add_argument("--quality-threshold", type=float, default=0.0)
    parser.add_argument("--area-cost-weight", type=float, default=15.0)
    parser.add_argument("--dedupe-overlap-distance", type=float, default=45.0)
    parser.add_argument("--dedupe-perp-distance", type=float, default=18.0)
    parser.add_argument("--dedupe-direction-cos", type=float, default=0.92)
    parser.add_argument("--dedupe-min-overlap-frames", type=int, default=4)
    parser.add_argument("--dedupe-min-overlap-ratio", type=float, default=0.6)
    parser.add_argument("--dedupe-full-cover-ratio", type=float, default=0.95)
    parser.add_argument("--dedupe-polyline-distance", type=float, default=20.0)
    parser.add_argument("--dedupe-polyline-cover-ratio", type=float, default=0.75)
    parser.add_argument("--dedupe-short-track-cover-ratio", type=float, default=0.6)
    parser.add_argument("--dedupe-short-track-max-points", type=int, default=8)
    parser.add_argument("--dedupe-parallel-perp-distance", type=float, default=26.0)
    parser.add_argument("--dedupe-parallel-direction-cos", type=float, default=0.98)
    parser.add_argument("--dedupe-near-full-overlap-ratio", type=float, default=0.9)
    parser.add_argument("--dedupe-bbox-contain-ratio", type=float, default=0.7)
    parser.add_argument("--dedupe-bbox-contain-frames", type=int, default=3)
    parser.add_argument("--dedupe-frame-slack", type=int, default=1)
    return parser.parse_args()


def load_zone_mask(mask_path: str, shape: Tuple[int, int]) -> Optional[np.ndarray]:
    if not mask_path:
        return None
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read zone mask: {mask_path}")
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def estimate_video_noise(cap: cv2.VideoCapture, fps: float, sample_seconds: int = 10) -> dict:
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    block_size = 10
    num_blocks = 3
    segment_len = max(1, total_frames // (num_blocks + 1)) if total_frames > 0 else int(fps * 5)

    block_variances = []

    for blk in range(num_blocks):
        start = segment_len * blk
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(block_size):
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray.astype(np.float32))
        if len(frames) >= 4:
            stack = np.stack(frames, axis=0)
            pvar = np.var(stack, axis=0)
            block_variances.append(float(np.median(pvar)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)

    if not block_variances:
        return {"noise_level": 0.0, "samples": 0}

    noise_level = float(np.median(block_variances))

    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

    return {
        "noise_level": noise_level,
        "codec_fourcc": fourcc_str,
        "samples": sum(1 for _ in block_variances),
    }


def calibrate_noise_params(args: argparse.Namespace, noise_info: dict) -> None:
    noise_level = noise_info["noise_level"]
    fourcc = noise_info.get("codec_fourcc", "")

    if noise_level > 9.0:
        args.blur_kernel = 7
        args.temporal_smooth = 0.35
        args.morph_open_iters = max(args.morph_open_iters, 2)
    elif noise_level > 6.0:
        args.blur_kernel = 5
        args.temporal_smooth = 0.2
        args.morph_open_iters = max(args.morph_open_iters, 2)
    elif noise_level > 3.0:
        args.blur_kernel = 3
        args.temporal_smooth = 0.1
    else:
        args.blur_kernel = 3
        args.temporal_smooth = 0.0

    print(f"[auto-calibrate] noise_level={noise_level:.2f} codec={fourcc} "
          f"-> blur={args.blur_kernel} temporal_smooth={args.temporal_smooth:.2f} "
          f"morph_open={args.morph_open_iters}")


def auto_calibrate(args: argparse.Namespace, width: int, height: int, fps: float, cap: Optional[cv2.VideoCapture] = None) -> None:
    if not args.auto_calibrate:
        return
    diag = math.hypot(width, height)
    fps_scale = fps / 30.0 if fps > 0 else 1.0
    args.max_distance = diag / 10.0
    args.max_segment = diag / 8.0
    args.min_displacement = diag / 8.0
    args.min_path_length = diag / 7.0
    args.max_track_speed = diag / 4.0
    args.min_points = max(4, int(round(6 * fps_scale)))
    args.max_missing = max(2, int(round(8 * fps_scale)))
    args.dedupe_overlap_distance = max(20.0, diag / 25.0)
    args.dedupe_perp_distance = max(10.0, diag / 60.0)
    args.dedupe_polyline_distance = max(12.0, diag / 55.0)
    args.dedupe_parallel_perp_distance = max(18.0, diag / 42.0)

    if cap is not None:
        noise_info = estimate_video_noise(cap, fps)
        if noise_info["samples"] > 0:
            calibrate_noise_params(args, noise_info)


def detect_blobs(
    frame_gray: np.ndarray,
    subtractor: cv2.BackgroundSubtractor,
    args: argparse.Namespace,
    zone_mask: Optional[np.ndarray],
) -> Tuple[np.ndarray, List[Detection]]:
    fgmask = subtractor.apply(frame_gray)
    _, fgmask = cv2.threshold(fgmask, args.fg_threshold, 255, cv2.THRESH_BINARY)

    if zone_mask is not None:
        fgmask = np.where(zone_mask, fgmask, 0).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    if args.morph_open_iters > 0:
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=args.morph_open_iters)
    if args.morph_close_iters > 0:
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=args.morph_close_iters)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: List[Detection] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < args.min_area or area > args.max_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cx = x + w / 2.0
        cy = y + h / 2.0
        bbox = (x, y, w, h)
        if zone_mask is not None and args.zone_require != "off":
            if not bbox_allowed(zone_mask, bbox, (cx, cy), args.zone_require):
                continue
        detections.append(Detection(centroid=(cx, cy), area=area, bbox=bbox))
    return fgmask, detections


def bbox_allowed(
    zone_mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    centroid: Tuple[float, float],
    mode: str,
) -> bool:
    x, y, w, h = bbox
    cx, cy = int(round(centroid[0])), int(round(centroid[1]))
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(zone_mask.shape[1], x + w)
    y1 = min(zone_mask.shape[0], y + h)
    if mode == "center":
        return 0 <= cx < zone_mask.shape[1] and 0 <= cy < zone_mask.shape[0] and bool(zone_mask[cy, cx])
    crop = zone_mask[y0:y1, x0:x1]
    if crop.size == 0:
        return False
    if mode == "full":
        return bool(np.all(crop))
    return bool(np.any(crop))


def compute_track_metrics(track: Track, args: argparse.Namespace) -> Dict[str, float]:
    pts = track.points
    plength = path_length(pts)
    disp = displacement(pts)
    duration_frames = max(0, pts[-1][0] - pts[0][0]) if pts else 0
    directionality = disp / plength if plength > 0 else 0.0

    segment_lengths: List[float] = []
    turn_angles: List[float] = []
    gaps = 0
    for prev, cur in zip(pts, pts[1:]):
        segment = math.hypot(cur[1] - prev[1], cur[2] - prev[2])
        segment_lengths.append(segment)
        if cur[0] - prev[0] > 1:
            gaps += cur[0] - prev[0] - 1
    for a, b, c in zip(pts, pts[1:], pts[2:]):
        v1 = np.array([b[1] - a[1], b[2] - a[2]], dtype=np.float32)
        v2 = np.array([c[1] - b[1], c[2] - b[2]], dtype=np.float32)
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        cos_sim = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        turn_angles.append(math.degrees(math.acos(cos_sim)))

    avg_speed = float(np.mean(segment_lengths)) if segment_lengths else 0.0
    moving_ratio = float(np.mean([1.0 if seg <= args.max_segment else 0.0 for seg in segment_lengths])) if segment_lengths else 0.0
    areas = np.array([p[3] for p in pts], dtype=np.float32)
    area_mean = float(np.mean(areas)) if len(areas) else 0.0
    area_std = float(np.std(areas)) if len(areas) else 0.0
    area_consistency = 1.0 / (1.0 + (area_std / max(area_mean, 1.0)))
    smoothness = 1.0 - min(float(np.mean(turn_angles)) / 180.0, 1.0) if turn_angles else 1.0
    sharp_turns = float(sum(1 for angle in turn_angles if angle > 60.0))

    return {
        "points": float(len(pts)),
        "start_frame": float(pts[0][0]) if pts else 0.0,
        "end_frame": float(pts[-1][0]) if pts else 0.0,
        "duration_frames": float(duration_frames),
        "displacement": safe_float(disp),
        "path_length": safe_float(plength),
        "directionality": safe_float(directionality),
        "avg_speed": safe_float(avg_speed),
        "moving_ratio": safe_float(moving_ratio),
        "area_mean": safe_float(area_mean),
        "area_std": safe_float(area_std),
        "area_consistency": safe_float(area_consistency),
        "smoothness": safe_float(smoothness),
        "sharp_turns": safe_float(sharp_turns),
        "gaps": float(gaps),
    }


def compute_quality_score(metrics: Dict[str, float]) -> float:
    if metrics["points"] < 4 or metrics["displacement"] < 80.0:
        return 0.0
    score = (
        0.35 * metrics["directionality"]
        + 0.35 * sigmoid_score(metrics["displacement"], center=200.0, scale=150.0)
        + 0.10 * metrics["smoothness"]
        + 0.10 * metrics["area_consistency"]
        + 0.10 * metrics["moving_ratio"]
    )
    return max(0.0, min(1.0, safe_float(score)))


def vector_cosine(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a < 1e-6 or norm_b < 1e-6:
        return 1.0
    return float(np.clip(np.dot(vec_a, vec_b) / (norm_a * norm_b), -1.0, 1.0))


def track_direction_vector(track: Track) -> np.ndarray:
    if len(track.points) < 2:
        return np.zeros(2, dtype=np.float32)
    start = track.points[0]
    end = track.points[-1]
    return np.array([end[1] - start[1], end[2] - start[2]], dtype=np.float32)


def direction_cosine(track_a: Track, track_b: Track) -> float:
    vec_a = track_direction_vector(track_a)
    vec_b = track_direction_vector(track_b)
    return vector_cosine(vec_a, vec_b)


def overlap_frame_distances(track_a: Track, track_b: Track) -> List[float]:
    points_b = {point[0]: point for point in track_b.points}
    distances: List[float] = []
    for point_a in track_a.points:
        point_b = points_b.get(point_a[0])
        if point_b is None:
            continue
        distances.append(math.hypot(point_a[1] - point_b[1], point_a[2] - point_b[2]))
    return distances


def alignment_axis(track_a: Track, track_b: Track) -> np.ndarray:
    axis = track_direction_vector(track_a) + track_direction_vector(track_b)
    norm = np.linalg.norm(axis)
    if norm < 1e-6:
        axis = track_direction_vector(track_a)
        norm = np.linalg.norm(axis)
    if norm < 1e-6:
        return np.array([1.0, 0.0], dtype=np.float32)
    return axis / norm


def aligned_point_pairs(
    track_a: Track,
    track_b: Track,
    frame_slack: int,
) -> List[Tuple[Point, Point, int]]:
    pairs: List[Tuple[Point, Point, int]] = []
    used_b: set[int] = set()
    for point_a in track_a.points:
        best_idx: Optional[int] = None
        best_key: Optional[Tuple[int, float]] = None
        for idx_b, point_b in enumerate(track_b.points):
            if idx_b in used_b:
                continue
            frame_gap = abs(point_a[0] - point_b[0])
            if frame_gap > frame_slack:
                continue
            dist = math.hypot(point_a[1] - point_b[1], point_a[2] - point_b[2])
            key = (frame_gap, dist)
            if best_key is None or key < best_key:
                best_key = key
                best_idx = idx_b
        if best_idx is not None:
            used_b.add(best_idx)
            pairs.append((point_a, track_b.points[best_idx], abs(point_a[0] - track_b.points[best_idx][0])))
    return pairs


def duplicate_alignment_stats(
    track_a: Track,
    track_b: Track,
    frame_slack: int,
) -> Dict[str, float]:
    pairs = aligned_point_pairs(track_a, track_b, frame_slack)
    if not pairs:
        return {"pairs": 0.0, "overlap_ratio": 0.0, "mean_perp": 1e9, "mean_long_abs": 1e9}

    axis = alignment_axis(track_a, track_b)
    perp_values: List[float] = []
    long_values: List[float] = []
    for point_a, point_b, _ in pairs:
        delta = np.array([point_b[1] - point_a[1], point_b[2] - point_a[2]], dtype=np.float32)
        longitudinal = float(np.dot(delta, axis))
        perpendicular = float(np.linalg.norm(delta - longitudinal * axis))
        perp_values.append(perpendicular)
        long_values.append(abs(longitudinal))

    overlap_ratio = len(pairs) / max(1, min(len(track_a.points), len(track_b.points)))
    return {
        "pairs": float(len(pairs)),
        "overlap_ratio": float(overlap_ratio),
        "mean_perp": float(np.mean(perp_values)),
        "mean_long_abs": float(np.mean(long_values)),
    }


def best_duplicate_alignment_stats(
    track_a: Track,
    track_b: Track,
    frame_slack: int,
) -> Dict[str, float]:
    stats_ab = duplicate_alignment_stats(track_a, track_b, frame_slack)
    stats_ba = duplicate_alignment_stats(track_b, track_a, frame_slack)
    ranked = sorted(
        [stats_ab, stats_ba],
        key=lambda stats: (
            -stats["overlap_ratio"],
            -stats["pairs"],
            stats["mean_perp"],
            stats["mean_long_abs"],
        ),
    )
    return ranked[0]


def point_to_segment_distance(point: Point, seg_a: Point, seg_b: Point) -> float:
    px = np.array([point[1], point[2]], dtype=np.float32)
    a = np.array([seg_a[1], seg_a[2]], dtype=np.float32)
    b = np.array([seg_b[1], seg_b[2]], dtype=np.float32)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-9:
        return float(np.linalg.norm(px - a))
    t = float(np.clip(np.dot(px - a, ab) / denom, 0.0, 1.0))
    proj = a + t * ab
    return float(np.linalg.norm(px - proj))


def bbox_contains(outer: Tuple[int, int, int, int], inner: Tuple[int, int, int, int], pad: int = 2) -> bool:
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner
    return (
        ox - pad <= ix
        and oy - pad <= iy
        and ox + ow + pad >= ix + iw
        and oy + oh + pad >= iy + ih
    )


def bbox_containment_stats(track_a: Track, track_b: Track) -> Dict[str, float]:
    points_a = {point[0]: point for point in track_a.points}
    points_b = {point[0]: point for point in track_b.points}
    common_frames = sorted(set(points_a) & set(points_b))
    if not common_frames:
        return {"common_frames": 0.0, "contain_ratio": 0.0}

    contain_hits = 0
    for frame in common_frames:
        bbox_a = points_a[frame][4]
        bbox_b = points_b[frame][4]
        if bbox_contains(bbox_a, bbox_b) or bbox_contains(bbox_b, bbox_a):
            contain_hits += 1

    return {
        "common_frames": float(len(common_frames)),
        "contain_ratio": contain_hits / len(common_frames),
    }


def polyline_coverage_stats(reference: Track, probe: Track, corridor_distance: float) -> Dict[str, float]:
    if len(reference.points) < 2 or not probe.points:
        return {"cover_ratio": 0.0, "mean_distance": 1e9}

    distances: List[float] = []
    for point in probe.points:
        best = min(
            point_to_segment_distance(point, seg_a, seg_b)
            for seg_a, seg_b in zip(reference.points, reference.points[1:])
        )
        distances.append(best)

    covered = sum(1 for dist in distances if dist <= corridor_distance)
    return {
        "cover_ratio": covered / len(distances),
        "mean_distance": float(np.mean(distances)),
    }


def effective_polyline_cover_ratio(track: Track, args: argparse.Namespace) -> float:
    if len(track.points) <= args.dedupe_short_track_max_points:
        return min(args.dedupe_polyline_cover_ratio, args.dedupe_short_track_cover_ratio)
    return args.dedupe_polyline_cover_ratio


def join_is_smooth(prefix: Sequence[Point], suffix: Sequence[Point], args: argparse.Namespace) -> bool:
    if not prefix or not suffix:
        return True
    gap_frames = suffix[0][0] - prefix[-1][0]
    if gap_frames <= 0:
        return True

    bridge_vec = np.array([suffix[0][1] - prefix[-1][1], suffix[0][2] - prefix[-1][2]], dtype=np.float32)
    bridge_speed = np.linalg.norm(bridge_vec) / max(1, gap_frames)
    if bridge_speed > args.max_track_speed:
        return False

    if len(prefix) >= 2:
        prev_vec = np.array([prefix[-1][1] - prefix[-2][1], prefix[-1][2] - prefix[-2][2]], dtype=np.float32)
        if vector_cosine(prev_vec, bridge_vec) < 0.1:
            return False

    if len(suffix) >= 2:
        next_vec = np.array([suffix[1][1] - suffix[0][1], suffix[1][2] - suffix[0][2]], dtype=np.float32)
        if vector_cosine(bridge_vec, next_vec) < 0.1:
            return False

    return True


def merge_duplicate_tracks(leader: Track, candidate: Track, args: argparse.Namespace) -> bool:
    leader_start = leader.points[0][0]
    leader_end = leader.points[-1][0]
    prefix = [point for point in candidate.points if point[0] < leader_start]
    suffix = [point for point in candidate.points if point[0] > leader_end]

    merged = False
    new_points = list(leader.points)

    if prefix and join_is_smooth(prefix, new_points, args):
        new_points = prefix + new_points
        merged = True

    if suffix and join_is_smooth(new_points, suffix, args):
        new_points = new_points + suffix
        merged = True

    if not merged:
        return False

    deduped: Dict[int, Point] = {}
    for point in new_points:
        deduped[point[0]] = point
    leader.points = [deduped[frame] for frame in sorted(deduped)]
    leader.merged_track_ids.append(candidate.track_id)
    return True


def duplicate_match_details(
    leader: Track,
    candidate: Track,
    args: argparse.Namespace,
) -> Optional[Dict[str, float | bool | str]]:
    stats = best_duplicate_alignment_stats(leader, candidate, args.dedupe_frame_slack)
    if stats["pairs"] < args.dedupe_min_overlap_frames:
        return None

    overlap_ratio = stats["overlap_ratio"]
    if overlap_ratio < args.dedupe_min_overlap_ratio:
        return None

    poly_stats = polyline_coverage_stats(leader, candidate, args.dedupe_polyline_distance)
    bbox_stats = bbox_containment_stats(leader, candidate)
    required_cover = effective_polyline_cover_ratio(candidate, args)
    dir_cos = direction_cosine(leader, candidate)
    full_cover = overlap_ratio >= args.dedupe_full_cover_ratio
    fully_parallel = (
        full_cover
        and dir_cos >= args.dedupe_parallel_direction_cos
        and stats["mean_perp"] <= args.dedupe_parallel_perp_distance
        and stats["mean_long_abs"] <= args.dedupe_overlap_distance
    )
    nearly_fully_parallel = (
        overlap_ratio >= args.dedupe_near_full_overlap_ratio
        and dir_cos >= args.dedupe_parallel_direction_cos
        and stats["mean_perp"] <= args.dedupe_parallel_perp_distance
        and stats["mean_long_abs"] <= args.dedupe_overlap_distance
    )
    fully_parallel_short = (
        len(candidate.points) <= args.dedupe_short_track_max_points
        and full_cover
        and dir_cos >= args.dedupe_parallel_direction_cos
        and stats["mean_perp"] <= args.dedupe_parallel_perp_distance
        and stats["mean_long_abs"] <= args.dedupe_overlap_distance * 1.2
    )
    bbox_contained = (
        bbox_stats["common_frames"] >= args.dedupe_bbox_contain_frames
        and bbox_stats["contain_ratio"] >= args.dedupe_bbox_contain_ratio
        and dir_cos >= 0.75
        and overlap_ratio >= args.dedupe_min_overlap_ratio
    )

    if stats["mean_perp"] > args.dedupe_perp_distance:
        if poly_stats["cover_ratio"] < required_cover and not fully_parallel and not nearly_fully_parallel and not fully_parallel_short and not bbox_contained:
            return None
    if not full_cover and stats["mean_long_abs"] > args.dedupe_overlap_distance:
        if poly_stats["cover_ratio"] < required_cover and not bbox_contained:
            return None
    if dir_cos < args.dedupe_direction_cos:
        if poly_stats["cover_ratio"] < required_cover or poly_stats["mean_distance"] > args.dedupe_polyline_distance * 0.6:
            if not bbox_contained:
                return None
        if candidate.metrics.get("start_frame", 0.0) < leader.metrics.get("start_frame", 0.0):
            return None
        if candidate.metrics.get("end_frame", 0.0) > leader.metrics.get("end_frame", 0.0):
            return None
    if poly_stats["cover_ratio"] < required_cover and dir_cos < args.dedupe_direction_cos and not fully_parallel and not nearly_fully_parallel and not fully_parallel_short and not bbox_contained:
        return None

    return {
        "overlap_ratio": overlap_ratio,
        "mean_perp": stats["mean_perp"],
        "mean_long_abs": stats["mean_long_abs"],
        "poly_cover": poly_stats["cover_ratio"],
        "poly_mean": poly_stats["mean_distance"],
        "bbox_contain": bbox_stats["contain_ratio"],
        "dir_cos": dir_cos,
        "full_cover": full_cover,
    }


def track_priority(track: Track) -> Tuple[float, float, float]:
    return (
        track.metrics.get("path_length", 0.0),
        track.metrics.get("points", 0.0),
        track.quality_score,
    )


def suppress_duplicate_tracks(tracks: List[Track], args: argparse.Namespace) -> int:
    accepted_tracks = [track for track in tracks if track.accepted]
    accepted_tracks.sort(key=track_priority, reverse=True)
    by_id = {track.track_id: track for track in accepted_tracks}
    edge_details: Dict[Tuple[int, int], Dict[str, float | bool | str]] = {}
    adjacency: Dict[int, set[int]] = {track.track_id: set() for track in accepted_tracks}

    for idx, first in enumerate(accepted_tracks):
        for second in accepted_tracks[idx + 1 :]:
            leader, candidate = (first, second) if track_priority(first) >= track_priority(second) else (second, first)
            details = duplicate_match_details(leader, candidate, args)
            if details is None:
                continue
            edge_details[(leader.track_id, candidate.track_id)] = details
            adjacency[first.track_id].add(second.track_id)
            adjacency[second.track_id].add(first.track_id)

    suppressed = 0
    seen: set[int] = set()
    for track in accepted_tracks:
        if track.track_id in seen:
            continue
        stack = [track.track_id]
        component: List[int] = []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.append(current)
            stack.extend(adjacency[current] - seen)

        if len(component) <= 1:
            continue

        members = sorted((by_id[track_id] for track_id in component), key=track_priority, reverse=True)
        leader = members[0]
        for candidate in members[1:]:
            details = edge_details.get((leader.track_id, candidate.track_id))
            if details is None:
                details = edge_details.get((candidate.track_id, leader.track_id))
            merged = merge_duplicate_tracks(leader, candidate, args)
            candidate.accepted = False
            candidate.suppressed = True
            if details is None:
                candidate.suppress_reason = f"duplicate_family_of={leader.track_id};via_family=True"
            else:
                candidate.suppress_reason = (
                    f"duplicate_family_of={leader.track_id};"
                    f"overlap_ratio={float(details['overlap_ratio']):.2f};"
                    f"mean_perp={float(details['mean_perp']):.2f};"
                    f"mean_long_abs={float(details['mean_long_abs']):.2f};"
                    f"poly_cover={float(details['poly_cover']):.2f};"
                    f"poly_mean={float(details['poly_mean']):.2f};"
                    f"bbox_contain={float(details['bbox_contain']):.2f};"
                    f"dir_cos={float(details['dir_cos']):.3f}"
                )
            if merged:
                candidate.suppress_reason += ";merged_extremes=True"
            suppressed += 1

        if leader.merged_track_ids:
            leader.metrics = compute_track_metrics(leader, args)
            leader.quality_score = 1.0 if args.no_quality else compute_quality_score(leader.metrics)

    return suppressed


def assign_quality(tracks: List[Track], args: argparse.Namespace) -> None:
    if not tracks:
        return
    for track in tracks:
        track.metrics = compute_track_metrics(track, args)
        track.quality_score = 1.0 if args.no_quality else compute_quality_score(track.metrics)

    threshold = args.quality_threshold
    if threshold <= 0 and not args.no_quality:
        threshold = float(np.percentile([track.quality_score for track in tracks], args.quality_percentile))
    elif args.no_quality:
        threshold = 0.0

    for track in tracks:
        metrics = track.metrics
        base_ok = (
            metrics["points"] >= args.min_points
            and metrics["displacement"] >= args.min_displacement
            and metrics["path_length"] >= args.min_path_length
        )
        track.accepted = bool(base_ok and track.quality_score >= threshold)

    suppress_duplicate_tracks(tracks, args)


def draw_overlay(frame: np.ndarray, tracks: Sequence[Track], include_rejected: bool) -> np.ndarray:
    canvas = frame.copy()
    for track in tracks:
        if not track.accepted and not include_rejected:
            continue
        color = color_for_track(track.track_id) if track.accepted else (128, 128, 128)
        pts = np.array([[int(round(p[1])), int(round(p[2]))] for p in track.points], dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(canvas, [pts], False, color, 2, cv2.LINE_AA)
        elif len(pts) == 1:
            cv2.circle(canvas, tuple(pts[0]), 2, color, -1, cv2.LINE_AA)
    return canvas


def write_report(report_path: str, tracks: Sequence[Track]) -> None:
    ensure_dir(report_path)
    fieldnames = [
        "track_id",
        "accepted",
        "quality_score",
        "suppressed",
        "suppress_reason",
        "merged_track_ids",
        "points",
        "start_frame",
        "end_frame",
        "duration_frames",
        "displacement",
        "path_length",
        "directionality",
        "avg_speed",
        "moving_ratio",
        "area_mean",
        "area_std",
        "area_consistency",
        "smoothness",
        "sharp_turns",
        "gaps",
    ]
    ordered = sorted(tracks, key=lambda track: track.metrics.get("path_length", 0.0), reverse=True)
    with open(report_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for track in ordered:
            row = {
                "track_id": track.track_id,
                "accepted": track.accepted,
                "quality_score": f"{track.quality_score:.4f}",
                "suppressed": track.suppressed,
                "suppress_reason": track.suppress_reason,
                "merged_track_ids": ",".join(str(track_id) for track_id in track.merged_track_ids),
            }
            for key in fieldnames[6:]:
                row[key] = f"{track.metrics.get(key, 0.0):.4f}"
            writer.writerow(row)


def build_tracks_by_frame(tracks: Sequence[Track], accepted_only: bool, include_rejected_long: bool) -> Dict[int, List[Track]]:
    frame_map: Dict[int, List[Track]] = {}
    for track in tracks:
        if track.accepted:
            pass
        elif not include_rejected_long or len(track.points) < 2:
            continue
        if accepted_only and not track.accepted:
            continue
        for point in track.points:
            frame_map.setdefault(point[0], []).append(track)
    return frame_map


def annotate_video(
    video_path: str,
    output_path: str,
    tracks: Sequence[Track],
    start_frame: int,
    max_frame: Optional[int],
    resize: float,
    fps_out: float,
    include_rejected_long: bool,
) -> None:
    cap = cv2.VideoCapture(video_path)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Cannot read video for annotation.")
    if resize != 1.0:
        frame = cv2.resize(frame, None, fx=resize, fy=resize, interpolation=cv2.INTER_AREA)

    ensure_dir(output_path)
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps_out,
        (width, height),
    )
    tracks_per_frame = build_tracks_by_frame(tracks, accepted_only=False, include_rejected_long=include_rejected_long)

    frame_idx = start_frame
    while ok:
        if max_frame is not None and frame_idx > max_frame:
            break
        draw_frame = frame.copy()
        for track in tracks_per_frame.get(frame_idx, []):
            point = next((p for p in track.points if p[0] == frame_idx), None)
            if point is None:
                continue
            _, x, y, _, bbox = point
            color = color_for_track(track.track_id) if track.accepted else (128, 128, 128)
            bx, by, bw, bh = bbox
            cv2.rectangle(draw_frame, (bx, by), (bx + bw, by + bh), color, 1)
            cv2.circle(draw_frame, (int(round(x)), int(round(y))), 2, color, -1, cv2.LINE_AA)
            trail = np.array([[int(round(p[1])), int(round(p[2]))] for p in track.points if p[0] <= frame_idx], dtype=np.int32)
            if len(trail) >= 2:
                cv2.polylines(draw_frame, [trail], False, color, 2, cv2.LINE_AA)
            cv2.putText(
                draw_frame,
                f"{track.track_id}:{track.quality_score:.2f}",
                (bx, max(12, by - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            draw_frame,
            f"frame={frame_idx}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(draw_frame)
        if max_frame is not None and frame_idx == max_frame:
            break
        ok, frame = cap.read()
        frame_idx += 1
        if ok and resize != 1.0:
            frame = cv2.resize(frame, None, fx=resize, fy=resize, interpolation=cv2.INTER_AREA)

    writer.release()
    cap.release()


def main() -> None:
    args = parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    auto_calibrate(args, source_width * args.resize, source_height * args.resize, source_fps, cap)

    args.blur_kernel = positive_odd(args.blur_kernel)
    args.temporal_smooth = float(np.clip(args.temporal_smooth, 0.0, 0.95))

    start_frame = max(0, int(round(args.skip_seconds * source_fps)))
    max_frames = int(round(args.max_seconds * source_fps)) if args.max_seconds > 0 else 0
    max_frame = start_frame + max_frames - 1 if max_frames > 0 else None
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Cannot read initial frame from video.")

    if args.resize != 1.0:
        frame = cv2.resize(frame, None, fx=args.resize, fy=args.resize, interpolation=cv2.INTER_AREA)
    if args.save_reference_frame:
        ensure_dir(args.save_reference_frame)
        cv2.imwrite(args.save_reference_frame, frame)

    frame_height, frame_width = frame.shape[:2]
    zone_mask = load_zone_mask(args.zone_mask, (frame_height, frame_width))

    subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
    tracker = BatTracker(args)

    overlay_reference = frame.copy()
    prev_gray: Optional[np.ndarray] = None
    frame_idx = start_frame

    while ok:
        if max_frame is not None and frame_idx > max_frame:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if args.blur_kernel > 1:
            gray = cv2.GaussianBlur(gray, (args.blur_kernel, args.blur_kernel), 0)
        if prev_gray is not None and args.temporal_smooth > 0:
            gray = cv2.addWeighted(gray, 1.0 - args.temporal_smooth, prev_gray, args.temporal_smooth, 0.0)
        fgmask, detections = detect_blobs(gray, subtractor, args, zone_mask)
        tracker.step(frame_idx, detections)

        if args.show:
            debug = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
            cv2.imshow("tracking-mask", debug)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        prev_gray = gray
        ok, frame = cap.read()
        frame_idx += 1
        if ok and args.resize != 1.0:
            frame = cv2.resize(frame, None, fx=args.resize, fy=args.resize, interpolation=cv2.INTER_AREA)

    cap.release()
    if args.show:
        cv2.destroyAllWindows()

    tracks = tracker.finalize()
    assign_quality(tracks, args)
    write_report(args.report, tracks)

    if args.overlay_output:
        ensure_dir(args.overlay_output)
        overlay = draw_overlay(overlay_reference, tracks, include_rejected=args.draw_rejected_long)
        cv2.imwrite(args.overlay_output, overlay)

    annotate_video(
        video_path=args.video,
        output_path=args.output,
        tracks=tracks,
        start_frame=start_frame,
        max_frame=max_frame,
        resize=args.resize,
        fps_out=source_fps,
        include_rejected_long=args.draw_rejected_long,
    )

    accepted = sum(1 for track in tracks if track.accepted)
    print(f"Processed tracks: total={len(tracks)} accepted={accepted} report={args.report} output={args.output}")


if __name__ == "__main__":
    main()

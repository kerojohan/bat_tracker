from __future__ import annotations

import argparse
import csv
import json
from math import hypot
from pathlib import Path

import cv2
import numpy as np


def _parse_xy(value: str) -> np.ndarray:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected X,Y")
    return np.array([float(parts[0]), float(parts[1])], dtype=np.float64)


def _build_motion_heatmap(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    threshold: int,
    blur_kernel: int,
) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    start_frame = int(round(start_sec * fps))
    end_frame = int(round(end_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ok, previous = cap.read()
    if not ok:
        raise RuntimeError(f"Cannot read frame {start_frame} from {video_path}")

    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    acc = np.zeros_like(previous_gray, dtype=np.float32)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for _frame_idx in range(start_frame + 1, end_frame + 1):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_blur = cv2.GaussianBlur(previous_gray, (blur_kernel, blur_kernel), 0)
        gray_blur = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        diff = cv2.absdiff(gray_blur, prev_blur)
        _, binary = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
        acc += diff.astype(np.float32) * (binary.astype(np.float32) / 255.0)
        previous_gray = gray

    cap.release()
    return cv2.GaussianBlur(acc, (9, 9), 0), fps


def _reconstruct_path_from_corridor(
    heatmap: np.ndarray,
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    *,
    percentile: float,
    corridor_width: float,
    bins: int,
    y_max: float,
) -> list[tuple[float, float]]:
    vector = end_xy - start_xy
    length = float(np.linalg.norm(vector))
    if length <= 1e-6:
        raise ValueError("start and end seeds must be different")
    unit = vector / length

    height, width = heatmap.shape[:2]
    ys, xs = np.indices((height, width))
    rel_x = xs.astype(np.float64) - start_xy[0]
    rel_y = ys.astype(np.float64) - start_xy[1]
    projection = rel_x * unit[0] + rel_y * unit[1]
    perpendicular = np.abs(rel_x * (-unit[1]) + rel_y * unit[0])

    nonzero = heatmap[heatmap > 0]
    if nonzero.size == 0:
        raise RuntimeError("motion heatmap is empty")
    threshold = float(np.percentile(nonzero, percentile))
    corridor = (
        (projection >= -40.0)
        & (projection <= length + 80.0)
        & (perpendicular <= corridor_width)
        & (heatmap >= threshold)
    )
    if y_max > 0:
        corridor &= ys < y_max

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(corridor.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel_open)
    route_y, route_x = np.where(mask > 0)
    if route_x.size < 10:
        raise RuntimeError("not enough heatmap pixels inside the motion corridor")

    route_projection = projection[route_y, route_x]
    route_weights = heatmap[route_y, route_x] ** 1.35
    route_points = np.column_stack([route_x, route_y]).astype(np.float64)

    path: list[tuple[float, float]] = [tuple(start_xy)]
    for lo, hi in zip(np.linspace(-20.0, length + 50.0, bins)[:-1], np.linspace(-20.0, length + 50.0, bins)[1:]):
        selected = (route_projection >= lo) & (route_projection < hi)
        mid_projection = (lo + hi) / 2.0
        if selected.sum() < 5:
            if 0.0 <= mid_projection <= length:
                center = start_xy + unit * mid_projection
                path.append((float(center[0]), float(center[1])))
            continue
        xy = np.average(route_points[selected], axis=0, weights=route_weights[selected])
        line_xy = start_xy + unit * float(np.dot(xy - start_xy, unit))
        xy = 0.75 * xy + 0.25 * line_xy
        path.append((float(xy[0]), float(xy[1])))
    path.append(tuple(end_xy))
    return _smooth_path(path)


def _smooth_path(path: list[tuple[float, float]]) -> list[tuple[float, float]]:
    smoothed: list[tuple[float, float]] = []
    for idx in range(len(path)):
        if idx == 0 or idx == len(path) - 1:
            smoothed.append(path[idx])
            continue
        lo = max(0, idx - 2)
        hi = min(len(path), idx + 3)
        smoothed.append(
            (
                sum(point[0] for point in path[lo:hi]) / float(hi - lo),
                sum(point[1] for point in path[lo:hi]) / float(hi - lo),
            )
        )
    return smoothed


def _path_length(path: list[tuple[float, float]]) -> float:
    return sum(hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:]))


def _draw_path(
    image: np.ndarray,
    path: list[tuple[float, float]],
    title: str,
) -> np.ndarray:
    out = image.copy()
    color = (0, 255, 255)
    for a, b in zip(path, path[1:]):
        cv2.line(out, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), color, 5, cv2.LINE_AA)
    for idx, (a, b) in enumerate(zip(path, path[1:])):
        if idx % 4 == 0:
            cv2.arrowedLine(
                out,
                (int(a[0]), int(a[1])),
                (int(b[0]), int(b[1])),
                color,
                2,
                cv2.LINE_AA,
                tipLength=0.25,
            )
    cv2.circle(out, (int(path[0][0]), int(path[0][1])), 12, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(out, (int(path[-1][0]), int(path[-1][1])), 12, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.putText(out, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct one fast bat event from an inter-frame motion heatmap.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start-sec", required=True, type=float)
    parser.add_argument("--end-sec", required=True, type=float)
    parser.add_argument("--start-xy", required=True, type=_parse_xy, help="Start seed as X,Y")
    parser.add_argument("--end-xy", required=True, type=_parse_xy, help="End seed as X,Y")
    parser.add_argument("--background", type=Path, default=None)
    parser.add_argument("--threshold", type=int, default=14)
    parser.add_argument("--blur-kernel", type=int, default=5)
    parser.add_argument("--percentile", type=float, default=76.0)
    parser.add_argument("--corridor-width", type=float, default=180.0)
    parser.add_argument("--bins", type=int, default=22)
    parser.add_argument("--y-max", type=float, default=830.0)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    heatmap, fps = _build_motion_heatmap(args.input, args.start_sec, args.end_sec, args.threshold, args.blur_kernel)
    path = _reconstruct_path_from_corridor(
        heatmap,
        args.start_xy,
        args.end_xy,
        percentile=args.percentile,
        corridor_width=args.corridor_width,
        bins=args.bins,
        y_max=args.y_max,
    )

    cap = cv2.VideoCapture(str(args.input))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(((args.start_sec + args.end_sec) / 2.0) * fps)))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        frame = np.zeros((*heatmap.shape, 3), dtype=np.uint8)

    heat_norm = np.clip(heatmap / max(1e-6, float(heatmap.max())) * 255, 0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_TURBO)
    heat_overlay = cv2.addWeighted(frame, 0.62, heat_color, 0.6, 0)
    heat_overlay = _draw_path(heat_overlay, path, "ruta reconstruida por heatmap")
    cv2.imwrite(str(args.output / "heatmap_track_overlay.png"), heat_overlay)

    if args.background is not None:
        bg = cv2.imread(str(args.background), cv2.IMREAD_GRAYSCALE)
    else:
        bg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if bg is None:
        raise RuntimeError(f"Cannot read background: {args.background}")
    background_overlay = _draw_path(cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR), path, "ruta reconstruida por heatmap")
    cv2.imwrite(str(args.output / "heatmap_track_on_background.png"), background_overlay)

    xs = [point[0] for point in path]
    ys = [point[1] for point in path]
    pad = 140
    x1 = max(0, int(min(xs) - pad))
    y1 = max(0, int(min(ys) - pad))
    x2 = min(background_overlay.shape[1], int(max(xs) + pad))
    y2 = min(background_overlay.shape[0], int(max(ys) + pad))
    cv2.imwrite(str(args.output / "heatmap_track_crop.png"), background_overlay[y1:y2, x1:x2])

    with (args.output / "heatmap_track.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["idx", "time_sec_est", "x", "y"])
        for idx, point in enumerate(path):
            time_sec = args.start_sec + (args.end_sec - args.start_sec) * idx / max(1, len(path) - 1)
            writer.writerow([idx, round(time_sec, 4), round(point[0], 2), round(point[1], 2)])

    path_len = _path_length(path)
    displacement = hypot(path[-1][0] - path[0][0], path[-1][1] - path[0][1])
    meta = {
        "input": str(args.input.resolve()),
        "start_sec": args.start_sec,
        "end_sec": args.end_sec,
        "start_xy": [float(args.start_xy[0]), float(args.start_xy[1])],
        "end_xy": [float(args.end_xy[0]), float(args.end_xy[1])],
        "points": len(path),
        "displacement_px": displacement,
        "path_length_px": path_len,
        "straightness": displacement / path_len if path_len > 0 else 0.0,
        "outputs": {
            "heatmap_track_overlay": str((args.output / "heatmap_track_overlay.png").resolve()),
            "heatmap_track_on_background": str((args.output / "heatmap_track_on_background.png").resolve()),
            "heatmap_track_crop": str((args.output / "heatmap_track_crop.png").resolve()),
            "heatmap_track_csv": str((args.output / "heatmap_track.csv").resolve()),
        },
    }
    (args.output / "heatmap_track_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

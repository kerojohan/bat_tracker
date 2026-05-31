#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass
class TrackPoint:
    frame: int
    x: float
    y: float


@dataclass
class TrackSummary:
    direction: str = ""
    num_detections: int = 0


class TrackDebugGUI:
    def __init__(
        self,
        video_path: Path,
        tracks_csv: Path,
        events_csv: Path | None,
        mask_input: Path | None,
        out_dir: Path,
        window: str = "bat_tracker_debug",
    ) -> None:
        self.video_path = video_path
        self.tracks_csv = tracks_csv
        self.events_csv = events_csv
        self.mask_input = mask_input
        self.out_dir = out_dir
        self.window = window

        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0)

        self.track_points = self._load_tracks(tracks_csv)
        self.track_summaries = self._load_events(events_csv) if events_csv else {}
        self.points_by_frame = self._points_by_frame(self.track_points)

        self.play = False
        self.current_frame = 0
        self.wait_ms = 30
        self.trail_len = 20
        self.line_thickness = 2
        self.show_only_entry_exit = False
        self.show_only_selected = False
        self.min_track_len = 1

        self.selected_tracks: List[int] = []
        self.suppressed_tracks: set[int] = set()
        self.merge_pairs: set[Tuple[int, int]] = set()

        self.paint_mode = "draw"
        self.brush_px = 12
        self.mouse_down = False
        self.mouse_x = 0
        self.mouse_y = 0

        self.mask = self._init_mask()
        self.frame_cache: Dict[int, np.ndarray] = {}

    def _init_mask(self) -> np.ndarray:
        if self.mask_input and self.mask_input.exists():
            mask = cv2.imread(str(self.mask_input), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"Cannot read mask: {self.mask_input}")
            if mask.shape[:2] != (self.height, self.width):
                mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            return np.where(mask > 0, 255, 0).astype(np.uint8)
        return np.zeros((self.height, self.width), dtype=np.uint8)

    @staticmethod
    def _load_tracks(path: Path) -> Dict[int, List[TrackPoint]]:
        by_track: Dict[int, List[TrackPoint]] = {}
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                tid = int(row["track_id"])
                by_track.setdefault(tid, []).append(
                    TrackPoint(
                        frame=int(row["frame"]),
                        x=float(row["x"]),
                        y=float(row["y"]),
                    )
                )
        for tid in by_track:
            by_track[tid].sort(key=lambda p: p.frame)
        return by_track

    @staticmethod
    def _load_events(path: Path) -> Dict[int, TrackSummary]:
        summaries: Dict[int, TrackSummary] = {}
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                tid = int(row["track_id"])
                summaries[tid] = TrackSummary(
                    direction=str(row.get("direction", "")).strip().lower(),
                    num_detections=int(float(row.get("num_detections", "0") or "0")),
                )
        return summaries

    @staticmethod
    def _points_by_frame(track_points: Dict[int, List[TrackPoint]]) -> Dict[int, List[Tuple[int, TrackPoint]]]:
        by_frame: Dict[int, List[Tuple[int, TrackPoint]]] = {}
        for tid, pts in track_points.items():
            for pt in pts:
                by_frame.setdefault(pt.frame, []).append((tid, pt))
        return by_frame

    def _get_frame(self, idx: int) -> np.ndarray:
        if idx in self.frame_cache:
            return self.frame_cache[idx].copy()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.frame_cache[idx] = frame
        return frame.copy()

    def _track_visible(self, tid: int) -> bool:
        if tid in self.suppressed_tracks:
            return False
        summary = self.track_summaries.get(tid)
        n = summary.num_detections if summary else len(self.track_points.get(tid, []))
        if n < self.min_track_len:
            return False
        if self.show_only_entry_exit:
            if not summary or summary.direction not in {"entry", "exit"}:
                return False
        if self.show_only_selected and tid not in self.selected_tracks:
            return False
        return True

    def _track_color(self, tid: int) -> Tuple[int, int, int]:
        summary = self.track_summaries.get(tid)
        if summary:
            if summary.direction == "entry":
                return (0, 255, 0)
            if summary.direction == "exit":
                return (0, 0, 255)
        seed = np.random.default_rng(tid)
        b, g, r = seed.integers(40, 255, size=3).tolist()
        return int(b), int(g), int(r)

    def _draw_overlay(self, frame_idx: int) -> np.ndarray:
        frame = self._get_frame(frame_idx)
        out = frame.copy()

        tint = np.zeros_like(out)
        tint[self.mask > 0] = (0, 200, 200)
        out = cv2.addWeighted(out, 1.0, tint, 0.28, 0)
        contours, _ = cv2.findContours((self.mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)

        start_frame = max(0, frame_idx - self.trail_len)
        for tid, pts in self.track_points.items():
            if not self._track_visible(tid):
                continue
            seg = [p for p in pts if start_frame <= p.frame <= frame_idx]
            if len(seg) < 1:
                continue
            color = self._track_color(tid)
            poly = np.array([[int(round(p.x)), int(round(p.y))] for p in seg], dtype=np.int32)
            if len(poly) >= 2:
                cv2.polylines(out, [poly], False, color, self.line_thickness, cv2.LINE_AA)
            cv2.circle(out, tuple(poly[-1]), 2, color, -1, cv2.LINE_AA)
            if tid in self.selected_tracks:
                cv2.circle(out, tuple(poly[-1]), 7, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(out, str(tid), (poly[-1][0] + 5, poly[-1][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        self._draw_status(out, frame_idx)
        if self.mouse_down:
            cv2.circle(out, (self.mouse_x, self.mouse_y), self.brush_px, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    def _draw_status(self, img: np.ndarray, frame_idx: int) -> None:
        selected = ",".join(str(t) for t in self.selected_tracks) if self.selected_tracks else "-"
        msg = (
            f"frame {frame_idx+1}/{self.frame_count}  play={'on' if self.play else 'off'}  "
            f"trail={self.trail_len}  min_len={self.min_track_len}  brush={self.brush_px}  mode={self.paint_mode}  "
            f"entry_exit_only={'on' if self.show_only_entry_exit else 'off'}  selected={selected}"
        )
        cv2.rectangle(img, (0, 0), (img.shape[1], 48), (0, 0, 0), -1)
        cv2.putText(img, msg, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(
            img,
            "space play | a/d frame | z/x trail | j/k min_len | [,] brush | t entry/exit | v selected-only | s suppress | m merge | p draw/erase | e export | q quit",
            (8, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

    def _paint(self, x: int, y: int) -> None:
        value = 255 if self.paint_mode == "draw" else 0
        cv2.circle(self.mask, (x, y), self.brush_px, value, -1, cv2.LINE_AA)

    def _nearest_track_at_cursor(self, frame_idx: int, x: int, y: int, max_dist_px: float = 20.0) -> int | None:
        candidates = self.points_by_frame.get(frame_idx, [])
        if not candidates:
            return None
        best_tid = None
        best_d = float("inf")
        for tid, pt in candidates:
            if not self._track_visible(tid):
                continue
            d = float(np.hypot(pt.x - x, pt.y - y))
            if d < best_d:
                best_d = d
                best_tid = tid
        if best_tid is None or best_d > max_dist_px:
            return None
        return best_tid

    def _export_outputs(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        mask_path = self.out_dir / "manual_valid_region_mask.png"
        review_path = self.out_dir / "track_review_actions.json"
        merge_csv = self.out_dir / "suggested_merges.csv"
        suppress_csv = self.out_dir / "suggested_suppressions.csv"

        cv2.imwrite(str(mask_path), self.mask)
        payload = {
            "video_path": str(self.video_path.resolve()),
            "tracks_csv": str(self.tracks_csv.resolve()),
            "events_csv": str(self.events_csv.resolve()) if self.events_csv else "",
            "selected_tracks": sorted(self.selected_tracks),
            "suppressed_tracks": sorted(self.suppressed_tracks),
            "merge_pairs": sorted([list(pair) for pair in self.merge_pairs]),
            "params": {
                "trail_len": self.trail_len,
                "min_track_len": self.min_track_len,
                "entry_exit_only": self.show_only_entry_exit,
            },
        }
        review_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        with merge_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["track_a", "track_b"])
            for a, b in sorted(self.merge_pairs):
                writer.writerow([a, b])

        with suppress_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["track_id"])
            for tid in sorted(self.suppressed_tracks):
                writer.writerow([tid])

        print(f"[export] mask: {mask_path}")
        print(f"[export] review: {review_path}")
        print(f"[export] merges: {merge_csv}")
        print(f"[export] suppressions: {suppress_csv}")

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        self.mouse_x = x
        self.mouse_y = y
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_down = True
            self._paint(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.mouse_down:
            self._paint(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_down = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            tid = self._nearest_track_at_cursor(self.current_frame, x, y)
            if tid is None:
                return
            if tid in self.selected_tracks:
                self.selected_tracks = [t for t in self.selected_tracks if t != tid]
            else:
                self.selected_tracks.append(tid)
                self.selected_tracks = self.selected_tracks[-2:]

    def run(self) -> None:
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, min(1400, self.width), min(900, self.height + 60))
        cv2.setMouseCallback(self.window, self._on_mouse)

        while True:
            vis = self._draw_overlay(self.current_frame)
            cv2.imshow(self.window, vis)

            key = cv2.waitKey(self.wait_ms if self.play else 0) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                self.play = not self.play
            elif key == ord("d"):
                self.current_frame = min(self.frame_count - 1, self.current_frame + 1)
            elif key == ord("a"):
                self.current_frame = max(0, self.current_frame - 1)
            elif key == ord("z"):
                self.trail_len = max(1, self.trail_len - 2)
            elif key == ord("x"):
                self.trail_len = min(300, self.trail_len + 2)
            elif key == ord("j"):
                self.min_track_len = max(1, self.min_track_len - 1)
            elif key == ord("k"):
                self.min_track_len = min(999, self.min_track_len + 1)
            elif key == ord("["):
                self.brush_px = max(1, self.brush_px - 1)
            elif key == ord("]"):
                self.brush_px = min(200, self.brush_px + 1)
            elif key == ord("t"):
                self.show_only_entry_exit = not self.show_only_entry_exit
            elif key == ord("v"):
                self.show_only_selected = not self.show_only_selected
            elif key == ord("p"):
                self.paint_mode = "erase" if self.paint_mode == "draw" else "draw"
            elif key == ord("c"):
                self.mask.fill(0)
            elif key == ord("s"):
                for tid in self.selected_tracks:
                    if tid in self.suppressed_tracks:
                        self.suppressed_tracks.remove(tid)
                    else:
                        self.suppressed_tracks.add(tid)
            elif key == ord("m"):
                if len(self.selected_tracks) == 2:
                    a, b = sorted(self.selected_tracks)
                    pair = (a, b)
                    if pair in self.merge_pairs:
                        self.merge_pairs.remove(pair)
                    else:
                        self.merge_pairs.add(pair)
            elif key == ord("e"):
                self._export_outputs()

            if self.play:
                self.current_frame += 1
                if self.current_frame >= self.frame_count:
                    self.current_frame = self.frame_count - 1
                    self.play = False

        self.cap.release()
        cv2.destroyAllWindows()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="track_debug_gui",
        description="GUI for manual cave-mask drawing and temporal track validation.",
    )
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--tracks", required=True, help="Path to tracks.csv")
    parser.add_argument("--events", default="", help="Path to events.csv (optional, used for entry/exit filter)")
    parser.add_argument("--mask", default="", help="Initial mask path (optional)")
    parser.add_argument("--out-dir", default="track_debug_review", help="Output directory for exported mask/actions")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    gui = TrackDebugGUI(
        video_path=Path(args.video),
        tracks_csv=Path(args.tracks),
        events_csv=Path(args.events) if args.events else None,
        mask_input=Path(args.mask) if args.mask else None,
        out_dir=Path(args.out_dir),
    )
    gui.run()


if __name__ == "__main__":
    main()

"""
Revisió visual de la geometria de valid_region vs. semàntica inside/outside dels esdeveniments.

Genera panells per track (inici/fi × màscara actual / màscara invertida) i CSV amb camps automàtics.
No aplica heurístiques noves al pipeline principal.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .tracker import TrackPoint
from .video import frame_to_gray
from .video import open_video_capture
from .video import read_video_meta


def _read_gray_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame_to_gray(frame)


def _to_bgr(gray: np.ndarray) -> np.ndarray:
    if gray.ndim == 2:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return gray


def _overlay_mask(
    gray_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    inside_bgr: Tuple[int, int, int],
    alpha: float = 0.35,
) -> np.ndarray:
    """Destaca mask>0 amb color semitransparent."""
    out = gray_bgr.astype(np.float32)
    m = (mask > 0).astype(np.float32)[..., None]
    col = np.array(inside_bgr, dtype=np.float32).reshape(1, 1, 3)
    out = out * (1.0 - alpha * m) + col * (alpha * m)
    return np.clip(out, 0, 255).astype(np.uint8)


def _point_in_mask(x: float, y: float, mask: np.ndarray) -> bool:
    xi, yi = int(round(x)), int(round(y))
    if yi < 0 or yi >= mask.shape[0] or xi < 0 or xi >= mask.shape[1]:
        return False
    return bool(mask[yi, xi] > 0)


def _draw_track_on_image(
    img: np.ndarray,
    points: List[TrackPoint],
    *,
    frame_idx: int,
    radius: int = 6,
) -> None:
    """Dibuixa segment fins al punt d'aquest frame (inclos) si existeix."""
    pts = sorted(points, key=lambda p: p.frame)
    sub = [p for p in pts if p.frame <= frame_idx]
    if not sub:
        return
    for i in range(len(sub) - 1):
        p0, p1 = sub[i], sub[i + 1]
        cv2.line(
            img,
            (int(round(p0.x)), int(round(p0.y))),
            (int(round(p1.x)), int(round(p1.y))),
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    last = sub[-1]
    cv2.circle(img, (int(round(last.x)), int(round(last.y))), radius, (0, 255, 255), 2, cv2.LINE_AA)
    first = pts[0]
    cv2.circle(img, (int(round(first.x)), int(round(first.y))), radius + 2, (0, 200, 0), 2, cv2.LINE_AA)
    cv2.circle(img, (int(round(pts[-1].x)), int(round(pts[-1].y))), radius + 2, (0, 0, 255), 2, cv2.LINE_AA)


def _panel(
    gray: np.ndarray,
    mask: np.ndarray,
    inv_mask: np.ndarray,
    track_points: List[TrackPoint],
    frame_idx: int,
    *,
    use_inverted: bool,
) -> np.ndarray:
    g = _to_bgr(gray)
    m = inv_mask if use_inverted else mask
    # Màscara actual: cian a la zona valid_region (opening). Invertida: magenta al complement (prova de semàntica inversa).
    color = (255, 128, 0) if use_inverted else (0, 200, 255)
    out = _overlay_mask(g, m, inside_bgr=color, alpha=0.4)
    _draw_track_on_image(out, track_points, frame_idx=frame_idx)
    return out


def _hstack_scale(a: np.ndarray, b: np.ndarray, max_w: int = 1600) -> np.ndarray:
    h = max(a.shape[0], b.shape[0])
    def resize(img, target_h):
        if img.shape[0] == target_h:
            return img
        s = target_h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    a = resize(a, h)
    b = resize(b, h)
    return np.hstack([a, b])


def _vstack(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w = max(a.shape[1], b.shape[1])
    def pad(img):
        if img.shape[1] == w:
            return img
        p = np.zeros((img.shape[0], w, 3), dtype=np.uint8)
        p[:, : img.shape[1]] = img
        return p
    return np.vstack([pad(a), pad(b)])


def run_review(run_dir: Path, video_path: Path, out_subdir: str = "geometry_review") -> Path:
    run_dir = run_dir.resolve()
    out_dir = run_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_path = run_dir / "valid_region" / "mask.png"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Màscara no trobada: {mask_path}")

    bg_path = run_dir / "background.png"
    background = cv2.imread(str(bg_path), cv2.IMREAD_GRAYSCALE)
    if background is None:
        raise FileNotFoundError(f"Fons no trobat: {bg_path}")

    inv_mask = cv2.bitwise_not(mask)

    # Context: fons mediana + màscara
    bg_bgr = _to_bgr(background)
    ctx_std = _overlay_mask(bg_bgr, mask, inside_bgr=(0, 200, 255), alpha=0.45)
    ctx_inv = _overlay_mask(bg_bgr, inv_mask, inside_bgr=(255, 128, 0), alpha=0.35)
    cv2.imwrite(str(out_dir / "00_context_mask_on_median.png"), ctx_std)
    cv2.imwrite(str(out_dir / "01_context_inverted_on_median.png"), ctx_inv)
    cv2.imwrite(str(out_dir / "02_context_side_by_side.png"), _hstack_scale(ctx_std, ctx_inv))

    tracks_csv = run_dir / "tracks.csv"
    events_csv = run_dir / "events.csv"
    if not tracks_csv.exists():
        raise FileNotFoundError(tracks_csv)

    by_track: Dict[int, List[TrackPoint]] = defaultdict(list)
    with tracks_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            by_track[int(row["track_id"])].append(
                TrackPoint(
                    video_id=row["video_id"],
                    track_id=int(row["track_id"]),
                    frame=int(row["frame"]),
                    time_sec=float(row["time_sec"]),
                    x=float(row["x"]),
                    y=float(row["y"]),
                    vx=float(row["vx"]),
                    vy=float(row["vy"]),
                    bbox_x1=int(row["bbox_x1"]),
                    bbox_y1=int(row["bbox_y1"]),
                    bbox_x2=int(row["bbox_x2"]),
                    bbox_y2=int(row["bbox_y2"]),
                    area=float(row["area"]),
                )
            )

    events_by_id: Dict[int, Dict] = {}
    if events_csv.exists():
        with events_csv.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                events_by_id[int(row["track_id"])] = row

    cap = open_video_capture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No es pot obrir el vídeo: {video_path}")

    review_rows: List[Dict[str, object]] = []

    try:
        for track_id in sorted(by_track.keys()):
            tps = sorted(by_track[track_id], key=lambda p: p.frame)
            start = tps[0]
            end = tps[-1]
            ev = events_by_id.get(track_id, {})
            direction = ev.get("direction", "")
            s_in = str(ev.get("start_in_valid_region", "")).lower() == "true"
            e_in = str(ev.get("end_in_valid_region", "")).lower() == "true"

            g0 = _read_gray_frame(cap, start.frame)
            g1 = _read_gray_frame(cap, end.frame)
            if g0 is None or g1 is None:
                continue

            p00 = _panel(g0, mask, inv_mask, tps, start.frame, use_inverted=False)
            p01 = _panel(g0, mask, inv_mask, tps, start.frame, use_inverted=True)
            p10 = _panel(g1, mask, inv_mask, tps, end.frame, use_inverted=False)
            p11 = _panel(g1, mask, inv_mask, tps, end.frame, use_inverted=True)

            top = _hstack_scale(p00, p01)
            bot = _hstack_scale(p10, p11)
            grid = _vstack(top, bot)
            legend_h = 120
            leg = np.full((legend_h, grid.shape[1], 3), 240, dtype=np.uint8)
            txt = (
                f"track {track_id} | auto={direction} | start_in_mask={s_in} end_in_mask={e_in} | "
                f"cols: [frame start + mask actual] [frame start + complement invertit] "
                f"| row2: mateix per frame final"
            )
            y0 = 28
            for i, line in enumerate([txt[i : i + 110] for i in range(0, len(txt), 110)][:3]):
                cv2.putText(leg, line, (8, y0 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
            cv2.putText(
                leg,
                "Verd=primer punt | Vermell=ultim punt | Groc=trajecte fins al frame del panell",
                (8, y0 + 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (60, 60, 60),
                1,
                cv2.LINE_AA,
            )
            final_img = np.vstack([leg, grid])
            out_png = out_dir / f"track_{track_id:04d}_review.png"
            cv2.imwrite(str(out_png), final_img)

            dx = end.x - start.x
            dy = end.y - start.y
            review_rows.append(
                {
                    "track_id": track_id,
                    "direction_auto": direction,
                    "frame_start": start.frame,
                    "frame_end": end.frame,
                    "x_start": round(start.x, 2),
                    "y_start": round(start.y, 2),
                    "x_end": round(end.x, 2),
                    "y_end": round(end.y, 2),
                    "start_in_mask_raw": s_in,
                    "end_in_mask_raw": e_in,
                    "displacement_dx": round(dx, 2),
                    "displacement_dy": round(dy, 2),
                    "num_points": len(tps),
                    "review_image": str(out_png.name),
                    "visual_motion_estimate": "",
                    "auto_label_visual_ok": "",
                    "visual_review_notes": "",
                    "visual_category": "",
                }
            )
    finally:
        cap.release()

    # Mostra de vídeo per comparar màscara sobre escena real (mig del clip)
    vmeta = read_video_meta(str(video_path))
    sample_f = min(max(0, vmeta.frame_count // 2), max(0, vmeta.frame_count - 1))
    cap2 = open_video_capture(str(video_path))
    if cap2.isOpened():
        g = _read_gray_frame(cap2, sample_f)
        cap2.release()
        if g is not None:
            gb = _to_bgr(g)
            s = _overlay_mask(gb, mask, inside_bgr=(0, 200, 255), alpha=0.4)
            inv = _overlay_mask(gb, inv_mask, inside_bgr=(255, 128, 0), alpha=0.35)
            cv2.imwrite(str(out_dir / "03_video_frame_sample_mask_compare.png"), _hstack_scale(s, inv))

    csv_path = out_dir / "geometry_review_per_track.csv"
    if review_rows:
        keys = list(review_rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(review_rows)

    summary_path = out_dir / "geometry_review_README.txt"
    summary_path.write_text(
        "Fitxers:\n"
        "- 00_context_mask_on_median.png : valid_region (mask.png) sobre fons mediana; "
        "tinta sobre píxels mask>0 = 'inside' al pipeline.\n"
        "- 01_context_inverted_on_median.png : complement (prova de semàntica invertida).\n"
        "- 02_context_side_by_side.png : comparació.\n"
        "- 03_video_frame_sample_mask_compare.png : fotograma del vídeo (mig de clip) amb comparació.\n"
        "- track_XXXX_review.png : per cada track, inici (fila 1) i fi (fila 2); "
        "columnes [mask actual] [complement invertit].\n"
        "- geometry_review_per_track.csv : dades automàtics; columnes visual_* buides per omplir a mà.\n"
        "  Després de revisió humana es pot afegir geometry_review_visual_verdict.csv i "
        "geometry_review_category_summary.json (no generats automàticament).\n",
        encoding="utf-8",
    )

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Revisió visual valid_region vs esdeveniments.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Directori de sortida d'una corrida bat-tracker")
    parser.add_argument("--video", type=Path, default=None, help="Vídeo font (per defecte meta.json del run)")
    parser.add_argument("--out-subdir", type=str, default="geometry_review")
    args = parser.parse_args()

    video = args.video
    if video is None:
        meta_path = args.run_dir / "meta.json"
        if not meta_path.exists():
            raise SystemExit("Cal --video o meta.json al run-dir.")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        video = Path(meta["video"]["input_path"])

    out = run_review(args.run_dir, video, out_subdir=args.out_subdir)
    print(json.dumps({"geometry_review_dir": str(out.resolve())}, indent=2))


if __name__ == "__main__":
    main()

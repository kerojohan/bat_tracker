"""
Barrido fino: solo valid_region.mask_geometry dilate_px. Resto idéntico al YAML base.

Genera dilate_fine_sweep_summary.csv: exits, eventos, tracks 6 y 232, área de máscara,
IoU frente a la máscara en dilate_px=28 (estabilidad geométrica).
Los falsos positivos visuales requieren revisión humana de overlays; se incluye events_total como proxy.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import yaml


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config ha de ser un mapping YAML")
    return data


def _mask_area_and_iou(mask_path: Path, ref_bin: np.ndarray | None) -> tuple[int, float | str]:
    if not mask_path.exists():
        return 0, ""
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return 0, ""
    bin_a = (m > 0).astype(np.uint8)
    area = int(np.sum(bin_a))
    if ref_bin is None or ref_bin.shape != m.shape:
        return area, ""
    inter = np.logical_and(bin_a > 0, ref_bin > 0).sum()
    union = np.logical_or(bin_a > 0, ref_bin > 0).sum()
    iou = float(inter) / float(union) if union > 0 else 0.0
    return area, round(iou, 4)


def _events_stats(events_csv: Path, watch: List[int]) -> Dict[str, Any]:
    if not events_csv.exists():
        return {"exits": 0, "events_total": 0, "watch": {}}
    rows = list(csv.DictReader(events_csv.open(newline="", encoding="utf-8")))
    exits = sum(1 for r in rows if r.get("direction") == "exits")
    by_id = {int(r["track_id"]): r for r in rows}
    w: Dict[str, Any] = {}
    for tid in watch:
        r = by_id.get(tid)
        if r is None:
            w[str(tid)] = {"present": False, "direction": ""}
        else:
            w[str(tid)] = {"present": True, "direction": r.get("direction", "")}
    return {"exits": exits, "events_total": len(rows), "watch": w}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--reference-px", type=int, default=28, help="IoU de cada máscara frente a este dilate_px")
    parser.add_argument("--pixels", type=str, default="20,24,28,32,36")
    args = parser.parse_args()

    pixels = [int(x.strip()) for x in args.pixels.split(",") if x.strip()]
    base = _load_yaml(args.base_config)
    args.out_root.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parent.parent

    summary_rows: List[Dict[str, Any]] = []
    watch_ids = [6, 232]

    for px in pixels:
        name = f"px{px:03d}"
        run_dir = args.out_root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = deepcopy(base)
        vr = cfg.setdefault("valid_region", {})
        vr["mask_geometry"] = {
            "mode": "dilate",
            "dilate_px": px,
            "iterations": 1,
            "clip_to_profile_mask": True,
        }
        cfg_path = run_dir / "_sweep_config.yaml"
        with cfg_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)

        cmd = [
            str(args.python),
            "-m",
            "bat_tracker",
            "--input",
            str(args.video),
            "--output",
            str(run_dir),
            "--config",
            str(cfg_path),
        ]
        print(f"[sweep] dilate_px={px}", flush=True)
        proc = subprocess.run(cmd, cwd=str(repo_root))
        if proc.returncode != 0:
            summary_rows.append({"dilate_px": px, "ok": False})
            continue

        st = _events_stats(run_dir / "events.csv", watch_ids)
        w6 = st["watch"].get("6", {})
        w232 = st["watch"].get("232", {})
        area, _ = _mask_area_and_iou(run_dir / "valid_region" / "mask.png", None)

        summary_rows.append(
            {
                "dilate_px": px,
                "ok": True,
                "exits": st["exits"],
                "events_total": st["events_total"],
                "track_6_present": w6.get("present", False),
                "track_6_direction": w6.get("direction", ""),
                "track_232_present": w232.get("present", False),
                "track_232_direction": w232.get("direction", ""),
                "mask_area_px": area,
            }
        )

    ref_name = f"px{args.reference_px:03d}"
    ref_mask_path = args.out_root / ref_name / "valid_region" / "mask.png"
    ref_img = cv2.imread(str(ref_mask_path), cv2.IMREAD_GRAYSCALE)
    ref_bin = (ref_img > 0).astype(np.uint8) if ref_img is not None else None

    min_ev = min((r["events_total"] for r in summary_rows if r.get("ok")), default=0)

    for row in summary_rows:
        if not row.get("ok"):
            continue
        px = row["dilate_px"]
        mp = args.out_root / f"px{px:03d}" / "valid_region" / "mask.png"
        area, iou = _mask_area_and_iou(mp, ref_bin)
        row["mask_area_px"] = area
        row["mask_iou_vs_ref"] = 1.0 if px == args.reference_px else iou
        row["events_total_delta_vs_min_sweep"] = row["events_total"] - min_ev

    out_csv = args.out_root / "dilate_fine_sweep_summary.csv"
    if summary_rows:
        keys = [
            "dilate_px",
            "ok",
            "exits",
            "events_total",
            "events_total_delta_vs_min_sweep",
            "track_6_present",
            "track_6_direction",
            "track_232_present",
            "track_232_direction",
            "mask_area_px",
            "mask_iou_vs_ref",
        ]
        with out_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in summary_rows:
                w.writerow({k: r.get(k, "") for k in keys})

    note = args.out_root / "dilate_fine_sweep_NOTES.txt"
    note.write_text(
        "Falsos positius visuals: revisar manualment tracks_overlay.png de cada px***.\n"
        "events_total i events_total_delta_vs_min_sweep serveixen com a proxy de fragmentació/ruïna.\n"
        f"mask_iou_vs_ref: IoU de la màscara respecte px{args.reference_px:03d} (estabilitat geomètrica).\n",
        encoding="utf-8",
    )

    print(json.dumps({"summary_csv": str(out_csv.resolve())}, indent=2))


if __name__ == "__main__":
    main()

"""
Estudi comparatiu: variants de valid_region.mask_geometry sense canviar detecció/tracking/events.

Executa diverses configuracions i resumeix exits + tracks 6 i 232.
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

import yaml


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config YAML must be a mapping")
    return data


def _count_exits_and_tracks(events_csv: Path, watch_ids: List[int]) -> Dict[str, Any]:
    if not events_csv.exists():
        return {"exits": 0, "error": "missing events.csv"}
    rows = list(csv.DictReader(events_csv.open(newline="", encoding="utf-8")))
    exits = sum(1 for r in rows if r.get("direction") == "exits")
    by_id = {int(r["track_id"]): r for r in rows}
    watch = {}
    for tid in watch_ids:
        r = by_id.get(tid)
        if r is None:
            watch[str(tid)] = {"present": False}
        else:
            watch[str(tid)] = {
                "present": True,
                "direction": r.get("direction"),
                "start_in_valid_region": r.get("start_in_valid_region"),
                "end_in_valid_region": r.get("end_in_valid_region"),
            }
    return {"exits": exits, "events_total": len(rows), "watch": watch}


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparar mask_geometry sobre el mateix pipeline.")
    parser.add_argument("--base-config", type=Path, required=True, help="YAML base (p.ex. config.retallat_recommended.yaml)")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True, help="Directori pare per a subcarpetes per experiment")
    parser.add_argument("--python", type=Path, default=Path(sys.executable), help="Intèrpret Python per bat_tracker")
    args = parser.parse_args()

    base = _load_yaml(args.base_config)
    args.out_root.mkdir(parents=True, exist_ok=True)

    experiments: List[tuple[str, Dict[str, Any] | None]] = [
        ("01_baseline", None),
        ("02_dilate_px28", {"mode": "dilate", "dilate_px": 28, "clip_to_profile_mask": True}),
        ("03_dilate_px55", {"mode": "dilate", "dilate_px": 55, "clip_to_profile_mask": True}),
        ("04_dilate_px85", {"mode": "dilate", "dilate_px": 85, "clip_to_profile_mask": True}),
        ("05_convex_hull", {"mode": "convex_hull", "clip_to_profile_mask": True}),
        ("06_gradient_band", {
            "mode": "gradient_band_union",
            "gradient_percentile": 86.0,
            "band_dilate_px": 10,
            "band_close_px": 21,
            "clip_to_profile_mask": True,
        }),
    ]

    summary_rows: List[Dict[str, Any]] = []
    watch = [6, 232]

    for name, mg in experiments:
        cfg = deepcopy(base)
        vr = cfg.setdefault("valid_region", {})
        if mg is None:
            vr.pop("mask_geometry", None)
            label = "baseline (sense mask_geometry)"
        else:
            vr["mask_geometry"] = mg
            label = json.dumps(mg, sort_keys=True)

        run_dir = args.out_root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = run_dir / "_study_config.yaml"
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
        print(f"[study] Running {name} ...", flush=True)
        repo_root = Path(__file__).resolve().parent.parent
        proc = subprocess.run(cmd, cwd=str(repo_root))
        if proc.returncode != 0:
            print(f"[study] ERROR {name} exit {proc.returncode}", flush=True)
            summary_rows.append({"experiment": name, "label": label, "ok": False})
            continue

        events_path = run_dir / "events.csv"
        stats = _count_exits_and_tracks(events_path, watch)
        meta_path = run_dir / "meta.json"
        mg_meta = ""
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            vr_meta = meta.get("valid_region", {})
            mg_meta = str(vr_meta.get("mask_geometry_mode", ""))

        row = {
            "experiment": name,
            "mask_geometry_config": label,
            "mask_geometry_mode_meta": mg_meta,
            "ok": True,
            "exits": stats.get("exits", 0),
            "events_total": stats.get("events_total", 0),
            "track_6_direction": stats.get("watch", {}).get("6", {}).get("direction", ""),
            "track_6_present": stats.get("watch", {}).get("6", {}).get("present", False),
            "track_232_direction": stats.get("watch", {}).get("232", {}).get("direction", ""),
            "track_232_present": stats.get("watch", {}).get("232", {}).get("present", False),
            "overlay_png": str((run_dir / "tracks_overlay.png").resolve()),
        }
        summary_rows.append(row)

    summary_path = args.out_root / "mask_geometry_study_summary.csv"
    if summary_rows:
        keys = list(summary_rows[0].keys())
        with summary_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(summary_rows)

    print(json.dumps({"summary_csv": str(summary_path.resolve()), "rows": len(summary_rows)}, indent=2))


if __name__ == "__main__":
    main()

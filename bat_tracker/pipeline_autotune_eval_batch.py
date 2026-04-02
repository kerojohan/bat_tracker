"""
Avaluació batch: pipeline complet en mode manual vs scene_auto_tune (sense canviar regles ni el pipeline).

Escriu YAML efímer per cas/mode, executa run_pipeline, agrega mètriques i informes.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import yaml

from .pipeline import run_pipeline
from .scene_auto_tune import get_nested

logger = logging.getLogger(__name__)

SAT_MANUAL = {
    "enabled": False,
    "write_profile": False,
    "use_recommended_values": False,
    "overrides_allowed": False,
}

SAT_AUTOTUNE = {
    "enabled": True,
    "write_profile": False,
    "use_recommended_values": True,
    "overrides_allowed": True,
}


def _parse_videos_arg(items: List[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"Esperat CASE=ruta, rebut: {raw!r}")
        case, path = raw.split("=", 1)
        case = case.strip()
        path = path.strip()
        if not case or not path:
            raise ValueError(f"CASE o ruta buits: {raw!r}")
        out.append((case, path))
    return out


def _read_user_yaml(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    if not isinstance(doc, dict):
        raise ValueError("El YAML de configuració ha de ser un mapping")
    return doc


def _write_run_config(user_doc: Dict[str, Any], sat: Dict[str, Any], dest: Path) -> None:
    merged = dict(user_doc)
    base_sat = user_doc.get("scene_auto_tune")
    if isinstance(base_sat, dict):
        extra = {k: v for k, v in base_sat.items() if k not in sat}
        sat_out = {**extra, **sat}
    else:
        sat_out = dict(sat)
    if "sample_frames" not in sat_out:
        sat_out["sample_frames"] = 48
    if "profile_subdir" not in sat_out:
        sat_out["profile_subdir"] = "scene_profile"
    merged["scene_auto_tune"] = sat_out
    out_prev = merged.get("output")
    if isinstance(out_prev, dict):
        merged["output"] = {**out_prev, "progress_enabled": False}
    else:
        merged["output"] = {"progress_enabled": False}
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(merged, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _count_events(events_csv: Path) -> Dict[str, int]:
    if not events_csv.is_file():
        return {
            "events_total": 0,
            "exits": 0,
            "enters": 0,
            "outside": 0,
            "inside": 0,
        }
    counts: Counter[str] = Counter()
    with events_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            d = str(row.get("direction", "")).strip().lower()
            if d:
                counts[d] += 1
    return {
        "events_total": sum(counts.values()),
        "exits": int(counts.get("exits", 0)),
        "enters": int(counts.get("enters", 0)),
        "outside": int(counts.get("outside", 0)),
        "inside": int(counts.get("inside", 0)),
    }


def _mask_area_pixels(run_dir: Path, valid_subdir: str) -> Optional[int]:
    mask_path = run_dir / valid_subdir / "mask.png"
    if not mask_path.is_file():
        return None
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return int(cv2.countNonZero(m))


def _effective_params(meta: Dict[str, Any]) -> Dict[str, Any]:
    params = meta.get("parameters") or {}
    return {
        "dilate_px": get_nested(params, "valid_region.mask_geometry.dilate_px"),
        "diff_threshold": get_nested(params, "detection.diff_threshold"),
        "min_area": get_nested(params, "detection.min_area"),
        "max_missed": get_nested(params, "tracking.max_missed"),
        "max_distance": get_nested(params, "tracking.max_distance"),
    }


def _run_mode(
    *,
    case: str,
    mode: str,
    video_path: Path,
    config_path: Path,
    run_dir: Path,
    user_doc: Dict[str, Any],
    sat: Dict[str, Any],
) -> Dict[str, Any]:
    cfg_path = run_dir / "_eval_config.yaml"
    _write_run_config(user_doc, sat, cfg_path)
    meta = run_pipeline(
        input_video=str(video_path.resolve()),
        output_dir=str(run_dir.resolve()),
        config_path=str(cfg_path.resolve()),
    )
    meta_path = run_dir / "meta.json"
    with meta_path.open("r", encoding="utf-8") as fh:
        meta_disk = json.load(fh)
    metrics = meta_disk.get("metrics") or {}
    events_csv = run_dir / "events.csv"
    ev = _count_events(events_csv)
    valid_sub = str((meta_disk.get("parameters") or {}).get("valid_region", {}).get("output_subdir", "valid_region"))
    mask_area = _mask_area_pixels(run_dir, valid_sub)
    eff = _effective_params(meta_disk)
    sat_block = meta_disk.get("scene_autotune") or {}

    return {
        "case": case,
        "mode": mode,
        "tracks": int(metrics.get("tracks_total", 0)),
        "detections": int(metrics.get("detections_kept", 0)),
        "events_total": ev["events_total"],
        "exits": ev["exits"],
        "enters": ev["enters"],
        "outside": ev["outside"],
        "inside": ev["inside"],
        "dilate_px": eff.get("dilate_px"),
        "diff_threshold": eff.get("diff_threshold"),
        "min_area": eff.get("min_area"),
        "max_missed": eff.get("max_missed"),
        "max_distance": eff.get("max_distance"),
        "mask_area_px": mask_area,
        "final_parameter_sources": sat_block.get("final_parameter_sources"),
        "run_dir": str(run_dir.resolve()),
        "wall_sec": float((meta_disk.get("performance") or {}).get("pipeline_total_wall_sec", 0.0)),
    }


def _comparison_warnings(
    manual: Dict[str, Any],
    auto: Dict[str, Any],
) -> List[str]:
    w: List[str] = []
    md = float(manual.get("max_distance") or 0)
    ad = float(auto.get("max_distance") or 0)
    if ad >= 160:
        w.append(f"autotune max_distance={ad} (≥160)")
    ma = auto.get("min_area")
    if ma is not None and float(ma) <= 4:
        w.append(f"autotune min_area={ma} (≤4)")
    dil = auto.get("dilate_px")
    area = auto.get("mask_area_px")
    if dil is not None and area and area > 0:
        ratio = float(dil) / (float(area) ** 0.5)
        if float(dil) >= 55 or ratio > 0.14:
            w.append(
                f"autotune dilate_px={dil} elevat vs sqrt(màscara)≈{area**0.5:.0f} (ratio dilate/sqrt(area)={ratio:.3f})"
            )
    dt = auto.get("detections", 0)
    dm = manual.get("detections", 0)
    et = auto.get("events_total", 0)
    em = manual.get("events_total", 0)
    xe = int(auto.get("exits", 0)) - int(manual.get("exits", 0))
    if dt > dm + max(50, int(0.15 * max(dm, 1))) and et <= em + 2 and xe <= 0:
        w.append("més deteccions amb autotune sense guany d’esdeveniments ni de sortides (possible soroll)")
    if et > em + max(3, int(0.15 * max(em, 1))) and int(auto.get("exits", 0)) <= int(manual.get("exits", 0)):
        w.append("events_total puja amb autotune sense increment d’exits (proxy de més FP possibles)")
    return w


def run_eval_batch(
    *,
    cases: List[Tuple[str, str]],
    config_path: Path,
    out_dir: Path,
) -> Dict[str, Any]:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_path.resolve()

    user_doc = _read_user_yaml(config_path)
    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    by_case: Dict[str, Any] = {}

    csv_columns = [
        "case",
        "mode",
        "tracks",
        "detections",
        "events_total",
        "exits",
        "enters",
        "outside",
        "inside",
        "dilate_px",
        "diff_threshold",
        "min_area",
        "max_missed",
        "max_distance",
    ]

    for case, vid_s in cases:
        video_path = Path(vid_s).expanduser().resolve()
        if not video_path.is_file():
            errors.append({"case": case, "error": f"vídeo no trobat: {video_path}"})
            continue

        case_dir = out_dir / case
        manual_dir = case_dir / "manual"
        autotune_dir = case_dir / "autotune"
        manual_row: Optional[Dict[str, Any]] = None
        auto_row: Optional[Dict[str, Any]] = None

        try:
            manual_row = _run_mode(
                case=case,
                mode="manual",
                video_path=video_path,
                config_path=config_path,
                run_dir=manual_dir,
                user_doc=user_doc,
                sat=SAT_MANUAL,
            )
            rows.append({k: manual_row.get(k) for k in csv_columns})
            logger.info("OK %s manual tracks=%s", case, manual_row["tracks"])
        except Exception as exc:
            logger.exception("Fallo %s manual", case)
            errors.append({"case": case, "mode": "manual", "error": str(exc)})

        try:
            auto_row = _run_mode(
                case=case,
                mode="autotune",
                video_path=video_path,
                config_path=config_path,
                run_dir=autotune_dir,
                user_doc=user_doc,
                sat=SAT_AUTOTUNE,
            )
            rows.append({k: auto_row.get(k) for k in csv_columns})
            logger.info("OK %s autotune tracks=%s", case, auto_row["tracks"])
        except Exception as exc:
            logger.exception("Fallo %s autotune", case)
            errors.append({"case": case, "mode": "autotune", "error": str(exc)})

        if manual_row and auto_row:
            d_exits = int(auto_row["exits"]) - int(manual_row["exits"])
            d_ev = int(auto_row["events_total"]) - int(manual_row["events_total"])
            d_tracks = int(auto_row["tracks"]) - int(manual_row["tracks"])
            d_det = int(auto_row["detections"]) - int(manual_row["detections"])
            warns = _comparison_warnings(manual_row, auto_row)
            extra_keys = (
                "mask_area_px",
                "wall_sec",
                "final_parameter_sources",
                "run_dir",
            )
            def _pick(d: Dict[str, Any]) -> Dict[str, Any]:
                base = {k: d.get(k) for k in csv_columns}
                for ek in extra_keys:
                    base[ek] = d.get(ek)
                return base

            by_case[case] = {
                "delta_exits": d_exits,
                "delta_events_total": d_ev,
                "delta_tracks": d_tracks,
                "delta_detections": d_det,
                "manual": _pick(manual_row),
                "autotune": _pick(auto_row),
                "warnings": warns,
            }

    summary = {
        "out_dir": str(out_dir),
        "config": str(config_path),
        "rows": rows,
        "by_case": by_case,
        "errors": errors,
    }

    csv_path = out_dir / "summary.csv"
    cmp_path = out_dir / "pipeline_autotune_comparison.csv"
    for path in (csv_path, cmp_path):
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=csv_columns, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: "" if r.get(k) is None else r[k] for k in csv_columns})

    json_path = out_dir / "summary.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    lines: List[str] = [
        "# Pipeline manual vs scene_auto_tune",
        "",
        f"- Config base: `{config_path}`",
        f"- Sortida: `{out_dir}`",
        f"- Casos comparats: {len(by_case)}, errors: {len(errors)}",
        "",
        "## Per cas (diferències autotune − manual)",
        "",
        "| cas | Δ exits | Δ events | Δ tracks | Δ deteccions | Avisos |",
        "|-----|---------|----------|----------|--------------|--------|",
    ]
    for case, data in sorted(by_case.items()):
        wtxt = "; ".join(data.get("warnings") or []) or "—"
        lines.append(
            f"| {case} | {data['delta_exits']} | {data['delta_events_total']} | "
            f"{data['delta_tracks']} | {data['delta_detections']} | {wtxt} |"
        )
    lines.extend(
        [
            "",
            "## Taula completa",
            "",
            "Vegeu `summary.csv` o `pipeline_autotune_comparison.csv`.",
            "",
            "## Errors",
            "",
        ]
    )
    if errors:
        for e in errors:
            lines.append(f"- `{e}`")
    else:
        lines.append("- Cap.")
    lines.append("")
    report_text = "\n".join(lines)
    (out_dir / "report.md").write_text(report_text, encoding="utf-8")
    (out_dir / "pipeline_autotune_report.md").write_text(report_text, encoding="utf-8")

    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Executa el pipeline en manual vs autotune per diversos vídeos.",
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        required=True,
        metavar="CASE=/ruta/video",
        help="Parelles case=ruta (una o més).",
    )
    parser.add_argument("--config", type=Path, required=True, help="YAML base (mateix per tots els casos).")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directori arrel de l’avaluació.")
    args = parser.parse_args()
    cases = _parse_videos_arg(args.videos)
    summary = run_eval_batch(cases=cases, config_path=args.config, out_dir=args.out_dir)
    print(
        json.dumps(
            {"cases_compared": len(summary["by_case"]), "errors": len(summary["errors"])},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""
Execució batch de perfilat d'escena sobre diversos vídeos (avaluació multi-cas, sense retocar regles).

Genera CSV + JSON agregat, una carpeta per cas amb artefactes de scene_profile, i un informe heurístic.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .background import compute_background_median
from .compute import build_execution_plan
from .config import load_config
from .scene_auto_tune import RULES_VERSION, get_nested
from .scene_profile import build_scene_profile
from .video import read_video_meta

logger = logging.getLogger(__name__)

FLAT_KEYS = [
    "valid_region.mask_geometry.dilate_px",
    "detection.diff_threshold",
    "detection.min_area",
    "tracking.max_missed",
    "tracking.max_distance",
]

# (clau plana YAML, sufix per columnes CSV)
_ROW_METRIC_KEYS: List[Tuple[str, str]] = [
    ("valid_region.mask_geometry.dilate_px", "dilate_px"),
    ("detection.diff_threshold", "diff_threshold"),
    ("detection.min_area", "min_area"),
    ("tracking.max_missed", "max_missed"),
    ("tracking.max_distance", "max_distance"),
]


def _resolve_path(base_dir: Path, p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _manual_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in FLAT_KEYS:
        out[k] = get_nested(cfg, k)
    return out


def _as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _delta(auto: Any, manual: Any) -> Optional[float]:
    a, m = _as_float(auto), _as_float(manual)
    if a is None or m is None:
        return None
    return a - m


def _run_one_case(
    *,
    name: str,
    video_resolved: Path,
    cfg: Dict[str, Any],
    case_out: Path,
    sample_frames: int,
    write_artifacts: bool,
    strict_parity: bool,
) -> Dict[str, Any]:
    case_out.mkdir(parents=True, exist_ok=True)
    profile_subdir = case_out / "scene_profile"
    meta = read_video_meta(str(video_resolved))
    plan = build_execution_plan(cfg)
    bg_stats: Dict[str, int] = {}
    background = compute_background_median(
        video_path=str(video_resolved),
        meta=meta,
        sample_frames=int(cfg["background"]["sample_frames"]),
        uniform_sampling=bool(cfg["background"]["uniform_sampling"]),
        compute_device=plan.selected_device,
        strict_parity=strict_parity,
        runtime_stats=bg_stats,
    )
    profile = build_scene_profile(
        video_path=str(video_resolved),
        background=background,
        merged_config=cfg,
        out_dir=profile_subdir,
        sample_frames=sample_frames,
        write_artifacts=write_artifacts,
    )
    manual = _manual_from_config(cfg)
    rec = profile.get("recommended") or {}
    geo = profile.get("opening_geometry") or {}

    row: Dict[str, Any] = {
        "case_name": name,
        "video_path": str(video_resolved),
        "config_path": "",  # omplert pel caller
        "video_id": meta.video_id,
        "fps": meta.fps,
        "frame_count": meta.frame_count,
        "frame_width": meta.width,
        "frame_height": meta.height,
        "opening_area_px": geo.get("area_px"),
        "opening_bbox_w": geo.get("width_px"),
        "opening_bbox_h": geo.get("height_px"),
        "opening_frame_fraction": geo.get("frame_area_fraction"),
        "opening_solidity": geo.get("solidity"),
    }
    for flat_k, suffix in _ROW_METRIC_KEYS:
        row[f"manual_{suffix}"] = manual.get(flat_k)
        row[f"auto_{suffix}"] = rec.get(flat_k)
        row[f"delta_{suffix}"] = _delta(rec.get(flat_k), manual.get(flat_k))

    row["profile_dir"] = str(profile_subdir.resolve())
    row["scene_profile_json"] = str((profile_subdir / "scene_profile.json").resolve()) if write_artifacts else ""
    row["rules_version"] = (profile.get("decision_inputs") or {}).get("rules_version", RULES_VERSION)
    return row


def _numeric_stats(rows: List[Dict[str, Any]], col: str) -> Dict[str, Any]:
    vals: List[float] = []
    for r in rows:
        v = _as_float(r.get(col))
        if v is not None:
            vals.append(v)
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": statistics.mean(vals),
        "stdev": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


def _analyze_batch(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Heurístiques d'avaluació: extrems, variància, parelles de geometria similar."""
    observations: List[str] = []
    per_case_flags: Dict[str, List[str]] = {str(r["case_name"]): [] for r in rows}

    def flag(case: str, msg: str) -> None:
        observations.append(f"[{case}] {msg}")
        per_case_flags.setdefault(case, []).append(msg)

    # Extrems (clamps de les regles actuals)
    for r in rows:
        c = str(r["case_name"])
        d = _as_float(r.get("auto_dilate_px"))
        if d is not None:
            if d <= 8.5:
                flag(c, "auto_dilate_px prop del clamp inferior (8)")
            if d >= 84.5:
                flag(c, "auto_dilate_px prop del clamp superior (85)")
        dt = _as_float(r.get("auto_diff_threshold"))
        if dt is not None:
            if dt <= 5.5:
                flag(c, "auto_diff_threshold prop del clamp inferior (5)")
            if dt >= 34.5:
                flag(c, "auto_diff_threshold prop del clamp superior (35)")
        md = _as_float(r.get("auto_max_distance"))
        if md is not None:
            if md <= 46.0:
                flag(c, "auto_max_distance prop del clamp inferior (45)")
            if md >= 169.0:
                flag(c, "auto_max_distance prop del clamp superior (170)")

    # Coherència dilatació vs escala (ratio dilate / sqrt(area))
    ratios: List[Tuple[str, float]] = []
    for r in rows:
        a = _as_float(r.get("opening_area_px"))
        d = _as_float(r.get("auto_dilate_px"))
        if a and a > 0 and d is not None:
            ratios.append((str(r["case_name"]), d / (a**0.5)))
    if ratios:
        rvals = [x[1] for x in ratios]
        m = statistics.mean(rvals)
        s = statistics.pstdev(rvals) if len(rvals) > 1 else 0.0
        observations.append(
            f"Ratio auto_dilate_px/sqrt(area): mean={m:.4f}, stdev={s:.4f} (n={len(ratios)})."
        )
        if len(rvals) > 1 and s > 0.02:
            observations.append(
                "El ratio dilate/sqrt(area) varia molt entre casos — possible sensibilitat de la regla a l'escena."
            )
        for name, rv in ratios:
            if s > 1e-6 and abs(rv - m) > 2.5 * max(s, 0.001):
                flag(name, f"ratio dilate/sqrt(area)={rv:.4f} allunyat de la mitjana del lot ({m:.4f})")

    # Parelles amb geometria similar però dilate molt diferent
    seen_pairs: set[Tuple[str, str]] = set()
    for i, ri in enumerate(rows):
        ai = _as_float(ri.get("opening_area_px"))
        di = _as_float(ri.get("auto_dilate_px"))
        if ai is None or di is None:
            continue
        for j, rj in enumerate(rows):
            if j <= i:
                continue
            aj = _as_float(rj.get("opening_area_px"))
            dj = _as_float(rj.get("auto_dilate_px"))
            if aj is None or dj is None:
                continue
            ar = max(ai, aj) / max(min(ai, aj), 1.0)
            if ar <= 1.25 and abs(di - dj) >= 10:
                a, b = str(ri["case_name"]), str(rj["case_name"])
                key = tuple(sorted((a, b)))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                observations.append(
                    f"Parella similar en àrea ({a} vs {b}): "
                    f"ratio àrees {ar:.2f} però |Δdilate|={abs(di - dj):.0f} px — revisar si és plausible."
                )

    # max_distance vs diagonal
    for r in rows:
        w = _as_float(r.get("frame_width"))
        h = _as_float(r.get("frame_height"))
        md = _as_float(r.get("auto_max_distance"))
        if w and h and md:
            diag = (w * w + h * h) ** 0.5
            frac = md / diag if diag else 0.0
            if frac < 0.04:
                flag(
                    str(r["case_name"]),
                    f"auto_max_distance ({md:.0f}) és <4% de la diagonal ({diag:.0f}) — possible subestimació.",
                )
            if frac > 0.12:
                flag(
                    str(r["case_name"]),
                    f"auto_max_distance ({md:.0f}) és >12% de la diagonal — revisar coherència amb escena.",
                )

    # Variància global destacada (dilate i max_distance)
    st_d = _numeric_stats(rows, "auto_dilate_px")
    st_md = _numeric_stats(rows, "auto_max_distance")
    if st_d.get("n", 0) > 1 and st_d["stdev"] > 0.15 * max(st_d["mean"], 1.0):
        observations.append(
            f"Alta variància inter-vídeo en auto_dilate_px (σ={st_d['stdev']:.2f}, μ={st_d['mean']:.2f})."
        )
    if st_md.get("n", 0) > 1 and st_md["stdev"] > 0.2 * max(st_md["mean"], 1.0):
        observations.append(
            f"Alta variància inter-vídeo en auto_max_distance (σ={st_md['stdev']:.2f}, μ={st_md['mean']:.2f})."
        )

    return {
        "observations": observations,
        "per_case_flags": per_case_flags,
        "aggregate_stats": {
            "auto_dilate_px": st_d,
            "auto_max_distance": st_md,
            "auto_diff_threshold": _numeric_stats(rows, "auto_diff_threshold"),
            "auto_min_area": _numeric_stats(rows, "auto_min_area"),
            "auto_max_missed": _numeric_stats(rows, "auto_max_missed"),
            "opening_area_px": _numeric_stats(rows, "opening_area_px"),
        },
        "focus_notes": {
            "dilate_px": "Regla basada en sqrt(area) i costat mínim del bbox; comparar ratio dilate/sqrt(area) entre casos similars.",
            "max_distance": "Regla basada en diagonal del frame i obertura; validar fracció respecte diagonal per a cada resolució.",
        },
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            flat = {k: ("" if v is None else v) for k, v in r.items()}
            w.writerow(flat)


def run_batch(
    *,
    manifest_path: Path,
    out_dir: Path,
    write_artifacts: bool = True,
) -> Dict[str, Any]:
    manifest_path = manifest_path.resolve()
    base_dir = manifest_path.parent
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    videos = doc.get("videos")
    if not isinstance(videos, list) or not videos:
        raise ValueError("manifest must contain a non-empty 'videos' list")

    global_sample = int(doc.get("sample_frames", 48))
    global_strict = bool(doc.get("strict_parity", True))

    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for entry in videos:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip()
        vid = entry.get("video")
        cfg_p = entry.get("config")
        if not name or not vid or not cfg_p:
            errors.append({"entry": entry, "error": "missing name, video, or config"})
            continue
        video_path = _resolve_path(base_dir, str(vid))
        config_path = _resolve_path(base_dir, str(cfg_p))
        if not video_path.is_file():
            errors.append({"case": name, "error": f"video not found: {video_path}"})
            continue
        if not config_path.is_file():
            errors.append({"case": name, "error": f"config not found: {config_path}"})
            continue

        sample_frames = int(entry.get("sample_frames", global_sample))
        strict = bool(entry.get("strict_parity", global_strict))
        try:
            cfg = load_config(config_path)
            case_out = out_dir / name
            row = _run_one_case(
                name=name,
                video_resolved=video_path,
                cfg=cfg,
                case_out=case_out,
                sample_frames=sample_frames,
                write_artifacts=write_artifacts,
                strict_parity=strict,
            )
            row["config_path"] = str(config_path)
            rows.append(row)
            logger.info("OK case=%s video=%s", name, video_path)
        except Exception as exc:
            logger.exception("Fallo case=%s", name)
            errors.append({"case": name, "error": str(exc)})

    analysis = _analyze_batch(rows)
    summary = {
        "manifest": str(manifest_path),
        "out_dir": str(out_dir),
        "rules_version": RULES_VERSION,
        "cases_ok": len(rows),
        "cases_failed": len(errors),
        "errors": errors,
        "rows": rows,
        "analysis": analysis,
    }

    _write_csv(out_dir / "batch_scene_profile_summary.csv", rows)
    with (out_dir / "batch_scene_profile_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    # Informe llegible curt
    report_lines = [
        "# Batch scene profile / autotune (avaluació)",
        "",
        f"- Casos OK: {len(rows)}, fallits: {len(errors)}",
        f"- Sortida: `{out_dir}`",
        "",
        "## Observacions",
        "",
    ]
    for o in analysis.get("observations", []):
        report_lines.append(f"- {o}")
    report_lines.extend(["", "## Estadístiques agregades (auto)", ""])
    for key, st in analysis.get("aggregate_stats", {}).items():
        report_lines.append(f"- **{key}**: {st}")
    report_lines.append("")
    with (out_dir / "batch_scene_profile_report.md").open("w", encoding="utf-8") as fh:
        fh.write("\n".join(report_lines))

    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Perfilat d'escena en batch (múltiples vídeos, informe agregat).",
    )
    parser.add_argument("--manifest", required=True, type=Path, help="YAML amb llista videos (name, video, config).")
    parser.add_argument("--out-dir", required=True, type=Path, help="Directori de sortida (subcarpetes per cas).")
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="No escriure PNG/CSV per cas (només resum agregat).",
    )
    args = parser.parse_args()
    summary = run_batch(
        manifest_path=args.manifest,
        out_dir=args.out_dir,
        write_artifacts=not args.no_artifacts,
    )
    print(json.dumps({"cases_ok": summary["cases_ok"], "cases_failed": summary["cases_failed"]}, indent=2))


if __name__ == "__main__":
    main()

"""
Regles d'autoajust explícites: mapatge de mètriques de perfil a paràmetres del pipeline.

Totes les decisions es documenten amb `decision_inputs` per traçabilitat.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Versió de les regles (incrementar si canvia la lògica per comparar perfils).
RULES_VERSION = "1.0"


def _clamp_int(x: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(x))))


def recommend_dilate_px_from_geometry(geo: Dict[str, Any], contrast: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    """
    Escala amb sqrt(area) i amb el costat menor del bbox (alçada/amplada de l'obertura base).

    Calibrat per acostar-se a estudi manual ~28 px sobre vídeos tipus retallat (màscara base ~59k px²).
    """
    area = max(1, int(geo.get("area_px") or 0))
    char = float(area**0.5)
    h = max(1, int(geo.get("height_px") or 1))
    w = max(1, int(geo.get("width_px") or 1))
    side_min = float(min(h, w))
    solidity = float(geo.get("solidity") or 0.0)
    boost_irregular = (1.0 - min(1.0, max(0.0, solidity))) * 5.0
    raw = 0.034 * char + 0.076 * side_min + boost_irregular
    dil = _clamp_int(raw, 8, 85)
    inputs = {
        "area_px": area,
        "sqrt_area": char,
        "min_side_bbox_px": side_min,
        "solidity": solidity,
        "boost_irregular_px": boost_irregular,
        "raw_before_clamp": raw,
        "clamp": [8, 85],
    }
    return dil, inputs


def recommend_min_area(blob_stats: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    """
    Percentil 25 robust de les àrees de candidats (sense filtres forts), escalat per sota del blob típic petit.
    """
    p25 = float(blob_stats.get("area_p25") or 0.0)
    p10 = float(blob_stats.get("area_p10") or 0.0)
    dust = float(blob_stats.get("dust_proxy_high_frequency_small") or 0.0)
    if p25 < 0.5:
        m = 6
        note = "fallback_p25_too_small"
    else:
        scale = 0.42 if dust < 0.55 else 0.35
        m = _clamp_int(p25 * scale, 3, 50)
        note = "p25_scaled"
    inputs = {
        "area_p10": p10,
        "area_p25": p25,
        "dust_proxy": dust,
        "scale_used": 0.42 if dust < 0.55 else 0.35,
        "note": note,
        "clamp": [3, 50],
    }
    return int(m), inputs


def recommend_diff_threshold(
    temporal_noise: Dict[str, Any],
    contrast: Dict[str, Any],
) -> Tuple[int, Dict[str, Any]]:
    """
    La mediana per píxel dins la ROI sol ser ~0 (fons estable); usem p95 per frame (mediana)
    i cues de cua global (p99) + estabilitat de la mitjana per frame.
    """
    mf_p95 = float(temporal_noise.get("median_of_frame_p95_absdiff") or 0.0)
    mf_mean = float(temporal_noise.get("median_of_frame_mean_absdiff") or 0.0)
    p99 = float(temporal_noise.get("p99_absdiff_roi") or 0.0)
    stab = float(temporal_noise.get("absdiff_temporal_stability_std") or 0.0)
    delta = float(contrast.get("delta_mean") or 0.0)
    raw_signal = 0.48 * mf_p95 + 0.32 * mf_mean + 0.22 * p99 + stab * 1.25 + max(0.0, -delta) * 0.02
    # Escala affine: la senyal agregada és petita en uint8 però el pipeline usa llindars ~8–20 típics.
    raw = raw_signal * 4.5 + 1.5
    thr = _clamp_int(raw, 5, 35)
    inputs = {
        "median_of_frame_p95_absdiff": mf_p95,
        "median_of_frame_mean_absdiff": mf_mean,
        "p99_absdiff_roi_pooled": p99,
        "absdiff_temporal_stability_std_frame_means": stab,
        "delta_mean_interior_minus_corona": delta,
        "combined_signal_before_affine": raw_signal,
        "affine_scale_offset": [4.5, 1.5],
        "raw_before_clamp": raw,
        "clamp": [5, 35],
    }
    return int(thr), inputs


def recommend_tracking_params(
    video: Dict[str, Any],
    opening_geometry: Dict[str, Any],
    blob_stats: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    max_missed ~ 0.35 * fps (objectes ràpids vs fps).
    max_distance: diagonal del frame (escena) + contribució de l'obertura; els blobs candidats solen ser massa
    petits per estimar velocitat sense flux òptic, per això es fa servir geometria global.
    """
    fps = float(video.get("fps") or 25.0)
    w = float(video.get("width") or 0.0)
    h = float(video.get("height") or 0.0)
    diag = float((w * w + h * h) ** 0.5) if w > 0 and h > 0 else 0.0
    area_open = max(1, int(opening_geometry.get("area_px") or 1))
    char_open = float(area_open**0.5)
    p50a = float(blob_stats.get("area_p50") or 0.0)
    char_blob = float(max(1.0, p50a**0.5))

    max_missed = _clamp_int(0.35 * fps + 3.0, 6, 24)
    md_raw = 0.052 * diag + 0.12 * char_open + 1.5 * char_blob
    max_distance = float(_clamp_int(md_raw, 45, 170))

    inputs_mm = {"fps": fps, "formula": "0.35*fps+3", "clamp": [6, 24]}
    inputs_md = {
        "frame_diagonal_px": diag,
        "sqrt_opening_area": char_open,
        "sqrt_blob_area_p50": char_blob,
        "raw_before_clamp": md_raw,
        "clamp": [45, 170],
    }
    rec = {"tracking.max_missed": max_missed, "tracking.max_distance": max_distance}
    dec = {"tracking.max_missed": inputs_mm, "tracking.max_distance": inputs_md}
    return rec, dec


def compute_recommendations(profile: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str], Dict[str, Any]]:
    """
    Retorna (recommended_flat, rationale_text, decision_inputs_per_key).
    """
    geo = profile.get("opening_geometry") or {}
    contrast = profile.get("contrast") or {}
    tn = profile.get("temporal_noise") or {}
    blobs = profile.get("blob_stats") or {}
    video = profile.get("video") or {}

    dil, in_dil = recommend_dilate_px_from_geometry(geo, contrast)
    min_a, in_ma = recommend_min_area(blobs)
    diff_t, in_dt = recommend_diff_threshold(tn, contrast)
    track_extra, in_tr = recommend_tracking_params(video, geo, blobs)

    recommended: Dict[str, Any] = {
        "valid_region.mask_geometry.mode": "dilate",
        "valid_region.mask_geometry.dilate_px": dil,
        "valid_region.mask_geometry.iterations": 1,
        "valid_region.mask_geometry.clip_to_profile_mask": True,
        "detection.diff_threshold": diff_t,
        "detection.min_area": min_a,
        **track_extra,
    }

    rationale: Dict[str, str] = {
        "valid_region.mask_geometry.mode": "mode fix dilate per geometria morfològica de l'obertura",
        "valid_region.mask_geometry.dilate_px": "0.034*sqrt(area_px) + 0.076*min(w,h) bbox + boost si solidity baixa; clamp [8,85]",
        "valid_region.mask_geometry.iterations": "1 per defecte (mateix que configs d'estudi)",
        "valid_region.mask_geometry.clip_to_profile_mask": "retall al perfil horitzontal per evitar fuites",
        "detection.diff_threshold": "senyal combinat (p95/mediana frame, mitjanes, p99) * 4.5 + 1.5; clamp [5,35]",
        "detection.min_area": "percentil 25 d'àrees de candidats * factor (0.35–0.42 segons pols de pols/insectes); clamp [3,50]",
        "tracking.max_missed": "0.35*fps+3 clamp [6,24]",
        "tracking.max_distance": "0.052*diagonal_frame + 0.12*sqrt(area_obertura) + 1.5*sqrt(p50_area_blob); clamp [45,170]",
    }

    decision_inputs: Dict[str, Any] = {
        "rules_version": RULES_VERSION,
        "valid_region.mask_geometry.dilate_px": in_dil,
        "detection.min_area": in_ma,
        "detection.diff_threshold": in_dt,
        **{k: v for k, v in in_tr.items()},
    }

    return recommended, rationale, decision_inputs


def get_nested(cfg: Dict[str, Any], dotted: str) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def set_nested(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = cfg
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value


def apply_recommendations_to_config(
    cfg: Dict[str, Any],
    recommended_flat: Dict[str, Any],
    explicit_yaml_paths: set[str],
    allow_override_explicit: bool,
) -> Dict[str, Any]:
    """
    Aplica claus planes tipus 'detection.min_area'. Les claus presents a explicit_yaml_paths no es toquen
    tret que allow_override_explicit sigui True.
    """
    applied: Dict[str, Any] = {}
    skipped: Dict[str, Any] = {}
    cfg_mut = cfg
    for path, value in recommended_flat.items():
        is_explicit = path in explicit_yaml_paths
        if is_explicit and not allow_override_explicit:
            skipped[path] = {
                "reason": "explicit_yaml",
                "current_value": get_nested(cfg_mut, path),
            }
            continue
        set_nested(cfg_mut, path, value)
        applied[path] = {
            "value": value,
            "source": "scene_auto_tune_override_explicit" if is_explicit else "scene_auto_tune",
        }
    return {
        "applied": applied,
        "skipped_explicit": skipped,
        "allow_override_explicit": allow_override_explicit,
    }


def write_decisions_summary(path: str | Path, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def final_parameter_sources(
    cfg: Dict[str, Any],
    tune_keys: List[str],
    apply_report: Dict[str, Any],
) -> Dict[str, Any]:
    """Resum per meta.json: valor final i origen (manual vs auto)."""
    out: Dict[str, Any] = {}
    applied = apply_report.get("applied") or {}
    skipped = apply_report.get("skipped_explicit") or {}
    for key in tune_keys:
        src = "default_or_merged_yaml"
        if key in applied:
            src = applied[key].get("source", "scene_auto_tune")
        elif key in skipped:
            src = "explicit_yaml"
        out[key] = {
            "value": get_nested(cfg, key),
            "source": src,
        }
    return out


def snapshot_tunable_keys(cfg: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "valid_region.mask_geometry.mode",
        "valid_region.mask_geometry.dilate_px",
        "valid_region.mask_geometry.iterations",
        "valid_region.mask_geometry.clip_to_profile_mask",
        "detection.diff_threshold",
        "detection.min_area",
        "tracking.max_missed",
        "tracking.max_distance",
    ]
    return {k: deepcopy(get_nested(cfg, k)) for k in keys}

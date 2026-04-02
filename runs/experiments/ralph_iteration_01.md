# Ralph — Iteració 01 (registre)

## 1. Hipòtesi

**H1 (falsable):** En escenes amb murciàlags petits, el `detection.diff_threshold` recomanat per `scene_auto_tune` (affine 4.5× + 1.5 sobre la senyal temporal) queda **sistemàticament per sobre** del baseline manual i **n’és la causa principal** de la pèrdua d’`exits` respecte al mode manual.

*Baseline numèric (ja existent, `runs/pipeline_autotune_eval_smoke/summary.csv`, case_03):* manual `diff_threshold=12`, 10 exits; autotune `diff_threshold=17`, 9 exits (−1 exit, −23 deteccions).

## 2. Canvi realitzat (prova A → revertit)

- **Una sola palanca:** `recommend_diff_threshold` — affine `4.5*s + 1.5` → `3.7*s + 1.2` (RULES_VERSION provisional 1.1).
- **Motiu:** Si H1 és certa, abaixar el llindar recomanat hauria d’acostar exits al manual sense tocar `min_area`, dilate ni tracking.

*Després de mesurar, el codi s’ha **revertit** a affine 4.5+1.5 i RULES_VERSION 1.0.*

## 3. Casos avaluats

- **case_03** (`rabella_20211016_DSCF0005.mp4`, ~1502 frames): ràpid, ja tenia baseline al smoke.
- Config: `config.out3_clean.yaml`, `pipeline_autotune_eval_batch` (manual vs autotune).

## 4. Evidència numèrica

| Mètrica | Baseline smoke (autotune v1.0) | Prova affine 3.7+1.2 (`runs/ralph_iter1_case03`) |
|---------|--------------------------------|--------------------------------------------------|
| diff_threshold (autotune) | 17 | **14** |
| min_area (autotune) | 10 | 10 (sense canvi) |
| dilate_px | 31 | 31 |
| max_missed / max_distance | 12 / 114 | 12 / 114 |
| tracks | 16 | 16 |
| detections | 185 | **203** (+18) |
| events_total | 16 | 16 |
| **exits** | **9** | **8** (−1) |
| enters | 1 | **0** |
| inside | 6 | 8 |

Manual sense canvis: 10 exits, 208 deteccions, 18 events.

## 5. Evidència visual / traça

- Sortides completes: `runs/ralph_iter1_case03/case_03/{manual,autotune}/` (`tracks_overlay.png`, `events.csv`, `meta.json`).
- `meta.json` autotune confirma `recommended.detection.diff_threshold: 14` (vs 17 abans).

## 6. Conclusió causal (no només descriptiva)

- Baixar **només** `diff_threshold` ha **augmentat** deteccions (+18) però ha **reduït** exits (9→8) i ha eliminat l’únic `enters`.
- Per tant **H1 és falsada com a explicació principal**: en aquest clip, la pèrdua d’exits respecte al manual **no es corregeix** abaixant el llindar amb aquesta magnitud; el sistema respon amb més deteccions que **no es tradueixen** en més esdeveniments de sortida (possible soroll, fragmentació de tracks, o altres gates: `min_area`, geometria `valid_region`, classificació d’events).
- La correlació “autotune més conservador en diff → menys exits” al baseline **no implica** que relaxar diff **recuperi** exits: la direcció causal simple no es confirma.

## 7. Decisió

- **Descartar** el canvi affine 3.7+1.2 (revertit al repositori).
- **Mantenir** baseline autotune v1.0 fins a nova iteració.

## 8. Següent hipòtesi proposada (iteració 02)

**H2:** En case_03, el coll principal respecte al manual és **`detection.min_area`** (10 vs 8 manual) o la **combinació** min_area + geometria de màscara dilatada, més que `diff_threshold` aïllat.

**Intervenció única suggerida (pròxima iteració):** reduir lleument el factor d’escala a `recommend_min_area` (p.ex. 0.42→0.38 / 0.35→0.32) **sense** tocar diff_threshold; tornar a mesurar case_03 i, si promet, case_02.

**Alternativa si H2 falla:** instrumentar o revisar **classificació d’events** (p. ex. `direction_mode`, gate `require_start_or_end_in_valid_region`) amb el mateix protocol Ralph.

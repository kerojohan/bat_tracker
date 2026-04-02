# Ralph — Tanda: ajust `min_area` autotune (case_03)

Baseline fix: `runs/pipeline_autotune_eval_smoke/summary.csv` (manual vs autotune v1.0).

## Iteració 1

- **Hipòtesi:** El gap d’exits (manual 10 vs autotune 9) es deu en part a `min_area` massa alt (10 vs 8 manual); baixar els factors d’escala de `recommend_min_area` (única palanca) recuperarà exits.
- **Canvi:** `0.42/0.35` → `0.38/0.32`, `RULES_VERSION` 1.1.
- **Casos:** case_03, `config.out3_clean.yaml`, `runs/ralph_tanda_iter1_minarea/`.
- **Baseline:** autotune v1.0: exits 9, min_area 10, deteccions 185, enters 1.
- **Evidència numèrica:** exits **9**, min_area **9**, deteccions **185**, events 16, tracks 16 (mateixos exits; min_area més proper a manual).
- **Visual:** `.../case_03/autotune/tracks_overlay.png`, `events.csv`.
- **Conclusió causal:** Reduir escala **sí** abaixa `min_area` recomanat però **no** canvia exits; el coll no era només el valor enter de `min_area` en aquest pas.
- **Decisió:** Refinar (iter 2 més agressiu) o descartar; es prova iter 2.
- **Següent hipòtesi:** Amb `min_area` encara per sobre del manual (9 vs 8), un pas més en escala portarà a 8 i desbloquejarà l’exit que falta.

## Iteració 2

- **Hipòtesi:** Amb `min_area=9` encara hi ha blob vàlid filtrat; forçar escala `0.34/0.28` donarà `min_area=8` (com manual) i pujarà exits.
- **Canvi:** `0.38/0.32` → `0.34/0.28`, `RULES_VERSION` 1.2.
- **Casos:** case_03, `runs/ralph_tanda_iter2_minarea/`.
- **Baseline:** iter 1 / v1.0 autotune (exits 9).
- **Evidència numèrica:** min_area **8**, exits **9** (sense canvi), deteccions **189** (+4 vs v1.0), **enters 0** (abans 1), inside 7.
- **Conclusió causal:** Igualar `min_area` al manual **no** igualar exits; el dèficit d’1 exit respecte al manual **no** és explicable com a efecte dominant de `min_area` en aquest clip. Més deteccions sense més exits → soroll o trajectòries que no es classifiquen com exit per altres motius (màscara dilatada, tracking, events).
- **Decisió:** **Descartar** la línia d’ajust d’escala `min_area` (revertit a v1.0 al repositori).
- **Següent hipòtesi (fora d’aquesta tanda):** Geometria `valid_region` (dilate 31 vs sense dilate al manual) o classificació `direction` / gates d’events.

## Iteració 3

- **Parada:** Dos canvis seguits sense millora en **exits**; a més, iter 2 empitjora **enters** sense guany d’exits → criteri de parada de la tanda.
- **Acció:** Sense tercer canvi de codi; codi restaurat a **autotune v1.0**.

## Resum tanda

| Versió | min_area (auto) | exits | enters | deteccions |
|--------|-----------------|-------|--------|------------|
| Manual | 8 | 10 | 1 | 208 |
| v1.0 autotune | 10 | 9 | 1 | 185 |
| v1.1 (iter1) | 9 | 9 | 1 | 185 |
| v1.2 (iter2) | 8 | 9 | 0 | 189 |

**Mantenits:** cap (revertit).  
**Revertits:** escales min_area v1.1/v1.2.  
**Millora neta:** 0 exits.  
**Coll dominant sospitós:** diferència de pipeline entre manual (sense `mask_geometry.dilate` al YAML) i autotune (dilate 31 → màscara més gran), o vincle track/event, no `min_area` aïllat.

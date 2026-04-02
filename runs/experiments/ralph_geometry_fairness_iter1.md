# Ralph — Geometria justa (case_03) — Iteració 1

## Iteració 1

### Hipòtesi (H1)

La diferència manual vs autotune en **exits** a case_03 es deu en part important a **geometries de màscara distintes** (manual sense `mask_geometry` vs autotune amb dilate). Si el manual usa la **mateixa** `valid_region.mask_geometry` que l’autotune, el gap d’**exits** hauria de **reduir-se o desaparèixer** respecte al manual original.

### Canvi realitzat

- **Cap canvi al codi d’autotune** ni a detecció/tracking/events.
- S’han creat dos YAML sota `runs/ralph_geometry_fairness_case03/configs/`:
  - `config_manual_original.yaml` — còpia de `config.out3_clean.yaml` amb `scene_auto_tune` desactivat i **sense** `mask_geometry`.
  - `config_manual_geom_matched.yaml` — igual que l’anterior però amb la geometria extreta de l’autotune a case_03 (smoke): `mode: dilate`, `dilate_px: 31`, `iterations: 1`, `clip_to_profile_mask: true`.

### Casos avaluats

- **case_03:** `/home/jcaparros/BORRAR/rabella_20211016_DSCF0005.mp4`
- Tres sortides:
  - `runs/ralph_geometry_fairness_case03/manual/`
  - `runs/ralph_geometry_fairness_case03/manual_geom_matched/`
  - `runs/ralph_geometry_fairness_case03/autotune/` (config `config.out3_clean.yaml`, autotune actiu v1.0)

### Baselines

1. **Manual original:** YAML propi, sense `mask_geometry`, autotune off.
2. **Autotune original:** `config.out3_clean.yaml` (com en avaluacions anteriors).

### Evidència numèrica

| Run | tracks | detections | events_total | exits | enters | outside | inside | mg dilate_px | diff_thr | min_area | max_missed | max_dist |
|-----|--------|------------|--------------|-------|--------|---------|--------|--------------|----------|----------|------------|----------|
| manual | 18 | 208 | 18 | **10** | 1 | 0 | 7 | — | 12 | 8 | 10 | 120 |
| manual_geom_matched | 17 | 216 | 17 | **9** | 1 | 0 | 7 | 31 | 12 | 8 | 10 | 120 |
| autotune | 16 | 185 | 16 | **9** | 1 | 0 | 6 | 31 | 17 | 10 | 12 | 114 |

**Àrea màscara (`valid_region/mask.png`, píxels no zero):**

| Run | mask_nonzero |
|-----|----------------|
| manual | 66 714 |
| manual_geom_matched | 103 990 |
| autotune | 103 990 |

`manual_geom_matched` i **autotune** tenen **mateixa àrea de màscara** (coherent amb mateixa `mask_geometry`).

Fitxers agregats: `comparison_case03.csv`, `comparison_case03.json` al directori arrel del cas.

### Evidència visual

- `manual/tracks_overlay.png`, `manual_geom_matched/tracks_overlay.png`, `autotune/tracks_overlay.png`
- `*/valid_region/overlay.png` — canvi visible d’obertura entre manual i els dos amb dilate 31.

### Conclusió causal

- Només afegir **dilate 31** al manual (sense tocar diff/min_area/tracking) fa baixar els **exits de 10 a 9**.
- **Autotune** també dona **9 exits** amb la **mateixa geometria** (mateixa àrea de màscara que `manual_geom_matched`).
- Per tant, el **salto d’1 exit** entre manual original i autotune en aquest clip **s’explica de forma dominant per la geometria** (màscara més gran), **no** per la resta de recomanacions en el compte d’exits (que sí alteren deteccions i tracks, però el nombre d’exits coincideix amb el manual ja geometria-igualada).
- La comparació “manual `out3_clean` sense `mask_geometry` vs autotune amb dilate” **contaminava** la lectura: es barrejaven **dues màscares** i **dos paquets de paràmetres**.

### Decisió

- **Mantenir** com a artefacte d’avaluació els YAML `config_manual_original.yaml` i `config_manual_geom_matched.yaml` i les tres carpetes de sortida.
- **No cal Iteració 2:** la causa dominant (geometria → −1 exit en passar de màscara base a dilatada igual que autotune) queda **prou clara**.

### Lectura del gap

- **Explicat per geometria (respecte exits):** la transició manual (10 exits) → manual amb dilate 31 (9 exits) = **−1 exit**; coincideix amb autotune (9 exits).
- **No explicat per geometria en el compte d’exits:** la diferència entre `manual_geom_matched` i `autotune` (deteccions, tracks, `inside`, diff/min_area/tracking) és **altra història** (sensibilitat i vincle track), però **no** canvia el nombre d’exits en aquest cas.

### Següent hipòtesi (propera tanda, no executada aquí)

Per comparar autotune vs manual **sense contaminació geomètrica**, usar sempre **la mateixa** `mask_geometry` (manual fixada o tots sense dilate extra) i aleshores aïllar efectes de diff/min_area/tracking; o acceptar que la dilatació és una elecció de producte i avaluar-la explícitament.

---

## Resum de tanda

| Concepte | Resultat |
|----------|----------|
| **Canvis mantinguts** | Configs i runs sota `runs/ralph_geometry_fairness_case03/`; cap canvi al codi del repositori. |
| **Canvis revertits** | N/A |
| **Coll dominant** | **Geometria de `valid_region.mask_geometry`** (dilate 31): explica el descens d’exits 10→9 en case_03 quan s’alinia amb autotune. |
| **Recomanació següent** | En avaluacions Ralph futures, definir **baseline geomètric comú** (p. ex. sempre `manual_geom_matched` vs autotune, o tots sense dilate) abans d’atribuir efectes a diff/min_area. |

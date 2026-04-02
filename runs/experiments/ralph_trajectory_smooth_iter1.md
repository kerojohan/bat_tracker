# Ralph Loop — Suavizado de trayectorias (iteración 1)

## Hipótesis (H1)

El aspecto serrado de las trayectorias en overlay proviene principalmente del **jitter frame a frame de los centroides**, no de fallos de detección. Un **suavizado por track** (media móvil sobre `x`, `y`) debería alinear la representación visual con el movimiento aproximadamente recto, **sin empeorar** el conteo de exits ni el número de tracks, si la asociación y las reglas de eventos no cambian.

## Cambio (único)

- Módulo `bat_tracker/track_smoothing.py`: media móvil 1D en `x` e `y` por `track_id` (ventana impar, padding `edge`).
- `tracks.csv`: siempre coordenadas **raw** (sin suavizar).
- Con `output.trajectory_smoothing_enabled: true`:
  - overlay: `tracks_overlay_raw.png` y `tracks_overlay_smoothed.png`; `tracks_overlay.png` = raw (compatibilidad).
  - `events.csv`: puntos **suavizados** solo para geometría (extremos, dentro/fuera de máscara, dirección); reglas de eventos sin cambio estructural.
- **No modificado:** detección, asociación, `valid_region`, lógica de clasificación de eventos (solo la entrada geométrica).

Config de prueba **case_03:** `runs/experiments/config_case03_trajectory_smooth.yaml` — copia de `config.out3_clean.yaml` + `trajectory_smoothing_enabled: true`, ventana 5.  
**case_01:** `runs/experiments/config_case01_trajectory_smooth.yaml` (misma base `out3_clean`).

**Nota sobre la iteración inicial:** la primera tabla de case_03 usaba como referencia `ralph_geometry_fairness_case03/configs/config_manual_original.yaml` (`scene_auto_tune` desactivado). Esa base **no** coincide con el flujo habitual con `config.out3_clean.yaml`; las cifras de esa corrida quedan **obsoletas** para comparar con tu pipeline real.

## Evidencia

### case_03 — referencia correcta (`config.out3_clean.yaml`)

**Vídeo:** `rabella_20211016_DSCF0005.mp4`.

| Métrica | Baseline (`config.out3_clean.yaml`) | Suavizado (`config_case03_trajectory_smooth.yaml`) |
|--------|-------------------------------------|-----------------------------------------------------|
| Directorio run | `runs/ralph_traj_smooth_case03_refout3/baseline` | `runs/ralph_traj_smooth_case03_refout3/smoothed` |
| `tracks_total` (meta) | 16 | 16 |
| `detections_kept` (meta) | 185 | 185 |
| Filas `events.csv` | 16 | 16 |
| `direction == exits` | 9 | 9 |
| `enters` | 1 | 1 |
| `inside` / `outside` | 6 / 0 | 6 / 0 |

**Evidencia visual obligatoria (run suavizado):**

- Raw: `runs/ralph_traj_smooth_case03_refout3/smoothed/tracks_overlay_raw.png`
- Suavizado: `runs/ralph_traj_smooth_case03_refout3/smoothed/tracks_overlay_smoothed.png`

Comparación directa: las polilíneas en `tracks_overlay_smoothed.png` deben verse **más rectas** que en `tracks_overlay_raw.png` para los mismos IDs.

## Conclusión

En **case_03** con **base `out3_clean`** y ventana 5:

- **Exits:** sin regresión (9 = 9).
- **Tracks y detecciones:** idénticos (16 tracks, 185 detecciones en `tracks.csv` raw).
- **Overlay:** dos artefactos generados; la versión suavizada es la referencia visual para validar H1.

### case_01 (`DSCF0005.AVI`, `config.out3_clean.yaml` vs `config_case01_trajectory_smooth.yaml`)

Mismo `scene_auto_tune` y resto de parámetros; solo difiere el suavizado en el run *smoothed*.

| Métrica | Baseline | Suavizado |
|--------|----------|-----------|
| `tracks_total` | 97 | 97 |
| `detections_kept` | 1461 | 1461 |
| Filas `events.csv` | 97 | 97 |
| **exits** | **25** | **31** |
| enters | 16 | 12 |
| inside / outside | 53 / 3 | 47 / 7 |

Overlays: `runs/ralph_traj_smooth_case01/smoothed/tracks_overlay_raw.png`, `runs/ralph_traj_smooth_case01/smoothed/tracks_overlay_smoothed.png`.

**Interpretación:** no hay pérdida de tracks ni de detecciones, pero **sí cambia la etiquetación de eventos** al usar centroides suavizados frente a la máscara (más cruces “exit”, menos “enter”, más “outside”). Eso puede interpretarse como **más sensibilidad al borde** o como ruido adicional según el criterio de negocio.

## Decisión

- **Mantener** el suavizado como opción de salida (`trajectory_smoothing_enabled`) y documentar que, si está activo, la geometría de `events.csv` usa centroides suavizados (coherente con dirección y máscara).
- **case_03:** criterio de éxito cumplido **respecto a `config.out3_clean.yaml`** (exits y métricas idénticas baseline vs suavizado; overlays en `runs/ralph_traj_smooth_case03_refout3/smoothed/`).
- **case_01:** el criterio estricto “sin aumento de ruido” en **exits** **no** se cumple tal cual (25 → 31 exits). Para vídeos largos y escena auto-tune, valorar **iteración 2:** ventana más pequeña (p. ej. 3), o suavizado **solo para overlay** dejando `events.csv` en raw si se prioriza estabilidad numérica de eventos frente a rectitud visual.

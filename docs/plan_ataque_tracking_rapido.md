# Plan de ataque para mejorar tracks rápidos en `palomeres_20230516_16052023_palo_conteo.mp4`

## Resumen
Evidencia observada en el vídeo y en el pipeline actual:

- El tracker actual no usa Kalman ni ByteTrack. Es un `GreedyTracker` con asignación húngara, predicción lineal por velocidad y gate fijo por distancia (`max_distance=120`, `max_missed=12`).
- El detector sí ve muchos vuelos rápidos: en este vídeo salen `225` tracks candidatos, con `117` auto-merges, pero sólo `9` tracks finales.
- El principal cuello de botella en este clip no es `temporal_burst`: se suprimen `72` frames, pero ninguno coincide con tracks rápidos (`>=500 px/s`).
- El principal rechazo es espacial y de longitud:
  - `42` tracks rápidos detectados.
  - `3` aceptados.
  - `39` rechazados.
  - `27` por `min_track_length;valid_region_gate`.
  - `11` por `valid_region_gate` solamente.
  - `1` por `min_track_length;min_track_path_length`.
- Para tracks rápidos rechazados, la dirección dominante es `NW/W/N` y casi todos viven fuera de la gate: `38/39` empiezan y acaban fuera.
- Bajo la decisión funcional elegida aquí, la gate sigue siendo restricción dura. Por tanto:
  - No trataré como bug los vuelos rápidos completos que jamás tocan la gate.
  - Sí trataré como bug la fragmentación/duplicación de vuelos rápidos que sí tocan o cruzan la gate.
- Hay evidencia de duplicación no resuelta dentro de gate:
  - Tracks aceptados `214` y `215` se solapan en 2 frames con distancia media `23 px`; esto parece el mismo evento fragmentado/duplicado.
- Hay evidencia menor de pérdida por filtros incluso tocando gate:
  - `track 63`: rápido, toca gate, pero cae por `min_track_length` y `min_track_path_length`.

## Diagnóstico priorizado

### 1. Problema más probable: política de retención demasiado rígida para eventos rápidos cortos
Impacto esperado: alto

Evidencia:

- La mayoría de vuelos rápidos que el sistema sí detecta mueren en filtrado final.
- El rescate actual sólo permite salvar tracks que fallen exclusivamente por `min_track_length`, y no cubre bien eventos rápidos muy cortos.
- En este clip hay al menos un caso dentro de gate que cae por filtros (`track 63`).

Verificación:

1. Medir cuántos tracks rápidos que tocan gate se recuperan al introducir un perfil “fast-track”.
2. Comparar `accepted_fast_gate_touching` y `fragmentation_per_fast_gate_event` antes/después.

### 2. Problema muy probable: merge insuficiente para subtracks rápidos solapados
Impacto esperado: alto

Evidencia:

- `117` merges automáticos indican fragmentación masiva antes del filtrado.
- Aun así sobreviven duplicados plausibles, como `214` y `215`.
- El merge actual acepta algunos casos de overlap, pero falla cuando el solape es corto y el segmento rápido es sólo una parte del vuelo.

Verificación:

1. Medir pares aceptados con solape temporal y distancia media baja.
2. Objetivo: `duplicate_accepted_pairs` debe bajar a `0` en este clip.

### 3. Problema probable: modelo de movimiento demasiado simple para saltos grandes y centroides inestables
Impacto esperado: medio

Evidencia:

- No hay Kalman, ni estado de confianza, ni gate adaptativo por velocidad/tamaño.
- En vuelos con motion blur, el centroide cambia mucho entre frames y el gate fijo de `120 px` es ciego a contexto.
- Algunos eventos rápidos sí generan track largo pero con múltiples subtracks paralelos.

Verificación:

1. Loggear por asociación: posición predicha, distancia al match, `missed`, tamaño del bbox y velocidad estimada.
2. Medir reducción en `track_births_per_fast_event` y aumento en `single_track_coverage`.

### 4. Problema secundario: preprocesado de detección demasiado suavizado para murciélagos muy pequeños
Impacto esperado: medio-bajo en este clip, mayor en otros

Evidencia:

- `blur_kernel=9` es agresivo para objetivos de 2-5 frames.
- En este vídeo los vuelos rápidos grandes suelen detectarse igual, así que no parece el cuello principal.
- Sí puede explicar pérdidas marginales cerca de gate para sujetos muy pequeños.

Verificación:

1. A/B con `blur_kernel=5`, `morph_open=1`, `min_area=8`.
2. Medir recall de eventos cortos tocando gate y FP/minuto.

### 5. Problema no prioritario aquí: burst suppression
Impacto esperado: nulo en este clip

Evidencia:

- `0` tracks rápidos se solapan con frames suprimidos.

## Cambios de interfaz/config recomendados
Añadir al bloque `tracking`:

```yaml
tracking:
  fast_track_enabled: true
  fast_track_speed_px_sec: 500
  fast_track_min_length: 3
  fast_track_min_duration_sec: 0.08
  fast_track_min_displacement: 40
  fast_track_min_path_length: 60
  fast_track_min_straightness: 0.65
  fast_track_require_gate_touch: true
  fast_track_gate_crossing_margin_px: 35

  adaptive_max_distance_enabled: true
  adaptive_max_distance_base: 120
  adaptive_max_distance_speed_gain: 1.25
  adaptive_max_distance_bbox_gain: 0.35
  adaptive_max_distance_cap: 220

  merge_fast_overlap_enabled: true
  merge_fast_overlap_max_mean_distance: 35
  merge_fast_overlap_min_common_frames: 2
  merge_fast_handoff_max_gap_frames: 3
  merge_fast_handoff_projection_tolerance_px: 45
```

Cambios de tipo/API internos:

- `Detection` debe incorporar `score`.
- `ActiveTrack` debe incorporar al menos:
  - `bbox_w`, `bbox_h`
  - `speed_px_sec`
  - `last_detection_score`
  - `consecutive_hits`
- Añadir export opcional `tracking_debug.csv` con lifecycle por frame.

## Quick Wins

### Q1. Perfil de aceptación específico para fast tracks tocando gate
Qué cambiar:

- Mantener el perfil actual para tracks lentos.
- Si un track cumple:
  - `mean_speed_px_sec >= 500`
  - `num_detections >= 3` o `duration_sec >= 0.08`
  - `path_length_px >= 60`
  - `straightness >= 0.65`
  - toca gate en inicio o fin
  entonces aceptar con umbrales rápidos, aunque no llegue a `min_track_length=6`.

Por qué ayuda:

- Ataca el único fallo claro dentro de gate que ya aparece en este clip.
- Mejora captura de sujetos que sólo viven 3-5 frames.

Dificultad: baja
Coste computacional: despreciable
Riesgo de FP: bajo-medio

Validación:

- `fast_gate_touching_rejected_count` debe bajar.
- `FP/min` no debe subir más de `+10%`.

Snippet:

```python
is_fast = mean_speed >= cfg.fast_track_speed_px_sec
touches_gate = s_in or e_in

if is_fast and touches_gate:
    accept = (
        len(track_points) >= cfg.fast_track_min_length
        or duration >= cfg.fast_track_min_duration_sec
    ) and path_length >= cfg.fast_track_min_path_length \
      and displacement >= cfg.fast_track_min_displacement \
      and straightness >= cfg.fast_track_min_straightness
```

### Q2. Merge extra para subtracks rápidos con solape corto
Qué cambiar:

- Añadir una segunda pasada de merge sólo para tracks rápidos.
- Condición:
  - solape `>=2` frames
  - distancia media `<=35 px`
  - o handoff con gap `<=3` frames y extrapolación consistente

Por qué ayuda:

- Debe fusionar casos tipo `214 + 215`.
- Reduce fragmentación y duplicados sin abrir el gate al ruido global.

Dificultad: baja-media
Coste computacional: bajo
Riesgo de FP: medio

Validación:

- `duplicate_accepted_pairs` debe pasar a `0`.
- `tracks_total` puede bajar sin perder cobertura visual.

### Q3. Logging de asociación y rechazo
Qué cambiar:

- Exportar por frame y track:
  - `pred_x/pred_y`
  - `match_distance`
  - `adaptive_gate`
  - `missed_before/after`
  - `track_birth_reason`
  - `track_kill_reason`
  - `filter_reject_reasons`

Por qué ayuda:

- Permite distinguir detector vs asociación vs filtro en una sola pasada.
- Hará verificables los cambios siguientes.

Dificultad: baja
Coste computacional: bajo
Riesgo de FP: nulo

Validación:

- No es mejora funcional; es requisito para iterar rápido.

## Mejoras intermedias

### M1. Gate adaptativo de asociación según velocidad y tamaño
Qué cambiar:

- Sustituir `max_distance` fijo por uno por track:
  - `gate = clamp(base, base + speed_per_frame*1.25 + bbox_diag*0.35, 220)`
- Mantener `base=120`.

Por qué ayuda:

- El track rápido puede dar saltos >120 px aparentes por blur/centroide.
- El track lento conserva una gate más contenida.

Dificultad: media
Coste computacional: bajo
Riesgo de FP: medio

Validación:

- Medir `association_breaks_on_fast_tracks`.
- Medir `track_births_near_existing_fast_track`.
- Mantener `ID switches / duplicate tracks` controlados.

### M2. Sustituir el predictor lineal simple por Kalman CV 2D
Qué cambiar:

- Estado mínimo: `[x, y, vx, vy]`.
- Ruido de proceso más alto cuando `bbox_diag` crece o el `score` cae.
- Medición: centroide del blob.

Por qué ayuda:

- Más robusto a 1-2 frames con motion blur y centroides inestables.
- Permite gate elíptico por covarianza, no sólo radio fijo.

Dificultad: media
Coste computacional: bajo
Riesgo de FP: medio

Validación:

- Reducción de `fragmentation_per_fast_gate_event`.
- Aumento de `track_continuity_frames`.

### M3. Detector más sensible para objetivo pequeño, pero sólo cerca de gate
Qué cambiar:

- Experimento local:
  - `blur_kernel: 9 -> 5`
  - `morph_open: 3 -> 1`
  - `min_area: 12 -> 8`
- Aplicar sólo en una banda dilatada alrededor de la gate para contener FP.

Por qué ayuda:

- Reduce pérdida de sujetos muy pequeños antes de salir disparados.
- Limita el coste de FPs a la zona funcional.

Dificultad: baja-media
Coste computacional: bajo
Riesgo de FP: medio

Validación:

- `recall_short_fast_gate_events`
- `FP/min` dentro de banda de gate
- `detections_per_frame` en gate

### M4. Añadir score de detección para preparar asociación en dos etapas
Qué cambiar:

- Calcular `Detection.score` con mezcla de:
  - contraste máximo/medio en `diff`
  - consistencia área/aspect ratio
  - proximidad a gate
- Primera asociación con score alto.
- Segunda asociación con score medio para rescatar fast tracks.

Por qué ayuda:

- Emula la idea útil de ByteTrack sin necesitar detector neuronal.
- Reduce pérdida de detecciones débiles de 1-3 frames.

Dificultad: media
Coste computacional: bajo
Riesgo de FP: medio

Validación:

- `recovered_low_score_matches`
- `false_births_from_low_score`
- `single_track_coverage`

## Cambios estructurales

### S1. Asociación en dos etapas estilo ByteTrack adaptada a blobs
Qué cambiar:

1. Separar detecciones en `high_score` y `low_score`.
2. Asociar tracks confirmados con `high_score`.
3. Reintentar tracks no emparejados con `low_score`.
4. Crear tracks nuevos sólo con `high_score`, salvo modo fast-track cerca de gate.

Por qué ayuda:

- Para vuelos rápidos de 2-5 frames, la detección débil no debería crear un track nuevo siempre; a veces debe prolongar uno existente.
- Es la mejora con mejor equilibrio entre recall y control de FPs.

Dificultad: media-alta
Coste computacional: bajo-medio
Riesgo de FP: medio

Validación:

- `track_fragmentation_index`
- `new_track_birth_rate`
- `accepted_fast_gate_touching`

### S2. Rama temporal multi-frame para “fast exits” cerca de gate
Qué cambiar:

- Detector auxiliar con ventana de `3` frames:
  - `max(diff_t, diff_t-1, diff_t-2)` o suma temporal
  - búsqueda de streaks elongados cerca de gate
- El resultado no reemplaza al detector principal; propone tracklets cortos.

Por qué ayuda:

- El murciélago muy rápido deja huella más estable en 3 frames que en 1 frame.
- Captura sujetos que el frame-by-frame ve de forma fragmentaria.

Dificultad: alta
Coste computacional: medio
Riesgo de FP: medio-alto

Validación:

- `recall_3to5_frame_gate_events`
- `fast_track_duplication_rate`
- inspección visual de montajes

### S3. Optical flow local para puenteo de gaps de 1-2 frames
Qué cambiar:

- Cuando un fast track tocando gate se queda sin match durante `<=2` frames:
  - estimar desplazamiento local con LK flow o template matching
  - generar pseudo-medición con baja confianza

Por qué ayuda:

- Útil cuando el detector falla un frame por blur o saturación.
- Más barato que rehacer el detector.

Dificultad: alta
Coste computacional: medio
Riesgo de FP: medio

Validación:

- `gap_recovery_success_rate`
- `pseudo_measurement_usage`
- revisión frame a frame de eventos recuperados

## Checklist experimental

### Bloque A. Baseline instrumentado
1. Añadir `tracking_debug.csv`.
2. Extraer baseline de este clip:
   - `accepted_fast_gate_touching`
   - `rejected_fast_gate_touching`
   - `duplicate_accepted_pairs`
   - `fragmentation_per_fast_gate_event`
   - `FP/min`

### Bloque B. Filtro rápido
1. Activar `fast_track_*`.
2. Validar que sólo cambian eventos tocando gate.
3. Revisar manualmente todos los tracks aceptados nuevos.

### Bloque C. Merge rápido
1. Añadir merge fast-overlap/handoff.
2. Confirmar fusión de casos tipo `214+215`.
3. Verificar que no une vuelos distintos.

### Bloque D. Asociación adaptativa
1. Activar gate adaptativo.
2. Medir reducción en births duplicados y cortes.
3. Si suben FPs, bajar `adaptive_max_distance_cap`.

### Bloque E. Detección local sensible
1. Ejecutar A/B con:
   - `blur_kernel=5`
   - `morph_open=1`
   - `min_area=8`
2. Limitarlo a gate dilatada.
3. Comparar recall corto vs FP.

### Bloque F. Score + dos etapas
1. Añadir `Detection.score`.
2. Habilitar asociación high/low score.
3. Medir rescates reales y nacimientos falsos.

### Bloque G. Rama temporal
1. Prototipo de detector 3-frame cerca de gate.
2. Sólo si A-F no resuelven bien los eventos de 3-5 frames.

## Métricas antes/después
Métricas principales:

- `complete_track_rate_fast_gate`
- `fragmentation_per_fast_gate_event`
- `duplicate_accepted_pairs`
- `accepted_fast_gate_touching`
- `rejected_fast_gate_touching`
- `FP_per_minute_gate_band`

Definiciones operativas:

- `fast_gate_event`: vuelo real con `>=3` frames, `>=500 px/s`, que toca gate.
- `complete_track`: un único track final cubre al menos `80%` de frames del evento.
- `fragmentation_per_event`: número de tracks finales asignados al mismo evento.
- `duplicate_accepted_pairs`: pares de tracks aceptados con solape temporal y distancia media `<30 px`.

Métricas secundarias:

- `mean_match_distance_fast`
- `unmatched_fast_frames`
- `merge_count_fast`
- `new_track_births_near_active_fast_track`
- `recovered_low_score_matches`

## Visualizaciones de debugging
Generar siempre:

- `scatter_speed_vs_length.png`
  - color por `accepted/reject_reason`
- `fast_track_endpoints.png`
  - inicios/finales de tracks rápidos sobre el frame
- `overlap_pairs_fast.csv`
  - pares candidatos a duplicado
- `event_strips/`
  - tiras de 5-8 frames para cada evento rápido tocando gate
- `association_debug_overlay.mp4`
  - predicción, gate adaptativa, match elegido y detecciones descartadas

Visualizaciones prioritarias para este clip:

- montage de `214/215`
- montage de `187/192`
- montage de `track 63`

## Logs recomendados
Por frame y track:

```csv
frame,track_id,state,pred_x,pred_y,gate_px,matched,det_x,det_y,match_dist,
det_score,bbox_w,bbox_h,speed_px_sec,missed_before,missed_after,birth_reason,kill_reason
```

Por merge:

```csv
track_a,track_b,reason,common_frames,mean_distance,connector_cosine,
projected_gap_px,accepted_merge
```

Por filtro final:

```csv
track_id,is_fast,touches_gate,num_detections,duration_sec,displacement_px,
path_length_px,straightness,mean_speed_px_sec,accepted,reject_reasons,rescue_mode
```

## Orden de implementación recomendado
1. Instrumentación y métricas.
2. Perfil `fast_track_*` en filtro final.
3. Merge rápido para overlaps/handoffs cortos.
4. Gate adaptativo por velocidad/tamaño.
5. `Detection.score` + asociación en dos etapas.
6. Kalman CV 2D.
7. Rama temporal multi-frame cerca de gate.
8. Optical flow local sólo si aún quedan gaps de 1-2 frames.

## Supuestos y decisiones fijadas
- La `valid_region_gate` sigue siendo restricción funcional dura.
- No se persigue recuperar vuelos rápidos que nunca tocan gate.
- El objetivo en este clip es mejorar continuidad y rescate de vuelos rápidos relevantes a la gate, no maximizar recall global en todo el frame.
- `temporal_burst` no se toca en la primera iteración porque no está causando las pérdidas rápidas aquí.
- NMS no es foco actual porque no existe una etapa NMS real en este pipeline.
- “Kalman” no es un parámetro a ajustar sino una capacidad nueva a introducir.

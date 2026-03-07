# bat_tracker

Proyecto Python para Linux orientado a CPU que procesa videos IR monocromos de cueva y genera:

- `background.png`: fondo por mediana temporal
- `valid_region/`: mascara vertical de zona valida por iluminacion horizontal
- `tracks.csv`: trayectorias 2D por objeto
- `tracks_overlay.png`: trayectorias sobre el fondo
- `meta.json`: parametros y metricas de ejecucion

No genera video anotado y no usa modelos entrenados.

## Instalacion

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Opcionalmente, instalacion como paquete:

```bash
pip install -e .
```

## Uso

```bash
bat-tracker --input /path/video.mp4 --output /path/out_dir --config /path/config.yaml
```

O sin `--config` para usar defaults.

Tambien puede ejecutarse sin instalar entrypoint:

```bash
python -m bat_tracker.cli --input /path/video.mp4 --output /path/out_dir --config /path/config.yaml
```

Generacion standalone de mascara vertical valida:

```bash
python -m bat_tracker.valid_region \
  --input /path/out_dir/background.png \
  --output /path/out_dir/valid_region \
  --blur-kernel-size 151 \
  --threshold-ratio 0.45 \
  --safety-margin 10
```

## Entradas

- `--input` (obligatorio): ruta a video IR monocromo, por ejemplo `.mp4`.
- `--output` (obligatorio): carpeta de salida donde se escriben resultados.
- `--config` (opcional): YAML con parametros; si no se pasa, usa defaults internos.

Ejemplos de configuracion incluidos:

- `config.yaml.example` (base)
- `config.thrutracker_like.yaml` (perfil similar a ThruTracker)
- `config.rabella.yaml` (perfil mas sensible)
- `config.out3_clean.yaml` (perfil limpio para escenas tipo out3 con menos ruido)

## Salidas

Se escriben en la carpeta indicada por `--output`:

- `background.png`: fondo estimado por mediana temporal.
- `valid_region/mask.png`: mascara binaria vertical (255 zona valida, 0 laterales invalidos).
- `valid_region/overlay.png`: debug visual de banda valida sobre la imagen.
- `valid_region/profile.png`: perfil horizontal (raw + suavizado + cortes).
- `tracks.csv`: trayectorias 2D por deteccion y frame.
- `tracks_overlay.png`: trayectorias dibujadas sobre `background.png`.
- `track_clips/` (opcional): clips de video por track (`track_0001_000120-000186.mp4`, etc.).
- `meta.json`: metadatos del video, parametros efectivos y metricas de ejecucion.
  - incluye bloque `valid_region` con `x_start`, `x_end`, `width` y `method`.

## Formato de tracks.csv

Columnas exactas:

`video_id,track_id,frame,time_sec,x,y,vx,vy,bbox_x1,bbox_y1,bbox_x2,bbox_y2,area`

## Pipeline implementado

1. Lectura del video y metadatos.
2. Generacion de `background.png` por mediana temporal de un muestreo de frames.
3. Deteccion de foreground por diferencia absoluta con el fondo.
4. Umbral binario (fijo u Otsu) + morfologia (open/close) + contornos.
5. Filtrado de blobs por area minima/maxima.
6. Tracking 2D frame a frame con asignacion greedy por distancia maxima y prediccion por velocidad para reducir cortes.
7. Export de `tracks.csv` y render final `tracks_overlay.png` (color por track, primer punto mas grande).
8. Si `valid_region.enabled`, calculo de banda vertical valida desde iluminacion horizontal y guardado en `valid_region/*`.
9. Export de `meta.json` con parametros, metadatos y metricas.
   - incluye `postprocess.auto_merges_applied` cuando `tracking.auto_merge_suggested` esta activo.

## Configuracion

Usa `config.yaml.example` como base.

- `background.sample_frames`: numero de frames para mediana temporal
- `background.uniform_sampling`: muestreo uniforme en todo el video
- `detection.*`: parametros de blur, threshold, morfologia y area
  - `detection.threshold_mode`: `fixed` o `otsu`
  - `detection.otsu_offset`: ajuste fino sobre umbral Otsu (negativo = mas sensible)
  - `detection.max_global_intensity_shift`: descarta frame si el brillo medio difiere demasiado del fondo (`-1` desactiva)
  - `detection.max_foreground_ratio`: descarta frame si el porcentaje de foreground es demasiado alto (`-1` desactiva)
  - `detection.max_detections_per_frame`: descarta frame si supera este numero de blobs (`0` desactiva)
  - `detection.roi_x_min/roi_x_max/roi_y_min/roi_y_max`: limita detecciones a una ROI por centroide (`-1` desactiva cada limite)
  - `detection.temporal_burst_*`: gate temporal por rafagas de detecciones (desactiva con `0`)
    - `temporal_burst_min_detections`: umbral de detecciones altas por frame
    - `temporal_burst_window_frames`: tamano de ventana temporal
    - `temporal_burst_trigger_frames`: frames altos dentro de ventana para activar suppression
    - `temporal_burst_cooldown_frames`: frames suprimidos tras activacion
- `tracking.*`: distancia maxima de asociacion, tolerancia a frames perdidos y filtros minimos por trayectoria
  - `tracking.min_track_length`: minimo de puntos por trayectoria
  - `tracking.min_track_displacement`: desplazamiento neto minimo (pixeles)
  - `tracking.min_track_path_length`: recorrido acumulado minimo (pixeles)
  - `tracking.min_track_straightness`: rectitud minima `desplazamiento/recorrido` (0..1)
  - `tracking.auto_merge_suggested`: fusion automatica postproceso de tracks potencialmente duplicados
  - `tracking.merge_max_gap_frames` y `tracking.merge_max_endpoint_distance`: merge por handoff cercano (fin->inicio)
  - `tracking.merge_overlap_min_common_frames`: minimo de frames comunes para evaluar merge por solape
  - `tracking.merge_overlap_max_mean_distance`: distancia media maxima en frames comunes
  - `tracking.merge_overlap_min_direction_cosine`: coherencia minima de direccion entre tracks solapados
- `valid_region.*`: mascara vertical valida para eliminar vignette lateral IR sin recortar interior oscuro de cueva
  - `valid_region.enabled`: activa/desactiva etapa
  - `valid_region.input_image`: si se define, usa esta imagen en vez de `background.png`
  - `valid_region.blur_kernel_size` y `valid_region.profile_smooth_window`: deben ser impares
  - `valid_region.threshold_ratio`: fraccion del pico del perfil para definir region valida
  - `valid_region.safety_margin`: recorte adicional en pixeles por lado
  - `valid_region.min_region_width_ratio`: evita regiones absurdamente estrechas
- `output.*`: estilo del overlay
  - `output.overlay_draw_track_labels`: dibuja el numero de `track_id` junto al inicio de cada track
  - `output.overlay_draw_track_labels_at_end`: dibuja el numero de `track_id` al final del track
  - `output.overlay_label_font_scale` y `output.overlay_label_thickness`: estilo de etiqueta
  - `output.export_track_clips`: exporta clips por track en una carpeta
  - `output.track_clips_subdir`: nombre de la carpeta de clips dentro del output
  - `output.track_clips_padding_frames`: frames extra antes/despues del rango del track

## Ajuste rapido para mejorar recall/continuidad

Si faltan sujetos o aparecen tracks cortados:

1. subir `tracking.max_distance` (ej. `70-90`)
2. subir `tracking.max_missed` (ej. `15-25`)
3. bajar `detection.min_area` (ej. `4-8`)
4. usar `detection.threshold_mode: otsu` y ajustar `detection.otsu_offset` (ej. `-6` mas sensible)

Si aparecen demasiados tracks de ruido:

1. subir `tracking.min_track_displacement` (ej. `20-40`)
2. subir `tracking.min_track_path_length` (ej. `30-80`)
3. subir `tracking.min_track_straightness` (ej. `0.1-0.3`)
4. subir `background.sample_frames` (ej. `100-300`) para estabilizar `background.png`
5. activar gates anti-flicker en `detection`: `max_global_intensity_shift`, `max_foreground_ratio`, `max_detections_per_frame`
6. activar gate temporal `detection.temporal_burst_*` para suprimir rafagas cortas de ruido

## Tests minimos

```bash
pytest
```

Los tests cubren deteccion, tracking y export/render de salida.

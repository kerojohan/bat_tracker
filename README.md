# bat_tracker

Proyecto Python para Linux orientado a CPU que procesa videos IR monocromos de cueva y genera:

- `background.png`: fondo por mediana temporal
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

## Entradas

- `--input` (obligatorio): ruta a video IR monocromo, por ejemplo `.mp4`.
- `--output` (obligatorio): carpeta de salida donde se escriben resultados.
- `--config` (opcional): YAML con parametros; si no se pasa, usa defaults internos.

Ejemplos de configuracion incluidos:

- `config.yaml.example` (base)
- `config.thrutracker_like.yaml` (perfil similar a ThruTracker)
- `config.rabella.yaml` (perfil mas sensible)

## Salidas

Se escriben en la carpeta indicada por `--output`:

- `background.png`: fondo estimado por mediana temporal.
- `tracks.csv`: trayectorias 2D por deteccion y frame.
- `tracks_overlay.png`: trayectorias dibujadas sobre `background.png`.
- `meta.json`: metadatos del video, parametros efectivos y metricas de ejecucion.

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
8. Export de `meta.json` con parametros, metadatos y metricas.

## Configuracion

Usa `config.yaml.example` como base.

- `background.sample_frames`: numero de frames para mediana temporal
- `background.uniform_sampling`: muestreo uniforme en todo el video
- `detection.*`: parametros de blur, threshold, morfologia y area
  - `detection.threshold_mode`: `fixed` o `otsu`
  - `detection.otsu_offset`: ajuste fino sobre umbral Otsu (negativo = mas sensible)
- `tracking.*`: distancia maxima de asociacion, tolerancia a frames perdidos y longitud minima de track
- `output.*`: estilo del overlay

## Ajuste rapido para mejorar recall/continuidad

Si faltan sujetos o aparecen tracks cortados:

1. subir `tracking.max_distance` (ej. `70-90`)
2. subir `tracking.max_missed` (ej. `15-25`)
3. bajar `detection.min_area` (ej. `4-8`)
4. usar `detection.threshold_mode: otsu` y ajustar `detection.otsu_offset` (ej. `-6` mas sensible)

## Tests minimos

```bash
pytest
```

Los tests cubren deteccion, tracking y export/render de salida.

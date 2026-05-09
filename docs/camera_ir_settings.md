# Configuración recomendada de cámara IR para optimizar detección de murciélagos

## Prioridad: congelar el movimiento

Un murciélago a 2000 px/s con una cámara a 25 fps se desplaza ~80 px entre frames.
Con obturación lenta se convierte en una mancha alargada indetectable.
**La prioridad absoluta es minimizar el motion blur.**

---

## Parámetros críticos

### Velocidad de obturación (shutter speed)

| fps | Shutter mínimo | Ideal |
|-----|---------------|-------|
| 25 | 1/500s | **1/1000s** |
| 30 | 1/500s | **1/1000s** |
| 60 | 1/500s | **1/2000s** |

Compensa la pérdida de luz subiendo ISO. Mejor ruido que motion blur.

### ISO / Ganancia

```
ISO 3200–6400 (cámara IR-modificada)
```

El ruido de ISO alto se filtra con `blur_kernel: 3-5` y `noise_mask` en el pipeline.
No usar ISO automático — provoca cambios de contraste que saturan el detector.

### Framerate

```
50 o 60 fps mínimo
```

Impacto directo en el algoritmo:
- A 25 fps: bat 2000 px/s = 80 px/frame → difícil de asociar
- A 60 fps: bat 2000 px/s = 33 px/frame → bien dentro del `adaptive_max_distance_cap`
- Además: 2.4× más puntos por trayectoria → tracks más largos, mejor straightness

### Iluminación IR

```
Potencia máxima. Longitud de onda: 850nm o 940nm.
Distribución uniforme en toda la escena.
```

Las zonas oscuras (x < 300 en palomeres) pierden el 40% del corredor de salida.
Un IR potente y bien distribuido elimina ese problema.
Si es posible, usar 2 o más focos IR en ángulos distintos para evitar sombras.

---

## Parámetros importantes

### Enfoque

```
Manual, fijado a la distancia de la entrada de la cueva.
NUNCA autofocus ni focus peaking automático.
```

El autofocus genera cambios bruscos de contraste global que saturan
`max_global_intensity_shift` y producen ráfagas de falsos positivos.

### Reducción de ruido

```
Desactivar reducción de ruido temporal (3D NR, temporal NR).
Desactivar reducción de ruido espacial fuerte.
```

La reducción de ruido temporal compara frames consecutivos y difumina
objetos en movimiento — exactamente lo que queremos detectar.

### Estabilización de imagen

```
Desactivar (OIS, IBIS, estabilización digital).
```

La estabilización mueve el fondo artificialmente entre frames, generando
diferencias falsas en el background subtraction.

### Perfil de imagen

```
Perfil plano o log (contraste bajo).
No usar perfiles de alto contraste (Vivid, Landscape).
```

Un perfil plano preserva detalle en zonas oscuras donde los murciélagos
son apenas visibles. El contraste se puede aplicar en post-procesado.

### Compresión / Formato

```
MJPEG (Motion JPEG) o H.264 a máximo bitrate.
Desactivar compresión H.265/HEVC.
```

La compresión agresiva (H.265, bitrate bajo) crea artefactos de bloque
en zonas oscuras que el detector interpreta como movimiento.

---

## Tabla resumen

| Parámetro | Valor típico (auto) | Recomendado |
|-----------|--------------------|-------------|
| Shutter | Auto / 1/30-1/60 | **1/1000** |
| ISO | Auto | **3200-6400 manual** |
| FPS | 25-30 | **60** |
| Enfoque | Auto / AF-C | **Manual fijo** |
| IR iluminación | Variable | **Máxima, uniforme** |
| Noise reduction | ON | **OFF** |
| Estabilización | ON | **OFF** |
| Picture profile | Standard / Vivid | **Flat / Log** |
| Formato | H.265 / H.264 bajo | **MJPEG / H.264 máx bitrate** |

---

## Impacto estimado en el algoritmo

| Cambio | Efecto |
|--------|--------|
| Shutter 1/1000 | Blobs nítidos, sin colas de motion blur |
| 60 fps | 2.4× más detecciones, saltos más cortos entre frames |
| IR uniforme | El bat no desaparece al salir de la zona iluminada |
| NR off | Sin fantasmas ni difuminado en objetos en movimiento |
| Flat profile | Más detalle en sombras donde el bat es más tenue |
| Enfoque manual | Sin picos de contraste que saturan el detector |

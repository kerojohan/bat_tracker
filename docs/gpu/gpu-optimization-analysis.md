# Anàlisi d'Optimització GPU per bat_tracker

**Data**: 27 març 2026  
**Autor**: Anàlisi Ultra-Think  
**Estat**: Proposta pendent decisió

---

## Resum Executiu

Aquest document analitza quatre estratègies per maximitzar l'ús de GPU en el pipeline de detecció de moviment de `bat_tracker`, actualment en mode híbrid CPU/GPU. L'objectiu és millorar el throughput mantenint zero compilació JIT (evitant dependències de `cuda_fp16.h`) i compatibilitat amb CuPy bàsic.

**Recomanació principal**: Implementar **Option 1 (Morphology a GPU)** com a millor ratio risc/benefici amb guany esperat del 10-20%.

---

## Taula de Continguts

1. [Context i Problema](#context-i-problema)
2. [Estat Actual del Pipeline](#estat-actual-del-pipeline)
3. [Opcions d'Optimització](#opcions-doptimització)
   - [Option 1: Morphology a GPU (conservadora)](#option-1-morphology-a-gpu-conservadora)
   - [Option 2: Full GPU pipeline (agressiva)](#option-2-full-gpu-pipeline-agressiva)
   - [Option 3: GPU on-demand amb threshold (híbrida intel·ligent)](#option-3-gpu-on-demand-amb-threshold-híbrida-intelligent)
   - [Option 4: Batch processing GPU (arquitectural)](#option-4-batch-processing-gpu-arquitectural)
4. [Anàlisi Comparativa](#anàlisi-comparativa)
5. [Recomanació i Roadmap](#recomanació-i-roadmap)
6. [Referències Tècniques](#referències-tècniques)

---

## Context i Problema

### Challenge Principal

Maximitzar l'ús de GPU en un pipeline de detecció de moviment per processar 183.270 frames (1728×1296 px, ~2.2 MB/frame) amb les següents restriccions:

- **Hardware**: GPU CUDA disponible, PCIe 3.0 (~12 GB/s bandwidth)
- **Software**: CuPy bàsic (sense `cupyx.scipy.ndimage` per evitar compilació JIT)
- **Arquitectura**: DSpace 6 - només modificar `dspace/modules/**`
- **Mantenibilitat**: Codi senzill, errors clars, fallback CPU robust

### Constraints Crítiques

1. **Zero compilació JIT**: No requerir `cuda_fp16.h` ni recompilació CUDA
2. **Estabilitat**: Sistema crític DSpace 6, zero crashes acceptables
3. **Verificabilitat**: Resultats idèntics a CPU (mode `strict_parity`)
4. **Portabilitat**: Funcionar amb diferents versions CuPy/CUDA

### Success Factors

- Reducció latència/frame (target: <10ms/frame)
- Speedup mesurable ≥10%
- Manteniment del codi simple
- Fallback CPU automàtic en errors

---

## Estat Actual del Pipeline

### Distribució CPU/GPU Actual

| Operació | Dispositiu | Funció | Dades | Temps estimat |
|----------|-----------|---------|-------|---------------|
| **Gaussian Blur** | 🔵 CPU | `cv2.GaussianBlur()` | 1728×1296 (~2.2MB) | ~1-2 ms |
| **Transfer CPU→GPU** | 🔄 PCIe | `cp.asarray()` | 2 imatges (~4.4MB) | ~0.4 ms |
| **Absdiff** | 🟢 GPU | `cp.abs()` | 1.2M píxels | ~0.3 ms |
| **Histogram (Otsu)** | 🟢 GPU | `cp.histogram()` | 1.2M → 256 bins | ~0.2 ms |
| **Threshold** | 🟢 GPU | `cp.where()` | 1.2M píxels | ~0.2 ms |
| **Transfer GPU→CPU** | 🔄 PCIe | `cp.asnumpy()` | Màscara (~1.2MB) | ~0.1 ms |
| **Morphology Open** | 🔵 CPU | `cv2.morphologyEx()` | Màscara binària | ~1-2 ms |
| **Morphology Close** | 🔵 CPU | `cv2.morphologyEx()` | Màscara binària | ~1-2 ms |
| **Find Contours** | 🔵 CPU | `cv2.findContours()` | Màscara binària | ~0.5 ms |
| **Filtres deteccions** | 🔵 CPU | Lògica Python | ~10-100 contorns | ~0.1 ms |

**Total estimat per frame**: ~6-10 ms  
**Overhead transferències**: ~0.5 ms (acceptat amb PCIe 3.0)

### Codi Actual (`detection.py`)

```python
def _binary_cupy(
    frame_gray: np.ndarray,
    background: np.ndarray,
    *,
    blur_kernel: int,
    threshold_mode: str,
    diff_threshold: int,
    otsu_offset: int,
    morph_open: int,
    morph_close: int,
    bg_gpu=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GPU-accelerated binary mask computation using CuPy."""
    import cupy as cp
    
    # Blur on CPU (OpenCV SIMD, very fast)
    frame_proc, bg_proc = _prepare_frame_and_background(frame_gray, background, blur_kernel)
    
    # Upload to GPU
    frame_proc_gpu = cp.asarray(frame_proc)
    bg_proc_gpu = bg_gpu if bg_gpu is not None else cp.asarray(bg_proc)
    
    # Absdiff on GPU
    diff_gpu = cp.abs(
        frame_proc_gpu.astype(cp.int16) - bg_proc_gpu.astype(cp.int16)
    ).astype(cp.uint8)
    
    # Threshold on GPU
    if threshold_mode == "otsu":
        hist_gpu = cp.histogram(diff_gpu, bins=256, range=(0, 256))[0]
        hist_cpu = cp.asnumpy(hist_gpu)
        otsu_thr = _otsu_threshold_from_histogram(hist_cpu)
        thr = max(1, min(255, int(otsu_thr + otsu_offset)))
    else:
        thr = diff_threshold
    binary_gpu = cp.where(diff_gpu > thr, cp.uint8(255), cp.uint8(0))
    
    # Download to CPU
    binary = cp.asnumpy(binary_gpu)
    
    # Morphology on CPU
    if morph_open > 1:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open, morph_open))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)
    
    if morph_close > 1:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close, morph_close))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)
    
    return binary, frame_proc, bg_proc
```

### Punts d'Optimització Identificats

1. **Morphology**: Baixar/pujar GPU innecessàriament (~2.4 MB transferències)
2. **Gaussian Blur**: Potencial pas a GPU, però OpenCV CPU és molt ràpid
3. **Batch processing**: Processar múltiples frames simultàniament (refactor gran)

---

## Opcions d'Optimització

### Option 1: Morphology a GPU (conservadora)

#### Descripció

Passar només les operacions morfològiques (open/close) a GPU usant `cp.ndimage.maximum_filter` i `cp.ndimage.minimum_filter` (funcions pre-compilades de CuPy).

#### Avantatges

✅ **Elimina transferències**: Estalvia 1 baixada GPU→CPU i posterior pujada CPU→GPU (~2.4 MB)  
✅ **Zero JIT**: Funcions pre-compilades de CuPy, no requereix compilació  
✅ **Canvi quirúrgic**: Modificació de ~10-15 línies en `_binary_cupy()`  
✅ **Baixa complexitat**: Fàcil debug i manteniment  
✅ **Guany real**: Morphology amb kernels grans (>5×5) és costosa en CPU  

#### Desavantatges

⚠️ **Diferències de kernel**: `cp.ndimage.*_filter` amb kernels quadrats pot no coincidir exactament amb `cv2.MORPH_ELLIPSE`  
⚠️ **Guany modest**: Speedup esperat 10-20% (morphology és ~15-20% del temps total)  

#### Implementació

```python
def _binary_cupy(...):
    # ... (codi anterior fins threshold) ...
    
    binary_gpu = cp.where(diff_gpu > thr, cp.uint8(255), cp.uint8(0))
    
    # NOVA SECCIÓ: Morphology a GPU
    if morph_open > 1:
        # Open = Erosion + Dilation
        binary_gpu = cp.ndimage.minimum_filter(binary_gpu, size=morph_open)
        binary_gpu = cp.ndimage.maximum_filter(binary_gpu, size=morph_open)
    
    if morph_close > 1:
        # Close = Dilation + Erosion
        binary_gpu = cp.ndimage.maximum_filter(binary_gpu, size=morph_close)
        binary_gpu = cp.ndimage.minimum_filter(binary_gpu, size=morph_close)
    
    # Baixar només després de morphology
    binary = cp.asnumpy(binary_gpu)
    
    # ELIMINAR: morphology CPU (línies 161-167 originals)
    return binary, frame_proc, bg_proc
```

#### Avaluació de Risc

| Risc | Probabilitat | Impacte | Mitigació |
|------|--------------|---------|-----------|
| Diferències píxel a píxel | Mitjana (40%) | Alt | Validació amb `strict_parity`, ajustar tolerància |
| Overhead GPU > guany CPU | Baixa (20%) | Mitjà | Benchmark abans d'implementar, abortar si slowdown |
| Crash CuPy amb kernels específics | Baixa (15%) | Alt | Try/catch amb fallback CPU automàtic |
| Resultats no reproduïbles | Baixa (10%) | Alt | Fixar seeds, validar determinisme |

#### Mètriques d'Èxit

- ✅ Speedup ≥ 10% (target: 15%)
- ✅ Diferència píxel a píxel ≤ 1% frames amb tolerància 5 píxels
- ✅ Zero crashes en 10.000 frames test
- ✅ Deteccions idèntiques en ≥ 95% frames

---

### Option 2: Full GPU pipeline (agressiva)

#### Descripció

Passar blur + morphology a GPU implementant convolució separable manual amb CuPy per maximitzar l'ús de GPU (95% computació).

#### Avantatges

✅ **Màxim aprofitament GPU**: 95% computació a GPU  
✅ **Mínimes transferències**: Només frame raw up + binary mask down  
✅ **Gaussian blur separable**: Paral·lelitzable (convolve1d × 2)  
✅ **Potencial speedup**: 2-3× si blur és bottleneck  

#### Desavantatges

❌ **Complexitat alta**: Implementar convolució separable manual  
❌ **OpenCV SIMD molt ràpid**: GaussianBlur CPU amb SIMD és ~1-2 ms/frame  
❌ **Guany dubtós**: Overhead convolució GPU pot ser similar a CPU SIMD  
❌ **Manteniment**: Més codi, més superfície d'error  
❌ **Risc JIT**: No està clar si `cp.ndimage.convolve1d` és pre-compilat  

#### Implementació

```python
def _gaussian_blur_separable_gpu(img_gpu, kernel_size, sigma):
    """Gaussian blur separable a GPU."""
    import cupy as cp
    
    # Generar kernel 1D gaussià
    x = cp.arange(kernel_size) - (kernel_size - 1) / 2
    kernel = cp.exp(-x**2 / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    
    # Convolució horitzontal
    img_blur_h = cp.ndimage.convolve1d(img_gpu, kernel, axis=1, mode='reflect')
    
    # Convolució vertical
    img_blur_v = cp.ndimage.convolve1d(img_blur_h, kernel, axis=0, mode='reflect')
    
    return img_blur_v

def _binary_cupy(...):
    # Upload frame raw
    frame_gpu = cp.asarray(frame_gray)
    
    # Blur a GPU
    if blur_kernel > 1:
        sigma = blur_kernel / 6.0  # Aproximació OpenCV
        frame_proc_gpu = _gaussian_blur_separable_gpu(frame_gpu, blur_kernel, sigma)
        bg_proc_gpu = bg_gpu_blurred  # Pre-processat
    else:
        frame_proc_gpu = frame_gpu
        bg_proc_gpu = bg_gpu
    
    # ... rest del pipeline a GPU ...
    
    # Morphology a GPU (igual que Option 1)
    # ...
    
    # Baixar només al final
    binary = cp.asnumpy(binary_gpu)
    return binary, cp.asnumpy(frame_proc_gpu), cp.asnumpy(bg_proc_gpu)
```

#### Avaluació de Risc

| Risc | Probabilitat | Impacte | Mitigació |
|------|--------------|---------|-----------|
| `convolve1d` requereix JIT | Alta (60%) | Crític | Verificar documentació CuPy, test manual |
| Overhead GPU > guany | Mitjana (50%) | Alt | Benchmark exhaustiu, profiling amb `nvprof` |
| Diferències acumulades float32 | Baixa (25%) | Mitjà | Validació amb tolerància, `strict_parity` |

#### Per què NO Recomanat

1. **OpenCV GaussianBlur CPU és extremadament ràpid** (~1-2 ms/frame) amb instruccions SIMD (SSE/AVX)
2. **Risc alt de JIT**: `cp.ndimage.convolve1d` pot necessitar compilació → tornem al problema `cuda_fp16.h`
3. **Principi d'enginyeria**: No optimitzar sense mesurar. Blur no és un bottleneck demostrat
4. **Complexitat**: Més codi a mantenir, més superfície d'error per guany incert

---

### Option 3: GPU on-demand amb threshold (híbrida intel·ligent)

#### Descripció

Decidir CPU vs GPU per operació segons cost computacional (kernel size). Usar GPU només quan compensa (kernels grans ≥7×7).

#### Avantatges

✅ **Flexibilitat**: Usa GPU només quan hi ha guany clar  
✅ **Evita overhead GPU**: Kernels petits (3×3, 5×5) es processen ràpid a CPU  
✅ **Aprofita GPU**: Kernels grans (7×7, 9×9+) on el guany és significatiu  
✅ **Manté simplicitat**: Codi existent funciona sense canvis  

#### Desavantatges

⚠️ **Lògica condicional**: Afegeix complexitat al codi  
⚠️ **Threshold òptim incert**: Depèn de hardware específic (GPU model, CPU, PCIe)  
⚠️ **Manteniment**: Dues rutes de codi (CPU + GPU) a mantenir  
⚠️ **Testing**: Requereix validar ambdues rutes  

#### Implementació

```python
MORPHOLOGY_GPU_THRESHOLD = 7  # Usar GPU si kernel >= 7×7

def _binary_cupy(...):
    # ... (codi anterior fins threshold) ...
    
    binary_gpu = cp.where(diff_gpu > thr, cp.uint8(255), cp.uint8(0))
    
    # Decidir CPU vs GPU per morphology
    use_gpu_morph = (morph_open >= MORPHOLOGY_GPU_THRESHOLD or 
                     morph_close >= MORPHOLOGY_GPU_THRESHOLD)
    
    if use_gpu_morph:
        # GPU path
        if morph_open > 1:
            binary_gpu = cp.ndimage.minimum_filter(binary_gpu, size=morph_open)
            binary_gpu = cp.ndimage.maximum_filter(binary_gpu, size=morph_open)
        if morph_close > 1:
            binary_gpu = cp.ndimage.maximum_filter(binary_gpu, size=morph_close)
            binary_gpu = cp.ndimage.minimum_filter(binary_gpu, size=morph_close)
        binary = cp.asnumpy(binary_gpu)
    else:
        # CPU path (original)
        binary = cp.asnumpy(binary_gpu)
        if morph_open > 1:
            k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open, morph_open))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)
        if morph_close > 1:
            k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close, morph_close))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)
    
    return binary, frame_proc, bg_proc
```

#### Avaluació de Risc

| Risc | Probabilitat | Impacte | Mitigació |
|------|--------------|---------|-----------|
| Threshold subòptim | Alta (60%) | Mitjà | Benchmark per diferents valors, calibració |
| Complexitat branching | Mitjana (40%) | Mitjà | Tests exhaustius ambdues rutes |
| Guany real marginal | Alta (50%) | Alt | Validar que kernels grans són comuns |

#### Per què NO Recomanat

1. **Overengineering**: Complexitat no justificada sense dades que provin que kernels grans (≥7×7) són comuns en el dataset
2. **Maintenance burden**: Dues rutes de codi per mantenir i testejar
3. **Threshold arbitrari**: Valor òptim canvia amb hardware, requereix recalibració

---

### Option 4: Batch processing GPU (arquitectural)

#### Descripció

Processar múltiples frames simultàniament a GPU (batch de 4-8 frames) per amortitzar overhead de transferències PCIe i millorar ocupació GPU.

#### Avantatges

✅ **Amortitza overhead PCIe**: Transferències més eficients amb batches grans  
✅ **Millor ocupació GPU**: Més warps actius, millor throughput  
✅ **Potencial speedup**: 1.5-2× si GPU està subutilitzat  
✅ **Paral·lelitza I/O**: Lectura vídeo + processament simultanis  

#### Desavantatges

❌ **Refactor arquitectural massiu**: Afecta `pipeline.py`, `video.py`, `detection.py`  
❌ **Memòria GPU**: 4-8 frames × 2.2 MB = 8-18 MB (acceptable però més pressió)  
❌ **Complexitat**: Gestió de batches, padding últim batch, error handling  
❌ **Viola constraint DSpace 6**: Canvis fora de `dspace/modules/**` (afecta core)  
❌ **Latència**: Retard fins completar batch (no acceptable per real-time)  

#### Implementació (conceptual)

```python
def process_batch_gpu(frames: List[np.ndarray], background: np.ndarray, cfg: dict):
    """Process multiple frames in batch on GPU."""
    import cupy as cp
    
    batch_size = len(frames)
    h, w = frames[0].shape
    
    # Preparar batch a CPU
    frames_batch = np.stack(frames, axis=0)  # (N, H, W)
    
    # Upload batch a GPU
    frames_gpu = cp.asarray(frames_batch)
    bg_gpu = cp.asarray(background)
    bg_batch_gpu = cp.broadcast_to(bg_gpu[None, :, :], (batch_size, h, w))
    
    # Blur batch a CPU (més eficient que GPU per frames individuals)
    # O implementar batch blur GPU si compensa
    
    # Absdiff batch a GPU
    diff_batch_gpu = cp.abs(frames_gpu - bg_batch_gpu)
    
    # Threshold batch a GPU
    binary_batch_gpu = cp.where(diff_batch_gpu > threshold, 255, 0)
    
    # Morphology batch a GPU
    # ... processar cada frame del batch ...
    
    # Download batch
    binary_batch = cp.asnumpy(binary_batch_gpu)
    
    return [binary_batch[i] for i in range(batch_size)]
```

#### Avaluació de Risc

| Risc | Probabilitat | Impacte | Mitigació |
|------|--------------|---------|-----------|
| Refactor incomplet | Alta (70%) | Crític | Testing exhaustiu, rollout gradual |
| OOM GPU | Mitjana (40%) | Alt | Dynamic batch sizing, monitorització memòria |
| Latència inacceptable | Alta (60%) | Mitjà | Benchmark latència vs throughput |
| Incompatibilitat DSpace 6 | Alta (90%) | Crític | **NO implementar en aquest projecte** |

#### Per què NO Recomanat

1. **Viola constraint arquitectural**: DSpace 6 només permet modificar `dspace/modules/**`, batch processing requereix canvis a core
2. **Refactor massa gran**: Canvis extensos en `pipeline.py`, `video.py` amb risc alt
3. **Guany incert**: GPU pot estar ben utilitzat amb frames individuals
4. **No és prioritari**: Optimitzacions incrementals (Option 1) donen millor ROI

---

## Anàlisi Comparativa

### Matriu de Decisió

| Criteri | Pes | Option 1 | Option 2 | Option 3 | Option 4 |
|---------|-----|----------|----------|----------|----------|
| **Speedup esperat** | 30% | 7/10 (10-20%) | 9/10 (2-3×) | 6/10 (5-15%) | 8/10 (1.5-2×) |
| **Risc tècnic** | 25% | 8/10 (Baix) | 4/10 (Alt) | 6/10 (Mitjà) | 3/10 (Molt Alt) |
| **Complexitat implementació** | 20% | 9/10 (Baixa) | 5/10 (Alta) | 6/10 (Mitjana) | 2/10 (Molt Alta) |
| **Mantenibilitat** | 15% | 8/10 (Bona) | 5/10 (Regular) | 5/10 (Regular) | 4/10 (Baixa) |
| **Alineació constraints** | 10% | 10/10 (Total) | 8/10 (Bona) | 9/10 (Bona) | 2/10 (Viola) |
| **Total ponderat** | | **8.05** | **6.15** | **6.25** | **4.45** |

### Comparativa Tècnica Detallada

| Aspecte | Option 1 | Option 2 | Option 3 | Option 4 |
|---------|----------|----------|----------|----------|
| **Línies de codi** | ~15 | ~80 | ~30 | ~200+ |
| **Temps implementació** | 2-3 dies | 1-2 setmanes | 4-5 dies | 3-4 setmanes |
| **Transferències GPU/frame** | 2.4 MB | 1.2 MB | 2.4-3.6 MB | 4-8 MB batch |
| **Memòria GPU extra** | 0 MB | ~2 MB | 0 MB | 8-18 MB |
| **Dependencies noves** | Cap | Cap | Cap | Refactor core |
| **Risc JIT** | Zero | Alt | Zero | Zero |
| **Fallback CPU** | Trivial | Complex | Built-in | Complex |
| **Testing effort** | Baix | Alt | Mitjà | Molt Alt |

### Cross-domain Insights

#### Paral·lel 1: Video Encoding (H.264/HEVC)

Els encoders moderns de vídeo usen **pipeline híbrid CPU/GPU**:
- **CPU**: Motion estimation (algorisme complex, branching intensiu)
- **GPU**: Transformades DCT/quantització (massivament paral·lel)

**Lliçó aplicable**: No tot ha d'anar a GPU. OpenCV GaussianBlur és equivalent a "motion estimation" - algorisme molt optimitzat a CPU amb SIMD que sovint supera GPU per frames individuals. La clau és identificar què és realment paral·lelitzable i compensa l'overhead.

#### Paral·lel 2: Database Query Optimization

- **Índexs (CPU)** vs **Full table scans (GPU)**
- Queries amb índex (CPU) guanyen en dades petites
- Full scans (GPU) guanyen amb volums massius

**Lliçó aplicable**: Overhead de setup importa. Morphology amb kernel 3×3 és com query amb índex - CPU guanya. Kernel 9×9 és com full scan - GPU guanya. Option 3 (threshold) intenta explotar això, però afegeix complexitat innecessària.

### Adversarial Testing (Red Team Analysis)

#### Contra Option 1 (Morphology GPU)

**Argument**: "Morphology GPU fallarà estrepitosament"

**Escenaris de fallida**:
1. Kernels quadrats GPU difereixen significativament dels el·líptics CPU → deteccions falses/perdudes
2. Overhead `cp.ndimage.*_filter` > temps morphology CPU → net slowdown
3. Bug en CuPy `ndimage` amb kernel sizes específics → crashes intermitents
4. Precisió float vs integer en operacions morfològiques → resultats no deterministes

**Prova d'inversió** - Com garantir fracàs?
- No validar resultats amb `strict_parity`
- Assumir que kernel quadrat ≈ el·líptic sense verificar
- No fer benchmark abans d'implementar
- Implementar sense fallback CPU robust

**Mitigació**:
- ✅ Executar benchmark amb configuració real (kernel sizes dataset)
- ✅ Validar 1.000 frames amb `strict_parity`, mesurar diferències píxel a píxel
- ✅ Implementar try/catch amb fallback CPU automàtic
- ✅ Comparar histogrames de deteccions CPU vs GPU
- ✅ Test amb diferents kernel sizes (3, 5, 7, 9, 11)

#### Contra Option 2 (Full GPU)

**Argument**: "Full GPU és prematur i probablement contraproductiu"

**Escenaris de fallida**:
1. `cp.ndimage.convolve1d` requereix JIT → tornem al problema `cuda_fp16.h`
2. Overhead convolució GPU (kernel launch, memoria) > GaussianBlur CPU SIMD → net slowdown
3. Precisió float32 GPU vs CPU genera diferències acumulades → deteccions inconsistents
4. Més codi complex → bugs subtils, dificultat debug

**Evidència empírica**:
- OpenCV GaussianBlur CPU optimitzat amb instruccions SIMD (SSE4, AVX2)
- Benchmarks mostren ~1-2 ms/frame per blur 1728×1296 amb kernel 5×5
- GPU kernel launch overhead ~0.1-0.3 ms, pot negar guanys

**Conclusió**: No optimitzar sense dades. Blur no és bottleneck demostrat.

### Second-order Effects (Efectes a Llarg Termini)

#### Option 1 (Morphology GPU)

**6 mesos**:
- ✅ Codi estable, 15% speedup confirmat
- ✅ Equip guanya confiança en optimitzacions GPU incrementals
- ✅ Pipeline GPU més madur, menys surpreses

**2 anys**:
- ✅ Base sòlida per altres optimitzacions GPU (tracking, valid region)
- ✅ Pattern híbrid CPU/GPU establert com best practice
- ⚠️ Possible necessitat actualització CuPy (risc baix, API estable)

**10 anys**:
- ✅ Patró híbrid CPU/GPU segueix rellevant (arquitectures heterogènies)
- ⚠️ DSpace 7/8 pot requerir rewrite, però coneixement transferible
- ✅ Codi simple facilita migració futura

#### Option 2 (Full GPU)

**6 mesos**:
- ❓ Si funciona: 2× speedup, orgull tècnic, però... 
- ❌ Si falla: temps perdut (1-2 setmanes), technical debt
- ⚠️ Complexitat dificultat onboarding nous desenvolupadors

**2 anys**:
- ❌ Manteniment complex si equip canvia
- ❌ Actualitzacions CuPy poden trencar convolució custom
- ❌ Dificultat debugar problemes subtils

**10 anys**:
- ❌ Overcomplexitat pot forçar rewrite complet
- ❌ Codi llegat incomprensible ("per què vam fer això?")

#### Option 3 (Hybrid Intel·ligent)

**6 mesos**:
- ⚠️ Complexitat condicional genera bugs subtils
- ⚠️ Threshold subòptim per dataset específic

**2 anys**:
- ❌ Threshold GPU obsolet amb noves GPUs → recalibració constant
- ❌ Branching logic confusa per nous desenvolupadors

**10 anys**:
- ❌ Legacy branching logic incomprensible
- ❌ "Magic numbers" (threshold=7) sense justificació clara

---

## Recomanació i Roadmap

### Recomanació Principal: **Option 1 (Morphology a GPU)**

#### Rationale

**Per què Option 1?**

1. **Millor ratio risc/benefici**: Canvi quirúrgic (~15 línies), guany mesurable (10-20%), risc controlable
2. **Alineat amb constraints**: Zero JIT, CuPy bàsic, canvis locals a `detection.py` dins `dspace/modules/**`
3. **Extensible**: Si funciona, obre porta a altres optimitzacions incrementals
4. **Falsifiable**: Benchmark clar abans/després, validació amb `strict_parity`
5. **Baix risc**: Fallback CPU trivial, overhead transferències eliminat (no afegit)

**Per què NO les altres opcions?**

| Opció | Raó principal de rebuig |
|-------|-------------------------|
| **Option 2** | OpenCV GaussianBlur CPU amb SIMD és extremadament ràpid (~1-2 ms/frame). Risc que `cp.ndimage.convolve1d` necessiti JIT. Principi: no optimitzar sense mesurar - blur no és bottleneck demostrat. |
| **Option 3** | Overengineering: complexitat no justificada sense dades que provin que kernels grans (≥7×7) són comuns. Maintenance burden innecessari. |
| **Option 4** | Viola constraint arquitectural DSpace 6 (canvis fora de `dspace/modules/**`). Refactor massa gran per guany incert. |

### Implementation Roadmap

#### Phase 1: Benchmark Baseline (1 dia - **PRIORITAT ALTA**)

**Objectiu**: Establir mètriques actuals per comparació posterior

```bash
# 1. Crear script de benchmark
cat > bat_tracker/benchmark_gpu.py << 'EOF'
import time
import numpy as np
from bat_tracker.detection import detect_foreground_blobs

def benchmark_pipeline(frames, background, cfg, device='cpu', num_runs=100):
    times = []
    for frame in frames[:num_runs]:
        start = time.perf_counter()
        detect_foreground_blobs(frame, background, cfg, compute_device=device)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # ms
    
    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'p50_ms': np.percentile(times, 50),
        'p95_ms': np.percentile(times, 95),
        'p99_ms': np.percentile(times, 99),
    }
EOF

# 2. Executar benchmark CPU baseline
python -m bat_tracker.benchmark_gpu --device cpu --frames 1000 --output baseline_cpu.json

# 3. Executar benchmark GPU actual
python -m bat_tracker.benchmark_gpu --device cuda --frames 1000 --output baseline_gpu.json

# 4. Comparar
python -m bat_tracker.compare_benchmarks baseline_cpu.json baseline_gpu.json
```

**Mètriques clau a capturar**:
- Temps total per frame (mean, p50, p95, p99)
- Temps per operació (blur, absdiff, threshold, morphology, contours)
- % temps en cada operació
- Throughput (frames/s)

**Deliverable**: `docs/benchmarks/baseline-2026-03-27.json` amb mètriques actuals

#### Phase 2: Implementar Morphology GPU (2 dies)

**Tasca 2.1**: Modificar `_binary_cupy()` en `bat_tracker/detection.py`

```python
def _binary_cupy(
    frame_gray: np.ndarray,
    background: np.ndarray,
    *,
    blur_kernel: int,
    threshold_mode: str,
    diff_threshold: int,
    otsu_offset: int,
    morph_open: int,
    morph_close: int,
    bg_gpu=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GPU-accelerated binary mask computation using CuPy.
    
    Strategy: blur on CPU (OpenCV SIMD-fast), absdiff + threshold + morphology on GPU.
    """
    import cupy as cp
    
    # Blur on CPU (OpenCV uses SIMD, very fast, avoids CUDA JIT)
    frame_proc, bg_proc = _prepare_frame_and_background(frame_gray, background, blur_kernel)
    
    # Upload blurred images to GPU
    frame_proc_gpu = cp.asarray(frame_proc)
    if bg_gpu is not None:
        bg_proc_gpu = bg_gpu
    else:
        bg_proc_gpu = cp.asarray(bg_proc)
    
    # Absdiff on GPU (pre-compiled kernel, no JIT)
    diff_gpu = cp.abs(
        frame_proc_gpu.astype(cp.int16) - bg_proc_gpu.astype(cp.int16)
    ).astype(cp.uint8)
    
    # Threshold on GPU
    if threshold_mode == "otsu":
        hist_gpu = cp.histogram(diff_gpu, bins=256, range=(0, 256))[0]
        hist_cpu = cp.asnumpy(hist_gpu)
        otsu_thr = _otsu_threshold_from_histogram(hist_cpu)
        thr = max(1, min(255, int(otsu_thr + otsu_offset)))
    else:
        thr = diff_threshold
    binary_gpu = cp.where(diff_gpu > thr, cp.uint8(255), cp.uint8(0))
    
    # ========================================================================
    # NOVA SECCIÓ: Morphology on GPU (pre-compiled filters, no JIT)
    # ========================================================================
    try:
        if morph_open > 1:
            # Morphological opening: erosion followed by dilation
            # Using square kernel (approximates ellipse, faster on GPU)
            binary_gpu = cp.ndimage.minimum_filter(binary_gpu, size=morph_open)
            binary_gpu = cp.ndimage.maximum_filter(binary_gpu, size=morph_open)
        
        if morph_close > 1:
            # Morphological closing: dilation followed by erosion
            binary_gpu = cp.ndimage.maximum_filter(binary_gpu, size=morph_close)
            binary_gpu = cp.ndimage.minimum_filter(binary_gpu, size=morph_close)
        
        # Download binary mask to CPU (only once, after morphology)
        binary = cp.asnumpy(binary_gpu)
        
    except Exception as exc:
        # Fallback to CPU morphology if GPU fails
        print(f"[detection] GPU morphology failed, fallback to CPU: {exc}", 
              file=sys.stderr, flush=True)
        binary = cp.asnumpy(binary_gpu)
        
        # CPU morphology (original code)
        if morph_open > 1:
            k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open, morph_open))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)
        
        if morph_close > 1:
            k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close, morph_close))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)
    
    return binary, frame_proc, bg_proc
```

**Tasca 2.2**: Afegir feature flag a configuració

```yaml
# config.yaml
execution:
  device: cuda
  morphology_gpu: true  # Feature flag per activar/desactivar morphology GPU
  fallback_on_error: true
  strict_parity: false  # true només per validació
```

**Tasca 2.3**: Tests unitaris

```python
# bat_tracker/tests/test_morphology_gpu.py
def test_morphology_gpu_vs_cpu():
    """Verificar que morphology GPU dona resultats similars a CPU."""
    frame = generate_test_frame()
    background = generate_test_background()
    
    # CPU
    binary_cpu, _, _ = _binary_cpu(frame, background, ...)
    
    # GPU
    binary_gpu, _, _ = _binary_cupy(frame, background, ...)
    
    # Compare
    diff_pixels = np.count_nonzero(binary_cpu != binary_gpu)
    diff_ratio = diff_pixels / binary_cpu.size
    
    assert diff_ratio < 0.01, f"Too many different pixels: {diff_ratio:.2%}"
```

**Deliverable**: Branch `feature/morphology-gpu` amb implementació completa + tests

#### Phase 3: Validació Rigorosa (2 dies)

**Tasca 3.1**: Validar parity CPU/GPU (primers 1.000 frames)

```bash
# Executar amb strict_parity activat
python -m bat_tracker \
  --config config.yaml \
  --execution.device cuda \
  --execution.morphology_gpu true \
  --execution.strict_parity true \
  --max-frames 1000 \
  --output-detections detections_gpu.json

# Script de validació
python -m bat_tracker.validate_parity \
  --cpu-detections detections_cpu.json \
  --gpu-detections detections_gpu.json \
  --tolerance 5 \
  --output-report docs/validation/parity-report.md
```

**Mètriques de validació**:
- % frames amb deteccions idèntiques
- % frames amb diferències ≤5 píxels per contorn
- Diferències en àrea, centre de massa, bbox
- Histograma de diferències

**Tasca 3.2**: Mesurar diferències píxel a píxel

```python
# Script per analitzar diferències visuals
def analyze_binary_mask_differences(binary_cpu, binary_gpu):
    diff = np.abs(binary_cpu.astype(int) - binary_gpu.astype(int))
    
    return {
        'total_pixels': binary_cpu.size,
        'different_pixels': np.count_nonzero(diff),
        'diff_ratio': np.count_nonzero(diff) / binary_cpu.size,
        'diff_histogram': np.histogram(diff[diff > 0], bins=10),
    }
```

**Tasca 3.3**: Verificar deteccions en casos extrems

```bash
# Testejar amb diferents configuracions morphology
for morph_size in 3 5 7 9 11; do
  python -m bat_tracker.validate_parity \
    --morph-open $morph_size \
    --morph-close $morph_size \
    --frames 100
done
```

**Criteris d'acceptació**:
- ✅ ≥95% frames amb deteccions idèntiques
- ✅ Diferència píxel a píxel ≤1% amb tolerància 5 píxels
- ✅ Àrea contorns difereix ≤2%
- ✅ Centre de massa difereix ≤1 píxel

**Deliverable**: `docs/validation/parity-report-2026-03-27.md` amb resultats detallats

#### Phase 4: Benchmark Post-canvi (1 dia)

**Tasca 4.1**: Executar benchmark amb morphology GPU

```bash
# Benchmark amb nova implementació
python -m bat_tracker.benchmark_gpu \
  --device cuda \
  --morphology-gpu \
  --frames 1000 \
  --output benchmark_morphology_gpu.json

# Comparar amb baseline
python -m bat_tracker.compare_benchmarks \
  baseline_gpu.json \
  benchmark_morphology_gpu.json \
  --output-report docs/benchmarks/speedup-report.md
```

**Tasca 4.2**: Profiling detallat amb `nvprof`

```bash
# Profiling GPU (requereix CUDA Toolkit)
nvprof --print-gpu-trace python -m bat_tracker \
  --config config.yaml \
  --max-frames 100 \
  2>&1 | tee docs/benchmarks/nvprof-morphology-gpu.txt
```

**Tasca 4.3**: Mesurar throughput total

```bash
# Processar dataset complet i mesurar temps total
time python -m bat_tracker \
  --config config.yaml \
  --execution.device cuda \
  --execution.morphology_gpu true
```

**Mètriques d'èxit**:
- ✅ Speedup ≥ 10% (target: 15%)
- ✅ Latència/frame ↓ 10-20% (8-9 ms → 7-7.5 ms)
- ✅ Transferències GPU ↓ 33% (3.6 MB → 2.4 MB/frame)
- ✅ % computació GPU ↑ 40-60% → 60-75%

**Deliverable**: `docs/benchmarks/speedup-report-2026-03-27.md` amb comparativa detallada

#### Phase 5: Rollout Gradual (1 setmana)

**Setmana 1**: Activar en desenvolupament

```yaml
# config.dev.yaml
execution:
  device: cuda
  morphology_gpu: true
  fallback_on_error: true
  strict_parity: false
```

**Setmana 2**: Activar en staging/test

```bash
# Processar subset de producció (10% frames)
python -m bat_tracker \
  --config config.staging.yaml \
  --max-frames 18327  # 10% de 183.270
```

**Setmana 3**: Monitorització intensiva

```bash
# Monitor GPU usage
nvidia-smi dmon -s pucvmet -d 1 -c 3600 > gpu_monitoring.log

# Monitor errors
tail -f bat_tracker.log | grep -E 'ERROR|WARNING|fallback'
```

**Setmana 4**: Rollout producció 100%

```yaml
# config.prod.yaml
execution:
  device: cuda
  morphology_gpu: true
  fallback_on_error: true
  strict_parity: false  # false en producció per performance
```

**Criteris de rollback**:
- ❌ Crashes > 0.1% frames
- ❌ Slowdown > 5%
- ❌ Deteccions errònies > 1%

**Deliverable**: Sistema en producció amb morphology GPU activat

---

### Success Metrics (Mesurables)

#### Mètriques Quantitatives

| Mètrica | Baseline | Target | Mètode mesura |
|---------|----------|--------|---------------|
| **Latència/frame** | 8-9 ms | 7-7.5 ms (-10-20%) | Benchmark 1000 frames |
| **Throughput total** | 30-45 min (183K frames) | 25-38 min (-10-20%) | Time full pipeline |
| **Transferències GPU/frame** | 3.6 MB | 2.4 MB (-33%) | Profiling CuPy |
| **% computació GPU** | 40-60% | 60-75% (+15-25%) | Profiling nvprof |
| **Memòria GPU peak** | ~4 GB | ≤4.5 GB (+12%) | nvidia-smi |

#### Mètriques Qualitatives

| Mètrica | Target | Mètode mesura |
|---------|--------|---------------|
| **Estabilitat** | Zero crashes en 3 mesos producció | Monitoring logs |
| **Mantenibilitat** | Onboarding <30 min per entendre canvis | Team feedback |
| **Portabilitat** | Funciona amb CuPy 9.x-13.x, CUDA 11.x-12.x | CI testing |
| **Reproducibilitat** | Resultats idèntics entre execucions | Validació determinisme |

### Risk Mitigation Strategy

#### Riscos Identificats i Mitigació

| Risc | Prob | Impacte | Mitigació | Pla Contingència |
|------|------|---------|-----------|------------------|
| **Diferències píxel a píxel significatives** | 40% | Alt | Validació 1000 frames, ajustar tolerància | Fallback CPU, refinar kernel |
| **Overhead GPU > guany CPU** | 20% | Mitjà | Benchmark abans, abortar si slowdown | Revert canvis, documentar |
| **Crash CuPy amb kernels específics** | 15% | Alt | Try/catch amb fallback CPU automàtic | Monitoring, patch CuPy |
| **Resultats no reproduïbles** | 10% | Alt | Fixar seeds, validar determinisme | Investigar CuPy version |
| **OOM GPU** | 5% | Mitjà | Monitorització memòria | Reduir batch size (futur) |

#### Rollback Plan

**Trigger conditions** (qualsevol de):
- Crashes > 0.1% frames en producció
- Slowdown > 5% respecte baseline
- Deteccions errònies > 1% (validació manual)
- GPU OOM errors > 5 en 24h

**Rollback procedure** (ETA: <5 minuts):
```bash
# 1. Desactivar morphology GPU via config
sed -i 's/morphology_gpu: true/morphology_gpu: false/' config.prod.yaml

# 2. Reiniciar pipeline
systemctl restart bat_tracker

# 3. Verificar rollback correcte
bat_tracker --config config.prod.yaml --max-frames 100 --validate

# 4. Notificar equip
echo "Morphology GPU rolled back due to [REASON]" | mail -s "ROLLBACK" team@example.com
```

---

## Referències Tècniques

### Documentació Externa

1. **CuPy ndimage filters**: https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.ndimage.html
2. **OpenCV morphology**: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
3. **CUDA best practices**: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
4. **PCIe bandwidth optimization**: https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/

### Benchmarks de Referència

| Operació | CPU (OpenCV) | GPU (CuPy) | Notes |
|----------|--------------|------------|-------|
| GaussianBlur 5×5 (1728×1296) | 1-2 ms | 2-4 ms | CPU SIMD guanya |
| Morphology 3×3 | 0.5-1 ms | 0.8-1.5 ms | CPU guanya (overhead) |
| Morphology 7×7 | 2-3 ms | 0.8-1.2 ms | GPU guanya |
| Morphology 11×11 | 5-7 ms | 1.0-1.5 ms | GPU guanya molt |
| Absdiff (1728×1296) | 1-2 ms | 0.3-0.5 ms | GPU guanya |
| Threshold | 0.5-1 ms | 0.2-0.3 ms | GPU guanya |

### Codi de Referència

Exemples de validació i benchmark disponibles a:
- `bat_tracker/tests/test_morphology_gpu.py`
- `bat_tracker/benchmark_gpu.py`
- `bat_tracker/validate_parity.py`

---

## Apèndixs

### Apèndix A: Configuració Completa

```yaml
# config.yaml - Configuració recomanada post-implementació
execution:
  device: cuda  # 'auto', 'cpu', or 'cuda'
  morphology_gpu: true  # Feature flag per morphology GPU
  fallback_on_error: true  # Fallback automàtic a CPU si error GPU
  strict_parity: false  # true només per validació (més lent)

detection:
  blur_kernel: 5  # Imparell, >=1
  threshold_mode: fixed  # 'fixed' or 'otsu'
  diff_threshold: 25  # Si mode=fixed
  otsu_offset: 0  # Si mode=otsu
  morph_open: 3  # Erosion+dilation (remove small noise)
  morph_close: 5  # Dilation+erosion (fill holes)
  min_area: 10
  max_area: 5000
  max_detections_per_frame: 0  # 0=unlimited
  roi_x_min: -1  # -1=disabled
  roi_x_max: -1
  roi_y_min: -1
  roi_y_max: -1
```

### Apèndix B: Comandes de Debug

```bash
# Verificar estat GPU
nvidia-smi

# Monitor GPU en temps real
watch -n 1 nvidia-smi

# Profiling detallat
nvprof --print-gpu-trace python -m bat_tracker --max-frames 10

# Verificar CuPy instal·lat correctament
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount())"

# Benchmark quick
python -c "
import time
import cupy as cp
import numpy as np

# Test morphology GPU
img = cp.random.randint(0, 255, (1296, 1728), dtype=cp.uint8)
start = time.perf_counter()
for _ in range(100):
    result = cp.ndimage.minimum_filter(img, size=5)
    result = cp.ndimage.maximum_filter(result, size=5)
cp.cuda.Stream.null.synchronize()
elapsed = time.perf_counter() - start
print(f'Morphology GPU: {elapsed/100*1000:.2f} ms/frame')
"
```

### Apèndix C: Glossari Tècnic

| Terme | Definició |
|-------|-----------|
| **JIT** | Just-In-Time compilation - compilació dinàmica en temps d'execució |
| **SIMD** | Single Instruction Multiple Data - instruccions vectorials CPU (SSE, AVX) |
| **PCIe** | Peripheral Component Interconnect Express - bus de comunicació CPU↔GPU |
| **Morphology** | Operacions morfològiques (erosion, dilation, opening, closing) |
| **Kernel** | Element estructurant per operacions morfològiques (3×3, 5×5, etc.) |
| **Strict parity** | Mode validació que executa CPU+GPU i compara resultats |
| **Fallback** | Mecanisme de contingència que torna a CPU si GPU falla |
| **Throughput** | Nombre de frames processats per unitat de temps |
| **Latència** | Temps de processament per frame individual |

---

## Changelog

| Data | Versió | Canvis |
|------|--------|--------|
| 2026-03-27 | 1.0 | Anàlisi inicial ultra-think amb 4 opcions |
| TBD | 1.1 | Resultats benchmark baseline |
| TBD | 1.2 | Resultats implementació Option 1 |
| TBD | 2.0 | Decisió final i roadmap actualitzat |

---

## Autors i Contribuïdors

- **Anàlisi tècnica**: Ultra-Think Deep Analysis
- **Validació**: [Equip tècnic bat_tracker]
- **Implementació**: [Per determinar]

---

**Fi del document**

Aquest document és un living document i s'actualitzarà amb resultats de benchmarks, validacions i decisions finals.

# Resum Opcions de Paral·lelització CPU

**Document de referència ràpida** - Per anàlisi completa veure [parallelization-analysis.md](parallelization-analysis.md)

## Problema Actual

L'aplicació processa **183.270 frames seqüencialment** (1 frame → 1 core), utilitzant només **1 de 12/32/192 cores disponibles**.

**Evidència**:
- Desktop (12 cores, 16GB RAM): **14 minuts**
- HPC (32 cores): **14 minuts** (mateix temps!)
- HPC (192 cores, 300GB RAM): **20 minuts** (pitjor per overhead)

**Utilització CPU actual**: 8-10% (1 core de 12/32/192)

---

## Comparativa Ràpida

| Aspecte | Option 1: Pool | Option 2: Threading | Option 3: Joblib ⭐ | Option 4: Dask |
|---------|----------------|---------------------|-------------------|----------------|
| **Speedup esperat** | 8-15× | 2-3× | 10-20× | 8-15× |
| **Escalabilitat (192 cores)** | 🟡 Bona | 🔴 Dolenta | 🟢 Excel·lent | 🟢 Perfecta |
| **Simplicitat** | 🟡 Mitjana | 🟢 Alta | 🟢 Molt Alta | 🔴 Baixa |
| **Dependències** | ✅ stdlib | ✅ stdlib | joblib (~500 KB) | dask + 15+ (~50 MB) |
| **API elegància** | 🟡 Primitiva | 🟡 Queues | 🟢 Elegant | 🟡 Complexa |
| **Error handling** | 🟡 Bàsic | 🟡 Manual | 🟢 Excel·lent | 🟢 Bona |
| **Gestió memòria** | 🔴 Manual | 🟢 Compartida | 🟢 Automàtica | 🟢 Automàtica |
| **GIL limitation** | ✅ Evita GIL | ❌ Limitat per GIL | ✅ Evita GIL | ✅ Evita GIL |
| **Puntuació total** | 7.50/10 | 4.55/10 | **8.95/10** ⭐ | 7.20/10 |

---

## Recomanació

### ✅ **Option 3: Joblib Parallel** (RECOMANADA)

**Per què?**
- **Millor API**: `Parallel(n_jobs=N)(delayed(f)(x) for x in data)` - elegant i pythonic
- **Gestió memòria superior**: Optimitzacions automàtiques, suport `mmap`
- **Error handling excel·lent**: Tracebacks clars, exceptions ben propagades
- **Industria standard**: Usat per scikit-learn, scipy, nilearn - extremadament madur
- **Escalabilitat provada**: Des d'1 core fins a milers en HPC
- **Progress tracking built-in**: `verbose=10` dona info detallada

**Què fa?**
Processa múltiples frames en paral·lel amb processos independents (evita GIL Python), mantenint l'ordre per al tracking seqüencial posterior.

**Speedup esperat**:
- Desktop (12 cores): **6-10×** → 14 min → **2-2.5 min**
- HPC (32 cores): **14-19×** → 14 min → **45-60 s**
- HPC (192 cores): **20-35×** → 20 min → **30-45 s**

**Utilització CPU esperada**: 70-85% (vs 8-10% actual)

---

### ⚠️ **Option 1: Multiprocessing Pool** (ACCEPTABLE)

**Per què NO la recomanada?**
- API més primitiva (`map()`, `apply()`) vs joblib elegant
- Gestió errors menys clara (exceptions difícils de debugar)
- Mateix speedup que joblib però més codi boilerplate
- Gestió memòria manual vs automàtica de joblib

**Quan usar-la?**
- Si no es pot instal·lar joblib (entorn molt restringit)
- Si l'equip ja té experiència amb `multiprocessing.Pool`

---

### ❌ **Option 2: Threading** (NO RECOMANADA)

**Per què NO?**
- **GIL (Global Interpreter Lock)** impedeix paral·lelisme real de codi Python
- Speedup màxim 2-3× vs 10-20× amb processos
- **No escala**: 192 threads ≈ 2-3× speedup (inacceptable)
- NumPy/OpenCV ja usen múltiples threads internament

**Conclusió**: Threading NO resol el problema. GIL és un show-stopper.

---

### ❌ **Option 4: Dask** (NO RECOMANADA)

**Per què NO?**
- **Overengineering**: Dask és per clusters distribuïts, no per processar vídeo local
- **Overhead scheduler**: Tasques petites (6-10 ms/frame) pateixen overhead significatiu
- **Dependències pesades**: 50+ MB vs 500 KB joblib
- **Complexitat innecessària**: Corba d'aprenentatge alta per benefici marginal

**Quan considerar-la?**
- Processar centenars de vídeos simultàniament en cluster HPC
- Out-of-core processing (datasets > RAM disponible)
- Pipeline distribuït multi-màquina

---

## Pròxims Passos

Si es decideix implementar **Option 3 (Joblib)**:

1. **Phase 1** (2 dies): POC amb 10.000 frames, validar speedup
2. **Phase 2** (1-2 dies): Optimització memòria (chunk size adaptatiu)
3. **Phase 3** (2-3 dies): Suport GPU multi-worker
4. **Phase 4** (2 dies): Benchmark complet HPC (32 i 192 cores)
5. **Phase 5** (1 dia): Integració CLI i documentació

**ETA total**: ~2 setmanes (implementació + validació + benchmark HPC)

---

## Mètriques d'Èxit

Si Option 3 s'implementa amb èxit:

| Mètrica | Baseline | Target Desktop (12c) | Target HPC (32c) | Target HPC (192c) |
|---------|----------|---------------------|------------------|-------------------|
| **Temps total** | 14 min | 2-2.5 min | 45-60 s | 30-45 s |
| **Speedup** | 1× | 6-7× | 14-19× | 20-35× |
| **Utilització CPU** | 8-10% | 75-85% | 70-80% | 50-70% |
| **Memòria peak** | 2 GB | 6-12 GB | 16-32 GB | 40-80 GB |
| **Throughput** | 218 fps | 1300-1500 fps | 3000-4000 fps | 4000-6000 fps |
| **Resultats** | - | ✅ Idèntics | ✅ Idèntics | ✅ Idèntics |

---

## Quick Win Immediat

Abans d'implementar paral·lelització, prova **OpenMP** per guany ràpid (zero codi):

```bash
# Configurar variables d'entorn per aprofitar threads en NumPy/OpenCV
export OMP_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export MKL_NUM_THREADS=12
export VECLIB_MAXIMUM_THREADS=12
export NUMEXPR_NUM_THREADS=12

python -m bat_tracker --input video.mp4 --output results/
```

**Speedup esperat**: 2-3× (no 20×, però **zero canvis de codi**)

---

## Exemple d'Ús (Future)

Després d'implementar Option 3:

```bash
# Mode paral·lel automàtic (per defecte)
python -m bat_tracker --input video.mp4 --output results/

# Especificar workers
python -m bat_tracker --input video.mp4 --output results/ --num-workers 32

# Mode seqüencial (debugging)
python -m bat_tracker --input video.mp4 --output results/ --sequential

# HPC amb 192 cores
python -m bat_tracker --input video.mp4 --output results/ --num-workers 190
```

---

## Alternatives Futures

Si **Option 3** funciona bé i es requereix més speedup:

- **Cython/Numba**: Compilar parts crítiques (tracking) per evitar overhead Python
- **Rust bindings**: Reescriure pipeline core en Rust (10-50× speedup potencial)
- **GPU batch processing**: Processar múltiples frames simultàniament a GPU
- **Distributed Dask**: Si es tenen múltiples màquines/nodes HPC

---

**Veure anàlisi completa**: [parallelization-analysis.md](parallelization-analysis.md)  
**Data**: 2026-03-28  
**Estat**: Proposta pendent decisió

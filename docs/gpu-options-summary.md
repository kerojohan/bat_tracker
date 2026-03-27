# Resum Opcions d'Optimització GPU

**Document de referència ràpida** - Per anàlisi completa veure [gpu-optimization-analysis.md](gpu-optimization-analysis.md)

## Comparativa Ràpida

| Aspecte | Option 1: Morphology GPU | Option 2: Full GPU | Option 3: GPU Híbrida | Option 4: Batch Processing |
|---------|--------------------------|-------------------|---------------------|---------------------------|
| **Speedup esperat** | 10-20% | 2-3× (dubtós) | 5-15% | 1.5-2× |
| **Risc tècnic** | 🟢 Baix | 🔴 Alt | 🟡 Mitjà | 🔴 Molt Alt |
| **Complexitat** | 🟢 Baixa (~15 línies) | 🔴 Alta (~80 línies) | 🟡 Mitjana (~30 línies) | 🔴 Molt Alta (refactor) |
| **Temps implementació** | 2-3 dies | 1-2 setmanes | 4-5 dies | 3-4 setmanes |
| **Risc JIT** | ✅ Zero | ⚠️ Alt | ✅ Zero | ✅ Zero |
| **Compatibilitat DSpace 6** | ✅ Total | ✅ Bona | ✅ Bona | ❌ Viola constraints |
| **Mantenibilitat** | ✅ Bona | ⚠️ Regular | ⚠️ Regular | ❌ Baixa |
| **Puntuació total** | **8.05/10** ⭐ | 6.15/10 | 6.25/10 | 4.45/10 |

## Recomanació

### ✅ **Option 1: Morphology a GPU** (RECOMANADA)

**Per què?**
- Millor ratio risc/benefici
- Canvi quirúrgic i reversible
- Guany mesurable 10-20%
- Zero risc de compilació JIT
- Extensible per futures optimitzacions

**Què fa?**
Passar operacions morfològiques (open/close) a GPU usant `cp.ndimage.maximum_filter` i `cp.ndimage.minimum_filter`.

**Impacte:**
- Elimina 2.4 MB transferències GPU↔CPU per frame
- Morphology més ràpida (especialment kernels >5×5)
- Fallback CPU automàtic si error

### ❌ **Option 2: Full GPU** (NO RECOMANADA)

**Per què NO?**
- OpenCV GaussianBlur CPU amb SIMD és brutalment ràpid (~1-2 ms)
- Risc alt que `cp.ndimage.convolve1d` necessiti JIT
- Complexitat alta per guany incert
- Principi: no optimitzar sense mesurar

### ⚠️ **Option 3: GPU Híbrida** (NO RECOMANADA)

**Per què NO?**
- Overengineering: complexitat no justificada
- Threshold arbitrari (7×7) sense dades que el suportin
- Maintenance burden: dues rutes de codi

### ❌ **Option 4: Batch Processing** (NO RECOMANADA)

**Per què NO?**
- Viola constraint DSpace 6 (canvis fora `dspace/modules/**`)
- Refactor arquitectural massa gran
- Guany incert per esforç molt alt

## Pròxims Passos

Si es decideix implementar **Option 1**:

1. **Phase 1** (1 dia): Benchmark baseline actual
2. **Phase 2** (2 dies): Implementar morphology GPU
3. **Phase 3** (2 dies): Validació rigorosa (parity checking)
4. **Phase 4** (1 dia): Benchmark post-canvi i mesurar speedup
5. **Phase 5** (1 setmana): Rollout gradual a producció

**ETA total**: ~2 setmanes (implementació + validació + rollout)

## Mètriques d'Èxit

Si Option 1 s'implementa amb èxit:

| Mètrica | Baseline | Target |
|---------|----------|--------|
| Latència/frame | 8-9 ms | 7-7.5 ms |
| Throughput total (183K frames) | 30-45 min | 25-38 min |
| Transferències GPU | 3.6 MB/frame | 2.4 MB/frame |
| % computació GPU | 40-60% | 60-75% |
| Crashes | 0 | 0 |

## Alternatives Futures

Si **Option 1** funciona bé, considerar (6-12 mesos):
- Valid region computation a GPU (median filter gran)
- Background subtraction més sofisticada (GMM a GPU)

Si emergeixen nous requisits (real-time, multi-camera):
- Reconsiderar **Option 2** (Full GPU) o **Option 4** (Batch)

---

**Veure anàlisi completa**: [gpu-optimization-analysis.md](gpu-optimization-analysis.md)  
**Data**: 2026-03-27  
**Estat**: Proposta pendent decisió

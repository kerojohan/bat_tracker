# Documentació bat_tracker

Aquesta carpeta conté documentació tècnica detallada del projecte `bat_tracker`.

## Índex de Documents

### Anàlisis Tècniques

#### Optimització GPU

- **[Anàlisi d'Optimització GPU](gpu/gpu-optimization-analysis.md)** (27 març 2026)
  - Anàlisi exhaustiva de 4 estratègies per maximitzar ús de GPU
  - Comparativa risc/benefici, roadmap d'implementació
  - Recomanació: Option 1 (Morphology a GPU)
  - Estat: Proposta pendent decisió
- **[Resum Opcions GPU](gpu/gpu-options-summary.md)** - Referència ràpida

#### Paral·lelització CPU

- **[Anàlisi de Paral·lelització CPU](cpu/parallelization-analysis.md)** (28 març 2026)
  - Anàlisi exhaustiva de 4 estratègies per paral·lelitzar processament
  - Problema: Temps idèntic amb 12, 32 o 192 cores (processament seqüencial)
  - Recomanació: Option 3 (Joblib Parallel) - speedup 10-20×
  - Estat: Proposta pendent decisió
- **[Resum Opcions CPU](cpu/parallelization-options-summary.md)** - Referència ràpida

## Estructura de Carpetes

```
docs/
├── README.md                          # Aquest fitxer
│
├── cpu/                               # Optimització CPU
│   ├── parallelization-analysis.md   # Anàlisi paral·lelització ultra-think
│   └── parallelization-options-summary.md  # Resum opcions
│
├── gpu/                               # Optimització GPU
│   ├── gpu-optimization-analysis.md  # Anàlisi GPU ultra-think
│   └── gpu-options-summary.md        # Resum opcions
│
├── benchmarks/                        # Resultats de benchmarks
├── validation/                        # Reports de validació
└── architecture/                      # Diagrames arquitectura
```

## Com Navegar

### Per Tema

- **Paral·lelització CPU**: [cpu/parallelization-analysis.md](cpu/parallelization-analysis.md)
- **Optimització GPU**: [gpu/gpu-optimization-analysis.md](gpu/gpu-optimization-analysis.md)
- **Benchmarks**: `benchmarks/` (disponible després implementació)
- **Validació**: `validation/` (disponible després implementació)

### Per Data

| Data | Document | Tipus | Speedup esperat |
|------|----------|-------|-----------------|
| 2026-03-28 | [Anàlisi Paral·lelització CPU](cpu/parallelization-analysis.md) | Anàlisi tècnica | 10-20× |
| 2026-03-27 | [Anàlisi Optimització GPU](gpu/gpu-optimization-analysis.md) | Anàlisi tècnica | 10-20% |

## Prioritats d'Optimització

Basant-se en les anàlisis realitzades, les prioritats són:

### 1. 🔥 **Paral·lelització CPU** (PRIORITAT ALTA)
- **Impacte**: 10-20× speedup (14 min → 1-2 min)
- **Esforç**: 2 setmanes (implementació + validació)
- **Risc**: Baix (joblib molt madur)
- **Next step**: Implementar POC (Phase 1)

### 2. 🟡 **Optimització GPU** (PRIORITAT MITJANA)
- **Impacte**: 10-20% speedup addicional
- **Esforç**: 1-2 setmanes
- **Risc**: Baix (morphology a GPU)
- **Next step**: Benchmark baseline (després CPU)

### 3. ⚪ **Altres Optimitzacions** (PRIORITAT BAIXA)
- Valid region computation a GPU
- Background subtraction més sofisticada
- Tracking optimitzat (si esdevé bottleneck)

## Convencions

### Noms de Fitxers

- Anàlisis tècniques: `[tema]-analysis.md`
- Resums d'opcions: `[tema]-options-summary.md`
- Reports de benchmark: `benchmark-[descripcio]-[YYYY-MM-DD].md`
- Reports de validació: `validation-[descripcio]-[YYYY-MM-DD].md`
- Diagrames: `diagram-[descripcio].[png|svg]`

### Format de Dates

Usar format ISO 8601: `YYYY-MM-DD` (ex: `2026-03-28`)

### Idioma

Tota la documentació tècnica s'escriu en **català**.

## Contribuir Documentació

### Crear Nova Documentació

1. Crear fitxer a la carpeta corresponent (`docs/cpu/`, `docs/gpu/`, `docs/benchmarks/`, etc.)
2. Seguir plantilla d'anàlisis existents si és aplicable
3. Actualitzar aquest `README.md` amb enllaç al nou document
4. Commit amb missatge descriptiu en català

### Plantilles Disponibles

- **Anàlisi tècnica**: Veure [cpu/parallelization-analysis.md](cpu/parallelization-analysis.md) o [gpu/gpu-optimization-analysis.md](gpu/gpu-optimization-analysis.md) com a referència
- **Resum opcions**: Veure [cpu/parallelization-options-summary.md](cpu/parallelization-options-summary.md) com a referència

## Referències Externes

### Documentació de Llibreries

#### GPU
- [CuPy Documentation](https://docs.cupy.dev/en/stable/)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

#### CPU / Paral·lelització
- [Joblib Documentation](https://joblib.readthedocs.io/en/latest/)
- [Python Multiprocessing](https://docs.python.org/3/library/multiprocessing.html)
- [scikit-learn Parallel Guide](https://scikit-learn.org/stable/computing/parallelism.html)

#### Computer Vision
- [OpenCV Documentation](https://docs.opencv.org/4.x/)
- [NumPy Documentation](https://numpy.org/doc/stable/)

### Papers i Articles

- [Embarrassingly Parallel Problems](https://en.wikipedia.org/wiki/Embarrassingly_parallel)
- [Amdahl's Law](https://en.wikipedia.org/wiki/Amdahl%27s_law) - Speedup teòric en paral·lelisme

---

**Última actualització**: 2026-03-28

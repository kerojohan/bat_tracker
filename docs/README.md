# Documentació bat_tracker

Aquesta carpeta conté documentació tècnica detallada del projecte `bat_tracker`.

## Índex de Documents

### Anàlisis Tècniques

- **[Anàlisi d'Optimització GPU](gpu-optimization-analysis.md)** (27 març 2026)
  - Anàlisi exhaustiva de 4 estratègies per maximitzar ús de GPU
  - Comparativa risc/benefici, roadmap d'implementació
  - Recomanació: Option 1 (Morphology a GPU)
  - Estat: Proposta pendent decisió

## Estructura de Carpetes

```
docs/
├── README.md                          # Aquest fitxer
├── gpu-optimization-analysis.md       # Anàlisi GPU ultra-think
├── benchmarks/                        # (futur) Resultats de benchmarks
├── validation/                        # (futur) Reports de validació
└── architecture/                      # (futur) Diagrames arquitectura
```

## Com Navegar

### Per Tema

- **Optimització GPU**: [gpu-optimization-analysis.md](gpu-optimization-analysis.md)
- **Benchmarks**: `benchmarks/` (disponible després Phase 1 roadmap)
- **Validació**: `validation/` (disponible després Phase 3 roadmap)

### Per Data

| Data | Document | Tipus |
|------|----------|-------|
| 2026-03-27 | [Anàlisi Optimització GPU](gpu-optimization-analysis.md) | Anàlisi tècnica |

## Convencions

### Noms de Fitxers

- Anàlisis tècniques: `[tema]-analysis.md`
- Reports de benchmark: `benchmark-[descripcio]-[YYYY-MM-DD].md`
- Reports de validació: `validation-[descripcio]-[YYYY-MM-DD].md`
- Diagrames: `diagram-[descripcio].[png|svg]`

### Format de Dates

Usar format ISO 8601: `YYYY-MM-DD` (ex: `2026-03-27`)

### Idioma

Tota la documentació tècnica s'escriu en **català**.

## Contribuir Documentació

### Crear Nova Documentació

1. Crear fitxer a la carpeta corresponent (`docs/`, `docs/benchmarks/`, etc.)
2. Seguir plantilla si n'hi ha disponible
3. Actualitzar aquest `README.md` amb enllaç al nou document
4. Commit amb missatge descriptiu en català

### Plantilles Disponibles

(Disponibles properament)

## Referències Externes

### Documentació de Llibreries

- [CuPy Documentation](https://docs.cupy.dev/en/stable/)
- [OpenCV Documentation](https://docs.opencv.org/4.x/)
- [NumPy Documentation](https://numpy.org/doc/stable/)

### Papers i Articles

(Per afegir quan sigui rellevant)

---

**Última actualització**: 2026-03-27

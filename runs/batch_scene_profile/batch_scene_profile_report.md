# Batch scene profile / autotune (avaluació)

- Casos OK: 4, fallits: 0
- Sortida: `/home/jcaparros/BORRAR/bat_tracker/runs/batch_scene_profile`

## Observacions

- [case_01] auto_max_distance prop del clamp superior (170)
- Ratio auto_dilate_px/sqrt(area): mean=0.1087, stdev=0.0088 (n=4). Valors massa dispersos podrien indicar regla fràgil respecte a la geometria.
- Alta variància inter-vídeo en auto_dilate_px (σ=16.47, μ=36.75).

## Estadístiques agregades (auto)

- **auto_dilate_px**: {'n': 4, 'mean': 36.75, 'stdev': 16.467771555374455, 'min': 25.0, 'max': 65.0}
- **auto_max_distance**: {'n': 4, 'mean': 144.25, 'stdev': 19.929563467371782, 'min': 114.0, 'max': 170.0}
- **auto_diff_threshold**: {'n': 4, 'mean': 18.0, 'stdev': 3.0, 'min': 15.0, 'max': 23.0}
- **auto_min_area**: {'n': 4, 'mean': 6.25, 'stdev': 2.48746859276655, 'min': 3.0, 'max': 10.0}
- **auto_max_missed**: {'n': 4, 'mean': 13.5, 'stdev': 0.8660254037844386, 'min': 12.0, 'max': 14.0}
- **opening_area_px**: {'n': 4, 'mean': 130030.5, 'stdev': 111768.44764400192, 'min': 59790.0, 'max': 323512.0}

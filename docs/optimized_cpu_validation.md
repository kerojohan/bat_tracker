# CPU-Optimized Stable Baseline

This repository contains a CPU-optimized pipeline variant that preserves exact
functional equivalence against `main` on the validated videos below.

## Stable runtime decisions

- OpenCV CPU threads are fixed to `2` in `bat_tracker/pipeline.py`.
- Frame prefetch queue depth is fixed to `64` in `bat_tracker/video.py`.
- The video reader keeps the stable `BGR -> gray` path in `bat_tracker/video.py`.
- Detection preprocessing reuses buffers and applies a separable Gaussian blur
  in `bat_tracker/detection.py`.

These values are not generic tuning advice. They are the best stable settings
 measured on the benchmark videos used during validation.

## Reproducible commands

Run from the repository root with the project virtualenv activated.

Video `/home/joan/BORRAR/2023_0919_212201_007.MOV`:

```bash
python -m bat_tracker.cli \
  --input /home/joan/BORRAR/2023_0919_212201_007.MOV \
  --output /tmp/bat_tracker_run_2023_0919_212201_007 \
  --config /home/joan/Projectes/bat_tracker/config.out3_clean.yaml
```

Video `/home/joan/BORRAR/rabella_20211016_DSCF0005.mp4`:

```bash
python -m bat_tracker.cli \
  --input /home/joan/BORRAR/rabella_20211016_DSCF0005.mp4 \
  --output /tmp/bat_tracker_run_rabella_20211016_DSCF0005 \
  --config /home/joan/Projectes/bat_tracker/config.out3_clean.yaml
```

The commands are stateless apart from their output directories. Removing the
target output directory before re-running gives a clean, reproducible run.

## Accepted optimizations

- Reused preprocessing buffers in detection.
- Switched frame blur to a separable Gaussian implementation.
- Added reader-thread prefetch for grayscale frames.
- Fixed OpenCV CPU worker count to `2`.

## Rejected experiments

- Larger decoder thread counts through the OpenCV FFMPEG constructor.
- External `ffmpeg` pipe decoding.
- Direct backend gray output with `CAP_PROP_CONVERT_RGB=0`.
- Additional queue-size tuning beyond the accepted stable point.
- Extra ROI tightening based on the valid-region mask.

## Known limits

- OpenCV/FFMPEG file decoding remains a dominant cost.
- `cvtColor` has no remaining low-risk optimization that preserves exact output.
- Blur is already materially optimized, but still expensive.

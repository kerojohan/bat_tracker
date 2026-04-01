from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from time import perf_counter
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class PhaseSummary:
    total_sec: float
    percent_total: float
    mean_sec_per_frame: float
    median_sec_per_frame: float
    p95_sec_per_frame: float
    executions: int


class PerformanceCollector:
    def __init__(self, frame_capacity: int):
        self.frame_capacity = max(0, int(frame_capacity))
        self._lock = Lock()
        self._totals: Dict[str, float] = {}
        self._executions: Dict[str, int] = {}
        self._per_frame: Dict[str, np.ndarray] = {}
        self._pipeline_started_at = perf_counter()
        self._pipeline_total_sec = 0.0
        self.frames_processed = 0

    def record(self, phase: str, duration_sec: float, *, frame_idx: int | None = None, executions: int = 1) -> None:
        duration = max(0.0, float(duration_sec))
        exec_count = max(0, int(executions))
        with self._lock:
            self._totals[phase] = self._totals.get(phase, 0.0) + duration
            self._executions[phase] = self._executions.get(phase, 0) + exec_count
            if frame_idx is not None and 0 <= frame_idx < self.frame_capacity:
                if phase not in self._per_frame:
                    self._per_frame[phase] = np.zeros(self.frame_capacity, dtype=np.float64)
                self._per_frame[phase][frame_idx] += duration

    def mark_frame_processed(self, frame_idx: int) -> None:
        with self._lock:
            self.frames_processed = max(self.frames_processed, int(frame_idx) + 1)

    def finish(self) -> None:
        self._pipeline_total_sec = max(0.0, perf_counter() - self._pipeline_started_at)

    def pipeline_total_sec(self) -> float:
        return self._pipeline_total_sec

    def summary(self) -> dict:
        total_sec = self._pipeline_total_sec
        frames = max(1, self.frames_processed)
        out: dict[str, dict] = {}
        for phase in sorted(self._totals):
            per_frame = self._per_frame.get(phase)
            if per_frame is None:
                values = np.zeros(frames, dtype=np.float64)
            else:
                values = per_frame[:frames]
            phase_total = float(self._totals[phase])
            percent = (100.0 * phase_total / total_sec) if total_sec > 0 else 0.0
            out[phase] = {
                "total_sec": phase_total,
                "percent_total": percent,
                "mean_sec_per_frame": float(np.mean(values)) if values.size else 0.0,
                "median_sec_per_frame": float(np.median(values)) if values.size else 0.0,
                "p95_sec_per_frame": float(np.percentile(values, 95)) if values.size else 0.0,
                "executions": int(self._executions.get(phase, 0)),
            }
        return {
            "pipeline_total_sec": total_sec,
            "frames_processed": self.frames_processed,
            "fps_effective": (float(self.frames_processed) / total_sec) if total_sec > 0 else 0.0,
            "phases": out,
        }

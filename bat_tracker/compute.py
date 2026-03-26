from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import cv2


@dataclass
class ExecutionPlan:
    requested_device: str
    selected_device: str
    gpu_available: bool
    reason: str


def _opencv_cuda_available() -> tuple[bool, str]:
    if not hasattr(cv2, "cuda"):
        return False, "opencv_cuda_namespace_missing"

    try:
        count = int(cv2.cuda.getCudaEnabledDeviceCount())
    except Exception as exc:
        return False, f"opencv_cuda_probe_failed:{exc.__class__.__name__}"

    if count <= 0:
        return False, "no_cuda_device"

    required = ["createGaussianFilter", "createMorphologyFilter", "threshold", "absdiff"]
    missing = [name for name in required if not hasattr(cv2.cuda, name)]
    if missing:
        return False, "missing_cuda_ops:" + ",".join(missing)

    if not hasattr(cv2, "cuda_GpuMat"):
        return False, "cuda_gpumat_missing"

    return True, "cuda_ready"


def build_execution_plan(cfg: Dict) -> ExecutionPlan:
    execution_cfg = cfg.get("execution", {}) if isinstance(cfg, dict) else {}
    requested = str(execution_cfg.get("device", "auto")).strip().lower()

    if requested not in {"auto", "cpu", "cuda"}:
        requested = "auto"

    gpu_available, gpu_reason = _opencv_cuda_available()

    if requested == "cpu":
        return ExecutionPlan(
            requested_device=requested,
            selected_device="cpu",
            gpu_available=gpu_available,
            reason="forced_cpu",
        )

    if requested == "cuda":
        if gpu_available:
            return ExecutionPlan(
                requested_device=requested,
                selected_device="cuda",
                gpu_available=True,
                reason="forced_cuda",
            )
        return ExecutionPlan(
            requested_device=requested,
            selected_device="cpu",
            gpu_available=False,
            reason=f"cuda_requested_but_unavailable:{gpu_reason}",
        )

    if gpu_available:
        return ExecutionPlan(
            requested_device=requested,
            selected_device="cuda",
            gpu_available=True,
            reason="auto_cuda",
        )

    return ExecutionPlan(
        requested_device=requested,
        selected_device="cpu",
        gpu_available=False,
        reason=f"auto_cpu:{gpu_reason}",
    )

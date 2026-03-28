from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import sys


@dataclass
class ExecutionPlan:
    requested_device: str
    selected_device: str
    gpu_available: bool
    reason: str


def _cupy_cuda_available() -> tuple[bool, str]:
    """Check if CuPy is installed and can see a CUDA device."""
    try:
        import cupy as cp  # type: ignore
    except ImportError:
        return False, "cupy_not_installed"
    except Exception as exc:
        return False, f"cupy_import_failed:{exc.__class__.__name__}"

    try:
        device_count = cp.cuda.runtime.getDeviceCount()
    except Exception as exc:
        return False, f"cupy_device_probe_failed:{exc.__class__.__name__}"

    if device_count <= 0:
        return False, "no_cuda_device"

    # Quick smoke test – allocate a tiny array on GPU
    try:
        _ = cp.array([1, 2, 3]).sum()
    except Exception as exc:
        return False, f"cupy_smoke_test_failed:{exc.__class__.__name__}"

    return True, "cupy_cuda_ready"


def build_execution_plan(cfg: Dict) -> ExecutionPlan:
    execution_cfg = cfg.get("execution", {}) if isinstance(cfg, dict) else {}
    requested = str(execution_cfg.get("device", "auto")).strip().lower()

    if requested not in {"auto", "cpu", "cuda"}:
        requested = "auto"

    gpu_available, gpu_reason = _cupy_cuda_available()

    if requested == "cpu":
        plan = ExecutionPlan(
            requested_device=requested,
            selected_device="cpu",
            gpu_available=gpu_available,
            reason="forced_cpu",
        )
        print(f"[compute] device=cpu (forced) | gpu_available={gpu_available}", file=sys.stderr, flush=True)
        return plan

    if requested == "cuda":
        if gpu_available:
            plan = ExecutionPlan(
                requested_device=requested,
                selected_device="cuda",
                gpu_available=True,
                reason="forced_cuda",
            )
            print("[compute] device=cuda (forced) | cupy ready", file=sys.stderr, flush=True)
            return plan
        plan = ExecutionPlan(
            requested_device=requested,
            selected_device="cpu",
            gpu_available=False,
            reason=f"cuda_requested_but_unavailable:{gpu_reason}",
        )
        print(f"[compute] device=cpu (cuda requested but unavailable: {gpu_reason})", file=sys.stderr, flush=True)
        return plan

    # auto mode
    if gpu_available:
        plan = ExecutionPlan(
            requested_device=requested,
            selected_device="cuda",
            gpu_available=True,
            reason="auto_cuda",
        )
        print("[compute] device=cuda (auto-detected via cupy)", file=sys.stderr, flush=True)
        return plan

    plan = ExecutionPlan(
        requested_device=requested,
        selected_device="cpu",
        gpu_available=False,
        reason=f"auto_cpu:{gpu_reason}",
    )
    print(f"[compute] device=cpu (auto, reason: {gpu_reason})", file=sys.stderr, flush=True)
    return plan

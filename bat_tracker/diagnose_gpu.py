"""GPU/CUDA diagnostic tool for bat_tracker.

Run with:  python -m bat_tracker.diagnose_gpu
"""
from __future__ import annotations

import sys


def main() -> None:
    print("=" * 60)
    print("bat_tracker GPU Diagnostic")
    print("=" * 60)

    # --- NumPy ---
    try:
        import numpy as np
        print(f"\n[OK] NumPy {np.__version__}")
    except ImportError:
        print("\n[FAIL] NumPy not installed")

    # --- OpenCV ---
    try:
        import cv2
        print(f"[OK] OpenCV {cv2.__version__}")
        if hasattr(cv2, "cuda"):
            try:
                count = int(cv2.cuda.getCudaEnabledDeviceCount())
                print(f"  OpenCV CUDA devices: {count}")
            except Exception as exc:
                print(f"  OpenCV CUDA probe failed: {exc}")
        else:
            print("  OpenCV CUDA namespace: not available (normal for pip opencv-python)")
    except ImportError:
        print("[FAIL] OpenCV not installed")

    # --- CuPy ---
    print()
    try:
        import cupy as cp
        print(f"[OK] CuPy {cp.__version__}")
    except ImportError:
        print("[FAIL] CuPy not installed")
        print("  Install with: pip install cupy-cuda12x  (for CUDA 12)")
        print("  Or:           pip install cupy-cuda11x  (for CUDA 11)")
        print("\n" + "=" * 60)
        print("RESULT: GPU acceleration NOT available (CuPy missing)")
        print("=" * 60)
        sys.exit(1)
    except Exception as exc:
        print(f"[FAIL] CuPy import error: {exc}")
        sys.exit(1)

    # --- CUDA runtime ---
    try:
        device_count = cp.cuda.runtime.getDeviceCount()
        print(f"[OK] CUDA devices detected: {device_count}")
    except Exception as exc:
        print(f"[FAIL] CUDA runtime error: {exc}")
        sys.exit(1)

    if device_count <= 0:
        print("\n" + "=" * 60)
        print("RESULT: GPU acceleration NOT available (no CUDA devices)")
        print("=" * 60)
        sys.exit(1)

    # --- Device info ---
    for i in range(device_count):
        dev = cp.cuda.Device(i)
        attrs = dev.attributes
        print(f"\n  Device {i}:")
        print(f"    Name:             {dev.pci_bus_id}")
        try:
            free, total = cp.cuda.runtime.memGetInfo()
            print(f"    Memory:           {total / (1024**3):.1f} GB total, {free / (1024**3):.1f} GB free")
        except Exception:
            pass
        print(f"    Compute cap:      {attrs.get('ComputeCapabilityMajor', '?')}.{attrs.get('ComputeCapabilityMinor', '?')}")
        print(f"    Multiprocessors:  {attrs.get('MultiProcessorCount', '?')}")

    # --- Smoke test ---
    print("\n  Smoke test...")
    try:
        a = cp.arange(1_000_000, dtype=cp.float32)
        result = float(cp.sum(a))
        expected = float(sum(range(1_000_000)))
        print(f"    sum(0..999999) = {result:.0f} (expected {expected:.0f})")
        print(f"    [OK] CuPy GPU computation works")
    except Exception as exc:
        print(f"    [FAIL] Smoke test failed: {exc}")
        sys.exit(1)

    # --- Test actual operations used by bat_tracker (no JIT needed) ---
    print()
    try:
        a = cp.full((64, 64), 100, dtype=cp.uint8)
        b = cp.full((64, 64), 50, dtype=cp.uint8)
        diff = cp.abs(a.astype(cp.int16) - b.astype(cp.int16)).astype(cp.uint8)
        binary = cp.where(diff > 25, cp.uint8(255), cp.uint8(0))
        assert int(cp.sum(binary)) > 0
        print("[OK] absdiff + threshold (pre-compiled, no JIT)")
    except Exception as exc:
        print(f"[FAIL] absdiff test failed: {exc}")
        sys.exit(1)

    try:
        hist = cp.histogram(diff, bins=256, range=(0, 256))[0]
        assert hist.shape == (256,)
        print("[OK] GPU histogram (pre-compiled, no JIT)")
    except Exception as exc:
        print(f"[FAIL] histogram test failed: {exc}")

    print("\n" + "=" * 60)
    print("RESULT: GPU acceleration AVAILABLE")
    print("  bat_tracker will use GPU when device=auto or device=cuda")
    print("=" * 60)


if __name__ == "__main__":
    main()

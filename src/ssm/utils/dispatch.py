from __future__ import annotations

import importlib


def _module_available(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def has_cuda_kernels() -> bool:
    """Return True if CUDA fused extension is importable.

    This is a lightweight heuristic and does not guarantee that all kernels
    are present; it is sufficient for feature gating in tests.
    """
    return _module_available("ssm.ops.cuda.bindings")


def has_cpu_kernels() -> bool:
    """Return True if CPU fused extension is importable."""
    return _module_available("ssm.ops.cpu.bindings")


def get_available_backend() -> str:
    """Return the best available backend: 'cuda', 'cpu', or 'python'."""
    if has_cuda_kernels():
        return "cuda"
    if has_cpu_kernels():
        return "cpu"
    return "python"

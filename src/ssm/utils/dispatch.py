from __future__ import annotations

import importlib
from typing import Callable


def _module_available(name: str) -> bool:
    """Return ``True`` if a module can be imported without raising."""

    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def has_cuda_kernels() -> bool:
    """Return ``True`` when the CUDA fused extension can be imported."""

    return _module_available("ssm.ops.cuda.bindings")


def has_cpu_kernels() -> bool:
    """Return ``True`` when the CPU fused extension can be imported."""

    return _module_available("ssm.ops.cpu.bindings")


def has_python_reference() -> bool:
    """Return ``True`` when the Python reference ops are importable."""

    return _module_available("ssm.ops.python.reference")


def _backend_checks() -> dict[str, Callable[[], bool]]:
    """Return availability predicates for the known backends."""

    return {
        "cuda": has_cuda_kernels,
        "cpu": has_cpu_kernels,
        "python": has_python_reference,
    }


def get_available_backend(preferred: str | None = None) -> str:
    """Return the best available backend.

    The search defaults to CUDA, then CPU, then the Python reference path.
    A caller may optionally supply a ``preferred`` backend; if that backend
    is unavailable the function gracefully falls back to the default order.

    Args:
        preferred: Optional backend name to prioritize (``"cuda"``, ``"cpu"``,
            or ``"python"``).

    Returns:
        The name of the first available backend.

    Raises:
        RuntimeError: If none of the known backends can be imported.
    """

    checks = _backend_checks()
    order: list[str] = []
    if preferred is not None:
        if preferred not in checks:
            raise ValueError(f"unknown backend preference: {preferred}")
        order.append(preferred)
    order.extend(name for name in ("cuda", "cpu", "python") if name not in order)

    for backend in order:
        if checks[backend]():
            return backend

    raise RuntimeError("no available backend detected")

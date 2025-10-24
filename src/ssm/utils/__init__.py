"""Utility namespace for generation, dispatch, and weight IO (stubs)."""

from .dispatch import has_cuda_kernels, has_cpu_kernels, get_available_backend

__all__ = [
    "has_cuda_kernels",
    "has_cpu_kernels",
    "get_available_backend",
]

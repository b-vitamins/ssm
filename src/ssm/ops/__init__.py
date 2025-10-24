"""Ops façade and backend detection.

This namespace exposes stable Python signatures and provides lightweight
backend detection helpers. Implementations for CPU/CUDA live in subpackages
and are registered later.
"""

from .python.reference import (
    selective_scan,
    selective_state_step,
    ssd_chunk_scan,
    dw_causal_conv,
    fused_layer_norm,
)

__all__ = [
    "selective_scan",
    "selective_state_step",
    "ssd_chunk_scan",
    "dw_causal_conv",
    "fused_layer_norm",
]

from __future__ import annotations

from typing import Any

import torch


def selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    softplus: bool = False,
    return_last_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Selective scan over sequences (Mamba v1 core op).

    Design-only stub. See DESIGN and API docs for the mathematical definition.

    Args:
        u: Input drive `(B, D, L)`.
        delta: dt values `(B, D, L)` in the same dtype as `u`.
        A: State matrix `(D, N)` in fp32 or complex fp32.
        B: Either `(D, N)` or time-varying `(B, G, N, L)` with group dim `G`.
        C: Either `(D, N)` or time-varying `(B, G, N, L)`.
        D: Optional skip `(D,)` in fp32.
        z: Optional gate `(B, D, L)` in the same dtype as `u`.
        dt_bias: Optional dt bias `(D,)` in fp32 (applied before softplus when requested).
        softplus: If True, apply softplus to `delta + dt_bias` in fp32.
        return_last_state: If True, also return last recurrent state `(B, D, N)`.

    Returns:
        If `return_last_state=False`: output tensor `(B, D, L)`.
        If `return_last_state=True`: tuple `(out, last_state)`.

    Raises:
        NotImplementedError: Implementation to be provided later.
    """
    raise NotImplementedError("selective_scan is not implemented in the scaffold.")


def selective_state_step(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    softplus: bool = True,
) -> torch.Tensor:
    """Single-timestep selective state update (decoding path).

    Args:
        state: Current SSM state `(B, D, N)` (same dtype as dt projection).
        x: Input `(B, D)`.
        dt: Per-dimension dt `(B, D)`.
        A: State matrix `(D, N)` in fp32 or complex fp32.
        B: Input map `(B, N)` or `(B, G, N)` depending on grouping.
        C: Output map `(B, N)` or `(B, G, N)`.
        D: Optional skip `(D,)`.
        z: Optional gate `(B, D)`.
        dt_bias: Optional dt bias `(D,)`.
        softplus: If True, apply softplus to `dt + dt_bias` in fp32.

    Returns:
        Tensor `(B, D)` output for the timestep.

    Raises:
        NotImplementedError: Implementation to be provided later.
    """
    raise NotImplementedError("selective_state_step is not implemented in the scaffold.")


def ssd_chunk_scan(
    X: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    seq_meta: dict[str, Any] | None = None,
    initial_states: torch.Tensor | None = None,
) -> torch.Tensor:
    """Chunked SSD scan (Mamba v2 core op).

    Args:
        X: Packed branch tensor `(B, L, H, P)`.
        dt: Per-head dt `(B, L, H)` (fp32 accumulation where appropriate).
        A: Per-head state decay `(H,)` or `(H, P?)` depending on formulation.
        B: Input map `(B, L, G, N)`.
        C: Output map `(B, L, G, N)`.
        chunk_size: Chunk length for intra-chunk scan.
        D: Optional skip `(H,)` or `(H, P)`.
        z: Optional gate for output.
        seq_meta: Optional varlen metadata, e.g., `{seq_idx, cu_seqlens}`.
        initial_states: Optional initial states per chunk or per sequence.

    Returns:
        Tensor `(B, L, H, P)` output.

    Raises:
        NotImplementedError: Implementation to be provided later.
    """
    raise NotImplementedError("ssd_chunk_scan is not implemented in the scaffold.")


def dw_causal_conv(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None, activation: str = "silu") -> torch.Tensor:
    """Depthwise causal 1D convolution with fused activation.

    Args:
        x: Input `(B, D, L)` or `(B, L, D)` (document the accepted memory layout and enforce in implementation).
        weight: Depthwise conv weights `(D, 1, K)` or `(D, K)` depending on chosen packing.
        bias: Optional bias `(D,)`.
        activation: Activation to apply; default 'silu'.

    Returns:
        Output tensor with same layout as input.

    Raises:
        NotImplementedError: Implementation to be provided later.
    """
    raise NotImplementedError("dw_causal_conv is not implemented in the scaffold.")


def fused_layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None = None,
    is_rms: bool = False,
    eps: float = 1e-5,
    prenorm: bool = True,
    residual_in_fp32: bool = True,
) -> torch.Tensor:
    """Fused add+LayerNorm or RMSNorm.

    Args:
        x: Input `(B, L, D)`.
        weight: Norm weight `(D,)`.
        bias: Optional norm bias `(D,)`.
        residual: Optional residual to fuse add before norm.
        is_rms: If True, use RMSNorm; else LayerNorm.
        eps: Epsilon for numerical stability.
        prenorm: If True, treat input as pre-norm; otherwise post-norm style.
        residual_in_fp32: If True, cast residual to fp32 for accumulation.

    Returns:
        Tensor `(B, L, D)` normalized.

    Raises:
        NotImplementedError: Implementation to be provided later.
    """
    raise NotImplementedError("fused_layer_norm is not implemented in the scaffold.")

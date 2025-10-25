from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _get_compute_dtype(tensor: torch.Tensor) -> torch.dtype:
    """Return the dtype used for accumulations for a given input tensor."""

    if tensor.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    if tensor.is_complex():
        # Promote complex16 inputs to complex64 for numerical stability.
        if tensor.dtype == torch.complex32:
            return torch.complex64
        return tensor.dtype
    return tensor.dtype


def _normalize_scan_param(
    name: str,
    param: torch.Tensor,
    batch: int,
    dim: int,
    state_dim: int,
    length: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Broadcast selective-scan parameters to `(B, D, N, L)` for time indexing."""

    if param.dim() == 2:
        if param.shape != (dim, state_dim):
            raise ValueError(f"{name} must have shape (D, N).")
        expanded = (
            param.to(dtype)
            .view(1, dim, state_dim, 1)
            .expand(batch, dim, state_dim, length)
        )
        return expanded
    if param.dim() == 3:
        if param.shape != (batch, dim, state_dim):
            raise ValueError(f"{name} must have shape (B, D, N) when 3-D.")
        return param.to(dtype).unsqueeze(-1).expand(batch, dim, state_dim, length)
    if param.dim() == 4:
        if param.shape[0] != batch or param.shape[-1] != length:
            raise ValueError(f"{name} must have shape (B, G, N, L) when 4-D.")
        groups = param.shape[1]
        if dim % groups != 0:
            raise ValueError("Group dimension must divide D.")
        repeated = param.repeat_interleave(dim // groups, dim=1)
        if repeated.shape[2] != state_dim:
            raise ValueError(f"{name} has mismatched state dimension.")
        return repeated.to(dtype)
    raise ValueError(f"Unsupported rank for {name}.")


def _normalize_chunk_param(
    name: str,
    param: torch.Tensor,
    batch: int,
    seqlen: int,
    heads: int,
    proj: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Broadcast SSD parameters to `(B, L, H, P)` for chunked scanning."""

    if param.dim() == 2:
        if param.shape != (heads, proj):
            raise ValueError(f"{name} must have shape (H, P).")
        return (
            param.to(dtype).view(1, 1, heads, proj).expand(batch, seqlen, heads, proj)
        )
    if param.dim() == 3:
        if param.shape == (batch, heads, proj):
            return param.to(dtype).unsqueeze(1).expand(batch, seqlen, heads, proj)
        raise ValueError(f"{name} must be (B, H, P) when 3-D.")
    if param.dim() == 4:
        if param.shape != (batch, seqlen, heads, proj):
            raise ValueError(f"{name} must have shape (B, L, H, P) when 4-D.")
        return param.to(dtype)
    raise ValueError(f"Unsupported rank for {name}.")


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
    """Reference selective scan (Mamba v1) implemented in pure PyTorch.

    The routine discretises a diagonal state-space model with per-dimension
    time-steps and applies optional gating/skip connections. Parameters are
    broadcast to match the batch and sequence dimensions following the API
    contract in :mod:`docs/API.md`.

    Args:
        u: Input drive ``(B, D, L)``.
        delta: Continuous-time step sizes ``(B, D, L)`` in the same dtype as ``u``.
        A: State matrix ``(D, N)`` stored in ``fp32`` (or complex variant).
        B: Input projection. Accepts ``(D, N)``, ``(B, D, N)`` or grouped/time-varying
            ``(B, G, N, L)`` where ``G`` divides ``D``.
        C: Output projection with the same broadcasting rules as ``B``.
        D: Optional skip connection ``(D,)`` accumulated in ``fp32``.
        z: Optional gating signal ``(B, D, L)``. SiLU activation is applied.
        dt_bias: Optional bias ``(D,)`` added to ``delta`` before softplus.
        softplus: Whether to apply ``softplus`` to ``delta`` in ``fp32``.
        return_last_state: If ``True`` also returns the final recurrent state
            ``(B, D, N)`` (in the input dtype).

    Returns:
        ``(B, D, L)`` tensor, or a tuple ``(output, last_state)`` when
        ``return_last_state`` is set.
    """
    if u.ndim != 3 or delta.ndim != 3:
        raise ValueError("u and delta must have shape (B, D, L).")

    batch, dim, length = u.shape
    if delta.shape != (batch, dim, length):
        raise ValueError("delta must match the shape of u.")

    state_dim = A.shape[-1]
    if A.shape != (dim, state_dim):
        raise ValueError("A must have shape (D, N).")

    compute_dtype = _get_compute_dtype(u)
    if A.dtype != torch.float32 and A.dtype != compute_dtype:
        compute_dtype = torch.promote_types(compute_dtype, A.dtype)

    u_compute = u.to(compute_dtype)
    delta_compute = delta.to(compute_dtype)
    if A.shape != (dim, state_dim):
        raise ValueError("A must have shape (D, N).")
    A_compute = A.to(compute_dtype)

    B_full = _normalize_scan_param("B", B, batch, dim, state_dim, length, compute_dtype)
    C_full = _normalize_scan_param("C", C, batch, dim, state_dim, length, compute_dtype)

    if D is not None:
        if D.shape != (dim,):
            raise ValueError("D must have shape (D,).")
        D_compute = D.to(compute_dtype)
    else:
        D_compute = None

    if z is not None:
        if z.shape != (batch, dim, length):
            raise ValueError("z must have shape (B, D, L).")
        z_compute = z.to(compute_dtype)
    else:
        z_compute = None

    if dt_bias is not None:
        if dt_bias.shape != (dim,):
            raise ValueError("dt_bias must have shape (D,).")
        delta_compute = delta_compute + dt_bias.to(compute_dtype).view(1, dim, 1)

    if softplus:
        delta_compute = F.softplus(delta_compute)

    state = torch.zeros(batch, dim, state_dim, dtype=compute_dtype, device=u.device)
    outputs: list[torch.Tensor] = []

    for t in range(length):
        delta_t = delta_compute[:, :, t]
        decay = torch.exp(torch.einsum("bd,dn->bdn", delta_t, A_compute))
        B_t = B_full[:, :, :, t]
        C_t = C_full[:, :, :, t]
        drive = B_t * u_compute[:, :, t].unsqueeze(-1)
        state = decay * state + delta_t.unsqueeze(-1) * drive
        y_t = (state * C_t).sum(-1)

        if D_compute is not None:
            y_t = y_t + D_compute.view(1, dim) * u_compute[:, :, t]

        if z_compute is not None:
            y_t = y_t * F.silu(z_compute[:, :, t])

        outputs.append(y_t)

    output = torch.stack(outputs, dim=-1).to(u.dtype)

    if return_last_state:
        return output, state.to(u.dtype)
    return output


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
    """Advance the selective state by a single timestep.

    This mirrors :func:`selective_scan` but operates on an explicit state tensor
    for decoding use-cases. Parameters follow the same broadcasting rules as the
    scan variant except the time dimension is omitted.

    Args:
        state: Current recurrent state ``(B, D, N)`` updated in-place.
        x: Input drive ``(B, D)``.
        dt: Time-step ``(B, D)`` in the same dtype as ``x``.
        A: State matrix ``(D, N)``.
        B: Input projection with shape ``(D, N)``, ``(B, D, N)`` or grouped
            ``(B, G, N)`` where ``G`` divides ``D``.
        C: Output projection mirroring the accepted shapes for ``B``.
        D: Optional skip connection ``(D,)``.
        z: Optional gating signal ``(B, D)`` (SiLU applied).
        dt_bias: Optional bias ``(D,)`` added before the ``softplus``.
        softplus: Whether to apply ``softplus`` to the biased ``dt``.

    Returns:
        Output tensor ``(B, D)`` for the timestep (matching ``x.dtype``).
    """
    if state.ndim != 3:
        raise ValueError("state must have shape (B, D, N).")
    if x.ndim != 2 or dt.ndim != 2:
        raise ValueError("x and dt must have shape (B, D).")
    batch, dim, state_dim = state.shape
    if x.shape != (batch, dim) or dt.shape != (batch, dim):
        raise ValueError("x and dt must match the first two dimensions of state.")
    if A.shape != (dim, state_dim):
        raise ValueError("A must have shape (D, N).")

    compute_dtype = _get_compute_dtype(state)
    if A.dtype != torch.float32 and A.dtype != compute_dtype:
        compute_dtype = torch.promote_types(compute_dtype, A.dtype)

    state_compute = state.to(compute_dtype)
    x_compute = x.to(compute_dtype)
    dt_compute = dt.to(compute_dtype)
    if A.shape != (dim, state_dim):
        raise ValueError("A must have shape (D, N).")
    A_compute = A.to(compute_dtype)

    def _expand_grouped_projection(name: str, param: torch.Tensor) -> torch.Tensor:
        """Expand grouped projections ``(B, G, N)`` to ``(B, D, N)`` when needed."""

        if param.dim() == 3 and param.shape[0] == batch and param.shape[1] != dim:
            groups = param.shape[1]
            if dim % groups != 0:
                raise ValueError(f"{name} group dimension must divide D.")
            if param.shape[2] != state_dim:
                raise ValueError(f"{name} must have matching state dimension.")
            param = param.repeat_interleave(dim // groups, dim=1)
        return param

    B_prepared = _expand_grouped_projection("B", B)
    C_prepared = _expand_grouped_projection("C", C)

    B_expanded = _normalize_scan_param(
        "B", B_prepared, batch, dim, state_dim, 1, compute_dtype
    )[:, :, :, 0]
    C_expanded = _normalize_scan_param(
        "C", C_prepared, batch, dim, state_dim, 1, compute_dtype
    )[:, :, :, 0]

    if D is not None:
        if D.shape != (dim,):
            raise ValueError("D must have shape (D,).")
        D_compute = D.to(compute_dtype)
    else:
        D_compute = None

    if z is not None:
        if z.shape != (batch, dim):
            raise ValueError("z must have shape (B, D).")
        z_compute = z.to(compute_dtype)
    else:
        z_compute = None

    if dt_bias is not None:
        if dt_bias.shape != (dim,):
            raise ValueError("dt_bias must have shape (D,).")
        dt_compute = dt_compute + dt_bias.to(compute_dtype).view(1, dim)

    if softplus:
        dt_compute = F.softplus(dt_compute)

    decay = torch.exp(torch.einsum("bd,dn->bdn", dt_compute, A_compute))
    drive = B_expanded * x_compute.unsqueeze(-1)
    new_state = decay * state_compute + dt_compute.unsqueeze(-1) * drive

    output = (new_state * C_expanded).sum(-1)

    if D_compute is not None:
        output = output + D_compute.view(1, dim) * x_compute

    if z_compute is not None:
        output = output * F.silu(z_compute)

    state.copy_(new_state.to(state.dtype))
    return output.to(x.dtype)


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
    """Reference SSD chunk scan (Mamba v2) implemented with explicit loops.

    The implementation follows the diagonal state-space recurrence while
    respecting chunk boundaries and optional ragged metadata.

    Args:
        X: Branch activations ``(B, L, H, P)``.
        dt: Per-head time-steps ``(B, L, H)``.
        A: State matrix ``(H, P)`` (or ``(H,)`` for scalar state per head).
        B: Input projection with shapes ``(H, P)``, ``(B, H, P)`` or
            time-varying ``(B, L, H, P)``.
        C: Output projection mirroring ``B``'s accepted shapes.
        chunk_size: Number of tokens processed per chunk.
        D: Optional skip connection ``(H,)`` or ``(H, P)``.
        z: Optional gating tensor ``(B, L, H)`` or ``(B, L, H, P)`` (SiLU applied).
        seq_meta: Optional metadata containing ``seq_lens`` or ``cu_seqlens`` for
            ragged batches; missing entries default to the full length ``L``.
        initial_states: Optional initial recurrent state ``(B, H, P)``.

    Returns:
        Tensor ``(B, L, H, P)`` with zero-padding beyond the valid lengths.
    """
    if X.ndim != 4:
        raise ValueError("X must have shape (B, L, H, P).")
    if dt.ndim != 3:
        raise ValueError("dt must have shape (B, L, H).")

    batch, seqlen, heads, proj = X.shape
    if dt.shape != (batch, seqlen, heads):
        raise ValueError("dt must align with (B, L, H).")

    compute_dtype = _get_compute_dtype(X)
    if A.dtype != torch.float32 and A.dtype != compute_dtype:
        compute_dtype = torch.promote_types(compute_dtype, A.dtype)

    X_compute = X.to(compute_dtype)
    dt_compute = dt.to(compute_dtype)

    if A.ndim == 1:
        if A.shape[0] != heads:
            raise ValueError("A must have shape (H,) when 1-D.")
        A_compute = A.to(compute_dtype).unsqueeze(-1).expand(heads, proj)
    elif A.ndim == 2:
        if A.shape != (heads, proj):
            raise ValueError("A must have shape (H, P) when 2-D.")
        A_compute = A.to(compute_dtype)
    else:
        raise ValueError("A must be 1-D or 2-D.")

    B_full = _normalize_chunk_param("B", B, batch, seqlen, heads, proj, compute_dtype)
    C_full = _normalize_chunk_param("C", C, batch, seqlen, heads, proj, compute_dtype)

    if D is not None:
        if D.ndim == 1:
            if D.shape[0] != heads:
                raise ValueError("D must have shape (H,).")
            D_compute = D.to(compute_dtype).view(1, 1, heads, 1)
        elif D.ndim == 2:
            if D.shape != (heads, proj):
                raise ValueError("D must have shape (H, P) when 2-D.")
            D_compute = D.to(compute_dtype).view(1, 1, heads, proj)
        else:
            raise ValueError("Unsupported rank for D.")
    else:
        D_compute = None

    if z is not None:
        if z.ndim == 3:
            if z.shape != (batch, seqlen, heads):
                raise ValueError("z must have shape (B, L, H) when 3-D.")
            z_compute = z.to(compute_dtype).unsqueeze(-1)
        elif z.ndim == 4:
            if z.shape != (batch, seqlen, heads, proj):
                raise ValueError("z must have shape (B, L, H, P) when 4-D.")
            z_compute = z.to(compute_dtype)
        else:
            raise ValueError("Unsupported rank for z.")
    else:
        z_compute = None

    if initial_states is not None:
        if initial_states.shape != (batch, heads, proj):
            raise ValueError("initial_states must have shape (B, H, P).")
        initial_state = initial_states.to(compute_dtype)
    else:
        initial_state = torch.zeros(
            batch, heads, proj, dtype=compute_dtype, device=X.device
        )

    if seq_meta is not None:
        if "cu_seqlens" in seq_meta:
            cu = seq_meta["cu_seqlens"].to(torch.long)
            if cu.numel() != batch + 1:
                raise ValueError("cu_seqlens must have length B + 1.")
            lengths = cu[1:] - cu[:-1]
        elif "seq_lens" in seq_meta:
            lengths = torch.tensor(
                seq_meta["seq_lens"], device=X.device, dtype=torch.long
            )
            if lengths.numel() != batch:
                raise ValueError("seq_lens must have length B.")
        else:
            raise ValueError("seq_meta must provide cu_seqlens or seq_lens.")
    else:
        lengths = torch.full((batch,), seqlen, dtype=torch.long, device=X.device)

    outputs = torch.zeros(
        batch, seqlen, heads, proj, dtype=compute_dtype, device=X.device
    )

    for b in range(batch):
        valid = int(lengths[b].item())
        state_b = initial_state[b]
        for start in range(0, valid, chunk_size):
            end = min(valid, start + chunk_size)
            for t in range(start, end):
                dt_bt = dt_compute[b, t]
                decay = torch.exp(dt_bt.unsqueeze(-1) * A_compute)

                drive = B_full[b, t] * X_compute[b, t]
                state_b = decay * state_b + dt_bt.unsqueeze(-1) * drive

                y_bt = state_b * C_full[b, t]

                if D_compute is not None:
                    skip = D_compute.view(heads, -1)
                    y_bt = y_bt + skip * X_compute[b, t]

                if z_compute is not None:
                    gate = z_compute[b, t]
                    if gate.shape != y_bt.shape:
                        gate = gate.expand_as(y_bt)
                    y_bt = y_bt * F.silu(gate)

                outputs[b, t] = y_bt

        if valid < seqlen:
            outputs[b, valid:] = 0

    return outputs.to(X.dtype)


def dw_causal_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str = "silu",
) -> torch.Tensor:
    """Depthwise causal 1D convolution with optional activation.

    Args:
        x: Input with layout ``(B, C, L)`` or ``(B, L, C)``.
        weight: Depthwise kernel ``(C, K)`` or ``(C, 1, K)``.
    bias: Optional bias ``(C,)``.
    activation: Post-convolution activation (``"silu"``, ``"swish"``, ``"identity"``, ``"none"``).

    Returns:
        Tensor matching the layout and dtype of ``x``.
    """
    if x.ndim != 3:
        raise ValueError("x must have shape (B, C, L) or (B, L, C).")

    in_channels = weight.shape[0]
    if x.shape[1] == in_channels:
        channels_first = True
    elif x.shape[2] == in_channels:
        channels_first = False
    else:
        raise ValueError("weight channel dimension must match x.")

    if channels_first:
        batch, channels, length = x.shape
        x_conv = x
    else:
        batch, length, channels = x.shape
        x_conv = x.permute(0, 2, 1)

    compute_dtype = _get_compute_dtype(x)

    if weight.ndim == 2:
        if weight.shape[0] != channels:
            raise ValueError("weight has incompatible shape.")
        kernel_size = weight.shape[1]
        weight_conv = weight.unsqueeze(1)
    elif weight.ndim == 3:
        if weight.shape[0] != channels or weight.shape[1] != 1:
            raise ValueError("Expected depthwise weights with shape (C, 1, K).")
        kernel_size = weight.shape[2]
        weight_conv = weight
    else:
        raise ValueError("weight must have 2 or 3 dimensions.")

    if bias is not None and bias.shape != (channels,):
        raise ValueError("bias must have shape (C,).")

    x_conv = x_conv.to(compute_dtype)
    weight_conv = weight_conv.to(compute_dtype)
    bias_conv = bias.to(compute_dtype) if bias is not None else None

    padding = (kernel_size - 1, 0)
    x_padded = F.pad(x_conv, padding)
    out = F.conv1d(x_padded, weight_conv, bias=bias_conv, groups=channels)

    activation = activation.lower()
    if activation in ("silu", "swish"):
        out = F.silu(out)
    elif activation in ("identity", "none"):
        pass
    else:
        raise ValueError(f"Unsupported activation '{activation}'.")

    if channels_first:
        return out.to(x.dtype)
    return out.permute(0, 2, 1).to(x.dtype)


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
    """Fused residual addition with LayerNorm or RMSNorm.

    Args:
        x: Input tensor ``(B, L, D)``.
        weight: Scale parameters ``(D,)``.
        bias: Optional shift parameters ``(D,)``.
        residual: Optional residual for fused addition.
        is_rms: Select RMSNorm (``True``) or LayerNorm (``False``).
        eps: Numerical stabiliser.
        prenorm: When ``True`` applies the residual before normalising.
        residual_in_fp32: Accumulate in ``float32`` when a residual is supplied.

    Returns:
        Normalised tensor with the same shape/dtype as ``x``.
    """
    if x.ndim != 3:
        raise ValueError("x must have shape (B, L, D).")
    if weight.shape != (x.shape[-1],):
        raise ValueError("weight must match the last dimension of x.")
    if bias is not None and bias.shape != (x.shape[-1],):
        raise ValueError("bias must match the last dimension of x.")
    if residual is not None and residual.shape != x.shape:
        raise ValueError("residual must match the shape of x.")

    compute_dtype = _get_compute_dtype(x)
    base_dtype = (
        torch.float32 if residual_in_fp32 and residual is not None else compute_dtype
    )

    x_compute = x.to(base_dtype)

    if residual is not None:
        residual_compute = residual.to(base_dtype)
        if prenorm:
            norm_input = x_compute + residual_compute
        else:
            norm_input = x_compute
    else:
        residual_compute = None
        norm_input = x_compute

    if is_rms:
        mean_square = norm_input.pow(2).mean(dim=-1, keepdim=True)
        normed = norm_input * torch.rsqrt(mean_square + eps)
    else:
        mean = norm_input.mean(dim=-1, keepdim=True)
        var = (norm_input - mean).pow(2).mean(dim=-1, keepdim=True)
        normed = (norm_input - mean) * torch.rsqrt(var + eps)

    weight_compute = weight.to(normed.dtype)
    bias_compute = bias.to(normed.dtype) if bias is not None else None

    output = normed * weight_compute
    if bias_compute is not None:
        output = output + bias_compute.view(1, 1, -1)

    if not prenorm and residual_compute is not None:
        output = output + residual_compute.to(output.dtype)

    return output.to(x.dtype)

"""Ops façade with backend dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from .python import reference as reference_ops

try:
    from torch.utils.cpp_extension import load as _load_extension
except ImportError:  # pragma: no cover - torch utils missing
    _load_extension = None


_CPU_OPS = None
_CPU_LOAD_ERROR: Optional[Exception] = None


@dataclass(frozen=True)
class _BackendCandidate:
    name: str
    available: bool


def _cpu_sources() -> list[str]:
    root = Path(__file__).resolve().parent / "cpu"
    return [
        str(root / "bindings.cpp"),
        str(root / "selective_scan.cpp"),
        str(root / "state_step.cpp"),
        str(root / "ssd_chunk_scan.cpp"),
        str(root / "dw_causal_conv.cpp"),
        str(root / "layer_norm.cpp"),
    ]


def _load_cpu_ops() -> Optional[object]:
    global _CPU_OPS, _CPU_LOAD_ERROR
    if _CPU_OPS is not None:
        return _CPU_OPS
    if _CPU_LOAD_ERROR is not None:
        return None
    if _load_extension is None:
        _CPU_LOAD_ERROR = RuntimeError("torch.utils.cpp_extension.load unavailable")
        return None

    try:
        _CPU_OPS = _load_extension(
            name="ssm_cpu_ops",
            sources=_cpu_sources(),
            extra_cflags=["-O3"],
            verbose=False,
        )
        return _CPU_OPS
    except Exception as exc:  # pragma: no cover - build environment specific
        _CPU_LOAD_ERROR = exc
        return None


def _should_use_cpu(*tensors: torch.Tensor) -> bool:
    return all(t.device.type == "cpu" for t in tensors if isinstance(t, torch.Tensor))


def _cpu_backend_status() -> _BackendCandidate:
    ops = _load_cpu_ops()
    return _BackendCandidate(name="cpu", available=ops is not None)


def _invoke_cpu(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except RuntimeError as exc:  # translate contract violations
        raise ValueError(str(exc)) from None


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
    if _should_use_cpu(u, delta, A, B, C, D, z, dt_bias):
        ops = _load_cpu_ops()
        if ops is not None:
            result = _invoke_cpu(
                ops.selective_scan,
                u,
                delta,
                A,
                B,
                C,
                D,
                z,
                dt_bias,
                softplus,
                return_last_state,
            )
            output, last_state = result
            if return_last_state:
                return output, last_state
            return output
    return reference_ops.selective_scan(
        u,
        delta,
        A,
        B,
        C,
        D=D,
        z=z,
        dt_bias=dt_bias,
        softplus=softplus,
        return_last_state=return_last_state,
    )


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
    if _should_use_cpu(state, x, dt, A, B, C, D, z, dt_bias):
        ops = _load_cpu_ops()
        if ops is not None:
            return _invoke_cpu(
                ops.selective_state_step,
                state,
                x,
                dt,
                A,
                B,
                C,
                D,
                z,
                dt_bias,
                softplus,
            )
    return reference_ops.selective_state_step(
        state,
        x,
        dt,
        A,
        B,
        C,
        D=D,
        z=z,
        dt_bias=dt_bias,
        softplus=softplus,
    )


def _prepare_seq_meta(
    seq_meta: Optional[dict[str, Any]], device: torch.device
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not seq_meta:
        return None, None

    seq_lens_tensor: torch.Tensor | None = None
    cu_seqlens_tensor: torch.Tensor | None = None

    if "seq_lens" in seq_meta and seq_meta["seq_lens"] is not None:
        seq_lens_tensor = torch.as_tensor(
            seq_meta["seq_lens"], device=device, dtype=torch.long
        )
    if "cu_seqlens" in seq_meta and seq_meta["cu_seqlens"] is not None:
        cu = seq_meta["cu_seqlens"]
        cu_seqlens_tensor = (
            cu.to(device=device, dtype=torch.long)
            if isinstance(cu, torch.Tensor)
            else torch.as_tensor(cu, device=device, dtype=torch.long)
        )

    return seq_lens_tensor, cu_seqlens_tensor


def ssd_chunk_scan(
    X: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    seq_meta: Optional[dict[str, Any]] = None,
    initial_states: torch.Tensor | None = None,
) -> torch.Tensor:
    if _should_use_cpu(X, dt, A, B, C, D, z, initial_states):
        ops = _load_cpu_ops()
        if ops is not None:
            seq_lens_tensor, cu_seqlens_tensor = _prepare_seq_meta(seq_meta, X.device)
            return _invoke_cpu(
                ops.ssd_chunk_scan,
                X,
                dt,
                A,
                B,
                C,
                chunk_size,
                D,
                z,
                seq_lens_tensor,
                cu_seqlens_tensor,
                initial_states,
            )
    return reference_ops.ssd_chunk_scan(
        X,
        dt,
        A,
        B,
        C,
        chunk_size,
        D=D,
        z=z,
        seq_meta=seq_meta,
        initial_states=initial_states,
    )


def dw_causal_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str = "silu",
) -> torch.Tensor:
    if _should_use_cpu(x, weight, bias):
        ops = _load_cpu_ops()
        if ops is not None:
            return _invoke_cpu(ops.dw_causal_conv, x, weight, bias, activation)
    return reference_ops.dw_causal_conv(x, weight, bias=bias, activation=activation)


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
    if _should_use_cpu(x, weight, bias, residual):
        ops = _load_cpu_ops()
        if ops is not None:
            return _invoke_cpu(
                ops.fused_layer_norm,
                x,
                weight,
                bias,
                residual,
                is_rms,
                eps,
                prenorm,
                residual_in_fp32,
            )
    return reference_ops.fused_layer_norm(
        x,
        weight,
        bias,
        residual=residual,
        is_rms=is_rms,
        eps=eps,
        prenorm=prenorm,
        residual_in_fp32=residual_in_fp32,
    )


__all__ = [
    "selective_scan",
    "selective_state_step",
    "ssd_chunk_scan",
    "dw_causal_conv",
    "fused_layer_norm",
]

selective_scan.__doc__ = reference_ops.selective_scan.__doc__
selective_state_step.__doc__ = reference_ops.selective_state_step.__doc__
ssd_chunk_scan.__doc__ = reference_ops.ssd_chunk_scan.__doc__
dw_causal_conv.__doc__ = reference_ops.dw_causal_conv.__doc__
fused_layer_norm.__doc__ = reference_ops.fused_layer_norm.__doc__


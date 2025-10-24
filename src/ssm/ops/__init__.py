"""Ops façade with backend dispatch."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, cast

import torch
import torch.nn.functional as F

from .python import reference as reference_ops

try:
    from torch.utils.cpp_extension import load as _load_extension
except ImportError:  # pragma: no cover - torch utils missing
    _load_extension = None

try:  # pragma: no cover - optional CUDA AMP import
    from torch.cuda.amp import autocast as _cuda_autocast
except Exception:  # pragma: no cover - CUDA AMP unavailable
    _cuda_autocast = None


_CPU_OPS: Any = None
_CPU_LOAD_ERROR: Optional[Exception] = None
_CUDA_OPS: Any = None
_CUDA_LOAD_ERROR: Optional[Exception] = None


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


def _cuda_sources() -> list[str]:
    root = Path(__file__).resolve().parent / "cuda"
    return [
        str(root / "bindings.cpp"),
        str(root / "selective_scan.cu"),
        str(root / "state_step.cu"),
        str(root / "ssd_chunk_scan.cu"),
        str(root / "dw_causal_conv.cu"),
        str(root / "layer_norm.cu"),
    ]


def _load_cpu_ops() -> Optional[Any]:
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


def _load_cuda_ops() -> Optional[Any]:
    global _CUDA_OPS, _CUDA_LOAD_ERROR
    if _CUDA_OPS is not None:
        return _CUDA_OPS
    if _CUDA_LOAD_ERROR is not None:
        return None
    if _load_extension is None:
        _CUDA_LOAD_ERROR = RuntimeError("torch.utils.cpp_extension.load unavailable")
        return None
    if not torch.cuda.is_available():
        _CUDA_LOAD_ERROR = RuntimeError("CUDA is not available")
        return None

    try:
        _CUDA_OPS = _load_extension(
            name="ssm_cuda_ops",
            sources=_cuda_sources(),
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
        return _CUDA_OPS
    except Exception as exc:  # pragma: no cover - build environment specific
        _CUDA_LOAD_ERROR = exc
        return None


def _should_use_cpu(*tensors: Optional[torch.Tensor]) -> bool:
    return all(t.device.type == "cpu" for t in tensors if isinstance(t, torch.Tensor))


def _should_use_cuda(*tensors: Optional[torch.Tensor]) -> bool:
    devices = [t.device.type for t in tensors if isinstance(t, torch.Tensor)]
    if not devices:
        return False
    if any(device == "cuda" for device in devices):
        return all(device == "cuda" for device in devices)
    return False


def _any_requires_grad(*tensors: Optional[torch.Tensor]) -> bool:
    return any(t is not None and t.requires_grad for t in tensors)


def _state_step_reference_no_inplace(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor],
    z: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    softplus: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, dim, state_dim = state.shape
    compute_dtype = reference_ops._get_compute_dtype(state)
    if A.dtype != torch.float32 and A.dtype != compute_dtype:
        compute_dtype = torch.promote_types(compute_dtype, A.dtype)

    state_compute = state.to(compute_dtype)
    x_compute = x.to(compute_dtype)
    dt_compute = dt.to(compute_dtype)
    A_compute = A.to(compute_dtype)

    def _expand_grouped(name: str, param: torch.Tensor) -> torch.Tensor:
        if param.dim() == 3 and param.shape[0] == batch and param.shape[1] != dim:
            groups = param.shape[1]
            if dim % groups != 0:
                raise ValueError(f"{name} group dimension must divide D.")
            if param.shape[2] != state_dim:
                raise ValueError(f"{name} must have matching state dimension.")
            return param.repeat_interleave(dim // groups, dim=1)
        return param

    B_prepared = _expand_grouped("B", B)
    C_prepared = _expand_grouped("C", C)

    B_expanded = reference_ops._normalize_scan_param(
        "B", B_prepared, batch, dim, state_dim, 1, compute_dtype
    )[:, :, :, 0]
    C_expanded = reference_ops._normalize_scan_param(
        "C", C_prepared, batch, dim, state_dim, 1, compute_dtype
    )[:, :, :, 0]

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

    if D is not None:
        if D.shape != (dim,):
            raise ValueError("D must have shape (D,).")
        output = output + D.to(compute_dtype).view(1, dim) * x_compute

    if z is not None:
        if z.shape != (batch, dim):
            raise ValueError("z must have shape (B, D).")
        output = output * F.silu(z.to(compute_dtype))

    return output, new_state


class _SelectiveScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        return_last_state: bool,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
        dt_bias: Optional[torch.Tensor],
        softplus: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        ops = _load_cpu_ops()
        assert ops is not None
        output, last_state = ops.selective_scan(
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
        ctx.save_for_backward(u, delta, A, B, C)
        ctx.D = D
        ctx.z = z
        ctx.dt_bias = dt_bias
        ctx.softplus = softplus
        ctx.input_requires_grad = (
            u.requires_grad,
            delta.requires_grad,
            A.requires_grad,
            B.requires_grad,
            C.requires_grad,
            D.requires_grad if D is not None else False,
            z.requires_grad if z is not None else False,
            dt_bias.requires_grad if dt_bias is not None else False,
        )
        if return_last_state:
            return output, last_state
        return output

    @staticmethod
    def backward(  # type: ignore[override]
        ctx,
        *grad_outputs: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], ...]:
        grad_output = grad_outputs[0] if grad_outputs else None
        grad_last_state = grad_outputs[1] if len(grad_outputs) > 1 else None
        u, delta, A, B, C = ctx.saved_tensors
        D: Optional[torch.Tensor] = ctx.D
        z: Optional[torch.Tensor] = ctx.z
        dt_bias: Optional[torch.Tensor] = ctx.dt_bias
        softplus: bool = ctx.softplus
        req = ctx.input_requires_grad

        with torch.enable_grad():
            u_ref = u.detach().clone().requires_grad_(req[0])
            delta_ref = delta.detach().clone().requires_grad_(req[1])
            A_ref = A.detach().clone().requires_grad_(req[2])
            B_ref = B.detach().clone().requires_grad_(req[3])
            C_ref = C.detach().clone().requires_grad_(req[4])
            D_ref = D.detach().clone().requires_grad_(req[5]) if D is not None else None
            z_ref = z.detach().clone().requires_grad_(req[6]) if z is not None else None
            dt_bias_ref = (
                dt_bias.detach().clone().requires_grad_(req[7])
                if dt_bias is not None
                else None
            )

            out_ref, last_state_ref = reference_ops.selective_scan(
                u_ref,
                delta_ref,
                A_ref,
                B_ref,
                C_ref,
                D=D_ref,
                z=z_ref,
                dt_bias=dt_bias_ref,
                softplus=softplus,
                return_last_state=True,
            )

            grad_out = (
                grad_output if grad_output is not None else torch.zeros_like(out_ref)
            )
            grad_last = (
                grad_last_state
                if grad_last_state is not None
                else torch.zeros_like(last_state_ref)
            )

            inputs: list[torch.Tensor] = []
            mapping: list[int] = []
            candidates = [
                u_ref,
                delta_ref,
                A_ref,
                B_ref,
                C_ref,
                D_ref,
                z_ref,
                dt_bias_ref,
            ]
            for idx, tensor in enumerate(candidates):
                if tensor is not None and tensor.requires_grad:
                    inputs.append(tensor)
                    mapping.append(idx)

            grads: tuple[torch.Tensor, ...]
            if inputs:
                grads = torch.autograd.grad(
                    outputs=(out_ref, last_state_ref),
                    inputs=inputs,
                    grad_outputs=(grad_out, grad_last),
                    allow_unused=True,
                )
            else:
                grads = tuple()

        result: list[Optional[torch.Tensor]] = [None] * 8
        for idx, grad in zip(mapping, grads):
            result[idx] = grad

        return (None, *result, None)


class _SelectiveStateStepFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
        dt_bias: Optional[torch.Tensor],
        softplus: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ops = _load_cpu_ops()
        assert ops is not None
        state_saved = state.clone()
        state_work = state.clone()
        output = ops.selective_state_step(
            state_work,
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
        ctx.save_for_backward(state_saved, x, dt, A, B, C)
        ctx.D = D
        ctx.z = z
        ctx.dt_bias = dt_bias
        ctx.softplus = softplus
        ctx.input_requires_grad = (
            state.requires_grad,
            x.requires_grad,
            dt.requires_grad,
            A.requires_grad,
            B.requires_grad,
            C.requires_grad,
            D.requires_grad if D is not None else False,
            z.requires_grad if z is not None else False,
            dt_bias.requires_grad if dt_bias is not None else False,
        )
        return output, state_work

    @staticmethod
    def backward(  # type: ignore[override]
        ctx,
        grad_output: Optional[torch.Tensor],
        grad_state: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], ...]:
        state_saved, x, dt, A, B, C = ctx.saved_tensors
        D: Optional[torch.Tensor] = ctx.D
        z: Optional[torch.Tensor] = ctx.z
        dt_bias: Optional[torch.Tensor] = ctx.dt_bias
        softplus: bool = ctx.softplus
        req = ctx.input_requires_grad

        with torch.enable_grad():
            state_ref = state_saved.detach().clone().requires_grad_(req[0])
            x_ref = x.detach().clone().requires_grad_(req[1])
            dt_ref = dt.detach().clone().requires_grad_(req[2])
            A_ref = A.detach().clone().requires_grad_(req[3])
            B_ref = B.detach().clone().requires_grad_(req[4])
            C_ref = C.detach().clone().requires_grad_(req[5])
            D_ref = D.detach().clone().requires_grad_(req[6]) if D is not None else None
            z_ref = z.detach().clone().requires_grad_(req[7]) if z is not None else None
            dt_bias_ref = (
                dt_bias.detach().clone().requires_grad_(req[8])
                if dt_bias is not None
                else None
            )

            out_ref, new_state_ref = _state_step_reference_no_inplace(
                state_ref,
                x_ref,
                dt_ref,
                A_ref,
                B_ref,
                C_ref,
                D_ref,
                z_ref,
                dt_bias_ref,
                softplus,
            )

            grad_out = (
                grad_output.to(out_ref.dtype)
                if grad_output is not None
                else torch.zeros_like(out_ref)
            )
            grad_state_ref = (
                grad_state.to(new_state_ref.dtype)
                if grad_state is not None
                else torch.zeros_like(new_state_ref)
            )

            inputs: list[torch.Tensor] = []
            mapping: list[int] = []
            candidates = [
                state_ref,
                x_ref,
                dt_ref,
                A_ref,
                B_ref,
                C_ref,
                D_ref,
                z_ref,
                dt_bias_ref,
            ]
            for idx, tensor in enumerate(candidates):
                if tensor is not None and tensor.requires_grad:
                    inputs.append(tensor)
                    mapping.append(idx)

            grads: tuple[torch.Tensor, ...]
            if inputs:
                grads = torch.autograd.grad(
                    outputs=(out_ref, new_state_ref),
                    inputs=inputs,
                    grad_outputs=(grad_out, grad_state_ref),
                    allow_unused=True,
                )
            else:
                grads = tuple()

        result: list[Optional[torch.Tensor]] = [None] * 9
        for idx, grad in zip(mapping, grads):
            result[idx] = grad

        return (*result, None)


class _SSDChunkScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        X: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        chunk_size: int,
        D: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
        seq_meta: Optional[dict[str, Any]],
        initial_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        ops = _load_cpu_ops()
        assert ops is not None
        seq_lens_tensor, cu_seqlens_tensor = _prepare_seq_meta(seq_meta, X.device)
        output = ops.ssd_chunk_scan(
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
        ctx.save_for_backward(X, dt, A, B, C)
        ctx.chunk_size = chunk_size
        ctx.D = D
        ctx.z = z
        ctx.seq_meta = seq_meta
        ctx.initial_states = initial_states
        ctx.input_requires_grad = (
            X.requires_grad,
            dt.requires_grad,
            A.requires_grad,
            B.requires_grad,
            C.requires_grad,
            D.requires_grad if D is not None else False,
            z.requires_grad if z is not None else False,
            initial_states.requires_grad if initial_states is not None else False,
        )
        return output

    @staticmethod
    def backward(  # type: ignore[override]
        ctx,
        grad_output: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], ...]:
        X, dt, A, B, C = ctx.saved_tensors
        D: Optional[torch.Tensor] = ctx.D
        z: Optional[torch.Tensor] = ctx.z
        seq_meta: Optional[dict[str, Any]] = ctx.seq_meta
        initial_states: Optional[torch.Tensor] = ctx.initial_states
        chunk_size: int = ctx.chunk_size
        req = ctx.input_requires_grad

        with torch.enable_grad():
            X_ref = X.detach().clone().requires_grad_(req[0])
            dt_ref = dt.detach().clone().requires_grad_(req[1])
            A_ref = A.detach().clone().requires_grad_(req[2])
            B_ref = B.detach().clone().requires_grad_(req[3])
            C_ref = C.detach().clone().requires_grad_(req[4])
            D_ref = D.detach().clone().requires_grad_(req[5]) if D is not None else None
            z_ref = z.detach().clone().requires_grad_(req[6]) if z is not None else None
            init_ref = (
                initial_states.detach().clone().requires_grad_(req[7])
                if initial_states is not None
                else None
            )

            out_ref = reference_ops.ssd_chunk_scan(
                X_ref,
                dt_ref,
                A_ref,
                B_ref,
                C_ref,
                chunk_size,
                D=D_ref,
                z=z_ref,
                seq_meta=seq_meta,
                initial_states=init_ref,
            )

            grad_out = (
                grad_output if grad_output is not None else torch.zeros_like(out_ref)
            )

            inputs: list[torch.Tensor] = []
            mapping: list[int] = []
            candidates = [
                X_ref,
                dt_ref,
                A_ref,
                B_ref,
                C_ref,
                D_ref,
                z_ref,
                init_ref,
            ]
            for idx, tensor in enumerate(candidates):
                if tensor is not None and tensor.requires_grad:
                    inputs.append(tensor)
                    mapping.append(idx)

            grads: tuple[torch.Tensor, ...]
            if inputs:
                grads = torch.autograd.grad(
                    outputs=out_ref,
                    inputs=inputs,
                    grad_outputs=grad_out,
                    allow_unused=True,
                )
            else:
                grads = tuple()

        full: list[Optional[torch.Tensor]] = [None] * 10
        positions = [0, 1, 2, 3, 4, 6, 7, 9]
        for idx, grad in zip(mapping, grads):
            full[positions[idx]] = grad

        return tuple(full)


class _DwCausalConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        activation: str,
    ) -> torch.Tensor:
        ops = _load_cpu_ops()
        assert ops is not None
        output = ops.dw_causal_conv(x, weight, bias, activation)
        placeholder = weight.new_empty(0)
        ctx.save_for_backward(x, weight, bias if bias is not None else placeholder)
        ctx.activation = activation
        ctx.has_bias = bias is not None
        ctx.input_requires_grad = (
            x.requires_grad,
            weight.requires_grad,
            bias.requires_grad if bias is not None else False,
        )
        return output

    @staticmethod
    def backward(  # type: ignore[override]
        ctx,
        grad_output: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], ...]:
        saved = ctx.saved_tensors
        x = saved[0]
        weight = saved[1]
        bias_tensor = saved[2] if ctx.has_bias else None
        activation: str = ctx.activation
        req = ctx.input_requires_grad

        with torch.enable_grad():
            x_ref = x.detach().clone().requires_grad_(req[0])
            weight_ref = weight.detach().clone().requires_grad_(req[1])
            if ctx.has_bias and bias_tensor is not None:
                bias_ref = bias_tensor.detach().clone().requires_grad_(req[2])
            else:
                bias_ref = None

            out_ref = reference_ops.dw_causal_conv(
                x_ref,
                weight_ref,
                bias=bias_ref,
                activation=activation,
            )

            grad_out = (
                grad_output if grad_output is not None else torch.zeros_like(out_ref)
            )

            inputs: list[torch.Tensor] = []
            mapping: list[int] = []
            candidates = [x_ref, weight_ref, bias_ref]
            for idx, tensor in enumerate(candidates):
                if tensor is not None and tensor.requires_grad:
                    inputs.append(tensor)
                    mapping.append(idx)

            grads: tuple[torch.Tensor, ...]
            if inputs:
                grads = torch.autograd.grad(
                    outputs=out_ref,
                    inputs=inputs,
                    grad_outputs=grad_out,
                    allow_unused=True,
                )
            else:
                grads = tuple()

        result: list[Optional[torch.Tensor]] = [None] * 3
        for idx, grad in zip(mapping, grads):
            result[idx] = grad

        return (*result, None)


class _FusedLayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        residual: Optional[torch.Tensor],
        is_rms: bool,
        eps: float,
        prenorm: bool,
        residual_in_fp32: bool,
    ) -> torch.Tensor:
        ops = _load_cpu_ops()
        assert ops is not None
        output = ops.fused_layer_norm(
            x,
            weight,
            bias,
            residual=residual,
            is_rms=is_rms,
            eps=eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
        )
        bias_placeholder = weight.new_empty(0)
        residual_placeholder = x.new_empty(0)
        ctx.save_for_backward(
            x,
            weight,
            bias if bias is not None else bias_placeholder,
            residual if residual is not None else residual_placeholder,
        )
        ctx.flags = (
            is_rms,
            eps,
            prenorm,
            residual_in_fp32,
            bias is not None,
            residual is not None,
        )
        ctx.input_requires_grad = (
            x.requires_grad,
            weight.requires_grad,
            bias.requires_grad if bias is not None else False,
            residual.requires_grad if residual is not None else False,
        )
        return output

    @staticmethod
    def backward(  # type: ignore[override]
        ctx,
        grad_output: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], ...]:
        x, weight, bias_tensor, residual_tensor = ctx.saved_tensors
        is_rms, eps, prenorm, residual_in_fp32, has_bias, has_residual = ctx.flags
        req = ctx.input_requires_grad

        with torch.enable_grad():
            x_ref = x.detach().clone().requires_grad_(req[0])
            weight_ref = weight.detach().clone().requires_grad_(req[1])
            bias_ref = (
                bias_tensor.detach().clone().requires_grad_(req[2])
                if has_bias
                else None
            )
            residual_ref = (
                residual_tensor.detach().clone().requires_grad_(req[3])
                if has_residual
                else None
            )

            out_ref = reference_ops.fused_layer_norm(
                x_ref,
                weight_ref,
                bias_ref,
                residual=residual_ref,
                is_rms=is_rms,
                eps=eps,
                prenorm=prenorm,
                residual_in_fp32=residual_in_fp32,
            )

            grad_out = (
                grad_output if grad_output is not None else torch.zeros_like(out_ref)
            )

            inputs: list[torch.Tensor] = []
            mapping: list[int] = []
            candidates = [x_ref, weight_ref, bias_ref, residual_ref]
            for idx, tensor in enumerate(candidates):
                if tensor is not None and tensor.requires_grad:
                    inputs.append(tensor)
                    mapping.append(idx)

            grads: tuple[torch.Tensor, ...]
            if inputs:
                grads = torch.autograd.grad(
                    outputs=out_ref,
                    inputs=inputs,
                    grad_outputs=grad_out,
                    allow_unused=True,
                )
            else:
                grads = tuple()

        result: list[Optional[torch.Tensor]] = [None] * 4
        for idx, grad in zip(mapping, grads):
            result[idx] = grad

        return (*result, None, None, None, None)


def _cpu_backend_status() -> _BackendCandidate:
    ops = _load_cpu_ops()
    return _BackendCandidate(name="cpu", available=ops is not None)


def _cuda_backend_status() -> _BackendCandidate:
    ops = _load_cuda_ops()
    return _BackendCandidate(name="cuda", available=ops is not None)


def _invoke_cpu(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except RuntimeError as exc:  # translate contract violations
        raise ValueError(str(exc)) from None


def _invoke_cuda(fn, *args, **kwargs):
    context = (
        _cuda_autocast(enabled=False) if _cuda_autocast is not None else nullcontext()
    )
    try:
        with context:
            return fn(*args, **kwargs)
    except RuntimeError as exc:
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
    if _should_use_cuda(u, delta, A, B, C, D, z, dt_bias):
        ops = _load_cuda_ops()
        if ops is not None:
            result = _invoke_cuda(
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
    if _should_use_cpu(u, delta, A, B, C, D, z, dt_bias):
        ops = _load_cpu_ops()
        if ops is not None:
            if _any_requires_grad(u, delta, A, B, C, D, z, dt_bias):
                outputs = _SelectiveScanFunction.apply(
                    return_last_state,
                    u,
                    delta,
                    A,
                    B,
                    C,
                    D,
                    z,
                    dt_bias,
                    softplus,
                )
                if return_last_state:
                    output, last_state = cast(
                        tuple[torch.Tensor, torch.Tensor], outputs
                    )
                    return output, last_state
                return cast(torch.Tensor, outputs)
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
    if _should_use_cuda(state, x, dt, A, B, C, D, z, dt_bias):
        ops = _load_cuda_ops()
        if ops is not None:
            return _invoke_cuda(
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
    if _should_use_cpu(state, x, dt, A, B, C, D, z, dt_bias):
        ops = _load_cpu_ops()
        if ops is not None:
            if _any_requires_grad(state, x, dt, A, B, C, D, z, dt_bias):
                output, new_state = cast(
                    tuple[torch.Tensor, torch.Tensor],
                    _SelectiveStateStepFunction.apply(
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
                    ),
                )
                state.copy_(new_state)
                return output
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
    if _should_use_cuda(X, dt, A, B, C, D, z, initial_states):
        ops = _load_cuda_ops()
        if ops is not None:
            seq_lens_tensor, cu_seqlens_tensor = _prepare_seq_meta(seq_meta, X.device)
            return _invoke_cuda(
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
    if _should_use_cpu(X, dt, A, B, C, D, z, initial_states):
        ops = _load_cpu_ops()
        if ops is not None:
            if _any_requires_grad(X, dt, A, B, C, D, z, initial_states):
                return cast(
                    torch.Tensor,
                    _SSDChunkScanFunction.apply(
                        X,
                        dt,
                        A,
                        B,
                        C,
                        chunk_size,
                        D,
                        z,
                        seq_meta,
                        initial_states,
                    ),
                )
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
    if _should_use_cuda(x, weight, bias):
        ops = _load_cuda_ops()
        if ops is not None:
            return _invoke_cuda(ops.dw_causal_conv, x, weight, bias, activation)
    if _should_use_cpu(x, weight, bias):
        ops = _load_cpu_ops()
        if ops is not None:
            if _any_requires_grad(x, weight, bias):
                return cast(
                    torch.Tensor,
                    _DwCausalConvFunction.apply(x, weight, bias, activation),
                )
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
    if _should_use_cuda(x, weight, bias, residual):
        ops = _load_cuda_ops()
        if ops is not None:
            return _invoke_cuda(
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
    if _should_use_cpu(x, weight, bias, residual):
        ops = _load_cpu_ops()
        if ops is not None:
            if _any_requires_grad(x, weight, bias, residual):
                return cast(
                    torch.Tensor,
                    _FusedLayerNormFunction.apply(
                        x,
                        weight,
                        bias,
                        residual,
                        is_rms,
                        eps,
                        prenorm,
                        residual_in_fp32,
                    ),
                )
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
    "_cpu_backend_status",
    "_cuda_backend_status",
]

selective_scan.__doc__ = reference_ops.selective_scan.__doc__
selective_state_step.__doc__ = reference_ops.selective_state_step.__doc__
ssd_chunk_scan.__doc__ = reference_ops.ssd_chunk_scan.__doc__
dw_causal_conv.__doc__ = reference_ops.dw_causal_conv.__doc__
fused_layer_norm.__doc__ = reference_ops.fused_layer_norm.__doc__

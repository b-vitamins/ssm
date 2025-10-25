"""Gradient parity tests for compiled ops against the Python reference."""

from __future__ import annotations

from typing import Iterable

import pytest
import torch

import ssm.ops as ops
from ssm.ops.python import reference as reference_ops


_DEVICE_DTYPE_CASES = [
    pytest.param(
        torch.device("cpu"),
        torch.float32,
        id="cpu-f32",
    ),
    pytest.param(
        torch.device("cuda"),
        torch.float32,
        id="cuda-f32",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA is not available"
        ),
    ),
    pytest.param(
        torch.device("cuda"),
        torch.float16,
        id="cuda-f16",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA is not available"
        ),
    ),
]

_GRADCHECK_DEVICES = [
    pytest.param(torch.device("cpu"), id="cpu"),
    pytest.param(
        torch.device("cuda"),
        id="cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA is not available"
        ),
    ),
]


def _require_backend(device: torch.device) -> None:
    if device.type == "cuda":
        status = ops._cuda_backend_status()
    else:
        status = ops._cpu_backend_status()
    if not status.available:
        pytest.skip(f"{device.type.upper()} backend unavailable: {status}")


def _clone_tensor(value: torch.Tensor) -> torch.Tensor:
    clone = value.detach().clone()
    clone.requires_grad_(value.requires_grad)
    return clone


def _clone_optional_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    return _clone_tensor(value)


def _require_tensor(value: torch.Tensor | None) -> torch.Tensor:
    if value is None:  # pragma: no cover - defensive
        raise AssertionError("Expected tensor, received None")
    return value


def _output_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float16:
        return 2e-3, 3e-3
    if dtype == torch.float64:
        return 1e-7, 1e-7
    return 1e-4, 1e-4


def _grad_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float16:
        return 5e-3, 5e-3
    if dtype == torch.float64:
        return 1e-6, 1e-6
    return 1e-4, 1e-4


def _assert_grads_close(
    tensors: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    atol: float,
    rtol: float,
) -> None:
    for test_tensor, ref_tensor in tensors:
        if not test_tensor.requires_grad:
            continue
        test_grad = test_tensor.grad
        ref_grad = ref_tensor.grad
        assert test_grad is not None
        assert ref_grad is not None
        torch.testing.assert_close(
            test_grad,
            ref_grad.to(test_grad.dtype),
            atol=atol,
            rtol=rtol,
        )


@pytest.mark.parametrize(("device", "dtype"), _DEVICE_DTYPE_CASES)
def test_selective_scan_backend_matches_reference(
    device: torch.device, dtype: torch.dtype
) -> None:
    _require_backend(device)
    torch.manual_seed(0)

    batch, dim, state_dim, length = 2, 3, 4, 5
    u = torch.randn(batch, dim, length, device=device, dtype=dtype, requires_grad=True)
    delta = torch.randn_like(u, requires_grad=True)
    A = torch.randn(
        dim, state_dim, device=device, dtype=torch.float32, requires_grad=True
    )
    B = torch.randn(dim, state_dim, device=device, dtype=dtype, requires_grad=True)
    C = torch.randn(dim, state_dim, device=device, dtype=dtype, requires_grad=True)
    D_skip = torch.randn(dim, device=device, dtype=dtype, requires_grad=True)
    z = torch.randn(batch, dim, length, device=device, dtype=dtype, requires_grad=True)
    dt_bias = torch.randn(dim, device=device, dtype=dtype, requires_grad=True)

    ref_args = (
        _clone_tensor(u),
        _clone_tensor(delta),
        _clone_tensor(A),
        _clone_tensor(B),
        _clone_tensor(C),
    )
    ref_kwargs = {
        "D": _clone_optional_tensor(D_skip),
        "z": _clone_optional_tensor(z),
        "dt_bias": _clone_optional_tensor(dt_bias),
        "softplus": True,
        "return_last_state": True,
    }

    out, state = ops.selective_scan(
        u,
        delta,
        A,
        B,
        C,
        D=D_skip,
        z=z,
        dt_bias=dt_bias,
        softplus=True,
        return_last_state=True,
    )
    ref_out, ref_state = reference_ops.selective_scan(*ref_args, **ref_kwargs)

    data_atol, data_rtol = _output_tolerances(dtype)
    torch.testing.assert_close(
        out, ref_out.to(out.dtype), atol=data_atol, rtol=data_rtol
    )
    torch.testing.assert_close(
        state, ref_state.to(state.dtype), atol=data_atol, rtol=data_rtol
    )

    (out.sum() + state.sum()).backward()
    (ref_out.sum() + ref_state.sum()).backward()

    grad_atol, grad_rtol = _grad_tolerances(dtype)
    _assert_grads_close(
        [
            (u, ref_args[0]),
            (delta, ref_args[1]),
            (A, ref_args[2]),
            (B, ref_args[3]),
            (C, ref_args[4]),
            (D_skip, _require_tensor(ref_kwargs["D"])),
            (z, _require_tensor(ref_kwargs["z"])),
            (dt_bias, _require_tensor(ref_kwargs["dt_bias"])),
        ],
        atol=grad_atol,
        rtol=grad_rtol,
    )


@pytest.mark.parametrize("device", _GRADCHECK_DEVICES)
def test_selective_scan_gradcheck(device: torch.device) -> None:
    _require_backend(device)
    torch.manual_seed(1)

    batch, dim, state_dim, length = 1, 2, 2, 3
    dtype = torch.float64

    u = torch.randn(batch, dim, length, device=device, dtype=dtype, requires_grad=True)
    delta = torch.randn_like(u, requires_grad=True)
    A = torch.randn(dim, state_dim, device=device, dtype=dtype, requires_grad=True)
    B = torch.randn(dim, state_dim, device=device, dtype=dtype, requires_grad=True)
    C = torch.randn(dim, state_dim, device=device, dtype=dtype, requires_grad=True)

    def func(u, delta, A, B, C):
        out, _ = ops.selective_scan(
            u,
            delta,
            A,
            B,
            C,
            D=None,
            z=None,
            dt_bias=None,
            softplus=False,
            return_last_state=True,
        )
        return out

    assert torch.autograd.gradcheck(func, (u, delta, A, B, C), eps=1e-4, atol=1e-4)


@pytest.mark.parametrize(("device", "dtype"), _DEVICE_DTYPE_CASES)
def test_selective_state_step_backend_matches_reference(
    device: torch.device, dtype: torch.dtype
) -> None:
    _require_backend(device)
    torch.manual_seed(2)

    batch, dim, state_dim = 2, 3, 4
    base_state = torch.randn(batch, dim, state_dim, device=device, dtype=dtype)
    state_test = base_state.clone()
    state_ref = base_state.clone()

    x = torch.randn(batch, dim, device=device, dtype=dtype, requires_grad=True)
    dt = torch.randn_like(x, requires_grad=True)
    A = torch.randn(
        dim, state_dim, device=device, dtype=torch.float32, requires_grad=True
    )
    B = torch.randn(dim, state_dim, device=device, dtype=dtype, requires_grad=True)
    C = torch.randn(dim, state_dim, device=device, dtype=dtype, requires_grad=True)
    D_skip = torch.randn(dim, device=device, dtype=dtype, requires_grad=True)
    z = torch.randn(batch, dim, device=device, dtype=dtype, requires_grad=True)
    dt_bias = torch.randn(dim, device=device, dtype=dtype, requires_grad=True)

    ref_inputs = (
        _clone_tensor(x),
        _clone_tensor(dt),
        _clone_tensor(A),
        _clone_tensor(B),
        _clone_tensor(C),
        _clone_optional_tensor(D_skip),
        _clone_optional_tensor(z),
        _clone_optional_tensor(dt_bias),
    )

    out = ops.selective_state_step(
        state_test,
        x,
        dt,
        A,
        B,
        C,
        D=D_skip,
        z=z,
        dt_bias=dt_bias,
        softplus=True,
    )
    ref_out = reference_ops.selective_state_step(
        state_ref,
        ref_inputs[0],
        ref_inputs[1],
        ref_inputs[2],
        ref_inputs[3],
        ref_inputs[4],
        D=ref_inputs[5],
        z=ref_inputs[6],
        dt_bias=ref_inputs[7],
        softplus=True,
    )

    data_atol, data_rtol = _output_tolerances(dtype)
    torch.testing.assert_close(
        out, ref_out.to(out.dtype), atol=data_atol, rtol=data_rtol
    )
    torch.testing.assert_close(
        state_test, state_ref.to(state_test.dtype), atol=data_atol, rtol=data_rtol
    )

    out.sum().backward()
    ref_out.sum().backward()

    grad_atol, grad_rtol = _grad_tolerances(dtype)
    _assert_grads_close(
        [
            (x, ref_inputs[0]),
            (dt, ref_inputs[1]),
            (A, ref_inputs[2]),
            (B, ref_inputs[3]),
            (C, ref_inputs[4]),
            (D_skip, _require_tensor(ref_inputs[5])),
            (z, _require_tensor(ref_inputs[6])),
            (dt_bias, _require_tensor(ref_inputs[7])),
        ],
        atol=grad_atol,
        rtol=grad_rtol,
    )


@pytest.mark.parametrize("device", _GRADCHECK_DEVICES)
def test_selective_state_step_gradcheck(device: torch.device) -> None:
    _require_backend(device)
    torch.manual_seed(3)

    batch, dim, state_dim = 1, 2, 3
    dtype = torch.float64

    base_state = torch.zeros(batch, dim, state_dim, device=device, dtype=dtype)
    x = torch.randn(batch, dim, device=device, dtype=dtype, requires_grad=True)
    dt = torch.randn_like(x, requires_grad=True)
    A = torch.randn(dim, state_dim, device=device, dtype=dtype, requires_grad=True)
    B = torch.randn(dim, state_dim, device=device, dtype=dtype, requires_grad=True)
    C = torch.randn(dim, state_dim, device=device, dtype=dtype, requires_grad=True)

    def func(x, dt, A, B, C):
        state_local = base_state.clone()
        return ops.selective_state_step(
            state_local,
            x,
            dt,
            A,
            B,
            C,
            D=None,
            z=None,
            dt_bias=None,
            softplus=False,
        )

    assert torch.autograd.gradcheck(func, (x, dt, A, B, C), eps=1e-4, atol=1e-4)


@pytest.mark.parametrize(("device", "dtype"), _DEVICE_DTYPE_CASES)
def test_ssd_chunk_scan_backend_matches_reference(
    device: torch.device, dtype: torch.dtype
) -> None:
    _require_backend(device)
    torch.manual_seed(4)

    batch, seqlen, heads, proj = 2, 4, 3, 2
    chunk_size = 2

    X = torch.randn(
        batch, seqlen, heads, proj, device=device, dtype=dtype, requires_grad=True
    )
    dt = torch.randn(
        batch, seqlen, heads, device=device, dtype=dtype, requires_grad=True
    )
    A = torch.randn(heads, proj, device=device, dtype=torch.float32, requires_grad=True)
    B = torch.randn(heads, proj, device=device, dtype=dtype, requires_grad=True)
    C = torch.randn(heads, proj, device=device, dtype=dtype, requires_grad=True)
    D_skip = torch.randn(heads, device=device, dtype=dtype, requires_grad=True)
    z = torch.randn(
        batch, seqlen, heads, device=device, dtype=dtype, requires_grad=True
    )
    initial_state = torch.randn(batch, heads, proj, device=device, dtype=dtype)
    initial_state_test = initial_state.clone()
    initial_state_ref = initial_state.clone()

    ref_args = (
        _clone_tensor(X),
        _clone_tensor(dt),
        _clone_tensor(A),
        _clone_tensor(B),
        _clone_tensor(C),
        _clone_optional_tensor(D_skip),
        _clone_optional_tensor(z),
    )

    out = ops.ssd_chunk_scan(
        X,
        dt,
        A,
        B,
        C,
        chunk_size=chunk_size,
        D=D_skip,
        z=z,
        seq_meta=None,
        initial_states=initial_state_test,
    )
    ref_out = reference_ops.ssd_chunk_scan(
        ref_args[0],
        ref_args[1],
        ref_args[2],
        ref_args[3],
        ref_args[4],
        chunk_size=chunk_size,
        D=ref_args[5],
        z=ref_args[6],
        seq_meta=None,
        initial_states=initial_state_ref,
    )

    data_atol, data_rtol = _output_tolerances(dtype)
    torch.testing.assert_close(
        out, ref_out.to(out.dtype), atol=data_atol, rtol=data_rtol
    )

    out.sum().backward()
    ref_out.sum().backward()

    grad_atol, grad_rtol = _grad_tolerances(dtype)
    _assert_grads_close(
        [
            (X, ref_args[0]),
            (dt, ref_args[1]),
            (A, ref_args[2]),
            (B, ref_args[3]),
            (C, ref_args[4]),
            (D_skip, _require_tensor(ref_args[5])),
            (z, _require_tensor(ref_args[6])),
        ],
        atol=grad_atol,
        rtol=grad_rtol,
    )


@pytest.mark.parametrize("device", _GRADCHECK_DEVICES)
def test_ssd_chunk_scan_gradcheck(device: torch.device) -> None:
    _require_backend(device)
    torch.manual_seed(5)

    batch, seqlen, heads, proj = 1, 3, 1, 2
    chunk_size = 2
    dtype = torch.float64

    X = torch.randn(
        batch, seqlen, heads, proj, device=device, dtype=dtype, requires_grad=True
    )
    dt = torch.randn(
        batch, seqlen, heads, device=device, dtype=dtype, requires_grad=True
    )
    A = torch.randn(heads, proj, device=device, dtype=dtype, requires_grad=True)
    B = torch.randn(heads, proj, device=device, dtype=dtype, requires_grad=True)
    C = torch.randn(heads, proj, device=device, dtype=dtype, requires_grad=True)

    def func(X, dt, A, B, C):
        return ops.ssd_chunk_scan(
            X,
            dt,
            A,
            B,
            C,
            chunk_size=chunk_size,
            D=None,
            z=None,
            seq_meta=None,
            initial_states=None,
        )

    assert torch.autograd.gradcheck(func, (X, dt, A, B, C), eps=1e-4, atol=1e-4)


@pytest.mark.parametrize(("device", "dtype"), _DEVICE_DTYPE_CASES)
def test_dw_causal_conv_backend_matches_reference(
    device: torch.device, dtype: torch.dtype
) -> None:
    _require_backend(device)
    torch.manual_seed(6)

    batch, channels, length = 2, 3, 5
    kernel_size = 3

    x = torch.randn(
        batch, channels, length, device=device, dtype=dtype, requires_grad=True
    )
    weight = torch.randn(
        channels, kernel_size, device=device, dtype=dtype, requires_grad=True
    )
    bias = torch.randn(channels, device=device, dtype=dtype, requires_grad=True)

    ref_args = (_clone_tensor(x), _clone_tensor(weight), _clone_tensor(bias))

    out = ops.dw_causal_conv(x, weight, bias=bias, activation="silu")
    ref_out = reference_ops.dw_causal_conv(
        ref_args[0], ref_args[1], bias=ref_args[2], activation="silu"
    )

    data_atol, data_rtol = _output_tolerances(dtype)
    torch.testing.assert_close(
        out, ref_out.to(out.dtype), atol=data_atol, rtol=data_rtol
    )

    out.sum().backward()
    ref_out.sum().backward()

    grad_atol, grad_rtol = _grad_tolerances(dtype)
    _assert_grads_close(
        [
            (x, ref_args[0]),
            (weight, ref_args[1]),
            (bias, ref_args[2]),
        ],
        atol=grad_atol,
        rtol=grad_rtol,
    )


@pytest.mark.parametrize("device", _GRADCHECK_DEVICES)
def test_dw_causal_conv_gradcheck(device: torch.device) -> None:
    _require_backend(device)
    torch.manual_seed(7)

    batch, channels, length = 1, 2, 4
    kernel_size = 3
    dtype = torch.float64

    x = torch.randn(
        batch, channels, length, device=device, dtype=dtype, requires_grad=True
    )
    weight = torch.randn(
        channels, kernel_size, device=device, dtype=dtype, requires_grad=True
    )
    bias = torch.randn(channels, device=device, dtype=dtype, requires_grad=True)

    def func(x, weight, bias):
        return ops.dw_causal_conv(x, weight, bias=bias, activation="identity")

    assert torch.autograd.gradcheck(func, (x, weight, bias), eps=1e-4, atol=1e-4)


@pytest.mark.parametrize(("device", "dtype"), _DEVICE_DTYPE_CASES)
@pytest.mark.parametrize("is_rms", [False, True])
@pytest.mark.parametrize("prenorm", [False, True])
def test_fused_layer_norm_backend_matches_reference(
    device: torch.device, dtype: torch.dtype, is_rms: bool, prenorm: bool
) -> None:
    _require_backend(device)
    torch.manual_seed(8)

    batch, length, hidden = 2, 3, 4

    x = torch.randn(
        batch, length, hidden, device=device, dtype=dtype, requires_grad=True
    )
    weight = torch.randn(hidden, device=device, dtype=dtype, requires_grad=True)
    bias = torch.randn(hidden, device=device, dtype=dtype, requires_grad=True)
    residual = torch.randn(
        batch, length, hidden, device=device, dtype=dtype, requires_grad=True
    )

    ref_args = (
        _clone_tensor(x),
        _clone_tensor(weight),
        _clone_tensor(bias),
        _clone_tensor(residual),
    )

    out = ops.fused_layer_norm(
        x,
        weight,
        bias,
        residual=residual,
        is_rms=is_rms,
        prenorm=prenorm,
    )
    ref_out = reference_ops.fused_layer_norm(
        ref_args[0],
        ref_args[1],
        ref_args[2],
        residual=ref_args[3],
        is_rms=is_rms,
        prenorm=prenorm,
    )

    data_atol, data_rtol = _output_tolerances(dtype)
    torch.testing.assert_close(
        out, ref_out.to(out.dtype), atol=data_atol, rtol=data_rtol
    )

    out.sum().backward()
    ref_out.sum().backward()

    grad_atol, grad_rtol = _grad_tolerances(dtype)
    _assert_grads_close(
        [
            (x, ref_args[0]),
            (weight, ref_args[1]),
            (bias, ref_args[2]),
            (residual, ref_args[3]),
        ],
        atol=grad_atol,
        rtol=grad_rtol,
    )


@pytest.mark.parametrize("device", _GRADCHECK_DEVICES)
@pytest.mark.parametrize("is_rms", [False, True])
@pytest.mark.parametrize("prenorm", [False, True])
def test_fused_layer_norm_gradcheck(
    device: torch.device, is_rms: bool, prenorm: bool
) -> None:
    _require_backend(device)
    torch.manual_seed(9)

    batch, length, hidden = 1, 2, 3
    dtype = torch.float64

    x = torch.randn(
        batch, length, hidden, device=device, dtype=dtype, requires_grad=True
    )
    weight = torch.randn(hidden, device=device, dtype=dtype, requires_grad=True)
    bias = torch.randn(hidden, device=device, dtype=dtype, requires_grad=True)
    residual = torch.randn(
        batch, length, hidden, device=device, dtype=dtype, requires_grad=True
    )

    def func(x, weight, bias, residual):
        return ops.fused_layer_norm(
            x,
            weight,
            bias,
            residual=residual,
            is_rms=is_rms,
            prenorm=prenorm,
        )

    assert torch.autograd.gradcheck(
        func, (x, weight, bias, residual), eps=1e-4, atol=1e-4
    )

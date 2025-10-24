"""Smoke tests ensuring the CPU backend matches the reference path."""

from __future__ import annotations

import itertools
from typing import cast

import pytest
import torch

import ssm.ops as ops
from ssm.ops.python import reference as reference_ops


def _require_cpu_backend() -> None:
    status = ops._cpu_backend_status()
    if not status.available:
        pytest.skip(f"CPU backend unavailable: {status}")


@pytest.mark.parametrize(
    "B_shape,C_shape",
    [
        ("(D,N)", "(D,N)"),
        ("(B,D,N)", "(B,D,N)"),
        ("(B,G,N,L)", "(B,G,N,L)"),
    ],
)
def test_selective_scan_cpu_agrees_with_reference(B_shape: str, C_shape: str) -> None:
    _require_cpu_backend()
    torch.manual_seed(0)
    batch, dim, state_dim, length = 2, 4, 3, 5
    groups = 2
    u = torch.randn(batch, dim, length)
    delta = torch.randn(batch, dim, length)
    A = torch.randn(dim, state_dim, dtype=torch.float32)
    dt_bias = torch.randn(dim)
    D = torch.randn(dim)
    z = torch.randn(batch, dim, length)

    B: torch.Tensor
    C: torch.Tensor
    if B_shape == "(D,N)":
        B = torch.randn(dim, state_dim)
    elif B_shape == "(B,D,N)":
        B = torch.randn(batch, dim, state_dim)
    else:
        B = torch.randn(batch, groups, state_dim, length)
    if C_shape == "(D,N)":
        C = torch.randn(dim, state_dim)
    elif C_shape == "(B,D,N)":
        C = torch.randn(batch, dim, state_dim)
    else:
        C = torch.randn(batch, groups, state_dim, length)

    ref_out, ref_state = reference_ops.selective_scan(
        u,
        delta,
        A,
        B,
        C,
        D=D,
        z=z,
        dt_bias=dt_bias,
        softplus=True,
        return_last_state=True,
    )
    cpu_out, cpu_state = ops.selective_scan(
        u,
        delta,
        A,
        B,
        C,
        D=D,
        z=z,
        dt_bias=dt_bias,
        softplus=True,
        return_last_state=True,
    )

    torch.testing.assert_close(cpu_out, ref_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(cpu_state, ref_state, atol=1e-5, rtol=1e-5)


def test_selective_scan_cpu_gradcheck() -> None:
    _require_cpu_backend()
    torch.manual_seed(1)
    batch, dim, state_dim, length = 1, 2, 2, 3
    groups = 2
    dtype = torch.double
    u = torch.randn(batch, dim, length, dtype=dtype, requires_grad=True)
    delta = torch.randn(batch, dim, length, dtype=dtype, requires_grad=True)
    A = torch.randn(dim, state_dim, dtype=dtype, requires_grad=True)
    B = torch.randn(batch, groups, state_dim, length, dtype=dtype, requires_grad=True)
    C = torch.randn(batch, groups, state_dim, length, dtype=dtype, requires_grad=True)

    def func(*params: torch.Tensor) -> torch.Tensor:
        uu, dd, AA, BB, CC = params
        out = cast(
            torch.Tensor,
            ops.selective_scan(
                uu,
                dd,
                AA,
                BB,
                CC,
                D=None,
                z=None,
                dt_bias=None,
                softplus=False,
                return_last_state=False,
            ),
        )
        return out.sum()

    assert torch.autograd.gradcheck(func, (u, delta, A, B, C), atol=1e-6, rtol=1e-6)


def test_selective_state_step_cpu_grouped_matches_reference() -> None:
    _require_cpu_backend()
    torch.manual_seed(2)
    batch, dim, state_dim = 2, 4, 3
    groups = 2
    state_ref = torch.randn(batch, dim, state_dim)
    state_cpu = state_ref.clone()
    x = torch.randn(batch, dim)
    dt = torch.randn(batch, dim)
    A = torch.randn(dim, state_dim, dtype=torch.float32)
    B = torch.randn(batch, groups, state_dim)
    C = torch.randn(batch, groups, state_dim)
    D = torch.randn(dim)
    z = torch.randn(batch, dim)

    ref_out = reference_ops.selective_state_step(
        state_ref,
        x,
        dt,
        A,
        B,
        C,
        D=D,
        z=z,
        dt_bias=None,
        softplus=True,
    )
    cpu_out = ops.selective_state_step(
        state_cpu,
        x,
        dt,
        A,
        B,
        C,
        D=D,
        z=z,
        dt_bias=None,
        softplus=True,
    )

    torch.testing.assert_close(cpu_out, ref_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(state_cpu, state_ref, atol=1e-5, rtol=1e-5)


def test_selective_state_step_cpu_gradcheck() -> None:
    _require_cpu_backend()
    torch.manual_seed(3)
    batch, dim, state_dim = 1, 3, 2
    dtype = torch.double
    state = torch.randn(batch, dim, state_dim, dtype=dtype, requires_grad=True)
    x = torch.randn(batch, dim, dtype=dtype, requires_grad=True)
    dt = torch.randn(batch, dim, dtype=dtype, requires_grad=True)
    A = torch.randn(dim, state_dim, dtype=dtype, requires_grad=True)
    B = torch.randn(dim, state_dim, dtype=dtype, requires_grad=True)
    C = torch.randn(dim, state_dim, dtype=dtype, requires_grad=True)

    def func(*params: torch.Tensor) -> torch.Tensor:
        st, xx, dd, AA, BB, CC = params
        st_clone = st.clone()
        out = ops.selective_state_step(
            st_clone,
            xx,
            dd,
            AA,
            BB,
            CC,
            D=None,
            z=None,
            dt_bias=None,
            softplus=False,
        )
        return out.sum() + st_clone.sum()

    assert torch.autograd.gradcheck(func, (state, x, dt, A, B, C), atol=1e-6, rtol=1e-6)


def test_ssd_chunk_scan_cpu_varlen_and_grouped() -> None:
    _require_cpu_backend()
    torch.manual_seed(4)
    batch, seqlen, heads, proj = 2, 7, 2, 3
    chunk_size = 4
    X = torch.randn(batch, seqlen, heads, proj)
    dt = torch.rand(batch, seqlen, heads)
    A = torch.randn(heads, proj, dtype=torch.float32)
    B = torch.randn(batch, seqlen, heads, proj)
    C = torch.randn(batch, heads, proj)
    D = torch.randn(heads, proj)
    z = torch.randn(batch, seqlen, heads, proj)
    cu_seqlens = torch.tensor([0, seqlen, seqlen + seqlen - 2])
    init_state = torch.randn(batch, heads, proj)

    ref_out = reference_ops.ssd_chunk_scan(
        X,
        dt,
        A,
        B,
        C,
        chunk_size,
        D=D,
        z=z,
        seq_meta={"cu_seqlens": cu_seqlens},
        initial_states=init_state,
    )
    cpu_out = ops.ssd_chunk_scan(
        X,
        dt,
        A,
        B,
        C,
        chunk_size,
        D=D,
        z=z,
        seq_meta={"cu_seqlens": cu_seqlens},
        initial_states=init_state,
    )

    torch.testing.assert_close(cpu_out, ref_out, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("channels_first", [True, False])
def test_dw_causal_conv_cpu_matches_reference(channels_first: bool) -> None:
    _require_cpu_backend()
    torch.manual_seed(5)
    batch, channels, length, kernel = 2, 3, 9, 4
    layout = (batch, channels, length) if channels_first else (batch, length, channels)
    x = torch.randn(layout)
    weight = torch.randn(channels, 1, kernel)
    bias = torch.randn(channels)

    ref_out = reference_ops.dw_causal_conv(
        x,
        weight,
        bias=bias,
        activation="identity",
    )
    cpu_out = ops.dw_causal_conv(
        x,
        weight,
        bias=bias,
        activation="identity",
    )

    torch.testing.assert_close(cpu_out, ref_out, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "is_rms,prenorm,residual_in_fp32",
    list(itertools.product([True, False], repeat=3)),
)
def test_fused_layer_norm_cpu_matches_reference(
    is_rms: bool, prenorm: bool, residual_in_fp32: bool
) -> None:
    _require_cpu_backend()
    torch.manual_seed(6)
    batch, seqlen, dim = 2, 4, 5
    x = torch.randn(batch, seqlen, dim)
    weight = torch.randn(dim)
    bias = torch.randn(dim)
    residual = torch.randn(batch, seqlen, dim)

    ref_out = reference_ops.fused_layer_norm(
        x,
        weight,
        bias,
        residual=residual,
        is_rms=is_rms,
        eps=1e-5,
        prenorm=prenorm,
        residual_in_fp32=residual_in_fp32,
    )
    cpu_out = ops.fused_layer_norm(
        x,
        weight,
        bias,
        residual=residual,
        is_rms=is_rms,
        eps=1e-5,
        prenorm=prenorm,
        residual_in_fp32=residual_in_fp32,
    )

    torch.testing.assert_close(cpu_out, ref_out, atol=1e-5, rtol=1e-5)


def test_fused_layer_norm_cpu_gradcheck() -> None:
    _require_cpu_backend()
    torch.manual_seed(7)
    batch, seqlen, dim = 1, 2, 3
    dtype = torch.double
    x = torch.randn(batch, seqlen, dim, dtype=dtype, requires_grad=True)
    weight = torch.randn(dim, dtype=dtype, requires_grad=True)
    bias = torch.randn(dim, dtype=dtype, requires_grad=True)

    def func(*params: torch.Tensor) -> torch.Tensor:
        xx, ww, bb = params
        out = ops.fused_layer_norm(
            xx,
            ww,
            bb,
            residual=None,
            is_rms=False,
            eps=1e-5,
            prenorm=True,
            residual_in_fp32=True,
        )
        return out.sum()

    assert torch.autograd.gradcheck(func, (x, weight, bias), atol=1e-6, rtol=1e-6)

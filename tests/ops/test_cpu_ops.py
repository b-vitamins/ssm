"""Smoke tests ensuring the CPU backend matches the reference path."""

from __future__ import annotations

import pytest
import torch

import ssm.ops as ops
from ssm.ops.python import reference as reference_ops


def _require_cpu_backend() -> None:
    status = ops._cpu_backend_status()
    if not status.available:
        pytest.skip(f"CPU backend unavailable: {status}")


def test_selective_scan_cpu_agrees_with_reference() -> None:
    _require_cpu_backend()
    torch.manual_seed(0)
    batch, dim, state_dim, length = 2, 3, 4, 5
    u = torch.randn(batch, dim, length)
    delta = torch.randn(batch, dim, length)
    A = torch.randn(dim, state_dim, dtype=torch.float32)
    B = torch.randn(dim, state_dim)
    C = torch.randn(dim, state_dim)
    D = torch.randn(dim)
    z = torch.randn(batch, dim, length)
    dt_bias = torch.randn(dim)

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


def test_selective_state_step_cpu_agrees_with_reference() -> None:
    _require_cpu_backend()
    torch.manual_seed(1)
    batch, dim, state_dim = 2, 4, 3
    state_ref = torch.randn(batch, dim, state_dim)
    state_cpu = state_ref.clone()
    x = torch.randn(batch, dim)
    dt = torch.randn(batch, dim)
    A = torch.randn(dim, state_dim, dtype=torch.float32)
    B = torch.randn(dim, state_dim)
    C = torch.randn(dim, state_dim)
    D = torch.randn(dim)
    z = torch.randn(batch, dim)
    dt_bias = torch.randn(dim)

    ref_out = reference_ops.selective_state_step(
        state_ref,
        x,
        dt,
        A,
        B,
        C,
        D=D,
        z=z,
        dt_bias=dt_bias,
        softplus=False,
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
        dt_bias=dt_bias,
        softplus=False,
    )

    torch.testing.assert_close(cpu_out, ref_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(state_cpu, state_ref, atol=1e-5, rtol=1e-5)


def test_ssd_chunk_scan_cpu_matches_reference() -> None:
    _require_cpu_backend()
    torch.manual_seed(2)
    batch, seqlen, heads, proj = 2, 6, 3, 2
    chunk_size = 3
    X = torch.randn(batch, seqlen, heads, proj)
    dt = torch.randn(batch, seqlen, heads)
    A = torch.randn(heads, proj, dtype=torch.float32)
    B = torch.randn(heads, proj)
    C = torch.randn(heads, proj)
    D = torch.randn(heads)
    z = torch.randn(batch, seqlen, heads)
    seq_lens = torch.tensor([seqlen, seqlen - 1])

    ref_out = reference_ops.ssd_chunk_scan(
        X,
        dt,
        A,
        B,
        C,
        chunk_size,
        D=D,
        z=z,
        seq_meta={"seq_lens": seq_lens},
        initial_states=None,
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
        seq_meta={"seq_lens": seq_lens},
        initial_states=None,
    )

    torch.testing.assert_close(cpu_out, ref_out, atol=1e-5, rtol=1e-5)


def test_dw_causal_conv_cpu_matches_reference() -> None:
    _require_cpu_backend()
    torch.manual_seed(3)
    batch, channels, length, kernel = 2, 4, 8, 3
    x = torch.randn(batch, channels, length)
    weight = torch.randn(channels, kernel)
    bias = torch.randn(channels)

    ref_out = reference_ops.dw_causal_conv(
        x,
        weight,
        bias=bias,
        activation="relu",
    )
    cpu_out = ops.dw_causal_conv(
        x,
        weight,
        bias=bias,
        activation="relu",
    )

    torch.testing.assert_close(cpu_out, ref_out, atol=1e-5, rtol=1e-5)


def test_fused_layer_norm_cpu_matches_reference() -> None:
    _require_cpu_backend()
    torch.manual_seed(4)
    batch, seqlen, dim = 2, 5, 6
    x = torch.randn(batch, seqlen, dim)
    weight = torch.randn(dim)
    bias = torch.randn(dim)
    residual = torch.randn(batch, seqlen, dim)

    ref_out = reference_ops.fused_layer_norm(
        x,
        weight,
        bias,
        residual=residual,
        is_rms=False,
        eps=1e-5,
        prenorm=False,
        residual_in_fp32=True,
    )
    cpu_out = ops.fused_layer_norm(
        x,
        weight,
        bias,
        residual=residual,
        is_rms=False,
        eps=1e-5,
        prenorm=False,
        residual_in_fp32=True,
    )

    torch.testing.assert_close(cpu_out, ref_out, atol=1e-5, rtol=1e-5)


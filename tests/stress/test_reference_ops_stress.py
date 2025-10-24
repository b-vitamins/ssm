from __future__ import annotations

from typing import cast

import torch

from ssm import ops


def _assert_close_half_precision(result: torch.Tensor, reference: torch.Tensor) -> None:
    torch.testing.assert_close(
        result.to(torch.float32),
        reference.to(torch.float32),
        atol=5e-3,
        rtol=5e-3,
    )


def test_selective_scan_long_sequence_fp32_accumulation() -> None:
    torch.manual_seed(0)
    batch, dim, state, length = 2, 6, 32, 512

    u_fp32 = torch.randn(batch, dim, length) * 1e-3
    delta_fp32 = torch.rand(batch, dim, length) * 1e-4
    A = torch.randn(dim, state) * 1e-3
    B = torch.randn(dim, state) * 1e-3
    C = torch.randn(dim, state) * 1e-3
    D = torch.randn(dim) * 1e-4

    reference = cast(
        torch.Tensor,
        ops.selective_scan(
            u=u_fp32,
            delta=delta_fp32,
            A=A,
            B=B,
            C=C,
            D=D,
            softplus=True,
        ),
    )

    u_bf16 = u_fp32.to(torch.bfloat16)
    delta_bf16 = delta_fp32.to(torch.bfloat16)

    result = cast(
        torch.Tensor,
        ops.selective_scan(
            u=u_bf16,
            delta=delta_bf16,
            A=A,
            B=B,
            C=C,
            D=D,
            softplus=True,
        ),
    )

    assert result.dtype == torch.bfloat16
    assert torch.isfinite(result).all()
    _assert_close_half_precision(result, reference)


def test_selective_scan_high_state_dimension() -> None:
    torch.manual_seed(1)
    batch, dim, state, length = 1, 4, 256, 64

    u_fp32 = torch.randn(batch, dim, length) * 1e-3
    delta_fp32 = torch.rand(batch, dim, length) * 1e-4
    A = torch.randn(dim, state) * 1e-3
    B = torch.randn(dim, state) * 1e-3
    C = torch.randn(dim, state) * 1e-3

    reference = cast(
        torch.Tensor,
        ops.selective_scan(
            u=u_fp32,
            delta=delta_fp32,
            A=A,
            B=B,
            C=C,
            softplus=False,
        ),
    )

    u_bf16 = u_fp32.to(torch.bfloat16)
    delta_bf16 = delta_fp32.to(torch.bfloat16)

    result = cast(
        torch.Tensor,
        ops.selective_scan(
            u=u_bf16,
            delta=delta_bf16,
            A=A,
            B=B,
            C=C,
            softplus=False,
        ),
    )

    assert torch.isfinite(result).all()
    _assert_close_half_precision(result, reference)


def test_selective_scan_multigroup_parameters() -> None:
    torch.manual_seed(42)
    batch, dim, state, length, groups = 2, 12, 8, 16, 3
    if dim % groups != 0:
        raise AssertionError("dim must be divisible by groups for the test")

    u = (torch.randn(batch, dim, length) * 1e-3).to(torch.bfloat16)
    delta = (torch.rand(batch, dim, length) * 1e-4).to(torch.bfloat16)
    A = torch.randn(dim, state) * 1e-3
    B_grouped = torch.randn(batch, groups, state, length) * 1e-3
    C_grouped = torch.randn(batch, groups, state, length) * 1e-3

    result = cast(
        torch.Tensor,
        ops.selective_scan(
            u=u,
            delta=delta,
            A=A,
            B=B_grouped,
            C=C_grouped,
            softplus=False,
        ),
    )

    B_expanded = B_grouped.repeat_interleave(dim // groups, dim=1)
    C_expanded = C_grouped.repeat_interleave(dim // groups, dim=1)
    reference = cast(
        torch.Tensor,
        ops.selective_scan(
            u=u,
            delta=delta,
            A=A,
            B=B_expanded,
            C=C_expanded,
            softplus=False,
        ),
    )

    _assert_close_half_precision(result, reference)

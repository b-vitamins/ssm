from __future__ import annotations

from typing import Callable

import pytest
import torch

import ssm.ops as ops
from ssm.ops.python import reference as reference_ops


def _require_cpu_backend() -> None:
    status = ops._cpu_backend_status()
    if not status.available:
        pytest.skip(f"CPU backend unavailable: {status}")


def _run_impl(
    impl: str,
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    D: torch.Tensor | None,
    z: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    softplus: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    runner: Callable[..., torch.Tensor]
    if impl == "cpu":
        _require_cpu_backend()
        runner = ops.selective_state_step
    elif impl == "reference":
        runner = reference_ops.selective_state_step
    else:  # pragma: no cover - defensive guard for parametrization
        raise AssertionError(f"Unknown implementation: {impl}")

    state_copy = state.clone()
    output = runner(
        state_copy,
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
    return output, state_copy


@pytest.mark.parametrize("impl", ["reference", "cpu"])
@pytest.mark.parametrize("use_grouped", [False, True])
@pytest.mark.parametrize("with_optionals", [False, True])
@pytest.mark.parametrize("use_bias", [False, True])
def test_selective_state_step_variants(
    impl: str,
    use_grouped: bool,
    with_optionals: bool,
    use_bias: bool,
) -> None:
    torch.manual_seed(5)
    batch, dim, state_dim = 2, 4, 8
    groups = 2

    state_ref = torch.randn(batch, dim, state_dim)
    x = torch.randn(batch, dim)
    dt = torch.randn(batch, dim)
    A = torch.randn(dim, state_dim, dtype=torch.float32)

    if use_grouped:
        B = torch.randn(batch, groups, state_dim)
        C = torch.randn(batch, groups, state_dim)
    else:
        B = torch.randn(dim, state_dim)
        C = torch.randn(dim, state_dim)

    D = torch.randn(dim) if with_optionals else None
    z = torch.randn(batch, dim) if with_optionals else None
    dt_bias = torch.randn(dim) if use_bias else None
    softplus = use_bias

    ref_out, ref_state = _run_impl(
        "reference",
        state_ref,
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

    test_out, test_state = _run_impl(
        impl,
        state_ref,
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

    torch.testing.assert_close(test_out, ref_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(test_state, ref_state, atol=1e-5, rtol=1e-5)

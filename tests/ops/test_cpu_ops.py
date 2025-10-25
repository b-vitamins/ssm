"""Smoke tests ensuring the CPU backend matches the reference path."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import TypedDict, cast

import pytest
import torch

import ssm.ops as ops
from ssm.ops.python import reference as reference_ops


_SELECTIVE_SCAN_GOLDEN = (
    Path(__file__).resolve().parents[1]
    / "goldens"
    / "ops"
    / "selective_scan_reference.json"
)


class _SelectiveScanInputs(TypedDict):
    u: torch.Tensor
    delta: torch.Tensor
    A: torch.Tensor
    B: torch.Tensor
    C: torch.Tensor
    D: torch.Tensor | None
    z: torch.Tensor | None
    dt_bias: torch.Tensor | None


class _SelectiveScanOutputs(TypedDict):
    y: torch.Tensor
    state: torch.Tensor


class _SelectiveScanCase(TypedDict):
    description: str
    inputs: _SelectiveScanInputs
    outputs: _SelectiveScanOutputs


def _require_cpu_backend() -> None:
    status = ops._cpu_backend_status()
    if not status.available:
        pytest.skip(f"CPU backend unavailable: {status}")


def _tensor_from_spec(spec: dict[str, object]) -> torch.Tensor:
    """Reconstruct a tensor from the serialized JSON specification."""

    dtype_name = cast(str, spec["dtype"])
    dtype = getattr(torch, dtype_name)
    data = torch.tensor(spec["data"], dtype=dtype)
    shape = tuple(cast(list[int], spec["shape"]))
    return data.view(shape)


def _maybe_tensor(value: object) -> object:
    if isinstance(value, dict) and {"dtype", "shape", "data"} <= value.keys():
        return _tensor_from_spec(cast(dict[str, object], value))
    return value


def _selective_scan_golden_cases() -> list[tuple[str, _SelectiveScanCase]]:
    if not _SELECTIVE_SCAN_GOLDEN.exists():
        return []
    with _SELECTIVE_SCAN_GOLDEN.open("r", encoding="utf-8") as handle:
        encoded_cases = json.load(handle)
    cases: list[tuple[str, _SelectiveScanCase]] = []
    for entry in encoded_cases:
        decoded = {
            "description": entry["description"],
            "inputs": {
                key: _maybe_tensor(value) for key, value in entry["inputs"].items()
            },
            "outputs": {
                key: _maybe_tensor(value) for key, value in entry["outputs"].items()
            },
        }
        case = cast(_SelectiveScanCase, decoded)
        cases.append((case["description"], case))
    return cases


def test_dw_causal_conv_cpu_matches_reference() -> None:
    _require_cpu_backend()
    torch.manual_seed(0)

    batch, channels, length = 2, 3, 11
    kernel_sizes = [1, 3, 5]
    activations = ["identity", "none", "silu", "swish"]
    dtypes = [torch.float32, torch.float64, torch.bfloat16]

    for dtype in dtypes:
        atol = 1e-5
        rtol = 1e-5
        if dtype == torch.bfloat16:
            atol = 2e-3
            rtol = 2e-3
        for channels_first in (True, False):
            base_shape = (batch, channels, length)
            x = torch.randn(base_shape, dtype=dtype)
            if not channels_first:
                x = x.permute(0, 2, 1).contiguous()
            for kernel_size in kernel_sizes:
                weight = torch.randn(channels, kernel_size, dtype=dtype)
                bias = torch.randn(channels, dtype=dtype)
                for activation in activations:
                    for use_bias in (False, True):
                        bias_arg = bias if use_bias else None
                        for weight_layout in ("ck", "c1k"):
                            weight_arg = (
                                weight if weight_layout == "ck" else weight.unsqueeze(1)
                            )
                            out_cpu = ops.dw_causal_conv(
                                x,
                                weight_arg,
                                bias_arg,
                                activation=activation,
                            )
                            out_ref = reference_ops.dw_causal_conv(
                                x,
                                weight_arg,
                                bias=bias_arg,
                                activation=activation,
                            )
                            assert out_cpu.shape == x.shape
                            assert out_cpu.dtype == x.dtype
                            torch.testing.assert_close(
                                out_cpu, out_ref, atol=atol, rtol=rtol
                            )


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


@pytest.mark.parametrize(
    "case", _selective_scan_golden_cases(), ids=lambda entry: entry[0]
)
def test_selective_scan_cpu_golden(case: tuple[str, _SelectiveScanCase]) -> None:
    name, payload = case
    if not name:
        pytest.skip("No golden cases available")
    _require_cpu_backend()
    inputs = payload["inputs"]
    outputs = payload["outputs"]

    def _maybe_clone(value: object) -> object:
        if isinstance(value, torch.Tensor):
            return value.clone()
        return value

    args = cast(
        _SelectiveScanInputs,
        {key: _maybe_clone(value) for key, value in inputs.items()},
    )
    golden_y = outputs["y"].clone()
    golden_state = outputs["state"].clone()

    cpu_out, cpu_state = ops.selective_scan(
        args["u"],
        args["delta"],
        args["A"],
        args["B"],
        args["C"],
        D=args["D"],
        z=args["z"],
        dt_bias=args["dt_bias"],
        softplus=True,
        return_last_state=True,
    )
    ref_out, ref_state = reference_ops.selective_scan(
        args["u"],
        args["delta"],
        args["A"],
        args["B"],
        args["C"],
        D=args["D"],
        z=args["z"],
        dt_bias=args["dt_bias"],
        softplus=True,
        return_last_state=True,
    )

    torch.testing.assert_close(cpu_out, golden_y, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(cpu_state, golden_state, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(ref_out, golden_y, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(ref_state, golden_state, atol=1e-6, rtol=1e-6)


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


@pytest.mark.parametrize(
    "is_rms,prenorm,residual_in_fp32,use_residual",
    list(itertools.product([True, False], repeat=4)),
)
def test_fused_layer_norm_cpu_matches_reference(
    is_rms: bool, prenorm: bool, residual_in_fp32: bool, use_residual: bool
) -> None:
    _require_cpu_backend()
    torch.manual_seed(6)
    batch, seqlen, dim = 2, 4, 5
    x = torch.randn(batch, seqlen, dim)
    weight = torch.randn(dim)
    bias = torch.randn(dim)
    residual = torch.randn(batch, seqlen, dim) if use_residual else None

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

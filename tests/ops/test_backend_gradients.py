"""Gradient parity tests for compiled ops against the Python reference."""

from __future__ import annotations

import importlib
from typing import Callable, Iterable, TypedDict, cast

import pytest
import torch

import ssm.ops as ops
from ssm.ops.python import reference as reference_ops


_UPSTREAM_MODULE_CANDIDATES = (
    "state_spaces.mamba.ops",
    "state_spaces.ops",
    "mamba_ssm.ops",
)

_UPSTREAM_CACHE: dict[str, Callable | None] = {}


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


def _load_upstream_op(name: str) -> Callable | None:
    if name in _UPSTREAM_CACHE:
        return _UPSTREAM_CACHE[name]
    for module_name in _UPSTREAM_MODULE_CANDIDATES:
        try:
            module = importlib.import_module(module_name)
        except Exception:  # pragma: no cover - optional dependency
            continue
        candidate = getattr(module, name, None)
        if callable(candidate):
            _UPSTREAM_CACHE[name] = candidate
            return candidate
    _UPSTREAM_CACHE[name] = None
    return None


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


def _unpack_maybe_tuple(
    value: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    expects_tuple: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(value, tuple):
        if len(value) != 2:  # pragma: no cover - defensive
            raise AssertionError("Expected 2-tuple from backend")
        return value[0], value[1]
    if expects_tuple:
        raise AssertionError("Backend did not return the expected tuple")
    return value, None


def _sum_outputs(output: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
    total = output.sum()
    if state is not None:
        total = total + state.sum()
    return total


def _clone_seq_meta(
    meta: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor] | None:
    if meta is None:
        return None
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in meta.items()
    }


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


class _SelectiveScanCase(TypedDict):
    name: str
    return_last_state: bool
    softplus: bool
    with_D: bool
    with_z: bool
    with_dt_bias: bool


class _SelectiveStateStepCase(TypedDict):
    name: str
    softplus: bool
    with_D: bool
    with_z: bool
    with_dt_bias: bool


class _SsdChunkScanCase(TypedDict):
    name: str
    with_D: bool
    with_z: bool
    with_seq_meta: bool
    with_initial_state: bool


class _DwConvCase(TypedDict):
    name: str
    with_bias: bool
    activation: str


class _FusedLayerNormCase(TypedDict):
    name: str
    with_bias: bool
    with_residual: bool
    is_rms: bool
    prenorm: bool
    residual_in_fp32: bool


_SELECTIVE_SCAN_CPU_CASES = [
    pytest.param(
        dict(
            name="full",
            return_last_state=True,
            softplus=True,
            with_D=True,
            with_z=True,
            with_dt_bias=True,
        ),
        id="full",
    ),
    pytest.param(
        dict(
            name="minimal",
            return_last_state=False,
            softplus=False,
            with_D=False,
            with_z=False,
            with_dt_bias=False,
        ),
        id="minimal",
    ),
    pytest.param(
        dict(
            name="partial",
            return_last_state=True,
            softplus=False,
            with_D=True,
            with_z=False,
            with_dt_bias=True,
        ),
        id="partial",
    ),
    pytest.param(
        dict(
            name="gate_only",
            return_last_state=False,
            softplus=True,
            with_D=False,
            with_z=True,
            with_dt_bias=False,
        ),
        id="gate-only",
    ),
]


_SELECTIVE_STATE_STEP_CPU_CASES = [
    pytest.param(
        dict(
            name="full",
            softplus=True,
            with_D=True,
            with_z=True,
            with_dt_bias=True,
        ),
        id="full",
    ),
    pytest.param(
        dict(
            name="minimal",
            softplus=False,
            with_D=False,
            with_z=False,
            with_dt_bias=False,
        ),
        id="minimal",
    ),
    pytest.param(
        dict(
            name="skip_z",
            softplus=True,
            with_D=True,
            with_z=False,
            with_dt_bias=True,
        ),
        id="skip-z",
    ),
]


_SSD_CHUNK_SCAN_CPU_CASES = [
    pytest.param(
        dict(
            name="full",
            with_D=True,
            with_z=True,
            with_seq_meta=True,
            with_initial_state=True,
        ),
        id="full",
    ),
    pytest.param(
        dict(
            name="minimal",
            with_D=False,
            with_z=False,
            with_seq_meta=False,
            with_initial_state=False,
        ),
        id="minimal",
    ),
    pytest.param(
        dict(
            name="ragged",
            with_D=False,
            with_z=True,
            with_seq_meta=True,
            with_initial_state=False,
        ),
        id="ragged",
    ),
]


_DW_CONV_CPU_CASES = [
    pytest.param(
        dict(name="identity_bias", with_bias=True, activation="identity"),
        id="identity-bias",
    ),
    pytest.param(
        dict(name="silu_no_bias", with_bias=False, activation="silu"), id="silu-no-bias"
    ),
]


_FUSED_LAYER_NORM_CPU_CASES = [
    pytest.param(
        dict(
            name="prenorm_full",
            with_bias=True,
            with_residual=True,
            is_rms=False,
            prenorm=True,
            residual_in_fp32=True,
        ),
        id="prenorm-full",
    ),
    pytest.param(
        dict(
            name="postnorm_rms",
            with_bias=False,
            with_residual=True,
            is_rms=True,
            prenorm=False,
            residual_in_fp32=False,
        ),
        id="postnorm-rms",
    ),
    pytest.param(
        dict(
            name="no_residual",
            with_bias=True,
            with_residual=False,
            is_rms=False,
            prenorm=False,
            residual_in_fp32=True,
        ),
        id="no-residual",
    ),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("case", _SELECTIVE_SCAN_CPU_CASES)
def test_selective_scan_cpu_compiled_matches_all_backends(
    dtype: torch.dtype, case: dict[str, object]
) -> None:
    status = ops._cpu_backend_status()
    if not status.available:
        pytest.skip(f"CPU backend unavailable: {status}")

    device = torch.device("cpu")
    torch.manual_seed(101)

    batch, dim, state_dim, length = 2, 3, 4, 5

    case_data = cast(_SelectiveScanCase, case)
    return_last_state = case_data["return_last_state"]
    softplus = case_data["softplus"]

    u_base = torch.randn(
        batch, dim, length, device=device, dtype=dtype, requires_grad=True
    )
    delta_base = torch.randn_like(u_base, requires_grad=True)
    A_base = torch.randn(
        dim, state_dim, device=device, dtype=torch.float32, requires_grad=True
    )
    B_base = torch.randn(dim, state_dim, device=device, dtype=dtype, requires_grad=True)
    C_base = torch.randn(dim, state_dim, device=device, dtype=dtype, requires_grad=True)
    D_base = (
        torch.randn(dim, device=device, dtype=dtype, requires_grad=True)
        if case_data["with_D"]
        else None
    )
    z_base = (
        torch.randn(batch, dim, length, device=device, dtype=dtype, requires_grad=True)
        if case_data["with_z"]
        else None
    )
    dt_bias_base = (
        torch.randn(dim, device=device, dtype=dtype, requires_grad=True)
        if case_data["with_dt_bias"]
        else None
    )

    args_base = (u_base, delta_base, A_base, B_base, C_base)

    compiled_args = tuple(_clone_tensor(arg) for arg in args_base)
    compiled_D = _clone_optional_tensor(D_base)
    compiled_z = _clone_optional_tensor(z_base)
    compiled_dt_bias = _clone_optional_tensor(dt_bias_base)
    compiled_result = ops.selective_scan(
        *compiled_args,
        D=compiled_D,
        z=compiled_z,
        dt_bias=compiled_dt_bias,
        softplus=softplus,
        return_last_state=return_last_state,
    )
    out_compiled, state_compiled = _unpack_maybe_tuple(
        compiled_result, return_last_state
    )

    ref_args = tuple(_clone_tensor(arg) for arg in args_base)
    ref_D = _clone_optional_tensor(D_base)
    ref_z = _clone_optional_tensor(z_base)
    ref_dt_bias = _clone_optional_tensor(dt_bias_base)
    ref_result = reference_ops.selective_scan(
        *ref_args,
        D=ref_D,
        z=ref_z,
        dt_bias=ref_dt_bias,
        softplus=softplus,
        return_last_state=return_last_state,
    )
    out_ref, state_ref = _unpack_maybe_tuple(ref_result, return_last_state)

    data_atol, data_rtol = _output_tolerances(dtype)
    torch.testing.assert_close(
        out_compiled, out_ref.to(out_compiled.dtype), atol=data_atol, rtol=data_rtol
    )
    if state_compiled is not None and state_ref is not None:
        torch.testing.assert_close(
            state_compiled,
            state_ref.to(state_compiled.dtype),
            atol=data_atol,
            rtol=data_rtol,
        )

    _sum_outputs(out_compiled, state_compiled).backward()
    _sum_outputs(out_ref, state_ref).backward()

    grad_atol, grad_rtol = _grad_tolerances(dtype)
    grad_pairs: list[tuple[torch.Tensor, torch.Tensor]] = [
        (compiled_args[0], ref_args[0]),
        (compiled_args[1], ref_args[1]),
        (compiled_args[2], ref_args[2]),
        (compiled_args[3], ref_args[3]),
        (compiled_args[4], ref_args[4]),
    ]
    if isinstance(compiled_D, torch.Tensor):
        grad_pairs.append((compiled_D, _require_tensor(ref_D)))
    if isinstance(compiled_z, torch.Tensor):
        grad_pairs.append((compiled_z, _require_tensor(ref_z)))
    if isinstance(compiled_dt_bias, torch.Tensor):
        grad_pairs.append((compiled_dt_bias, _require_tensor(ref_dt_bias)))
    _assert_grads_close(grad_pairs, atol=grad_atol, rtol=grad_rtol)

    upstream_fn = _load_upstream_op("selective_scan")
    if upstream_fn is not None:
        upstream_args = tuple(_clone_tensor(arg) for arg in args_base)
        upstream_D = _clone_optional_tensor(D_base)
        upstream_z = _clone_optional_tensor(z_base)
        upstream_dt_bias = _clone_optional_tensor(dt_bias_base)
        try:
            upstream_result = upstream_fn(
                *upstream_args,
                D=upstream_D,
                z=upstream_z,
                dt_bias=upstream_dt_bias,
                softplus=softplus,
                return_last_state=return_last_state,
            )
        except Exception:  # pragma: no cover - optional dependency differences
            upstream_fn = None
        else:
            out_upstream, state_upstream = _unpack_maybe_tuple(
                upstream_result, return_last_state
            )
            torch.testing.assert_close(
                out_compiled,
                out_upstream.to(out_compiled.dtype),
                atol=data_atol,
                rtol=data_rtol,
            )
            if state_compiled is not None and state_upstream is not None:
                torch.testing.assert_close(
                    state_compiled,
                    state_upstream.to(state_compiled.dtype),
                    atol=data_atol,
                    rtol=data_rtol,
                )

            _sum_outputs(out_upstream, state_upstream).backward()

            upstream_pairs: list[tuple[torch.Tensor, torch.Tensor]] = [
                (compiled_args[0], upstream_args[0]),
                (compiled_args[1], upstream_args[1]),
                (compiled_args[2], upstream_args[2]),
                (compiled_args[3], upstream_args[3]),
                (compiled_args[4], upstream_args[4]),
            ]
            if isinstance(compiled_D, torch.Tensor) and isinstance(
                upstream_D, torch.Tensor
            ):
                upstream_pairs.append((compiled_D, _require_tensor(upstream_D)))
            if isinstance(compiled_z, torch.Tensor) and isinstance(
                upstream_z, torch.Tensor
            ):
                upstream_pairs.append((compiled_z, _require_tensor(upstream_z)))
            if isinstance(compiled_dt_bias, torch.Tensor) and isinstance(
                upstream_dt_bias, torch.Tensor
            ):
                upstream_pairs.append(
                    (compiled_dt_bias, _require_tensor(upstream_dt_bias))
                )
            _assert_grads_close(upstream_pairs, atol=grad_atol, rtol=grad_rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("case", _SSD_CHUNK_SCAN_CPU_CASES)
def test_ssd_chunk_scan_cpu_compiled_matches_all_backends(
    dtype: torch.dtype, case: dict[str, object]
) -> None:
    status = ops._cpu_backend_status()
    if not status.available:
        pytest.skip(f"CPU backend unavailable: {status}")

    device = torch.device("cpu")
    torch.manual_seed(103)

    batch, seqlen, heads, proj = 2, 4, 3, 2
    chunk_size = 2

    case_data = cast(_SsdChunkScanCase, case)

    X_base = torch.randn(
        batch, seqlen, heads, proj, device=device, dtype=dtype, requires_grad=True
    )
    dt_base = torch.randn(
        batch, seqlen, heads, device=device, dtype=dtype, requires_grad=True
    )
    A_base = torch.randn(
        heads, proj, device=device, dtype=torch.float32, requires_grad=True
    )
    B_base = torch.randn(heads, proj, device=device, dtype=dtype, requires_grad=True)
    C_base = torch.randn(heads, proj, device=device, dtype=dtype, requires_grad=True)
    D_base = (
        torch.randn(heads, device=device, dtype=dtype, requires_grad=True)
        if case_data["with_D"]
        else None
    )
    if case_data["with_z"]:
        if case_data["name"] == "ragged":
            z_shape = (batch, seqlen, heads, proj)
        else:
            z_shape = (batch, seqlen, heads)
        z_base = torch.randn(*z_shape, device=device, dtype=dtype, requires_grad=True)
    else:
        z_base = None
    if case_data["with_seq_meta"]:
        seq_meta_base: dict[str, torch.Tensor] | None = {
            "seq_lens": torch.tensor(
                [seqlen, seqlen - 1], device=device, dtype=torch.int64
            )
        }
    else:
        seq_meta_base = None
    initial_state_base = (
        torch.randn(batch, heads, proj, device=device, dtype=dtype)
        if case_data["with_initial_state"]
        else None
    )

    args_base = (X_base, dt_base, A_base, B_base, C_base)

    compiled_args = tuple(_clone_tensor(arg) for arg in args_base)
    compiled_D = _clone_optional_tensor(D_base)
    compiled_z = _clone_optional_tensor(z_base)
    compiled_seq_meta = _clone_seq_meta(seq_meta_base)
    compiled_initial_state = _clone_optional_tensor(initial_state_base)
    out_compiled = ops.ssd_chunk_scan(
        *compiled_args,
        chunk_size=chunk_size,
        D=compiled_D,
        z=compiled_z,
        seq_meta=compiled_seq_meta,
        initial_states=compiled_initial_state,
    )

    ref_args = tuple(_clone_tensor(arg) for arg in args_base)
    ref_D = _clone_optional_tensor(D_base)
    ref_z = _clone_optional_tensor(z_base)
    ref_seq_meta = _clone_seq_meta(seq_meta_base)
    ref_initial_state = _clone_optional_tensor(initial_state_base)
    out_ref = reference_ops.ssd_chunk_scan(
        *ref_args,
        chunk_size=chunk_size,
        D=ref_D,
        z=ref_z,
        seq_meta=ref_seq_meta,
        initial_states=ref_initial_state,
    )

    data_atol, data_rtol = _output_tolerances(dtype)
    torch.testing.assert_close(
        out_compiled, out_ref.to(out_compiled.dtype), atol=data_atol, rtol=data_rtol
    )

    out_compiled.sum().backward()
    out_ref.sum().backward()

    grad_atol, grad_rtol = _grad_tolerances(dtype)
    grad_pairs: list[tuple[torch.Tensor, torch.Tensor]] = [
        (compiled_args[0], ref_args[0]),
        (compiled_args[1], ref_args[1]),
        (compiled_args[2], ref_args[2]),
        (compiled_args[3], ref_args[3]),
        (compiled_args[4], ref_args[4]),
    ]
    if isinstance(compiled_D, torch.Tensor):
        grad_pairs.append((compiled_D, _require_tensor(ref_D)))
    if isinstance(compiled_z, torch.Tensor):
        grad_pairs.append((compiled_z, _require_tensor(ref_z)))
    _assert_grads_close(grad_pairs, atol=grad_atol, rtol=grad_rtol)

    upstream_fn = _load_upstream_op("ssd_chunk_scan")
    if upstream_fn is not None:
        upstream_args = tuple(_clone_tensor(arg) for arg in args_base)
        upstream_D = _clone_optional_tensor(D_base)
        upstream_z = _clone_optional_tensor(z_base)
        upstream_seq_meta = _clone_seq_meta(seq_meta_base)
        upstream_initial_state = _clone_optional_tensor(initial_state_base)
        try:
            out_upstream = upstream_fn(
                *upstream_args,
                chunk_size=chunk_size,
                D=upstream_D,
                z=upstream_z,
                seq_meta=upstream_seq_meta,
                initial_states=upstream_initial_state,
            )
        except Exception:  # pragma: no cover - optional dependency differences
            upstream_fn = None
        else:
            torch.testing.assert_close(
                out_compiled,
                out_upstream.to(out_compiled.dtype),
                atol=data_atol,
                rtol=data_rtol,
            )

            out_upstream.sum().backward()

            upstream_pairs: list[tuple[torch.Tensor, torch.Tensor]] = [
                (compiled_args[0], upstream_args[0]),
                (compiled_args[1], upstream_args[1]),
                (compiled_args[2], upstream_args[2]),
                (compiled_args[3], upstream_args[3]),
                (compiled_args[4], upstream_args[4]),
            ]
            if isinstance(compiled_D, torch.Tensor) and isinstance(
                upstream_D, torch.Tensor
            ):
                upstream_pairs.append((compiled_D, _require_tensor(upstream_D)))
            if isinstance(compiled_z, torch.Tensor) and isinstance(
                upstream_z, torch.Tensor
            ):
                upstream_pairs.append((compiled_z, _require_tensor(upstream_z)))
            _assert_grads_close(upstream_pairs, atol=grad_atol, rtol=grad_rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("case", _SELECTIVE_STATE_STEP_CPU_CASES)
def test_selective_state_step_cpu_compiled_matches_all_backends(
    dtype: torch.dtype, case: dict[str, object]
) -> None:
    status = ops._cpu_backend_status()
    if not status.available:
        pytest.skip(f"CPU backend unavailable: {status}")

    device = torch.device("cpu")
    torch.manual_seed(102)

    batch, dim, state_dim = 2, 3, 4

    case_data = cast(_SelectiveStateStepCase, case)
    softplus = case_data["softplus"]

    base_state = torch.randn(batch, dim, state_dim, device=device, dtype=dtype)
    x_base = torch.randn(batch, dim, device=device, dtype=dtype, requires_grad=True)
    dt_base = torch.randn_like(x_base, requires_grad=True)
    A_base = torch.randn(
        dim, state_dim, device=device, dtype=torch.float32, requires_grad=True
    )
    B_base = torch.randn(dim, state_dim, device=device, dtype=dtype, requires_grad=True)
    C_base = torch.randn(dim, state_dim, device=device, dtype=dtype, requires_grad=True)
    D_base = (
        torch.randn(dim, device=device, dtype=dtype, requires_grad=True)
        if case_data["with_D"]
        else None
    )
    z_base = (
        torch.randn(batch, dim, device=device, dtype=dtype, requires_grad=True)
        if case_data["with_z"]
        else None
    )
    dt_bias_base = (
        torch.randn(dim, device=device, dtype=dtype, requires_grad=True)
        if case_data["with_dt_bias"]
        else None
    )

    state_compiled = base_state.clone()
    compiled_args = (
        state_compiled,
        _clone_tensor(x_base),
        _clone_tensor(dt_base),
        _clone_tensor(A_base),
        _clone_tensor(B_base),
        _clone_tensor(C_base),
    )
    compiled_D = _clone_optional_tensor(D_base)
    compiled_z = _clone_optional_tensor(z_base)
    compiled_dt_bias = _clone_optional_tensor(dt_bias_base)
    out_compiled = ops.selective_state_step(
        *compiled_args,
        D=compiled_D,
        z=compiled_z,
        dt_bias=compiled_dt_bias,
        softplus=softplus,
    )

    state_ref = base_state.clone()
    ref_args = (
        state_ref,
        _clone_tensor(x_base),
        _clone_tensor(dt_base),
        _clone_tensor(A_base),
        _clone_tensor(B_base),
        _clone_tensor(C_base),
    )
    ref_D = _clone_optional_tensor(D_base)
    ref_z = _clone_optional_tensor(z_base)
    ref_dt_bias = _clone_optional_tensor(dt_bias_base)
    out_ref = reference_ops.selective_state_step(
        *ref_args,
        D=ref_D,
        z=ref_z,
        dt_bias=ref_dt_bias,
        softplus=softplus,
    )

    data_atol, data_rtol = _output_tolerances(dtype)
    torch.testing.assert_close(
        out_compiled, out_ref.to(out_compiled.dtype), atol=data_atol, rtol=data_rtol
    )
    torch.testing.assert_close(
        state_compiled,
        state_ref.to(state_compiled.dtype),
        atol=data_atol,
        rtol=data_rtol,
    )

    out_compiled.sum().backward()
    out_ref.sum().backward()

    grad_atol, grad_rtol = _grad_tolerances(dtype)
    grad_pairs: list[tuple[torch.Tensor, torch.Tensor]] = [
        (compiled_args[1], ref_args[1]),
        (compiled_args[2], ref_args[2]),
        (compiled_args[3], ref_args[3]),
        (compiled_args[4], ref_args[4]),
        (compiled_args[5], ref_args[5]),
    ]
    if isinstance(compiled_D, torch.Tensor):
        grad_pairs.append((compiled_D, _require_tensor(ref_D)))
    if isinstance(compiled_z, torch.Tensor):
        grad_pairs.append((compiled_z, _require_tensor(ref_z)))
    if isinstance(compiled_dt_bias, torch.Tensor):
        grad_pairs.append((compiled_dt_bias, _require_tensor(ref_dt_bias)))
    _assert_grads_close(grad_pairs, atol=grad_atol, rtol=grad_rtol)

    upstream_fn = _load_upstream_op("selective_state_step")
    if upstream_fn is not None:
        state_upstream = base_state.clone()
        upstream_args = (
            state_upstream,
            _clone_tensor(x_base),
            _clone_tensor(dt_base),
            _clone_tensor(A_base),
            _clone_tensor(B_base),
            _clone_tensor(C_base),
        )
        upstream_D = _clone_optional_tensor(D_base)
        upstream_z = _clone_optional_tensor(z_base)
        upstream_dt_bias = _clone_optional_tensor(dt_bias_base)
        try:
            out_upstream = upstream_fn(
                *upstream_args,
                D=upstream_D,
                z=upstream_z,
                dt_bias=upstream_dt_bias,
                softplus=softplus,
            )
        except Exception:  # pragma: no cover - optional dependency differences
            upstream_fn = None
        else:
            torch.testing.assert_close(
                out_compiled,
                out_upstream.to(out_compiled.dtype),
                atol=data_atol,
                rtol=data_rtol,
            )
            torch.testing.assert_close(
                state_compiled,
                state_upstream.to(state_compiled.dtype),
                atol=data_atol,
                rtol=data_rtol,
            )

            out_upstream.sum().backward()

            upstream_pairs: list[tuple[torch.Tensor, torch.Tensor]] = [
                (compiled_args[1], upstream_args[1]),
                (compiled_args[2], upstream_args[2]),
                (compiled_args[3], upstream_args[3]),
                (compiled_args[4], upstream_args[4]),
                (compiled_args[5], upstream_args[5]),
            ]
            if isinstance(compiled_D, torch.Tensor) and isinstance(
                upstream_D, torch.Tensor
            ):
                upstream_pairs.append((compiled_D, _require_tensor(upstream_D)))
            if isinstance(compiled_z, torch.Tensor) and isinstance(
                upstream_z, torch.Tensor
            ):
                upstream_pairs.append((compiled_z, _require_tensor(upstream_z)))
            if isinstance(compiled_dt_bias, torch.Tensor) and isinstance(
                upstream_dt_bias, torch.Tensor
            ):
                upstream_pairs.append(
                    (compiled_dt_bias, _require_tensor(upstream_dt_bias))
                )
            _assert_grads_close(upstream_pairs, atol=grad_atol, rtol=grad_rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("case", _DW_CONV_CPU_CASES)
def test_dw_causal_conv_cpu_compiled_matches_all_backends(
    dtype: torch.dtype, case: dict[str, object]
) -> None:
    status = ops._cpu_backend_status()
    if not status.available:
        pytest.skip(f"CPU backend unavailable: {status}")

    device = torch.device("cpu")
    torch.manual_seed(104)

    case_data = cast(_DwConvCase, case)

    batch, channels, length = 2, 3, 6
    kernel_size = 3

    x_base = torch.randn(
        batch, channels, length, device=device, dtype=dtype, requires_grad=True
    )
    weight_base = torch.randn(
        channels, kernel_size, device=device, dtype=dtype, requires_grad=True
    )
    bias_base = (
        torch.randn(channels, device=device, dtype=dtype, requires_grad=True)
        if case_data["with_bias"]
        else None
    )
    activation = case_data["activation"]

    compiled_args = (
        _clone_tensor(x_base),
        _clone_tensor(weight_base),
    )
    compiled_bias = _clone_optional_tensor(bias_base)
    out_compiled = ops.dw_causal_conv(
        *compiled_args, bias=compiled_bias, activation=activation
    )

    ref_args = (
        _clone_tensor(x_base),
        _clone_tensor(weight_base),
    )
    ref_bias = _clone_optional_tensor(bias_base)
    out_ref = reference_ops.dw_causal_conv(
        *ref_args, bias=ref_bias, activation=activation
    )

    data_atol, data_rtol = _output_tolerances(dtype)
    torch.testing.assert_close(
        out_compiled, out_ref.to(out_compiled.dtype), atol=data_atol, rtol=data_rtol
    )

    out_compiled.sum().backward()
    out_ref.sum().backward()

    grad_atol, grad_rtol = _grad_tolerances(dtype)
    grad_pairs: list[tuple[torch.Tensor, torch.Tensor]] = [
        (compiled_args[0], ref_args[0]),
        (compiled_args[1], ref_args[1]),
    ]
    if isinstance(compiled_bias, torch.Tensor):
        grad_pairs.append((compiled_bias, _require_tensor(ref_bias)))
    _assert_grads_close(grad_pairs, atol=grad_atol, rtol=grad_rtol)

    upstream_fn = _load_upstream_op("dw_causal_conv")
    if upstream_fn is not None:
        upstream_args = (
            _clone_tensor(x_base),
            _clone_tensor(weight_base),
        )
        upstream_bias = _clone_optional_tensor(bias_base)
        try:
            out_upstream = upstream_fn(
                *upstream_args, bias=upstream_bias, activation=activation
            )
        except Exception:  # pragma: no cover - optional dependency differences
            upstream_fn = None
        else:
            torch.testing.assert_close(
                out_compiled,
                out_upstream.to(out_compiled.dtype),
                atol=data_atol,
                rtol=data_rtol,
            )

            out_upstream.sum().backward()

            upstream_pairs: list[tuple[torch.Tensor, torch.Tensor]] = [
                (compiled_args[0], upstream_args[0]),
                (compiled_args[1], upstream_args[1]),
            ]
            if isinstance(compiled_bias, torch.Tensor) and isinstance(
                upstream_bias, torch.Tensor
            ):
                upstream_pairs.append((compiled_bias, _require_tensor(upstream_bias)))
            _assert_grads_close(upstream_pairs, atol=grad_atol, rtol=grad_rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("case", _FUSED_LAYER_NORM_CPU_CASES)
def test_fused_layer_norm_cpu_compiled_matches_all_backends(
    dtype: torch.dtype, case: dict[str, object]
) -> None:
    status = ops._cpu_backend_status()
    if not status.available:
        pytest.skip(f"CPU backend unavailable: {status}")

    device = torch.device("cpu")
    torch.manual_seed(105)

    case_data = cast(_FusedLayerNormCase, case)

    batch, length, hidden = 2, 3, 4

    x_base = torch.randn(
        batch, length, hidden, device=device, dtype=dtype, requires_grad=True
    )
    weight_base = torch.randn(hidden, device=device, dtype=dtype, requires_grad=True)
    bias_base = (
        torch.randn(hidden, device=device, dtype=dtype, requires_grad=True)
        if case_data["with_bias"]
        else None
    )
    residual_base = (
        torch.randn(
            batch, length, hidden, device=device, dtype=dtype, requires_grad=True
        )
        if case_data["with_residual"]
        else None
    )

    compiled_args = (
        _clone_tensor(x_base),
        _clone_tensor(weight_base),
        _clone_optional_tensor(bias_base),
    )
    compiled_residual = _clone_optional_tensor(residual_base)
    out_compiled = ops.fused_layer_norm(
        *compiled_args,
        residual=compiled_residual,
        is_rms=case_data["is_rms"],
        prenorm=case_data["prenorm"],
        residual_in_fp32=case_data["residual_in_fp32"],
    )

    ref_args = (
        _clone_tensor(x_base),
        _clone_tensor(weight_base),
        _clone_optional_tensor(bias_base),
    )
    ref_residual = _clone_optional_tensor(residual_base)
    out_ref = reference_ops.fused_layer_norm(
        *ref_args,
        residual=ref_residual,
        is_rms=case_data["is_rms"],
        prenorm=case_data["prenorm"],
        residual_in_fp32=case_data["residual_in_fp32"],
    )

    data_atol, data_rtol = _output_tolerances(dtype)
    torch.testing.assert_close(
        out_compiled, out_ref.to(out_compiled.dtype), atol=data_atol, rtol=data_rtol
    )

    out_compiled.sum().backward()
    out_ref.sum().backward()

    grad_atol, grad_rtol = _grad_tolerances(dtype)
    grad_pairs: list[tuple[torch.Tensor, torch.Tensor]] = [
        (compiled_args[0], ref_args[0]),
        (compiled_args[1], ref_args[1]),
    ]
    if isinstance(compiled_args[2], torch.Tensor):
        grad_pairs.append((compiled_args[2], _require_tensor(ref_args[2])))
    if isinstance(compiled_residual, torch.Tensor):
        grad_pairs.append((compiled_residual, _require_tensor(ref_residual)))
    _assert_grads_close(grad_pairs, atol=grad_atol, rtol=grad_rtol)

    upstream_fn = _load_upstream_op("fused_layer_norm")
    if upstream_fn is not None:
        upstream_args = (
            _clone_tensor(x_base),
            _clone_tensor(weight_base),
            _clone_optional_tensor(bias_base),
        )
        upstream_residual = _clone_optional_tensor(residual_base)
        try:
            out_upstream = upstream_fn(
                *upstream_args,
                residual=upstream_residual,
                is_rms=case_data["is_rms"],
                prenorm=case_data["prenorm"],
                residual_in_fp32=case_data["residual_in_fp32"],
            )
        except Exception:  # pragma: no cover - optional dependency differences
            upstream_fn = None
        else:
            torch.testing.assert_close(
                out_compiled,
                out_upstream.to(out_compiled.dtype),
                atol=data_atol,
                rtol=data_rtol,
            )

            out_upstream.sum().backward()

            upstream_pairs: list[tuple[torch.Tensor, torch.Tensor]] = [
                (compiled_args[0], upstream_args[0]),
                (compiled_args[1], upstream_args[1]),
            ]
            if isinstance(compiled_args[2], torch.Tensor) and isinstance(
                upstream_args[2], torch.Tensor
            ):
                upstream_pairs.append(
                    (compiled_args[2], _require_tensor(upstream_args[2]))
                )
            if isinstance(compiled_residual, torch.Tensor) and isinstance(
                upstream_residual, torch.Tensor
            ):
                upstream_pairs.append(
                    (compiled_residual, _require_tensor(upstream_residual))
                )
            _assert_grads_close(upstream_pairs, atol=grad_atol, rtol=grad_rtol)


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

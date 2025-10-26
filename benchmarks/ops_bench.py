"""Micro-benchmarks for comparing SSM kernels across backends.

The harness intentionally mirrors the workloads used in the upstream
``state-spaces/mamba`` repository so we can sanity-check parity while
bringing the kernels online here.  It builds synthetic inputs that
exercise the selective scan (Mamba-1), selective state step, and SSD
chunk scan (Mamba-2) operators for representative sequence lengths,
state dimensions, and chunk sizes.

Example::

    $ python -m benchmarks.ops_bench --device cpu
    $ python -m benchmarks.ops_bench --device cuda --dtype float16
    $ python -m benchmarks.ops_bench --device cpu --mode backward

Use ``--json`` to emit the raw timing samples for downstream analysis
(e.g. plotting, regression gates).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import torch

from ssm import ops as dispatch_ops
from ssm.ops import _load_cpu_ops, _load_cuda_ops
from ssm.ops.python import reference as reference_ops


try:  # pragma: no cover - optional dependency for richer tables
    from tabulate import tabulate
except Exception:  # pragma: no cover - optional dependency
    tabulate = None  # type: ignore[assignment]


@dataclass(frozen=True)
class SelectiveScanCase:
    name: str
    batch: int
    length: int
    dim: int
    state_dim: int
    softplus: bool = False
    gated: bool = True
    skip_connection: bool = True


@dataclass(frozen=True)
class StateStepCase:
    name: str
    batch: int
    dim: int
    state_dim: int
    softplus: bool = True
    gated: bool = True
    skip_connection: bool = True


@dataclass(frozen=True)
class ChunkScanCase:
    name: str
    batch: int
    length: int
    heads: int
    proj: int
    chunk_size: int
    gated: bool = True
    skip_connection: bool = True


@dataclass
class Timing:
    op: str
    case: str
    device: str
    mode: str
    backend: str
    time_ms: float | None
    speedup_vs_python: float | None
    notes: str = ""


_SELECTIVE_SCAN_CASES: Sequence[SelectiveScanCase] = (
    SelectiveScanCase("prefill_2k_d512", batch=1, length=2048, dim=512, state_dim=16),
    SelectiveScanCase("prefill_4k_d768", batch=1, length=4096, dim=768, state_dim=16),
)

_STATE_STEP_CASES: Sequence[StateStepCase] = (
    StateStepCase("decode_step_d512", batch=4, dim=512, state_dim=16),
    StateStepCase("decode_step_d1024", batch=4, dim=1024, state_dim=16),
)

_CHUNK_SCAN_CASES: Sequence[ChunkScanCase] = (
    ChunkScanCase(
        "chunk_scan_l2k_h16_c128",
        batch=1,
        length=2048,
        heads=16,
        proj=64,
        chunk_size=128,
    ),
    ChunkScanCase(
        "chunk_scan_l4k_h32_c256",
        batch=1,
        length=4096,
        heads=32,
        proj=64,
        chunk_size=256,
    ),
)


def _manual_seed(seed: int) -> None:
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - cuda optional
        torch.cuda.manual_seed_all(seed)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":  # pragma: no cover - cuda optional
        torch.cuda.synchronize(device)


def _time_fn(
    fn: Callable[[], None], *, device: torch.device, iters: int
) -> list[float]:
    times: list[float] = []
    # Always perform a warmup iteration to amortize kernel compilation and cache effects.
    fn()
    _synchronize(device)
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        _synchronize(device)
        times.append((time.perf_counter() - start) * 1000.0)
    return times


def _backend_order(device: torch.device) -> tuple[str, ...]:
    order: list[str] = ["python"]
    if device.type == "cpu":
        order.append("cpu")
    if device.type == "cuda":  # pragma: no branch - only cuda targets
        order.append("cuda")
    order.append("dispatch")
    return tuple(order)


def _with_grad_disabled(fn: Callable[[], None]) -> Callable[[], None]:
    def wrapper() -> None:
        with torch.no_grad():
            fn()

    return wrapper


def _prepare_autograd_inputs(
    base_inputs: dict[str, torch.Tensor],
    *,
    grad_keys: Sequence[str],
    clone_keys: Sequence[str] = (),
) -> dict[str, torch.Tensor]:
    """Detach and clone tensors before enabling gradients for benchmarking."""

    prepared: dict[str, torch.Tensor] = {}
    grad_key_set = set(grad_keys)
    clone_key_set = set(clone_keys)

    for name, value in base_inputs.items():
        tensor = value
        if name in grad_key_set or name in clone_key_set:
            tensor = tensor.detach().clone()
        if name in grad_key_set and tensor.is_floating_point():
            tensor.requires_grad_(True)
        prepared[name] = tensor

    return prepared


def _grad_enabled_keys(
    inputs: dict[str, torch.Tensor], candidates: Sequence[str]
) -> tuple[str, ...]:
    return tuple(
        name
        for name in candidates
        if name in inputs and inputs[name].is_floating_point()
    )


def _autograd_targets(
    inputs: dict[str, torch.Tensor], grad_keys: Sequence[str]
) -> list[torch.Tensor]:
    return [inputs[name] for name in grad_keys if inputs[name].requires_grad]


def _ensure_tensor(
    output: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output


def _selective_scan_inputs(
    case: SelectiveScanCase, device: torch.device, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    _manual_seed(0)
    u = torch.randn(case.batch, case.dim, case.length, device=device, dtype=dtype)
    delta = torch.rand(case.batch, case.dim, case.length, device=device, dtype=dtype)
    # Broadcasted parameters live in fp32 in upstream kernels.
    A = torch.randn(case.dim, case.state_dim, device=device, dtype=torch.float32)
    B = torch.randn(case.dim, case.state_dim, device=device, dtype=torch.float32)
    C = torch.randn(case.dim, case.state_dim, device=device, dtype=torch.float32)
    dt_bias = torch.randn(case.dim, device=device, dtype=torch.float32)
    inputs: dict[str, torch.Tensor] = {
        "u": u,
        "delta": delta,
        "A": A,
        "B": B,
        "C": C,
        "dt_bias": dt_bias,
    }
    if case.skip_connection:
        inputs["D"] = torch.randn(case.dim, device=device, dtype=torch.float32)
    if case.gated:
        inputs["z"] = torch.randn(
            case.batch, case.dim, case.length, device=device, dtype=dtype
        )
    return inputs


def _state_step_inputs(
    case: StateStepCase, device: torch.device, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    _manual_seed(1)
    state = torch.randn(
        case.batch, case.dim, case.state_dim, device=device, dtype=dtype
    )
    x = torch.randn(case.batch, case.dim, device=device, dtype=dtype)
    dt = torch.rand(case.batch, case.dim, device=device, dtype=dtype)
    A = torch.randn(case.dim, case.state_dim, device=device, dtype=torch.float32)
    B = torch.randn(case.dim, case.state_dim, device=device, dtype=torch.float32)
    C = torch.randn(case.dim, case.state_dim, device=device, dtype=torch.float32)
    dt_bias = torch.randn(case.dim, device=device, dtype=torch.float32)
    inputs: dict[str, torch.Tensor] = {
        "state": state,
        "x": x,
        "dt": dt,
        "A": A,
        "B": B,
        "C": C,
        "dt_bias": dt_bias,
    }
    if case.skip_connection:
        inputs["D"] = torch.randn(case.dim, device=device, dtype=torch.float32)
    if case.gated:
        inputs["z"] = torch.randn(case.batch, case.dim, device=device, dtype=dtype)
    return inputs


def _chunk_scan_inputs(
    case: ChunkScanCase, device: torch.device, dtype: torch.dtype
) -> tuple[dict[str, torch.Tensor], int]:
    _manual_seed(2)
    X = torch.randn(
        case.batch, case.length, case.heads, case.proj, device=device, dtype=dtype
    )
    dt = torch.rand(case.batch, case.length, case.heads, device=device, dtype=dtype)
    A = torch.randn(case.heads, case.proj, device=device, dtype=torch.float32)
    B = torch.randn(
        case.batch,
        case.length,
        case.heads,
        case.proj,
        device=device,
        dtype=torch.float32,
    )
    C = torch.randn(
        case.batch,
        case.length,
        case.heads,
        case.proj,
        device=device,
        dtype=torch.float32,
    )
    inputs: dict[str, torch.Tensor] = {
        "X": X,
        "dt": dt,
        "A": A,
        "B": B,
        "C": C,
    }
    if case.skip_connection:
        inputs["D"] = torch.randn(case.heads, device=device, dtype=torch.float32)
    if case.gated:
        inputs["z"] = torch.randn(
            case.batch, case.length, case.heads, device=device, dtype=dtype
        )
    return inputs, case.chunk_size


def _benchmark_backend(
    name: str,
    fn_builder: Callable[[str], Callable[[], None] | None],
    device: torch.device,
    *,
    iters: int,
    mode: str,
) -> tuple[list[float] | None, str]:
    runner = fn_builder(name)
    if runner is None:
        return None, "unavailable"
    if mode == "forward":
        runner = _with_grad_disabled(runner)
    samples = _time_fn(runner, device=device, iters=iters)
    return samples, ""


def _format_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    if tabulate is not None:
        return tabulate(rows, headers=headers, tablefmt="github", floatfmt=".2f")

    # Fallback formatting when tabulate is missing.
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(str(value)))
    header_line = " | ".join(h.ljust(widths[idx]) for idx, h in enumerate(headers))
    sep = "-+-".join("-" * widths[idx] for idx in range(len(headers)))
    body = [
        " | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(row))
        for row in rows
    ]
    return "\n".join([header_line, sep, *body])


def _build_selective_scan_runners(
    case: SelectiveScanCase,
    device: torch.device,
    dtype: torch.dtype,
    *,
    iters: int,
    mode: str,
) -> list[Timing]:
    inputs = _selective_scan_inputs(case, device, dtype)
    grad_keys = _grad_enabled_keys(
        inputs, ("u", "delta", "A", "B", "C", "D", "z", "dt_bias")
    )
    backend_samples: dict[str, list[float]] = {}
    notes: dict[str, str] = {}
    extra_notes: dict[str, str] = {}

    def make_runner(backend: str) -> Callable[[], None] | None:
        if backend == "python":

            def call(current_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
                return _ensure_tensor(
                    reference_ops.selective_scan(
                        current_inputs["u"],
                        current_inputs["delta"],
                        current_inputs["A"],
                        current_inputs["B"],
                        current_inputs["C"],
                        D=current_inputs.get("D"),
                        z=current_inputs.get("z"),
                        dt_bias=current_inputs.get("dt_bias"),
                        softplus=case.softplus,
                        return_last_state=False,
                    )
                )

        elif backend == "cpu":
            module = _load_cpu_ops()
            if module is None:
                return None
            if mode == "backward":
                extra_notes[backend] = "autograd (dispatch wrapper)"

                def call(current_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
                    return _ensure_tensor(
                        dispatch_ops.selective_scan(
                            current_inputs["u"],
                            current_inputs["delta"],
                            current_inputs["A"],
                            current_inputs["B"],
                            current_inputs["C"],
                            D=current_inputs.get("D"),
                            z=current_inputs.get("z"),
                            dt_bias=current_inputs.get("dt_bias"),
                            softplus=case.softplus,
                            return_last_state=False,
                        )
                    )

            else:

                def call(current_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
                    return module.selective_scan(  # type: ignore[call-arg]
                        current_inputs["u"],
                        current_inputs["delta"],
                        current_inputs["A"],
                        current_inputs["B"],
                        current_inputs["C"],
                        current_inputs.get("D"),
                        current_inputs.get("z"),
                        current_inputs.get("dt_bias"),
                        case.softplus,
                        False,
                    )

        elif backend == "cuda":
            module = _load_cuda_ops()
            if module is None:
                return None
            if mode == "backward":
                extra_notes[backend] = "autograd (dispatch wrapper)"

                def call(current_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
                    return _ensure_tensor(
                        dispatch_ops.selective_scan(
                            current_inputs["u"],
                            current_inputs["delta"],
                            current_inputs["A"],
                            current_inputs["B"],
                            current_inputs["C"],
                            D=current_inputs.get("D"),
                            z=current_inputs.get("z"),
                            dt_bias=current_inputs.get("dt_bias"),
                            softplus=case.softplus,
                            return_last_state=False,
                        )
                    )

            else:

                def call(current_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
                    return module.selective_scan(  # type: ignore[call-arg]
                        current_inputs["u"],
                        current_inputs["delta"],
                        current_inputs["A"],
                        current_inputs["B"],
                        current_inputs["C"],
                        current_inputs.get("D"),
                        current_inputs.get("z"),
                        current_inputs.get("dt_bias"),
                        case.softplus,
                        False,
                    )

        elif backend == "dispatch":

            def call(current_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
                return _ensure_tensor(
                    dispatch_ops.selective_scan(
                        current_inputs["u"],
                        current_inputs["delta"],
                        current_inputs["A"],
                        current_inputs["B"],
                        current_inputs["C"],
                        D=current_inputs.get("D"),
                        z=current_inputs.get("z"),
                        dt_bias=current_inputs.get("dt_bias"),
                        softplus=case.softplus,
                        return_last_state=False,
                    )
                )

        else:
            raise ValueError(f"unknown backend {backend}")

        if mode == "forward":

            def run_forward() -> None:
                call(inputs)

            return run_forward

        def run_backward() -> None:
            prepared = _prepare_autograd_inputs(
                inputs, grad_keys=grad_keys, clone_keys=()
            )
            output = call(prepared)
            loss = output.sum()
            targets = _autograd_targets(prepared, grad_keys)
            if targets:
                torch.autograd.grad(loss, targets, allow_unused=True)

        return run_backward

    order = _backend_order(device)
    for backend in order:
        samples, note = _benchmark_backend(
            backend, make_runner, device, iters=iters, mode=mode
        )
        if samples is not None:
            backend_samples[backend] = samples
        combined_note = note or extra_notes.get(backend, "")
        notes[backend] = combined_note

    python_time = statistics.mean(backend_samples.get("python", [math.nan]))
    timings: list[Timing] = []
    for backend in order:
        if backend in backend_samples:
            samples = backend_samples[backend]
            mean_time = statistics.mean(samples)
            speedup = (
                python_time / mean_time
                if python_time and not math.isnan(python_time)
                else None
            )
            if backend == "python":
                speedup = 1.0
            timings.append(
                Timing(
                    op="selective_scan",
                    case=case.name,
                    device=device.type,
                    mode=mode,
                    backend=backend,
                    time_ms=mean_time,
                    speedup_vs_python=speedup,
                    notes=notes.get(backend, ""),
                )
            )
        else:
            timings.append(
                Timing(
                    op="selective_scan",
                    case=case.name,
                    device=device.type,
                    mode=mode,
                    backend=backend,
                    time_ms=None,
                    speedup_vs_python=None,
                    notes=notes.get(backend, ""),
                )
            )
    return timings


def _build_state_step_runners(
    case: StateStepCase,
    device: torch.device,
    dtype: torch.dtype,
    *,
    iters: int,
    mode: str,
) -> list[Timing]:
    inputs = _state_step_inputs(case, device, dtype)
    grad_keys = _grad_enabled_keys(
        inputs, ("state", "x", "dt", "A", "B", "C", "D", "z", "dt_bias")
    )
    backend_samples: dict[str, list[float]] = {}
    notes: dict[str, str] = {}
    extra_notes: dict[str, str] = {}

    def make_runner(backend: str) -> Callable[[], None] | None:
        if backend == "python":

            def call(current_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
                return reference_ops.selective_state_step(
                    current_inputs["state"],
                    current_inputs["x"],
                    current_inputs["dt"],
                    current_inputs["A"],
                    current_inputs["B"],
                    current_inputs["C"],
                    D=current_inputs.get("D"),
                    z=current_inputs.get("z"),
                    dt_bias=current_inputs.get("dt_bias"),
                    softplus=case.softplus,
                )

        elif backend == "cpu":
            module = _load_cpu_ops()
            if module is None:
                return None
            if mode == "backward":
                extra_notes[backend] = "autograd (dispatch wrapper)"

                def call(current_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
                    return dispatch_ops.selective_state_step(
                        current_inputs["state"],
                        current_inputs["x"],
                        current_inputs["dt"],
                        current_inputs["A"],
                        current_inputs["B"],
                        current_inputs["C"],
                        D=current_inputs.get("D"),
                        z=current_inputs.get("z"),
                        dt_bias=current_inputs.get("dt_bias"),
                        softplus=case.softplus,
                    )

            else:

                def call(current_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
                    return module.selective_state_step(  # type: ignore[call-arg]
                        current_inputs["state"],
                        current_inputs["x"],
                        current_inputs["dt"],
                        current_inputs["A"],
                        current_inputs["B"],
                        current_inputs["C"],
                        current_inputs.get("D"),
                        current_inputs.get("z"),
                        current_inputs.get("dt_bias"),
                        case.softplus,
                    )

        elif backend == "cuda":
            module = _load_cuda_ops()
            if module is None:
                return None
            if mode == "backward":
                extra_notes[backend] = "autograd (dispatch wrapper)"

                def call(current_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
                    return dispatch_ops.selective_state_step(
                        current_inputs["state"],
                        current_inputs["x"],
                        current_inputs["dt"],
                        current_inputs["A"],
                        current_inputs["B"],
                        current_inputs["C"],
                        D=current_inputs.get("D"),
                        z=current_inputs.get("z"),
                        dt_bias=current_inputs.get("dt_bias"),
                        softplus=case.softplus,
                    )

            else:

                def call(current_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
                    return module.selective_state_step(  # type: ignore[call-arg]
                        current_inputs["state"],
                        current_inputs["x"],
                        current_inputs["dt"],
                        current_inputs["A"],
                        current_inputs["B"],
                        current_inputs["C"],
                        current_inputs.get("D"),
                        current_inputs.get("z"),
                        current_inputs.get("dt_bias"),
                        case.softplus,
                    )

        elif backend == "dispatch":

            def call(current_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
                return dispatch_ops.selective_state_step(
                    current_inputs["state"],
                    current_inputs["x"],
                    current_inputs["dt"],
                    current_inputs["A"],
                    current_inputs["B"],
                    current_inputs["C"],
                    D=current_inputs.get("D"),
                    z=current_inputs.get("z"),
                    dt_bias=current_inputs.get("dt_bias"),
                    softplus=case.softplus,
                )

        else:
            raise ValueError(f"unknown backend {backend}")

        if mode == "forward":

            def run_forward() -> None:
                state_inputs = inputs.copy()
                state_inputs["state"] = state_inputs["state"].clone()
                call(state_inputs)

            return run_forward

        def run_backward() -> None:
            prepared = _prepare_autograd_inputs(
                inputs, grad_keys=grad_keys, clone_keys=("state",)
            )
            output = call(prepared)
            loss = output.sum()
            targets = _autograd_targets(prepared, grad_keys)
            if targets:
                torch.autograd.grad(loss, targets, allow_unused=True)

        return run_backward

    order = _backend_order(device)
    for backend in order:
        samples, note = _benchmark_backend(
            backend, make_runner, device, iters=iters, mode=mode
        )
        if samples is not None:
            backend_samples[backend] = samples
        combined_note = note or extra_notes.get(backend, "")
        notes[backend] = combined_note

    python_time = statistics.mean(backend_samples.get("python", [math.nan]))
    timings: list[Timing] = []
    for backend in order:
        if backend in backend_samples:
            samples = backend_samples[backend]
            mean_time = statistics.mean(samples)
            speedup = (
                python_time / mean_time
                if python_time and not math.isnan(python_time)
                else None
            )
            if backend == "python":
                speedup = 1.0
            timings.append(
                Timing(
                    op="selective_state_step",
                    case=case.name,
                    device=device.type,
                    mode=mode,
                    backend=backend,
                    time_ms=mean_time,
                    speedup_vs_python=speedup,
                    notes=notes.get(backend, ""),
                )
            )
        else:
            timings.append(
                Timing(
                    op="selective_state_step",
                    case=case.name,
                    device=device.type,
                    mode=mode,
                    backend=backend,
                    time_ms=None,
                    speedup_vs_python=None,
                    notes=notes.get(backend, ""),
                )
            )
    return timings


def _build_chunk_scan_runners(
    case: ChunkScanCase,
    device: torch.device,
    dtype: torch.dtype,
    *,
    iters: int,
    mode: str,
) -> list[Timing]:
    tensors, chunk_size = _chunk_scan_inputs(case, device, dtype)
    grad_keys = _grad_enabled_keys(tensors, ("X", "dt", "A", "B", "C", "D", "z"))
    backend_samples: dict[str, list[float]] = {}
    notes: dict[str, str] = {}
    extra_notes: dict[str, str] = {}

    def make_runner(backend: str) -> Callable[[], None] | None:
        if backend == "python":

            def call(current_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
                return reference_ops.ssd_chunk_scan(
                    current_inputs["X"],
                    current_inputs["dt"],
                    current_inputs["A"],
                    current_inputs["B"],
                    current_inputs["C"],
                    chunk_size,
                    D=current_inputs.get("D"),
                    z=current_inputs.get("z"),
                    seq_meta=None,
                    initial_states=None,
                )

        elif backend == "cpu":
            module = _load_cpu_ops()
            if module is None:
                return None
            if mode == "backward":
                extra_notes[backend] = "autograd (dispatch wrapper)"

                def call(current_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
                    return dispatch_ops.ssd_chunk_scan(
                        current_inputs["X"],
                        current_inputs["dt"],
                        current_inputs["A"],
                        current_inputs["B"],
                        current_inputs["C"],
                        chunk_size,
                        D=current_inputs.get("D"),
                        z=current_inputs.get("z"),
                        seq_meta=None,
                        initial_states=None,
                    )

            else:

                def call(current_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
                    return module.ssd_chunk_scan(  # type: ignore[call-arg]
                        current_inputs["X"],
                        current_inputs["dt"],
                        current_inputs["A"],
                        current_inputs["B"],
                        current_inputs["C"],
                        chunk_size,
                        current_inputs.get("D"),
                        current_inputs.get("z"),
                        None,
                        None,
                        None,
                    )

        elif backend == "cuda":
            module = _load_cuda_ops()
            if module is None:
                return None
            if mode == "backward":
                extra_notes[backend] = "autograd (dispatch wrapper)"

                def call(current_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
                    return dispatch_ops.ssd_chunk_scan(
                        current_inputs["X"],
                        current_inputs["dt"],
                        current_inputs["A"],
                        current_inputs["B"],
                        current_inputs["C"],
                        chunk_size,
                        D=current_inputs.get("D"),
                        z=current_inputs.get("z"),
                        seq_meta=None,
                        initial_states=None,
                    )

            else:

                def call(current_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
                    return module.ssd_chunk_scan(  # type: ignore[call-arg]
                        current_inputs["X"],
                        current_inputs["dt"],
                        current_inputs["A"],
                        current_inputs["B"],
                        current_inputs["C"],
                        chunk_size,
                        current_inputs.get("D"),
                        current_inputs.get("z"),
                        None,
                        None,
                        None,
                    )

        elif backend == "dispatch":

            def call(current_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
                return dispatch_ops.ssd_chunk_scan(
                    current_inputs["X"],
                    current_inputs["dt"],
                    current_inputs["A"],
                    current_inputs["B"],
                    current_inputs["C"],
                    chunk_size,
                    D=current_inputs.get("D"),
                    z=current_inputs.get("z"),
                    seq_meta=None,
                    initial_states=None,
                )

        else:
            raise ValueError(f"unknown backend {backend}")

        if mode == "forward":

            def run_forward() -> None:
                call(tensors)

            return run_forward

        def run_backward() -> None:
            prepared = _prepare_autograd_inputs(tensors, grad_keys=grad_keys)
            output = call(prepared)
            loss = output.sum()
            targets = _autograd_targets(prepared, grad_keys)
            if targets:
                torch.autograd.grad(loss, targets, allow_unused=True)

        return run_backward

    order = _backend_order(device)
    for backend in order:
        samples, note = _benchmark_backend(
            backend, make_runner, device, iters=iters, mode=mode
        )
        if samples is not None:
            backend_samples[backend] = samples
        combined_note = note or extra_notes.get(backend, "")
        notes[backend] = combined_note

    python_time = statistics.mean(backend_samples.get("python", [math.nan]))
    timings: list[Timing] = []
    for backend in order:
        if backend in backend_samples:
            samples = backend_samples[backend]
            mean_time = statistics.mean(samples)
            speedup = (
                python_time / mean_time
                if python_time and not math.isnan(python_time)
                else None
            )
            if backend == "python":
                speedup = 1.0
            timings.append(
                Timing(
                    op="ssd_chunk_scan",
                    case=case.name,
                    device=device.type,
                    mode=mode,
                    backend=backend,
                    time_ms=mean_time,
                    speedup_vs_python=speedup,
                    notes=notes.get(backend, ""),
                )
            )
        else:
            timings.append(
                Timing(
                    op="ssd_chunk_scan",
                    case=case.name,
                    device=device.type,
                    mode=mode,
                    backend=backend,
                    time_ms=None,
                    speedup_vs_python=None,
                    notes=notes.get(backend, ""),
                )
            )
    return timings


def _filter_devices(requested: str) -> Iterable[torch.device]:
    if requested == "all":
        yield torch.device("cpu")
        if torch.cuda.is_available():  # pragma: no cover - cuda optional
            yield torch.device("cuda")
        return
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():  # pragma: no cover
        raise RuntimeError(
            "CUDA device requested but torch.cuda.is_available() is False"
        )
    yield device


def _maybe_to_dtype(value: str | None, *, default: torch.dtype) -> torch.dtype:
    if value is None:
        return default
    try:
        return getattr(torch, value)
    except AttributeError as exc:  # pragma: no cover - invalid CLI usage
        raise ValueError(f"unknown dtype: {value}") from exc


def _emit_json(path: str, timings: Sequence[Timing]) -> None:
    serialisable = [
        {
            "op": t.op,
            "case": t.case,
            "device": t.device,
            "mode": t.mode,
            "backend": t.backend,
            "time_ms": t.time_ms,
            "speedup_vs_python": t.speedup_vs_python,
            "notes": t.notes,
        }
        for t in timings
    ]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(serialisable, handle, indent=2)


def _summarise(timings: Sequence[Timing]) -> str:
    grouped: dict[tuple[str, str, str, str], list[Timing]] = {}
    for timing in timings:
        key = (timing.op, timing.case, timing.device, timing.mode)
        grouped.setdefault(key, []).append(timing)

    lines: list[str] = []
    for (op, case, device, mode), group in sorted(grouped.items()):
        lines.append(f"== {op} :: {case} :: device={device} :: mode={mode} ==")
        rows: list[list[Any]] = []
        backend_order = {
            name: idx for idx, name in enumerate(("python", "cpu", "cuda", "dispatch"))
        }
        for timing in sorted(
            group, key=lambda t: backend_order.get(t.backend, len(backend_order))
        ):
            rows.append(
                [
                    timing.backend,
                    f"{timing.time_ms:.2f}" if timing.time_ms is not None else "n/a",
                    f"{timing.speedup_vs_python:.2f}x"
                    if timing.speedup_vs_python is not None
                    else "n/a",
                    timing.notes or "",
                ]
            )
        lines.append(
            _format_table(["backend", "time (ms)", "vs python", "notes"], rows)
        )
        lines.append("")
    return "\n".join(lines).strip()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark SSM kernels across backends"
    )
    parser.add_argument(
        "--device",
        default="all",
        choices=["cpu", "cuda", "all"],
        help="Device to run benchmarks on (default: all detected).",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help="Input dtype (torch attribute name). Defaults to float32 on CPU and float16 on CUDA.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=10,
        help="Number of timed iterations per backend (default: 10).",
    )
    parser.add_argument(
        "--mode",
        choices=["forward", "backward"],
        default="forward",
        help="Measure forward execution or forward+backward autograd timings.",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Optional path to emit raw timing samples as JSON.",
    )
    args = parser.parse_args(argv)

    timings: list[Timing] = []
    for device in _filter_devices(args.device):
        dtype_default = torch.float16 if device.type == "cuda" else torch.float32
        dtype = _maybe_to_dtype(args.dtype, default=dtype_default)
        for case in _SELECTIVE_SCAN_CASES:
            timings.extend(
                _build_selective_scan_runners(
                    case,
                    device=device,
                    dtype=dtype,
                    iters=args.iters,
                    mode=args.mode,
                )
            )
        for case in _STATE_STEP_CASES:
            timings.extend(
                _build_state_step_runners(
                    case,
                    device=device,
                    dtype=dtype,
                    iters=args.iters,
                    mode=args.mode,
                )
            )
        for case in _CHUNK_SCAN_CASES:
            timings.extend(
                _build_chunk_scan_runners(
                    case,
                    device=device,
                    dtype=dtype,
                    iters=args.iters,
                    mode=args.mode,
                )
            )

    summary = _summarise(timings)
    print(summary)
    if args.json:
        _emit_json(args.json, timings)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

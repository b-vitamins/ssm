#!/usr/bin/env python
"""Generate definitive forward + gradient goldens for upstream mamba-ssm CUDA/Triton ops.

This script executes small, deterministic forward passes for the key CUDA/Triton
operators in the upstream `mamba-ssm` repository and serializes inputs,
outputs, and parameter/input gradients to a single JSON file. Ports should
reproduce these numerics exactly.

Notes:
- Requires a CUDA-capable GPU and an environment where upstream mamba-ssm is
  importable and its CUDA extensions can load.
- We record forward outputs and gradients where autograd is available. Some
  component-only kernels that lack autograd wrappers are captured as
  forward-only.

Example:
    poetry run python scripts/refresh_mamba_goldens.py \
        --mamba-repo . \
        --device cuda:0 \
        --output tests/mamba_reference_cases.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import pprint  # noqa: F401
import subprocess
import sys
from dataclasses import dataclass  # noqa: F401
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from einops import rearrange
import os  # noqa: F401,F811
import shutil


# ---------------------------- Utilities ------------------------------------


def _validate_device(device: torch.device) -> None:
    if device.type != "cuda":
        raise ValueError("This script requires a CUDA device (e.g., --device cuda:0).")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Ensure an NVIDIA driver and CUDA runtime are present."
        )


def _append_repo_to_path(repo_path: Path) -> None:
    resolved = repo_path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Upstream repository not found at {resolved}.")
    if str(resolved) not in sys.path:
        sys.path.insert(0, str(resolved))


def _serialize_tensor(t: torch.Tensor) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "shape": list(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
    }
    if torch.is_complex(t):
        rec["data_real"] = t.detach().cpu().real.tolist()
        rec["data_imag"] = t.detach().cpu().imag.tolist()
    else:
        rec["data"] = t.detach().cpu().tolist()
    return rec


def _maybe_serialize(t: Optional[torch.Tensor]) -> Optional[Dict[str, Any]]:
    if t is None:
        return None
    return _serialize_tensor(t)


def _autograd_grads(
    output: torch.Tensor, inputs: Iterable[Optional[torch.Tensor]]
) -> List[Optional[torch.Tensor]]:
    tensors = [x for x in inputs if x is not None]
    if not tensors:
        return []
    grads = torch.autograd.grad(output, tensors, retain_graph=False, allow_unused=True)
    results: List[Optional[torch.Tensor]] = []
    it = iter(grads)
    for x in inputs:
        results.append(None if x is None else next(it))
    return results


def _resolve_commit(repo_path: Path) -> str:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_path)
    except Exception:
        return "unknown"
    return commit.decode().strip()


# Best-effort helpers for Triton on distros without ldconfig (e.g., Guix)
def _ensure_triton_env() -> None:
    # Point Triton at ptxas if not already set
    if not os.environ.get("TRITON_PTXAS_PATH"):
        ptxas = shutil.which("ptxas")
        if ptxas:
            os.environ["TRITON_PTXAS_PATH"] = ptxas
    # Locate libcuda.so.1 in common locations if not provided
    if not os.environ.get("TRITON_LIBCUDA_PATH"):
        candidates = []
        cuda_home = os.environ.get("CUDA_HOME")
        if cuda_home:
            candidates.append(os.path.join(cuda_home, "targets", "x86_64-linux", "lib"))
        candidates += [
            "/run/opengl-driver/lib",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib64",
            "/lib64",
            "/usr/lib",
        ]
        for d in candidates:
            if d and os.path.exists(os.path.join(d, "libcuda.so.1")):
                os.environ["TRITON_LIBCUDA_PATH"] = d
                break


# ---------------------------- Generators -----------------------------------


def gen_selective_scan(device: torch.device) -> Dict[str, Any]:
    """mamba_ssm.ops.selective_scan_interface.selective_scan_fn"""
    mod = importlib.import_module("mamba_ssm.ops.selective_scan_interface")
    selective_scan_fn = getattr(mod, "selective_scan_fn")

    torch.manual_seed(0)
    batch, dim, dstate, L = 2, 3, 4, 5

    # Case 1: Full options
    u = torch.randn(
        batch, dim, L, device=device, dtype=torch.float32, requires_grad=True
    )
    delta = torch.randn_like(u, requires_grad=True)
    A = torch.randn(dim, dstate, device=device, requires_grad=True)
    B = torch.randn(dim, dstate, device=device, requires_grad=True)
    C = torch.randn(dim, dstate, device=device, requires_grad=True)
    D = torch.randn(dim, device=device, requires_grad=True)
    z = torch.randn(batch, dim, L, device=device, requires_grad=True)
    delta_bias = torch.randn(dim, device=device, requires_grad=True)
    y, last_state = selective_scan_fn(
        u,
        delta,
        A,
        B,
        C,
        D=D,
        z=z,
        delta_bias=delta_bias,
        delta_softplus=True,
        return_last_state=True,
    )

    # Case 2: Minimal path (no D, z, bias, softplus=False)
    torch.manual_seed(1)
    u2 = torch.randn(batch, dim, L, device=device, requires_grad=True)
    delta2 = torch.randn_like(u2, requires_grad=True)
    A2 = torch.randn(dim, dstate, device=device, requires_grad=True)
    B2 = torch.randn(dim, dstate, device=device, requires_grad=True)
    C2 = torch.randn(dim, dstate, device=device, requires_grad=True)
    y2 = selective_scan_fn(
        u2, delta2, A2, B2, C2, D=None, z=None, delta_bias=None, delta_softplus=False
    )

    # Case 2b: Minimal fp16 and bf16 variants
    torch.manual_seed(11)
    u2h = torch.randn(
        batch, dim, L, device=device, dtype=torch.float16, requires_grad=True
    )
    delta2h = torch.randn_like(u2h, requires_grad=True)
    # Weights must be float32 or complex64 in upstream kernel
    A2h = torch.randn(
        dim, dstate, device=device, dtype=torch.float32, requires_grad=True
    )
    B2h = torch.randn(
        dim, dstate, device=device, dtype=torch.float32, requires_grad=True
    )
    C2h = torch.randn(
        dim, dstate, device=device, dtype=torch.float32, requires_grad=True
    )
    y2h = selective_scan_fn(
        u2h,
        delta2h,
        A2h,
        B2h,
        C2h,
        D=None,
        z=None,
        delta_bias=None,
        delta_softplus=False,
    )

    torch.manual_seed(12)
    u2b = torch.randn(
        batch, dim, L, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    delta2b = torch.randn_like(u2b, requires_grad=True)
    A2b = torch.randn(
        dim, dstate, device=device, dtype=torch.float32, requires_grad=True
    )
    B2b = torch.randn(
        dim, dstate, device=device, dtype=torch.float32, requires_grad=True
    )
    C2b = torch.randn(
        dim, dstate, device=device, dtype=torch.float32, requires_grad=True
    )
    y2b = selective_scan_fn(
        u2b,
        delta2b,
        A2b,
        B2b,
        C2b,
        D=None,
        z=None,
        delta_bias=None,
        delta_softplus=False,
    )

    # Case 3: Grouped/time-varying B, C
    torch.manual_seed(2)
    groups = 2
    u3 = torch.randn(batch, dim, L, device=device, requires_grad=True)
    delta3 = torch.randn_like(u3, requires_grad=True)
    A3 = torch.randn(dim, dstate, device=device, requires_grad=True)
    B3 = torch.randn(batch, groups, dstate, L, device=device, requires_grad=True)
    C3 = torch.randn(batch, groups, dstate, L, device=device, requires_grad=True)
    delta_bias3 = torch.randn(dim, device=device, requires_grad=True)
    y3 = selective_scan_fn(
        u3,
        delta3,
        A3,
        B3,
        C3,
        D=None,
        z=None,
        delta_bias=delta_bias3,
        delta_softplus=True,
        return_last_state=False,
    )

    # Case 4: 3D time-varying B, C (B, N, L)
    torch.manual_seed(3)
    u4 = torch.randn(batch, dim, L, device=device, requires_grad=True)
    delta4 = torch.randn_like(u4, requires_grad=True)
    A4 = torch.randn(dim, dstate, device=device, requires_grad=True)
    B4 = torch.randn(batch, dstate, L, device=device, requires_grad=True)
    C4 = torch.randn(batch, dstate, L, device=device, requires_grad=True)
    y4 = selective_scan_fn(
        u4, delta4, A4, B4, C4, D=None, z=None, delta_bias=None, delta_softplus=True
    )

    # Case 5: Complex A with real-packed B/C (B, N, 2L)
    torch.manual_seed(4)
    u5 = torch.randn(batch, dim, L, device=device, requires_grad=True)
    delta5 = torch.randn_like(u5, requires_grad=True)
    A5 = torch.randn(
        dim, dstate, device=device, dtype=torch.complex64, requires_grad=True
    )
    B5 = torch.randn(batch, dstate, 2 * L, device=device, requires_grad=True)
    C5 = torch.randn(batch, dstate, 2 * L, device=device, requires_grad=True)
    y5 = selective_scan_fn(
        u5, delta5, A5, B5, C5, D=None, z=None, delta_bias=None, delta_softplus=False
    )

    return {
        "full": {
            "inputs": {
                "u": _serialize_tensor(u),
                "delta": _serialize_tensor(delta),
                "A": _serialize_tensor(A),
                "B": _serialize_tensor(B),
                "C": _serialize_tensor(C),
                "D": _serialize_tensor(D),
                "z": _serialize_tensor(z),
                "delta_bias": _serialize_tensor(delta_bias),
            },
            "outputs": {
                "y": _serialize_tensor(y),
                "last_state": _serialize_tensor(last_state),
            },
            "gradients": {
                name: _maybe_serialize(grad)
                for name, grad in zip(
                    ("u", "delta", "A", "B", "C", "D", "z", "delta_bias"),
                    _autograd_grads(y.sum(), (u, delta, A, B, C, D, z, delta_bias)),
                )
            },
        },
        "minimal": {
            "inputs": {
                "u": _serialize_tensor(u2),
                "delta": _serialize_tensor(delta2),
                "A": _serialize_tensor(A2),
                "B": _serialize_tensor(B2),
                "C": _serialize_tensor(C2),
            },
            "outputs": {"y": _serialize_tensor(y2)},
            "gradients": {
                name: _maybe_serialize(grad)
                for name, grad in zip(
                    ("u", "delta", "A", "B", "C"),
                    _autograd_grads(y2.sum(), (u2, delta2, A2, B2, C2)),
                )
            },
        },
        "minimal_fp16": {
            "inputs": {
                "u": _serialize_tensor(u2h),
                "delta": _serialize_tensor(delta2h),
                "A": _serialize_tensor(A2h),
                "B": _serialize_tensor(B2h),
                "C": _serialize_tensor(C2h),
            },
            "outputs": {"y": _serialize_tensor(y2h)},
            "gradients": {
                name: _maybe_serialize(grad)
                for name, grad in zip(
                    ("u", "delta", "A", "B", "C"),
                    _autograd_grads(y2h.sum(), (u2h, delta2h, A2h, B2h, C2h)),
                )
            },
        },
        "minimal_bf16": {
            "inputs": {
                "u": _serialize_tensor(u2b),
                "delta": _serialize_tensor(delta2b),
                "A": _serialize_tensor(A2b),
                "B": _serialize_tensor(B2b),
                "C": _serialize_tensor(C2b),
            },
            "outputs": {"y": _serialize_tensor(y2b)},
            "gradients": {
                name: _maybe_serialize(grad)
                for name, grad in zip(
                    ("u", "delta", "A", "B", "C"),
                    _autograd_grads(y2b.sum(), (u2b, delta2b, A2b, B2b, C2b)),
                )
            },
        },
        "grouped_time_varying": {
            "inputs": {
                "u": _serialize_tensor(u3),
                "delta": _serialize_tensor(delta3),
                "A": _serialize_tensor(A3),
                "B": _serialize_tensor(B3),
                "C": _serialize_tensor(C3),
                "delta_bias": _serialize_tensor(delta_bias3),
            },
            "outputs": {"y": _serialize_tensor(y3)},
            "gradients": {
                name: _maybe_serialize(grad)
                for name, grad in zip(
                    ("u", "delta", "A", "B", "C", "delta_bias"),
                    _autograd_grads(y3.sum(), (u3, delta3, A3, B3, C3, delta_bias3)),
                )
            },
        },
        "time_varying_3d": {
            "inputs": {
                "u": _serialize_tensor(u4),
                "delta": _serialize_tensor(delta4),
                "A": _serialize_tensor(A4),
                "B": _serialize_tensor(B4),
                "C": _serialize_tensor(C4),
            },
            "outputs": {"y": _serialize_tensor(y4)},
            "gradients": {
                name: _maybe_serialize(grad)
                for name, grad in zip(
                    ("u", "delta", "A", "B", "C"),
                    _autograd_grads(y4.sum(), (u4, delta4, A4, B4, C4)),
                )
            },
        },
        "complex_A_packed_BC": {
            "inputs": {
                "u": _serialize_tensor(u5),
                "delta": _serialize_tensor(delta5),
                "A": _serialize_tensor(A5),
                "B": _serialize_tensor(B5),
                "C": _serialize_tensor(C5),
            },
            "outputs": {"y": _serialize_tensor(y5)},
            "gradients": {
                name: _maybe_serialize(grad)
                for name, grad in zip(
                    ("u", "delta", "A", "B", "C"),
                    _autograd_grads(y5.sum(), (u5, delta5, A5, B5, C5)),
                )
            },
        },
    }


def gen_ssd_chunk_components(device: torch.device) -> Dict[str, Any]:
    """Standalone SSD building blocks: cumsum, state, state_passing, scan."""
    st_mod = importlib.import_module("mamba_ssm.ops.triton.ssd_chunk_state")
    sp_mod = importlib.import_module("mamba_ssm.ops.triton.ssd_state_passing")
    sc_mod = importlib.import_module("mamba_ssm.ops.triton.ssd_chunk_scan")

    _chunk_cumsum_fwd = getattr(st_mod, "_chunk_cumsum_fwd")
    chunk_state = getattr(st_mod, "chunk_state")
    state_passing = getattr(sp_mod, "state_passing")
    chunk_scan = getattr(sc_mod, "chunk_scan")

    torch.manual_seed(10)
    batch, seqlen, nheads, hdim = 2, 6, 2, 3
    ngroups, dstate = 1, 4
    chunk_size = 3

    # Inputs
    x = torch.randn(batch, seqlen, nheads, hdim, device=device)
    dt = torch.randn(batch, seqlen, nheads, device=device)
    A = torch.randn(nheads, device=device)
    dt_bias = torch.randn(nheads, device=device)
    B = torch.randn(batch, seqlen, ngroups, dstate, device=device)
    C = torch.randn(batch, seqlen, ngroups, dstate, device=device)

    # Cumsum over chunks (softplus True)
    dA_cumsum, dt_out = _chunk_cumsum_fwd(
        dt, A, chunk_size, dt_bias=dt_bias, dt_softplus=True
    )
    dt_reshaped = dt_out  # (b, h, nchunks, chunk)

    # Cumsum with clamp and no softplus
    torch.manual_seed(13)
    dt2 = torch.randn(batch, seqlen, nheads, device=device) * 10.0  # large range
    dA_cumsum2, dt_out2 = _chunk_cumsum_fwd(
        dt2, A, chunk_size, dt_bias=None, dt_softplus=False, dt_limit=(0.01, 1.0)
    )

    # Per-chunk states and passing
    # Enable grads and compute states
    B.requires_grad_(True)
    x.requires_grad_(True)
    dt_reshaped.requires_grad_(True)
    dA_cumsum.requires_grad_(True)
    states = chunk_state(B, x, dt_reshaped, dA_cumsum)  # (b, nchunks, h, hdim, dstate)
    dA_last = dA_cumsum[:, :, :, -1]
    # state_passing expects (b, nchunks, h, dim). Flatten (hdim, dstate)
    states_flat = rearrange(states, "b c h p n -> b c h (p n)")
    # Also compute grads for state_passing
    states_flat.requires_grad_(True)
    dA_last.requires_grad_(True)
    states_passed_flat, final_states = state_passing(states_flat, dA_last)
    # Reshape back to (b, nchunks, h, hdim, dstate) for chunk_scan
    states_passed = rearrange(states_passed_flat, "b c h p n -> b c h p n", n=dstate)

    # Scan to outputs (D matrix and z provided)
    D = torch.randn(nheads, hdim, device=device)
    z = torch.randn(batch, seqlen, nheads, hdim, device=device)
    # chunk_scan has no autograd wrapper; keep forward only by detaching inputs
    y = chunk_scan(
        B.detach(),
        C.detach(),
        x.detach(),
        dt_reshaped.detach(),
        dA_cumsum.detach(),
        states_passed.detach(),
        D=D.detach(),
        z=z.detach(),
    )

    # Alt path: D as vector, no z
    torch.manual_seed(11)
    D_vec = torch.randn(nheads, device=device)
    y_vecD = chunk_scan(
        B.detach(),
        C.detach(),
        x.detach(),
        dt_reshaped.detach(),
        dA_cumsum.detach(),
        states_passed.detach(),
        D=D_vec.detach(),
        z=None,
    )

    return {
        "chunk_cumsum": {
            "inputs": {
                "dt": _serialize_tensor(dt),
                "A": _serialize_tensor(A),
                "dt_bias": _serialize_tensor(dt_bias),
            },
            "outputs": {
                "dA_cumsum": _serialize_tensor(dA_cumsum),
                "dt_out": _serialize_tensor(dt_out),
            },
        },
        "chunk_cumsum_clamped": {
            "inputs": {"dt": _serialize_tensor(dt2), "A": _serialize_tensor(A)},
            "outputs": {
                "dA_cumsum": _serialize_tensor(dA_cumsum2),
                "dt_out": _serialize_tensor(dt_out2),
            },
        },
        "chunk_state": {
            "inputs": {
                "B": _serialize_tensor(B),
                "x": _serialize_tensor(x),
                "dt": _serialize_tensor(dt_reshaped),
                "dA_cumsum": _serialize_tensor(dA_cumsum),
            },
            "outputs": {"states": _serialize_tensor(states)},
            "gradients": {
                name: _maybe_serialize(grad)
                for name, grad in zip(
                    ("B", "x", "dt", "dA_cumsum"),
                    _autograd_grads(states.sum(), (B, x, dt_reshaped, dA_cumsum)),
                )
            },
        },
        "state_passing": {
            "inputs": {
                "states": _serialize_tensor(states),
                "dA_last": _serialize_tensor(dA_last),
            },
            "outputs": {
                "states_passed": _serialize_tensor(states_passed),
                "final_states": _serialize_tensor(final_states),
            },
            "gradients": {
                name: _maybe_serialize(grad)
                for name, grad in zip(
                    ("states", "dA_last"),
                    _autograd_grads(states_passed_flat.sum(), (states_flat, dA_last)),
                )
            },
        },
        "chunk_scan_matrix_D_z": {
            "inputs": {
                "B": _serialize_tensor(B),
                "C": _serialize_tensor(C),
                "x": _serialize_tensor(x),
                "dt": _serialize_tensor(dt_reshaped),
                "dA_cumsum": _serialize_tensor(dA_cumsum),
                "states": _serialize_tensor(states_passed),
                "D": _serialize_tensor(D),
                "z": _serialize_tensor(z),
            },
            "outputs": {"y": _serialize_tensor(y)},
        },
        "chunk_scan_vector_D": {
            "inputs": {
                "B": _serialize_tensor(B),
                "C": _serialize_tensor(C),
                "x": _serialize_tensor(x),
                "dt": _serialize_tensor(dt_reshaped),
                "dA_cumsum": _serialize_tensor(dA_cumsum),
                "states": _serialize_tensor(states_passed),
                "D": _serialize_tensor(D_vec),
            },
            "outputs": {"y": _serialize_tensor(y_vecD)},
        },
    }


def gen_ssd_combined(device: torch.device) -> Dict[str, Any]:
    """mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined"""
    mod = importlib.import_module("mamba_ssm.ops.triton.ssd_combined")
    mamba_chunk_scan_combined = getattr(mod, "mamba_chunk_scan_combined")

    torch.manual_seed(20)
    batch, seqlen, nheads, hdim = 2, 6, 2, 3
    ngroups, dstate = 1, 4
    chunk_size = 3

    X = torch.randn(batch, seqlen, nheads, hdim, device=device, requires_grad=True)
    dt = torch.randn(batch, seqlen, nheads, device=device, requires_grad=True)
    A = torch.randn(nheads, device=device, requires_grad=True)
    B = torch.randn(batch, seqlen, ngroups, dstate, device=device, requires_grad=True)
    C = torch.randn(batch, seqlen, ngroups, dstate, device=device, requires_grad=True)
    D = torch.randn(nheads, hdim, device=device, requires_grad=True)
    z = torch.randn(batch, seqlen, nheads, hdim, device=device, requires_grad=True)
    dt_bias = torch.randn(nheads, device=device, requires_grad=True)
    initial_states = torch.randn(
        batch, nheads, hdim, dstate, device=device, requires_grad=True
    )

    # Base case with final states
    y, final_states = mamba_chunk_scan_combined(
        X,
        dt,
        A,
        B,
        C,
        chunk_size,
        D=D,
        z=z,
        dt_bias=dt_bias,
        initial_states=initial_states,
        dt_softplus=True,
        return_final_states=True,
    )
    grads_full = _autograd_grads(
        y.sum(), (X, dt, A, B, C, D, z, dt_bias, initial_states)
    )

    # Varlen states (no final states returned). Note: combined varlen path requires batch == 1.
    torch.manual_seed(21)
    batch_v, seqlen_v = 1, 6
    Xv = torch.randn(batch_v, seqlen_v, nheads, hdim, device=device)
    dtv = torch.randn(batch_v, seqlen_v, nheads, device=device)
    Bv = torch.randn(batch_v, seqlen_v, ngroups, dstate, device=device)
    Cv = torch.randn(batch_v, seqlen_v, ngroups, dstate, device=device)
    cu_seqlens = torch.tensor(
        [0, 3, 6], device=device, dtype=torch.int32
    )  # two sequences: lengths 3 and 3
    y2, varlen_states = mamba_chunk_scan_combined(
        Xv,
        dtv,
        A,
        Bv,
        Cv,
        chunk_size,
        D=None,
        z=None,
        dt_bias=None,
        cu_seqlens=cu_seqlens,
        dt_softplus=False,
        return_varlen_states=True,
    )

    return {
        "combined_full": {
            "inputs": {
                "X": _serialize_tensor(X),
                "dt": _serialize_tensor(dt),
                "A": _serialize_tensor(A),
                "B": _serialize_tensor(B),
                "C": _serialize_tensor(C),
                "chunk_size": chunk_size,
                "D": _serialize_tensor(D),
                "z": _serialize_tensor(z),
                "dt_bias": _serialize_tensor(dt_bias),
                "initial_states": _serialize_tensor(initial_states),
            },
            "outputs": {
                "y": _serialize_tensor(y),
                "final_states": _serialize_tensor(final_states),
            },
            "gradients": {
                name: _maybe_serialize(grad)
                for name, grad in zip(
                    ("X", "dt", "A", "B", "C", "D", "z", "dt_bias", "initial_states"),
                    grads_full,
                )
            },
        },
        "combined_varlen": {
            "inputs": {
                "X": _serialize_tensor(Xv),
                "dt": _serialize_tensor(dtv),
                "A": _serialize_tensor(A),
                "B": _serialize_tensor(Bv),
                "C": _serialize_tensor(Cv),
                "chunk_size": chunk_size,
            },
            "outputs": {
                "y": _serialize_tensor(y2),
                "varlen_states": _serialize_tensor(varlen_states),
            },
        },
    }


def gen_ssd_bmm(device: torch.device) -> Dict[str, Any]:
    """mamba_ssm.ops.triton.ssd_bmm._bmm_chunk_fwd (forward only)."""
    mod = importlib.import_module("mamba_ssm.ops.triton.ssd_bmm")
    _bmm_chunk_fwd = getattr(mod, "_bmm_chunk_fwd")

    torch.manual_seed(30)
    batch, seqlen, k = 2, 6, 4
    chunk_size = 3

    # Ungrouped, no causal, no seq_idx
    a = torch.randn(batch, seqlen, k, device=device)
    b = torch.randn_like(a)
    out = _bmm_chunk_fwd(a, b, chunk_size, seq_idx=None, causal=False)

    # Causal and seq_idx enabled
    torch.manual_seed(31)
    a2 = torch.randn(batch, seqlen, k, device=device)
    b2 = torch.randn_like(a2)
    seq_idx = torch.tensor(
        [[0, 0, 0, 1, 1, 1], [5, 5, 6, 6, 6, 6]], device=device, dtype=torch.int32
    )
    out2 = _bmm_chunk_fwd(a2, b2, chunk_size, seq_idx=seq_idx, causal=True)

    # Grouped (ngroups=2)
    torch.manual_seed(32)
    ngroups = 2
    a3 = torch.randn(batch, seqlen, ngroups, k, device=device)
    b3 = torch.randn_like(a3)
    out3 = _bmm_chunk_fwd(a3, b3, chunk_size, seq_idx=None, causal=False)

    return {
        "basic": {
            "inputs": {"a": _serialize_tensor(a), "b": _serialize_tensor(b)},
            "outputs": {"out": _serialize_tensor(out)},
        },
        "causal_seq": {
            "inputs": {
                "a": _serialize_tensor(a2),
                "b": _serialize_tensor(b2),
                "seq_idx": _serialize_tensor(seq_idx),
            },
            "outputs": {"out": _serialize_tensor(out2)},
        },
        "grouped": {
            "inputs": {"a": _serialize_tensor(a3), "b": _serialize_tensor(b3)},
            "outputs": {"out": _serialize_tensor(out3)},
        },
    }


def gen_selective_state_update(device: torch.device) -> Dict[str, Any]:
    mod = importlib.import_module("mamba_ssm.ops.triton.selective_state_update")
    selective_state_update = getattr(mod, "selective_state_update")

    torch.manual_seed(70)
    cases: Dict[str, Any] = {}
    batch, dim, state_dim = 2, 3, 4

    # Baseline with all options
    state = torch.randn(batch, dim, state_dim, device=device)
    x = torch.randn(batch, dim, device=device)
    dt = torch.randn_like(x)
    A = torch.randn(dim, state_dim, device=device)
    B = torch.randn(batch, state_dim, device=device)
    C = torch.randn(batch, state_dim, device=device)
    D = torch.randn(dim, device=device)
    z = torch.randn(batch, dim, device=device)
    dt_bias = torch.randn(dim, device=device)
    state_in = state.clone()
    y = selective_state_update(
        state, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True
    )

    cases["baseline_full"] = {
        "inputs": {
            "state": _serialize_tensor(state_in),
            "x": _serialize_tensor(x),
            "dt": _serialize_tensor(dt),
            "A": _serialize_tensor(A),
            "B": _serialize_tensor(B),
            "C": _serialize_tensor(C),
            "D": _serialize_tensor(D),
            "z": _serialize_tensor(z),
            "dt_bias": _serialize_tensor(dt_bias),
        },
        "outputs": {
            "y": _serialize_tensor(y),
            "updated_state": _serialize_tensor(state),
        },
    }

    # Grouped projections with heads, no D/z, no softplus
    torch.manual_seed(71)
    nheads, groups = 2, 2
    state = torch.randn(batch, nheads, dim, state_dim, device=device)
    x = torch.randn(batch, nheads, dim, device=device)
    dt = torch.randn_like(x)
    A = torch.randn(nheads, dim, state_dim, device=device)
    B = torch.randn(batch, groups, state_dim, device=device)
    C = torch.randn(batch, groups, state_dim, device=device)
    dt_bias = torch.randn(nheads, dim, device=device)
    state_in = state.clone()
    # Work around upstream star-unpack bug when D is None by passing zeros
    D_zero = torch.zeros(nheads, dim, device=device)
    y = selective_state_update(
        state, x, dt, A, B, C, D=D_zero, z=None, dt_bias=dt_bias, dt_softplus=False
    )
    cases["grouped"] = {
        "inputs": {
            "state": _serialize_tensor(state_in),
            "x": _serialize_tensor(x),
            "dt": _serialize_tensor(dt),
            "A": _serialize_tensor(A),
            "B": _serialize_tensor(B),
            "C": _serialize_tensor(C),
            "dt_bias": _serialize_tensor(dt_bias),
        },
        "outputs": {
            "y": _serialize_tensor(y),
            "updated_state": _serialize_tensor(state),
        },
    }

    return cases


def gen_layer_norms(device: torch.device) -> Dict[str, Any]:
    """LayerNorm/RMSNorm fused + gated RMSNorm."""
    ln_mod = importlib.import_module("mamba_ssm.ops.triton.layer_norm")
    gln_mod = importlib.import_module("mamba_ssm.ops.triton.layernorm_gated")

    layer_norm_fn = getattr(ln_mod, "layer_norm_fn")
    rms_norm_fn = getattr(ln_mod, "rms_norm_fn")
    gated_rms_ref = getattr(gln_mod, "rms_norm_ref")

    torch.manual_seed(40)
    batch, length, hidden = 2, 4, 6

    # Fused LayerNorm, prenorm True with residual
    x = torch.randn(batch, length, hidden, device=device, requires_grad=True)
    w = torch.randn(hidden, device=device, requires_grad=True)
    b = torch.randn(hidden, device=device, requires_grad=True)
    residual = torch.randn_like(x, requires_grad=True)
    y, residual_out = layer_norm_fn(
        x.reshape(-1, hidden),
        w,
        b,
        residual=residual.reshape(-1, hidden),
        prenorm=True,
        residual_in_fp32=True,
    )
    y = y.reshape(batch, length, hidden)
    residual_out = residual_out.reshape(batch, length, hidden)

    # Fused RMSNorm, no residual
    torch.manual_seed(41)
    x2 = torch.randn(batch, length, hidden, device=device, requires_grad=True)
    w2 = torch.randn(hidden, device=device, requires_grad=True)
    y2 = rms_norm_fn(
        x2.reshape(-1, hidden),
        w2,
        None,
        residual=None,
        prenorm=False,
        residual_in_fp32=False,
    ).reshape(batch, length, hidden)

    # Gated RMSNorm both orders
    torch.manual_seed(42)
    x3 = torch.randn(batch, length, hidden, device=device, requires_grad=True)
    w3 = torch.randn(hidden, device=device, requires_grad=True)
    b3 = torch.randn(hidden, device=device, requires_grad=True)
    z3 = torch.randn(batch, length, hidden, device=device, requires_grad=True)
    y3_before = gated_rms_ref(x3, w3, b3, z=z3, norm_before_gate=True)
    y3_after = gated_rms_ref(x3, w3, b3, z=z3, norm_before_gate=False)

    # LayerNorm extras: dropout, rowscale, dual branch, prenorm=False
    torch.manual_seed(43)
    x4 = torch.randn(batch, length, hidden, device=device, requires_grad=True)
    w4 = torch.randn(hidden, device=device, requires_grad=True)
    b4 = torch.randn(hidden, device=device, requires_grad=True)
    residual4 = torch.randn_like(x4, requires_grad=True)
    rowscale = torch.rand(batch * length, device=device)
    y4, dropout_mask4, _dropout_mask4_1 = layer_norm_fn(
        x4.reshape(-1, hidden),
        w4,
        b4,
        residual=residual4.reshape(-1, hidden),
        prenorm=False,
        residual_in_fp32=True,
        dropout_p=0.1,
        rowscale=rowscale,
        return_dropout_mask=True,
    )

    torch.manual_seed(44)
    # Dual branch: x1/w1/b1
    x5 = torch.randn(batch, length, hidden, device=device, requires_grad=True)
    w5 = torch.randn(hidden, device=device, requires_grad=True)
    b5 = torch.randn(hidden, device=device, requires_grad=True)
    x1 = torch.randn_like(x5, requires_grad=True)
    w1 = torch.randn(hidden, device=device, requires_grad=True)
    b1 = torch.randn(hidden, device=device, requires_grad=True)
    y5, y5_1 = layer_norm_fn(
        x5.reshape(-1, hidden),
        w5,
        b5,
        x1=x1.reshape(-1, hidden),
        weight1=w1,
        bias1=b1,
        prenorm=False,
        residual_in_fp32=False,
    )

    return {
        "layer_norm_prenorm": {
            "inputs": {
                "x": _serialize_tensor(x),
                "w": _serialize_tensor(w),
                "b": _serialize_tensor(b),
                "residual": _serialize_tensor(residual),
            },
            "outputs": {
                "y": _serialize_tensor(y),
                "residual_out": _serialize_tensor(residual_out),
            },
            "gradients": {
                name: _maybe_serialize(grad)
                for name, grad in zip(
                    ("x", "w", "b", "residual"),
                    _autograd_grads(y.sum(), (x, w, b, residual)),
                )
            },
        },
        "rms_norm": {
            "inputs": {"x": _serialize_tensor(x2), "w": _serialize_tensor(w2)},
            "outputs": {"y": _serialize_tensor(y2)},
            "gradients": {
                name: _maybe_serialize(grad)
                for name, grad in zip(("x", "w"), _autograd_grads(y2.sum(), (x2, w2)))
            },
        },
        "gated_rms_norm": {
            "inputs": {
                "x": _serialize_tensor(x3),
                "w": _serialize_tensor(w3),
                "b": _serialize_tensor(b3),
                "z": _serialize_tensor(z3),
            },
            "outputs": {
                "y_norm_before": _serialize_tensor(y3_before),
                "y_norm_after": _serialize_tensor(y3_after),
            },
        },
        "layer_norm_dropout_rowscale": {
            "inputs": {
                "x": _serialize_tensor(x4),
                "w": _serialize_tensor(w4),
                "b": _serialize_tensor(b4),
                "residual": _serialize_tensor(residual4),
                "rowscale": _serialize_tensor(rowscale),
            },
            "outputs": {
                "y": _serialize_tensor(y4),
                "dropout_mask": _serialize_tensor(dropout_mask4.to(torch.uint8)),
            },
            "gradients": {
                name: _maybe_serialize(grad)
                for name, grad in zip(
                    ("x", "w", "b", "residual"),
                    _autograd_grads(y4.sum(), (x4, w4, b4, residual4)),
                )
            },
        },
        "layer_norm_dual_branch": {
            "inputs": {
                "x": _serialize_tensor(x5),
                "w": _serialize_tensor(w5),
                "b": _serialize_tensor(b5),
                "x1": _serialize_tensor(x1),
                "w1": _serialize_tensor(w1),
                "b1": _serialize_tensor(b1),
            },
            "outputs": {"y": _serialize_tensor(y5), "y1": _serialize_tensor(y5_1)},
            "gradients": {
                name: _maybe_serialize(grad)
                for name, grad in zip(
                    ("x", "w", "b", "x1", "w1", "b1"),
                    _autograd_grads((y5.sum() + y5_1.sum()), (x5, w5, b5, x1, w1, b1)),
                )
            },
        },
    }


def gen_dw_causal_conv(device: torch.device) -> Dict[str, Any]:
    """causal_conv1d forward tests (channels-first and channels-last)."""
    try:
        causal_module = importlib.import_module("causal_conv1d")
    except ImportError as exc:
        raise ImportError(
            "The causal-conv1d package must be installed to generate these goldens."
        ) from exc
    causal_conv1d_fn = getattr(causal_module, "causal_conv1d_fn")

    torch.manual_seed(50)
    # causal_conv1d requires channels divisible by 8 in current builds
    batch, channels, length = 2, 8, 5
    ksz = 4

    # Channels-first with SiLU
    x = torch.randn(batch, channels, length, device=device, requires_grad=True)
    w = torch.randn(channels, ksz, device=device, requires_grad=True)
    b = torch.randn(channels, device=device, requires_grad=True)
    y = causal_conv1d_fn(x, w, b, activation="silu")
    grads_cf = _autograd_grads(y.sum(), (x, w, b))

    # Channels-last identity (activation=None), no bias
    torch.manual_seed(51)
    x2 = torch.randn(batch, length, channels, device=device, requires_grad=True)
    # causal_conv1d expects weight shape (channels, kernel_size)
    w2 = torch.randn(channels, ksz, device=device, requires_grad=True)
    y2 = causal_conv1d_fn(x2.permute(0, 2, 1), w2, None, activation=None).permute(
        0, 2, 1
    )
    grads_cl = _autograd_grads(y2.sum(), (x2, w2))

    return {
        "channels_first_silu": {
            "inputs": {
                "x": _serialize_tensor(x),
                "weight": _serialize_tensor(w),
                "bias": _serialize_tensor(b),
            },
            "outputs": {"y": _serialize_tensor(y)},
            "gradients": {
                name: _maybe_serialize(grad)
                for name, grad in zip(("x", "weight", "bias"), grads_cf)
            },
        },
        "channels_last_identity": {
            "inputs": {"x": _serialize_tensor(x2), "weight": _serialize_tensor(w2)},
            "outputs": {"y": _serialize_tensor(y2)},
            "gradients": {
                name: _maybe_serialize(grad)
                for name, grad in zip(("x", "weight"), grads_cl)
            },
        },
    }


def gen_swiglu(device: torch.device) -> Dict[str, Any]:
    """Triton SwiGLU forward/backward parity via autograd wrapper."""
    mod = importlib.import_module("mamba_ssm.ops.triton.k_activations")
    swiglu = getattr(mod, "swiglu")

    torch.manual_seed(60)
    batch, ncols = 2, 8
    xy = torch.randn(batch, ncols * 2, device=device, requires_grad=True)
    out = swiglu(xy)
    grads = _autograd_grads(out.sum(), (xy,))
    return {
        "swiglu": {
            "inputs": {"xy": _serialize_tensor(xy.detach())},
            "outputs": {"y": _serialize_tensor(out.detach())},
            "gradients": {"xy": _maybe_serialize(grads[0])},
        }
    }


# ---------------------------- CLI / Main -----------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mamba-repo",
        type=Path,
        required=True,
        help="Path to upstream mamba-ssm repo.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("tests/mamba_reference_cases.json"),
        help="Destination JSON path (default: tests/mamba_reference_cases.json)",
    )
    p.add_argument(
        "--device", type=str, default="cuda:0", help="CUDA device (default: cuda:0)"
    )
    p.add_argument(
        "--skip-dw-conv",
        action="store_true",
        help="Skip depthwise causal-conv goldens if the causal-conv1d package is unavailable.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    _validate_device(device)
    _ensure_triton_env()
    _append_repo_to_path(args.mamba_repo)

    meta = {
        "mamba_repo": str(args.mamba_repo.resolve()),
        "mamba_commit": _resolve_commit(args.mamba_repo),
        "device": args.device,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }

    goldens: Dict[str, Any] = {"meta": meta}

    goldens["selective_scan"] = gen_selective_scan(device)
    goldens["ssd_components"] = gen_ssd_chunk_components(device)
    goldens["ssd_combined"] = gen_ssd_combined(device)
    goldens["ssd_bmm"] = gen_ssd_bmm(device)
    goldens["selective_state_update"] = gen_selective_state_update(device)
    goldens["layer_norms"] = gen_layer_norms(device)
    if args.skip_dw_conv:
        goldens["dw_causal_conv"] = {"skipped": "causal-conv1d not required/installed"}
    else:
        goldens["dw_causal_conv"] = gen_dw_causal_conv(device)
    goldens["swiglu"] = gen_swiglu(device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(goldens, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":  # pragma: no cover
    main()

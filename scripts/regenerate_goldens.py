#!/usr/bin/env python3
"""Regenerate golden files for tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ssm.ops.python import reference as reference_ops


def _tensor_to_list(tensor: torch.Tensor) -> list:
    """Convert a tensor to a nested Python list for JSON serialization."""

    return tensor.detach().cpu().tolist()


def regenerate_selective_scan(out_dir: Path) -> None:
    torch.manual_seed(1234)
    B, D, L, N = 1, 2, 4, 2
    u = torch.randn(B, D, L)
    delta = torch.randn(B, D, L)
    A = torch.randn(D, N)
    Bm = torch.randn(D, N)
    Cm = torch.randn(D, N)
    D_skip = torch.randn(D)
    z = torch.randn(B, D, L)
    dt_bias = torch.randn(D)

    out, last_state = reference_ops.selective_scan(
        u,
        delta,
        A,
        Bm,
        Cm,
        D=D_skip,
        z=z,
        dt_bias=dt_bias,
        softplus=True,
        return_last_state=True,
    )

    payload = {
        "meta": {
            "description": "reference selective_scan",
            "B": B,
            "D": D,
            "L": L,
            "N": N,
            "dtype": "float32",
            "seed": 1234,
        },
        "inputs": {
            "u": _tensor_to_list(u),
            "delta": _tensor_to_list(delta),
            "A": _tensor_to_list(A),
            "B": _tensor_to_list(Bm),
            "C": _tensor_to_list(Cm),
            "D": _tensor_to_list(D_skip),
            "z": _tensor_to_list(z),
            "dt_bias": _tensor_to_list(dt_bias),
            "softplus": True,
        },
        "outputs": {
            "out": _tensor_to_list(out),
            "last_state": _tensor_to_list(last_state),
        },
    }

    (out_dir / "selective_scan_case1.json").write_text(json.dumps(payload, indent=2))

    torch.manual_seed(4321)
    B, D, L, N = 2, 4, 5, 3
    groups = 2
    u = torch.randn(B, D, L)
    delta = torch.randn(B, D, L)
    A = torch.randn(D, N)
    B_grouped = torch.randn(B, groups, N, L)
    C_grouped = torch.randn(B, groups, N, L)

    out_grouped, last_grouped = reference_ops.selective_scan(
        u,
        delta,
        A,
        B_grouped,
        C_grouped,
        D=None,
        z=None,
        dt_bias=None,
        softplus=False,
        return_last_state=True,
    )

    payload_grouped = {
        "meta": {
            "description": "grouped selective_scan",
            "B": B,
            "D": D,
            "L": L,
            "N": N,
            "groups": groups,
            "dtype": "float32",
            "seed": 4321,
        },
        "inputs": {
            "u": _tensor_to_list(u),
            "delta": _tensor_to_list(delta),
            "A": _tensor_to_list(A),
            "B": _tensor_to_list(B_grouped),
            "C": _tensor_to_list(C_grouped),
            "D": None,
            "z": None,
            "dt_bias": None,
            "softplus": False,
        },
        "outputs": {
            "out": _tensor_to_list(out_grouped),
            "last_state": _tensor_to_list(last_grouped),
        },
    }

    (out_dir / "selective_scan_grouped.json").write_text(
        json.dumps(payload_grouped, indent=2)
    )


def regenerate_ssd_chunk_scan(out_dir: Path) -> None:
    torch.manual_seed(5678)
    B, L, H, P = 1, 6, 2, 2
    X = torch.randn(B, L, H, P)
    dt = torch.randn(B, L, H)
    A = torch.randn(H, P)
    Bm = torch.randn(H, P)
    Cm = torch.randn(H, P)
    D_skip = torch.randn(H)
    z = torch.randn(B, L, H)
    init_state = torch.randn(B, H, P)
    seq_lens = [4]
    cu = torch.tensor([0, 4], dtype=torch.long)

    out = reference_ops.ssd_chunk_scan(
        X,
        dt,
        A,
        Bm,
        Cm,
        chunk_size=3,
        D=D_skip,
        z=z,
        seq_meta={"seq_lens": seq_lens, "cu_seqlens": cu},
        initial_states=init_state,
    )

    payload = {
        "meta": {
            "description": "reference ssd_chunk_scan",
            "B": B,
            "L": L,
            "H": H,
            "P": P,
            "dtype": "float32",
            "seed": 5678,
        },
        "inputs": {
            "X": _tensor_to_list(X),
            "dt": _tensor_to_list(dt),
            "A": _tensor_to_list(A),
            "B": _tensor_to_list(Bm),
            "C": _tensor_to_list(Cm),
            "D": _tensor_to_list(D_skip),
            "z": _tensor_to_list(z),
            "chunk_size": 3,
            "seq_meta": {"seq_lens": seq_lens, "cu_seqlens": cu.tolist()},
            "initial_states": _tensor_to_list(init_state),
        },
        "outputs": {"Y": _tensor_to_list(out)},
    }

    (out_dir / "ssd_chunk_scan_case1.json").write_text(json.dumps(payload, indent=2))

    torch.manual_seed(91011)
    B, L, H, P = 2, 5, 2, 3
    X = torch.randn(B, L, H, P)
    dt = torch.rand(B, L, H)
    A = torch.randn(H, P)
    B_time = torch.randn(B, L, H, P)
    C_batch = torch.randn(B, H, P)
    D_proj = torch.randn(H, P)
    Z_proj = torch.randn(B, L, H, P)
    init_state = torch.randn(B, H, P)
    cu = torch.tensor([0, 3, 5], dtype=torch.long)

    out_varlen = reference_ops.ssd_chunk_scan(
        X,
        dt,
        A,
        B_time,
        C_batch,
        chunk_size=2,
        D=D_proj,
        z=Z_proj,
        seq_meta={"cu_seqlens": cu},
        initial_states=init_state,
    )

    payload_varlen = {
        "meta": {
            "description": "varlen ssd_chunk_scan",
            "B": B,
            "L": L,
            "H": H,
            "P": P,
            "dtype": "float32",
            "seed": 91011,
        },
        "inputs": {
            "X": _tensor_to_list(X),
            "dt": _tensor_to_list(dt),
            "A": _tensor_to_list(A),
            "B": _tensor_to_list(B_time),
            "C": _tensor_to_list(C_batch),
            "D": _tensor_to_list(D_proj),
            "z": _tensor_to_list(Z_proj),
            "chunk_size": 2,
            "seq_meta": {"cu_seqlens": cu.tolist()},
            "initial_states": _tensor_to_list(init_state),
        },
        "outputs": {"Y": _tensor_to_list(out_varlen)},
    }

    (out_dir / "ssd_chunk_scan_varlen.json").write_text(
        json.dumps(payload_varlen, indent=2)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parents[1] / "tests" / "goldens" / "data"),
    )
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    regenerate_selective_scan(out_dir)
    regenerate_ssd_chunk_scan(out_dir)

    print(f"Goldens written to {out_dir}")


if __name__ == "__main__":
    main()

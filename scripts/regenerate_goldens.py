#!/usr/bin/env python3
"""Regenerate golden files for tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ssm.ops import selective_scan, ssd_chunk_scan


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

    out, last_state = selective_scan(
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

    out = ssd_chunk_scan(
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

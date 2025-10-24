from __future__ import annotations

from typing import Sequence

import pytest
import torch

from ssm import ops
from ssm.ops.python import reference as reference_ops


def _require_cpu_backend() -> None:
    status = ops._cpu_backend_status()
    if not status.available:
        pytest.skip(f"CPU backend unavailable: {status}")


def _make_seq_meta(lengths: Sequence[int]) -> dict[str, torch.Tensor]:
    return {"seq_lens": torch.tensor(lengths, dtype=torch.long)}


def test_ssd_chunk_scan_cpu_long_sequence_matches_reference() -> None:
    _require_cpu_backend()
    torch.manual_seed(123)

    batch, seqlen, heads, proj = 2, 384, 4, 64
    chunk_size = 64

    X = torch.randn(batch, seqlen, heads, proj)
    dt = torch.rand(batch, seqlen, heads) * 0.05
    A = torch.randn(heads, proj, dtype=torch.float32) * -0.1
    B = torch.randn(batch, seqlen, heads, proj, dtype=torch.float32) * 0.05
    C = torch.randn(heads, proj, dtype=torch.float32) * 0.05
    D = torch.randn(heads, proj, dtype=torch.float32) * 0.01
    z = torch.randn(batch, seqlen, heads, proj)

    ref_out = reference_ops.ssd_chunk_scan(
        X,
        dt,
        A,
        B,
        C,
        chunk_size,
        D=D,
        z=z,
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
    )

    torch.testing.assert_close(cpu_out, ref_out, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("lengths", [[128, 97, 64], [80, 80, 80]])
def test_ssd_chunk_scan_cpu_ragged_mixed_precision(lengths: Sequence[int]) -> None:
    _require_cpu_backend()
    torch.manual_seed(321)

    batch = len(lengths)
    seqlen, heads, proj = 160, 3, 32
    chunk_size = 32

    X = (torch.randn(batch, seqlen, heads, proj) * 0.1).to(torch.bfloat16)
    dt = (torch.rand(batch, seqlen, heads) * 0.01).to(torch.bfloat16)
    A = torch.randn(heads, proj, dtype=torch.float32) * -0.05
    B = torch.randn(batch, heads, proj, dtype=torch.float32) * 0.02
    C = torch.randn(batch, seqlen, heads, proj, dtype=torch.float32) * 0.03
    D = torch.randn(heads, dtype=torch.float32) * 0.01
    z = torch.randn(batch, seqlen, heads, dtype=torch.bfloat16)
    init_state = torch.randn(batch, heads, proj, dtype=torch.bfloat16) * 0.05

    seq_meta = _make_seq_meta(lengths)

    ref_out = reference_ops.ssd_chunk_scan(
        X,
        dt,
        A,
        B,
        C,
        chunk_size,
        D=D,
        z=z,
        seq_meta=seq_meta,
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
        seq_meta=seq_meta,
        initial_states=init_state,
    )

    assert cpu_out.dtype == X.dtype
    torch.testing.assert_close(cpu_out, ref_out, atol=2e-2, rtol=2e-2)

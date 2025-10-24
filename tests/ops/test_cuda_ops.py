import pytest
import torch

import ssm.ops as ops
from ssm.ops.python import reference as reference_ops


@pytest.fixture(autouse=True)
def _require_cuda_backend() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    status = ops._cuda_backend_status()
    if not status.available:
        pytest.skip(f"CUDA backend unavailable: {status}")


def test_selective_scan_cuda_matches_reference() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    batch, dim, state_dim, length = 2, 3, 4, 5

    u = torch.randn(batch, dim, length, device=device, dtype=torch.float16)
    delta = torch.randn(batch, dim, length, device=device, dtype=torch.float16)
    A = torch.randn(dim, state_dim, device=device, dtype=torch.float32)
    B = torch.randn(dim, state_dim, device=device)
    C = torch.randn(dim, state_dim, device=device)
    D = torch.randn(dim, device=device)
    z = torch.randn(batch, dim, length, device=device)
    dt_bias = torch.randn(dim, device=device)

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

    out, state = ops.selective_scan(
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

    with torch.cuda.amp.autocast():
        amp_out, amp_state = ops.selective_scan(
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

    torch.testing.assert_close(out, ref_out, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(state, ref_state, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(amp_out.float(), ref_out, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(amp_state.float(), ref_state, atol=1e-3, rtol=1e-3)


def test_ssd_chunk_scan_cuda_respects_varlen() -> None:
    torch.manual_seed(1)
    device = torch.device("cuda")
    batch, seqlen, heads, proj = 2, 6, 3, 2
    chunk_size = 3

    X = torch.randn(batch, seqlen, heads, proj, device=device)
    dt = torch.randn(batch, seqlen, heads, device=device)
    A = torch.randn(heads, proj, device=device, dtype=torch.float32)
    B = torch.randn(heads, proj, device=device)
    C = torch.randn(heads, proj, device=device)
    D = torch.randn(heads, device=device)
    z = torch.randn(batch, seqlen, heads, device=device)
    seq_lens = [seqlen, seqlen - 2]

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
    cuda_out = ops.ssd_chunk_scan(
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

    torch.testing.assert_close(cuda_out, ref_out, atol=1e-4, rtol=1e-4)


def test_selective_state_step_cuda_graph_capture() -> None:
    torch.manual_seed(2)
    device = torch.device("cuda")
    batch, dim, state_dim = 1, 2, 3

    base_state = torch.randn(batch, dim, state_dim, device=device)
    x = torch.randn(batch, dim, device=device)
    dt = torch.randn(batch, dim, device=device)
    A = torch.randn(dim, state_dim, device=device, dtype=torch.float32)
    B = torch.randn(dim, state_dim, device=device)
    C = torch.randn(dim, state_dim, device=device)

    ref_state = base_state.clone()
    ref_out = reference_ops.selective_state_step(
        ref_state,
        x,
        dt,
        A,
        B,
        C,
        softplus=False,
    )

    # Warm-up to populate CUDA memory pools
    ops.selective_state_step(
        base_state.clone(),
        x.clone(),
        dt.clone(),
        A,
        B,
        C,
        softplus=False,
    )

    state_cap = base_state.clone()
    x_cap = x.clone()
    dt_cap = dt.clone()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_cap = ops.selective_state_step(
            state_cap,
            x_cap,
            dt_cap,
            A,
            B,
            C,
            softplus=False,
        )

    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(out_cap, ref_out, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(state_cap, ref_state, atol=1e-4, rtol=1e-4)

    # Update inputs in-place and replay the captured graph again.
    new_x = torch.randn_like(x_cap)
    new_dt = torch.randn_like(dt_cap)
    x_cap.copy_(new_x)
    dt_cap.copy_(new_dt)
    state_cap.copy_(base_state)

    ref_state2 = base_state.clone()
    ref_out2 = reference_ops.selective_state_step(
        ref_state2,
        new_x,
        new_dt,
        A,
        B,
        C,
        softplus=False,
    )

    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(out_cap, ref_out2, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(state_cap, ref_state2, atol=1e-4, rtol=1e-4)

from typing import cast

import pytest
import torch
import torch.nn.functional as F

import ssm.ops as ops
from ssm.ops.python import reference as reference_ops


def _clone_grad(tensor: torch.Tensor) -> torch.Tensor:
    grad = tensor.grad
    assert grad is not None
    return grad.detach().clone()


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


def test_selective_scan_cuda_gradients_match_reference() -> None:
    torch.manual_seed(1)
    device = torch.device("cuda")
    batch, dim, state_dim, length = 1, 2, 3, 4

    u = torch.randn(
        batch, dim, length, device=device, dtype=torch.float32, requires_grad=True
    )
    delta = torch.randn_like(u, requires_grad=True)
    A = torch.randn(
        dim, state_dim, device=device, dtype=torch.float32, requires_grad=True
    )
    B = torch.randn(
        dim, state_dim, device=device, dtype=torch.float32, requires_grad=True
    )
    C = torch.randn(
        dim, state_dim, device=device, dtype=torch.float32, requires_grad=True
    )
    D = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
    z = torch.randn(
        batch, dim, length, device=device, dtype=torch.float32, requires_grad=True
    )
    dt_bias = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)

    ref_inputs = [
        u.detach().clone().requires_grad_(True),
        delta.detach().clone().requires_grad_(True),
        A.detach().clone().requires_grad_(True),
        B.detach().clone().requires_grad_(True),
        C.detach().clone().requires_grad_(True),
        D.detach().clone().requires_grad_(True),
        z.detach().clone().requires_grad_(True),
        dt_bias.detach().clone().requires_grad_(True),
    ]
    ref_out = cast(
        torch.Tensor,
        reference_ops.selective_scan(
            ref_inputs[0],
            ref_inputs[1],
            ref_inputs[2],
            ref_inputs[3],
            ref_inputs[4],
            D=ref_inputs[5],
            z=ref_inputs[6],
            dt_bias=ref_inputs[7],
            softplus=True,
        ),
    )
    ref_out.sum().backward()
    ref_grads = [_clone_grad(tensor) for tensor in ref_inputs]

    out = cast(
        torch.Tensor,
        ops.selective_scan(
            u,
            delta,
            A,
            B,
            C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            softplus=True,
        ),
    )
    out.sum().backward()
    cuda_grads = [_clone_grad(tensor) for tensor in (u, delta, A, B, C, D, z, dt_bias)]

    for grad_cuda, grad_ref in zip(cuda_grads, ref_grads):
        torch.testing.assert_close(grad_cuda, grad_ref, atol=1e-4, rtol=1e-4)


def test_selective_scan_cuda_gradients_grouped_match_reference() -> None:
    torch.manual_seed(2)
    device = torch.device("cuda")
    batch, dim, state_dim, length, groups = 2, 4, 3, 5, 2

    u = torch.randn(
        batch, dim, length, device=device, dtype=torch.float32, requires_grad=True
    )
    delta = torch.randn_like(u, requires_grad=True)
    A = torch.randn(
        dim, state_dim, device=device, dtype=torch.float32, requires_grad=True
    )
    B = torch.randn(
        batch,
        groups,
        state_dim,
        length,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    C = torch.randn(
        batch,
        groups,
        state_dim,
        1,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    dt_bias = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)

    ref_inputs = [
        u.detach().clone().requires_grad_(True),
        delta.detach().clone().requires_grad_(True),
        A.detach().clone().requires_grad_(True),
        B.detach().clone().requires_grad_(True),
        C.detach().clone().requires_grad_(True),
        dt_bias.detach().clone().requires_grad_(True),
    ]

    ref_out = cast(
        torch.Tensor,
        reference_ops.selective_scan(
            ref_inputs[0],
            ref_inputs[1],
            ref_inputs[2],
            ref_inputs[3],
            ref_inputs[4],
            D=None,
            z=None,
            dt_bias=ref_inputs[5],
            softplus=True,
        ),
    )
    ref_out.sum().backward()
    ref_grads = [_clone_grad(tensor) for tensor in ref_inputs]

    out = cast(
        torch.Tensor,
        ops.selective_scan(
            u,
            delta,
            A,
            B,
            C,
            D=None,
            z=None,
            dt_bias=dt_bias,
            softplus=True,
        ),
    )
    out.sum().backward()
    cuda_grads = [_clone_grad(tensor) for tensor in (u, delta, A, B, C, dt_bias)]

    for grad_cuda, grad_ref in zip(cuda_grads, ref_grads):
        torch.testing.assert_close(grad_cuda, grad_ref, atol=1e-4, rtol=1e-4)


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


def test_ssd_chunk_scan_cuda_gradients_match_reference() -> None:
    torch.manual_seed(2)
    device = torch.device("cuda")
    batch, seqlen, heads, proj = 1, 4, 2, 2
    chunk_size = 2

    X = torch.randn(batch, seqlen, heads, proj, device=device, requires_grad=True)
    dt = torch.randn(batch, seqlen, heads, device=device, requires_grad=True)
    A = torch.randn(heads, proj, device=device, dtype=torch.float32, requires_grad=True)
    B = torch.randn_like(A)
    C = torch.randn_like(A)
    D = torch.randn(heads, device=device, dtype=torch.float32, requires_grad=True)
    z = torch.randn(batch, seqlen, heads, device=device, requires_grad=True)

    ref_inputs = [
        X.detach().clone().requires_grad_(True),
        dt.detach().clone().requires_grad_(True),
        A.detach().clone().requires_grad_(True),
        B.detach().clone().requires_grad_(True),
        C.detach().clone().requires_grad_(True),
        D.detach().clone().requires_grad_(True),
        z.detach().clone().requires_grad_(True),
    ]
    ref_out = reference_ops.ssd_chunk_scan(
        ref_inputs[0],
        ref_inputs[1],
        ref_inputs[2],
        ref_inputs[3],
        ref_inputs[4],
        chunk_size,
        D=ref_inputs[5],
        z=ref_inputs[6],
    )
    ref_out.sum().backward()
    ref_grads = [_clone_grad(tensor) for tensor in ref_inputs]

    out = cast(
        torch.Tensor,
        ops.ssd_chunk_scan(
            X,
            dt,
            A,
            B,
            C,
            chunk_size,
            D=D,
            z=z,
        ),
    )
    out.sum().backward()
    cuda_grads = [_clone_grad(tensor) for tensor in (X, dt, A, B, C, D, z)]

    for grad_cuda, grad_ref in zip(cuda_grads, ref_grads):
        torch.testing.assert_close(grad_cuda, grad_ref, atol=1e-4, rtol=1e-4)


def test_ssd_chunk_scan_cuda_initial_state_and_cu_seqlens() -> None:
    torch.manual_seed(3)
    device = torch.device("cuda")

    batch, seqlen, heads, proj = 3, 20, 4, 8
    chunk_size = 6

    lengths = torch.tensor([20, 11, 7], device=device, dtype=torch.long)
    cu_seqlens = F.pad(lengths.cumsum(dim=0), (1, 0), value=0)

    X = torch.randn(batch, seqlen, heads, proj, device=device, dtype=torch.float32)
    dt = torch.rand(batch, seqlen, heads, device=device, dtype=torch.float32)
    A = torch.randn(heads, proj, device=device, dtype=torch.float32) * -0.01
    B = torch.randn(batch, heads, proj, device=device, dtype=torch.float32)
    C = torch.randn(batch, seqlen, heads, proj, device=device, dtype=torch.float32)
    D = torch.randn(heads, proj, device=device, dtype=torch.float32) * 0.1
    z = torch.randn(batch, seqlen, heads, 1, device=device, dtype=torch.float32)
    init_state = torch.randn(batch, heads, proj, device=device, dtype=torch.float32)

    seq_meta = {"cu_seqlens": cu_seqlens.to(device="cpu")}

    ref_out = reference_ops.ssd_chunk_scan(
        X.cpu(),
        dt.cpu(),
        A.cpu(),
        B.cpu(),
        C.cpu(),
        chunk_size,
        D=D.cpu(),
        z=z.cpu(),
        seq_meta=seq_meta,
        initial_states=init_state.cpu(),
    ).to(device)

    cuda_out = ops.ssd_chunk_scan(
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

    torch.testing.assert_close(cuda_out, ref_out, atol=1e-4, rtol=1e-4)


def test_ssd_chunk_scan_cuda_performance_smoke() -> None:
    torch.manual_seed(4)
    device = torch.device("cuda")

    batch, seqlen, heads, proj = 4, 256, 8, 64
    chunk_size = 64

    X = torch.randn(batch, seqlen, heads, proj, device=device, dtype=torch.float16)
    dt = torch.rand(batch, seqlen, heads, device=device, dtype=torch.float16)
    A = torch.randn(heads, proj, device=device, dtype=torch.float32) * -0.02
    B = torch.randn(heads, proj, device=device, dtype=torch.float16)
    C = torch.randn(heads, proj, device=device, dtype=torch.float16)

    warmup_iters, iters = 2, 5
    for _ in range(warmup_iters):
        ops.ssd_chunk_scan(X, dt, A, B, C, chunk_size)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(iters):
        ops.ssd_chunk_scan(X, dt, A, B, C, chunk_size)
    end_event.record()
    end_event.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event) / iters
    assert elapsed_ms > 0
    assert elapsed_ms < 2000


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


def test_selective_state_step_cuda_backward_matches_reference() -> None:
    torch.manual_seed(3)
    device = torch.device("cuda")
    batch, dim, state_dim = 1, 2, 3

    base_state = torch.randn(batch, dim, state_dim, device=device, requires_grad=True)
    x = torch.randn(batch, dim, device=device, requires_grad=True)
    dt = torch.randn(batch, dim, device=device, requires_grad=True)
    A = torch.randn(
        dim, state_dim, device=device, dtype=torch.float32, requires_grad=True
    )
    B = torch.randn(
        dim, state_dim, device=device, dtype=torch.float32, requires_grad=True
    )
    C = torch.randn(
        dim, state_dim, device=device, dtype=torch.float32, requires_grad=True
    )
    D = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
    z = torch.randn(batch, dim, device=device, requires_grad=True)

    ref_inputs = [
        base_state.detach().clone().requires_grad_(True),
        x.detach().clone().requires_grad_(True),
        dt.detach().clone().requires_grad_(True),
        A.detach().clone().requires_grad_(True),
        B.detach().clone().requires_grad_(True),
        C.detach().clone().requires_grad_(True),
        D.detach().clone().requires_grad_(True),
        z.detach().clone().requires_grad_(True),
    ]
    ref_out = reference_ops.selective_state_step(
        ref_inputs[0],
        ref_inputs[1],
        ref_inputs[2],
        ref_inputs[3],
        ref_inputs[4],
        ref_inputs[5],
        D=ref_inputs[6],
        z=ref_inputs[7],
        softplus=False,
    )
    ref_out.sum().backward()
    ref_grads = [_clone_grad(tensor) for tensor in ref_inputs]

    state_cuda = base_state.detach().clone().requires_grad_(True)
    x_cuda = x.detach().clone().requires_grad_(True)
    dt_cuda = dt.detach().clone().requires_grad_(True)
    A_cuda = A.detach().clone().requires_grad_(True)
    B_cuda = B.detach().clone().requires_grad_(True)
    C_cuda = C.detach().clone().requires_grad_(True)
    D_cuda = D.detach().clone().requires_grad_(True)
    z_cuda = z.detach().clone().requires_grad_(True)

    out_cuda = cast(
        torch.Tensor,
        ops.selective_state_step(
            state_cuda,
            x_cuda,
            dt_cuda,
            A_cuda,
            B_cuda,
            C_cuda,
            D=D_cuda,
            z=z_cuda,
            softplus=False,
        ),
    )
    out_cuda.sum().backward()

    cuda_grads = [
        _clone_grad(state_cuda),
        _clone_grad(x_cuda),
        _clone_grad(dt_cuda),
        _clone_grad(A_cuda),
        _clone_grad(B_cuda),
        _clone_grad(C_cuda),
        _clone_grad(D_cuda),
        _clone_grad(z_cuda),
    ]

    for grad_cuda, grad_ref in zip(cuda_grads, ref_grads):
        torch.testing.assert_close(grad_cuda, grad_ref, atol=1e-4, rtol=1e-4)


def test_selective_state_step_cuda_grouped_gradients_match_reference() -> None:
    torch.manual_seed(4)
    device = torch.device("cuda")
    batch, dim, state_dim, groups = 2, 4, 3, 2

    base_state = torch.randn(
        batch, dim, state_dim, device=device, dtype=torch.float32, requires_grad=True
    )
    x = torch.randn(batch, dim, device=device, dtype=torch.float32, requires_grad=True)
    dt = torch.randn(batch, dim, device=device, dtype=torch.float32, requires_grad=True)
    A = torch.randn(
        dim, state_dim, device=device, dtype=torch.float32, requires_grad=True
    )
    B = torch.randn(
        batch, groups, state_dim, device=device, dtype=torch.float32, requires_grad=True
    )
    C = torch.randn(
        batch, groups, state_dim, device=device, dtype=torch.float32, requires_grad=True
    )
    D = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
    z = torch.randn(batch, dim, device=device, dtype=torch.float32, requires_grad=True)
    dt_bias = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)

    ref_inputs = [
        base_state.detach().clone().requires_grad_(True),
        x.detach().clone().requires_grad_(True),
        dt.detach().clone().requires_grad_(True),
        A.detach().clone().requires_grad_(True),
        B.detach().clone().requires_grad_(True),
        C.detach().clone().requires_grad_(True),
        D.detach().clone().requires_grad_(True),
        z.detach().clone().requires_grad_(True),
        dt_bias.detach().clone().requires_grad_(True),
    ]

    ref_out = reference_ops.selective_state_step(
        ref_inputs[0],
        ref_inputs[1],
        ref_inputs[2],
        ref_inputs[3],
        ref_inputs[4],
        ref_inputs[5],
        D=ref_inputs[6],
        z=ref_inputs[7],
        dt_bias=ref_inputs[8],
        softplus=True,
    )
    ref_out.sum().backward()
    ref_grads = [_clone_grad(tensor) for tensor in ref_inputs]

    state_cuda = base_state.detach().clone().requires_grad_(True)
    x_cuda = x.detach().clone().requires_grad_(True)
    dt_cuda = dt.detach().clone().requires_grad_(True)
    A_cuda = A.detach().clone().requires_grad_(True)
    B_cuda = B.detach().clone().requires_grad_(True)
    C_cuda = C.detach().clone().requires_grad_(True)
    D_cuda = D.detach().clone().requires_grad_(True)
    z_cuda = z.detach().clone().requires_grad_(True)
    dt_bias_cuda = dt_bias.detach().clone().requires_grad_(True)

    out_cuda = cast(
        torch.Tensor,
        ops.selective_state_step(
            state_cuda,
            x_cuda,
            dt_cuda,
            A_cuda,
            B_cuda,
            C_cuda,
            D=D_cuda,
            z=z_cuda,
            dt_bias=dt_bias_cuda,
            softplus=True,
        ),
    )
    out_cuda.sum().backward()

    cuda_grads = [
        _clone_grad(state_cuda),
        _clone_grad(x_cuda),
        _clone_grad(dt_cuda),
        _clone_grad(A_cuda),
        _clone_grad(B_cuda),
        _clone_grad(C_cuda),
        _clone_grad(D_cuda),
        _clone_grad(z_cuda),
        _clone_grad(dt_bias_cuda),
    ]

    for grad_cuda, grad_ref in zip(cuda_grads, ref_grads):
        torch.testing.assert_close(grad_cuda, grad_ref, atol=1e-4, rtol=1e-4)


def test_dw_causal_conv_cuda_gradients_match_reference() -> None:
    torch.manual_seed(4)
    device = torch.device("cuda")
    batch, channels, length = 1, 3, 6

    x = torch.randn(batch, channels, length, device=device, requires_grad=True)
    weight = torch.randn(channels, 1, 3, device=device, requires_grad=True)
    bias = torch.randn(channels, device=device, requires_grad=True)

    ref_x = x.detach().clone().requires_grad_(True)
    ref_weight = weight.detach().clone().requires_grad_(True)
    ref_bias = bias.detach().clone().requires_grad_(True)
    ref_out = reference_ops.dw_causal_conv(
        ref_x, ref_weight, bias=ref_bias, activation="silu"
    )
    ref_out.sum().backward()
    ref_grads = [_clone_grad(ref_x), _clone_grad(ref_weight), _clone_grad(ref_bias)]

    out = cast(
        torch.Tensor, ops.dw_causal_conv(x, weight, bias=bias, activation="silu")
    )
    out.sum().backward()
    cuda_grads = [_clone_grad(x), _clone_grad(weight), _clone_grad(bias)]

    for grad_cuda, grad_ref in zip(cuda_grads, ref_grads):
        torch.testing.assert_close(grad_cuda, grad_ref, atol=1e-4, rtol=1e-4)


def test_fused_layer_norm_cuda_gradients_match_reference() -> None:
    torch.manual_seed(5)
    device = torch.device("cuda")
    batch, hidden = 2, 4

    x = torch.randn(batch, hidden, device=device, requires_grad=True)
    weight = torch.randn(hidden, device=device, requires_grad=True)
    bias = torch.randn(hidden, device=device, requires_grad=True)
    residual = torch.randn(batch, hidden, device=device, requires_grad=True)

    ref_inputs = [
        x.detach().clone().requires_grad_(True),
        weight.detach().clone().requires_grad_(True),
        bias.detach().clone().requires_grad_(True),
        residual.detach().clone().requires_grad_(True),
    ]
    ref_out = reference_ops.fused_layer_norm(
        ref_inputs[0],
        ref_inputs[1],
        ref_inputs[2],
        residual=ref_inputs[3],
        is_rms=False,
        eps=1e-5,
        prenorm=True,
        residual_in_fp32=True,
    )
    ref_out.sum().backward()
    ref_grads = [_clone_grad(tensor) for tensor in ref_inputs]

    out = cast(
        torch.Tensor,
        ops.fused_layer_norm(
            x,
            weight,
            bias,
            residual=residual,
            is_rms=False,
            eps=1e-5,
            prenorm=True,
            residual_in_fp32=True,
        ),
    )
    out.sum().backward()
    cuda_grads = [_clone_grad(tensor) for tensor in (x, weight, bias, residual)]

    for grad_cuda, grad_ref in zip(cuda_grads, ref_grads):
        torch.testing.assert_close(grad_cuda, grad_ref, atol=1e-4, rtol=1e-4)

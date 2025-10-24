from functools import partial

import pytest
import torch

from ssm.ops import (
    dw_causal_conv,
    fused_layer_norm,
    selective_scan,
    selective_state_step,
    ssd_chunk_scan,
)


@pytest.fixture
def ragged_seq_meta():
    """Provide simple ragged metadata with two sequences of lengths 2 and 4."""

    lengths = torch.tensor([2, 4], dtype=torch.long)
    cu_seqlens = torch.zeros(len(lengths) + 1, dtype=torch.long)
    cu_seqlens[1:] = torch.cumsum(lengths, dim=0)
    return {"seq_lens": lengths.tolist(), "cu_seqlens": cu_seqlens}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_selective_scan_shapes_and_dtype(dtype):
    torch.manual_seed(0)
    B, D, L, N = 2, 3, 5, 4
    u = torch.randn(B, D, L, dtype=dtype)
    delta = torch.randn(B, D, L, dtype=dtype)
    A = torch.randn(D, N, dtype=torch.float32)
    Bm = torch.randn(D, N, dtype=torch.float32)
    Cm = torch.randn(D, N, dtype=torch.float32)
    dt_bias = torch.randn(D, dtype=torch.float32)
    z = torch.randn(B, D, L, dtype=dtype)

    out, last_state = selective_scan(
        u,
        delta,
        A,
        Bm,
        Cm,
        D=torch.randn(D, dtype=torch.float32),
        z=z,
        dt_bias=dt_bias,
        softplus=True,
        return_last_state=True,
    )

    assert out.shape == (B, D, L)
    assert last_state.shape == (B, D, N)
    assert out.dtype == dtype
    assert last_state.dtype == dtype


def test_selective_scan_gradcheck():
    torch.manual_seed(1)
    B, D, L, N = 1, 2, 3, 2
    u = torch.randn(B, D, L, dtype=torch.float64, requires_grad=True)
    delta = torch.randn(B, D, L, dtype=torch.float64, requires_grad=True)
    A = torch.randn(D, N, dtype=torch.float64)
    Bm = torch.randn(D, N, dtype=torch.float64)
    Cm = torch.randn(D, N, dtype=torch.float64)

    func = partial(
        selective_scan, A=A, B=Bm, C=Cm, D=None, z=None, dt_bias=None, softplus=False
    )
    assert torch.autograd.gradcheck(func, (u, delta), eps=1e-6, atol=1e-6, rtol=1e-6)


def test_selective_state_step_updates_state_and_grad():
    torch.manual_seed(2)
    B, D, N = 2, 3, 4
    state = torch.zeros(B, D, N, dtype=torch.float32)
    x = torch.randn(B, D, dtype=torch.float32, requires_grad=True)
    dt = torch.randn(B, D, dtype=torch.float32)
    A = torch.randn(D, N, dtype=torch.float32)
    Bm = torch.randn(D, N, dtype=torch.float32)
    Cm = torch.randn(D, N, dtype=torch.float32)

    out = selective_state_step(state, x, dt, A, Bm, Cm)

    assert out.shape == (B, D)
    assert state.abs().sum() > 0

    loss = out.sum()
    loss.backward()
    assert x.grad is not None


def test_selective_state_step_accepts_grouped_projections():
    torch.manual_seed(20)
    B, D, N, groups = 2, 4, 3, 2
    state = torch.zeros(B, D, N, dtype=torch.float32)
    x = torch.randn(B, D, dtype=torch.float32)
    dt = torch.randn(B, D, dtype=torch.float32)
    A = torch.randn(D, N, dtype=torch.float32)
    B_grouped = torch.randn(B, groups, N, dtype=torch.float32)
    C_grouped = torch.randn(B, groups, N, dtype=torch.float32)

    out = selective_state_step(state, x, dt, A, B_grouped, C_grouped)

    assert out.shape == (B, D)
    assert state.abs().sum() > 0


def test_ssd_chunk_scan_matches_across_chunk_sizes(ragged_seq_meta):
    torch.manual_seed(3)
    B, L, H, P = 2, 5, 2, 3
    X = torch.randn(B, L, H, P, dtype=torch.float32)
    dt = torch.randn(B, L, H, dtype=torch.float32)
    A = torch.randn(H, P, dtype=torch.float32)
    Bm = torch.randn(H, P, dtype=torch.float32)
    Cm = torch.randn(H, P, dtype=torch.float32)

    out_small = ssd_chunk_scan(X, dt, A, Bm, Cm, chunk_size=1, seq_meta=ragged_seq_meta)
    out_large = ssd_chunk_scan(X, dt, A, Bm, Cm, chunk_size=4, seq_meta=ragged_seq_meta)

    assert torch.allclose(out_small, out_large)
    assert out_small.dtype == X.dtype

    lengths = torch.tensor(ragged_seq_meta["seq_lens"], dtype=torch.long)
    for b, length in enumerate(lengths):
        assert torch.all(out_small[b, int(length.item()) :] == 0)


def test_ssd_chunk_scan_gradcheck():
    torch.manual_seed(4)
    B, L, H, P = 1, 3, 1, 2
    X = torch.randn(B, L, H, P, dtype=torch.float64, requires_grad=True)
    dt = torch.randn(B, L, H, dtype=torch.float64, requires_grad=True)
    A = torch.randn(H, P, dtype=torch.float64)
    Bm = torch.randn(H, P, dtype=torch.float64)
    Cm = torch.randn(H, P, dtype=torch.float64)

    func = partial(
        ssd_chunk_scan,
        A=A,
        B=Bm,
        C=Cm,
        chunk_size=2,
        D=None,
        z=None,
        seq_meta=None,
        initial_states=None,
    )
    assert torch.autograd.gradcheck(func, (X, dt), eps=1e-6, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("layout", ["channels_first", "channels_last"])
def test_dw_causal_conv_layout_and_grad(layout):
    torch.manual_seed(5)
    B, C, L = 2, 3, 6
    kernel = 3
    if layout == "channels_first":
        x = torch.randn(B, C, L, dtype=torch.float32, requires_grad=True)
    else:
        x = torch.randn(B, L, C, dtype=torch.float32, requires_grad=True)

    weight = torch.randn(C, kernel, dtype=torch.float32)
    bias = torch.randn(C, dtype=torch.float32)

    out = dw_causal_conv(x, weight, bias=bias, activation="relu")
    assert out.shape == x.shape

    out.sum().backward()
    assert x.grad is not None


@pytest.mark.parametrize("is_rms", [False, True])
@pytest.mark.parametrize("prenorm", [True, False])
def test_fused_layer_norm_matches_torch(is_rms, prenorm):
    torch.manual_seed(6)
    B, L, D = 2, 4, 3
    x = torch.randn(B, L, D, dtype=torch.float32, requires_grad=True)
    residual = torch.randn(B, L, D, dtype=torch.float32)
    weight = torch.randn(D, dtype=torch.float32)
    bias = torch.randn(D, dtype=torch.float32)

    out = fused_layer_norm(
        x, weight, bias, residual=residual, is_rms=is_rms, prenorm=prenorm
    )

    if is_rms:
        ref_in = x + residual if prenorm else x
        norm = ref_in * torch.rsqrt(ref_in.pow(2).mean(-1, keepdim=True) + 1e-5)
        ref = norm * weight + bias
        if not prenorm:
            ref = ref + residual
    else:
        ref_in = x + residual if prenorm else x
        mean = ref_in.mean(-1, keepdim=True)
        var = (ref_in - mean).pow(2).mean(-1, keepdim=True)
        norm = (ref_in - mean) * torch.rsqrt(var + 1e-5)
        ref = norm * weight + bias
        if not prenorm:
            ref = ref + residual

    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)

    out.sum().backward()
    assert x.grad is not None

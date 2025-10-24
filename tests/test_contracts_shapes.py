import pytest
import torch

from ssm.ops import (
    selective_scan,
    selective_state_step,
    ssd_chunk_scan,
)


def test_selective_scan_contract_raises():
    B, D, L, N = 2, 8, 16, 4
    u = torch.randn(B, D, L)
    delta = torch.randn(B, D, L + 1)
    A = torch.randn(D, N)
    Bm = torch.randn(D, N)
    Cm = torch.randn(D, N)
    with pytest.raises(ValueError):
        selective_scan(u, delta, A, Bm, Cm)


def test_selective_state_step_contract_raises():
    B, D, N = 2, 8, 4
    state = torch.randn(B, D, N)
    x = torch.randn(B, D + 1)
    dt = torch.randn(B, D)
    A = torch.randn(D, N)
    Bv = torch.randn(D, N)
    Cv = torch.randn(D, N)
    with pytest.raises(ValueError):
        selective_state_step(state, x, dt, A, Bv, Cv)


def test_ssd_chunk_scan_contract_raises():
    B, L, H, P = 2, 64, 4, 8
    X = torch.randn(B, L, H, P)
    dt = torch.randn(B, L + 1, H)
    A = torch.randn(H)
    Bv = torch.randn(H, P)
    Cv = torch.randn(H, P)
    with pytest.raises(ValueError):
        ssd_chunk_scan(X, dt, A, Bv, Cv, chunk_size=32)
